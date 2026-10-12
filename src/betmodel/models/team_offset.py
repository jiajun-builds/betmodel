"""A per-club correction to the model, learned from where the market closed (D38).

The xG ratings of some clubs sit off the market's for a whole season at a time --
in 2026 CSL the model rated Qingdao Hainiu, Tianjin and Qingdao West Coast above
Pinnacle in both halves, and Shanghai Port below it -- and the market is the one
that is right. Its close moves further from the model's view of those clubs, not
toward it, and the error is the club's rather than the method's: shuffling the
club labels before fitting makes the forecast worse, not better.

So each match the model priced in the last ``window_months`` leaves a residual,

    logit(close home share) - logit(model home share)

in the no-vig close of ``team_offset.book``, and a weighted ridge regression
reads it as ``kappa + delta_home - delta_away``. Recent residuals weigh more
(``exp(-xi * age_days)``); ``lam`` pulls every club's delta toward zero, which is
where a club with little history stays. ``kappa`` is league-wide and unpenalised.

**The residual is against what the model said at the time.** That is why this
module refits the model before every past match day rather than asking the
current fit about matches it was trained on: an in-sample prediction has already
absorbed the match it is being compared with, and measured on 2024-25 that
version was worse than no correction at all.

The correction moves only the home/away split. The draw stays the model's.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from betmodel.config.schema import LeagueConfig, TeamOffset
from betmodel.dates import parse_date_only_series
from betmodel.models.dc import FitError, fit_production_model, prepare_training_frame
from betmodel.models.poisson import Outcome
from betmodel.signals.ev import devig

log = logging.getLogger(__name__)

#: Results a played match can carry. Anything else is a fixture still to come.
PLAYED = ("H", "D", "A")


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Offsets:
    """The fitted correction. ``Offsets()`` is the identity."""

    kappa: float = 0.0
    deltas: Mapping[str, float] = field(default_factory=dict)
    n_residuals: int = 0

    def shift(self, home: str, away: str) -> float:
        """Logit shift of the home share for this pairing. A club with no history
        contributes nothing."""
        if not self.deltas:
            return 0.0
        return self.kappa + self.deltas.get(home, 0.0) - self.deltas.get(away, 0.0)

    def adjust(self, outcome: Outcome, home: str, away: str) -> Outcome:
        """Move the home/away split by the shift; the draw is left as it is."""
        shift = self.shift(home, away)
        if shift == 0.0:
            return outcome
        decisive = outcome.home + outcome.away
        if decisive <= 0.0 or outcome.home <= 0.0 or outcome.away <= 0.0:
            return outcome
        share = _sigmoid(_logit(outcome.home / decisive) + shift)
        return Outcome(home=decisive * share, draw=outcome.draw,
                       away=decisive * (1.0 - share))


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #

def played_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Played matches, oldest first, with their day parsed into ``_day``."""
    frame = frame.copy()
    frame["_day"] = parse_date_only_series(frame["Date"])
    frame = frame[frame["Res"].isin(PLAYED)]
    return frame.sort_values("_day", kind="stable")


def walk_forward(history: pd.DataFrame, config: LeagueConfig,
                 days: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Price every match on each of ``days`` from a fit on strictly earlier matches.

    ``history`` comes from ``played_history``. Returns one row per priced match,
    indexed like ``history``, with ``raw_h/raw_d/raw_a`` and the weaker club's
    weighted evidence. A day the model cannot be fitted for is skipped, and so is
    a match with a club absent from that day's training window.
    """
    priced = []
    # One refit per past match day, each of which repeats the fitter's own
    # per-fit log lines (a dropped row, the fitted lambdas). The production fit
    # says them once; a hundred copies here would bury whatever else the run has
    # to say, so the fitter is quietened for the loop and nothing else.
    fitter_log = logging.getLogger("betmodel.models.dc")
    level = fitter_log.level
    fitter_log.setLevel(logging.ERROR)
    try:
        for day in days:
            priced.extend(_price_day(history, config, day))
    finally:
        fitter_log.setLevel(level)
    if not priced:
        return pd.DataFrame(columns=["raw_h", "raw_d", "raw_a", "evidence"])
    return pd.DataFrame(priced).set_index("index")


def _price_day(history: pd.DataFrame, config: LeagueConfig, day) -> list[dict]:
    try:
        train = prepare_training_frame(
            history[history["_day"] < day].drop(columns="_day"), config)
        fit = fit_production_model(train, config)
    except FitError:
        return []
    evidence = dict(zip(fit.teams, fit.model.appearances()[1]))
    priced = []
    for index, row in history[history["_day"] == day].iterrows():
        if row["Home"] not in evidence or row["Away"] not in evidence:
            continue
        raw = fit.model.outcome_probabilities(row["Home"], row["Away"])
        priced.append({
            "index": index, "raw_h": raw.home, "raw_d": raw.draw, "raw_a": raw.away,
            "evidence": min(evidence[row["Home"]], evidence[row["Away"]]),
        })
    return priced


def residuals(history: pd.DataFrame, priced: pd.DataFrame,
              config: LeagueConfig) -> pd.DataFrame:
    """``(day, home, away, residual)`` for every priced match with a usable close."""
    prefix = config.odds.book(config.model.team_offset.book).schema_prefix
    joined = priced.join(history, how="inner")
    rows = []
    for _, row in joined.iterrows():
        close = tuple(row.get(f"{prefix}_close_{leg}") for leg in "hda")
        if not all(pd.notna(o) and float(o) > 1.0 for o in close):
            continue
        market = devig(tuple(float(o) for o in close))
        decisive = row["raw_h"] + row["raw_a"]
        rows.append({
            "day": row["_day"], "home": row["Home"], "away": row["Away"],
            "residual": _logit(market[0] / (market[0] + market[2]))
            - _logit(row["raw_h"] / decisive),
        })
    return pd.DataFrame(rows, columns=["day", "home", "away", "residual"])


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #

def estimate(res: pd.DataFrame, as_of: pd.Timestamp, settings: TeamOffset) -> Offsets:
    """Fit the offsets on residuals of matches played before ``as_of``.

    Only the ``window_months`` before ``as_of`` count, so a residual dated on or
    after it can never reach the correction applied to it.
    """
    as_of = pd.Timestamp(as_of)
    lo = as_of - pd.DateOffset(months=settings.window_months)
    used = res[(res["day"] < as_of) & (res["day"] >= lo)]
    if len(used) < settings.min_history:
        return Offsets(n_residuals=len(used))

    teams = sorted(set(used["home"]) | set(used["away"]))
    position = {t: i for i, t in enumerate(teams)}
    rows = np.arange(len(used))
    design = np.zeros((len(used), len(teams) + 1))
    design[:, -1] = 1.0
    design[rows, [position[t] for t in used["home"]]] += 1.0
    design[rows, [position[t] for t in used["away"]]] -= 1.0

    age = (as_of - pd.to_datetime(used["day"])).dt.days.to_numpy(dtype=float)
    root = np.sqrt(np.exp(-settings.xi * age))
    a = design * root[:, None]
    y = used["residual"].to_numpy(dtype=float) * root
    penalty = np.eye(design.shape[1]) * settings.lam
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ y)
    return Offsets(kappa=float(coef[-1]),
                   deltas={t: float(coef[position[t]]) for t in teams},
                   n_residuals=len(used))


def from_history(frame: pd.DataFrame, config: LeagueConfig,
                 as_of: pd.Timestamp) -> Offsets:
    """The production entry point: refit the window, read the residuals, fit.

    ``frame`` is the match table as read, unplayed fixtures included.
    """
    settings = config.model.team_offset
    if not settings.enabled:
        return Offsets()
    as_of = pd.Timestamp(as_of)
    history = played_history(frame)
    lo = as_of - pd.DateOffset(months=settings.window_months)
    days = sorted(history.loc[(history["_day"] >= lo) & (history["_day"] < as_of),
                              "_day"].unique())
    offsets = estimate(residuals(history, walk_forward(history, config, days), config),
                       as_of, settings)
    log.info("team offsets as of %s: kappa %+.3f over %d residuals, %d clubs",
             as_of.date(), offsets.kappa, offsets.n_residuals, len(offsets.deltas))
    return offsets
