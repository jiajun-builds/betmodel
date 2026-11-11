"""The production fit recipe, in one place.

Both production and every research harness go through here, so a change to the
fit cannot silently apply to one and not the other. That matters more than it
sounds: the recipe used to be re-declared in eight backtest modules, which is how
a 27% error in the fitted scoring rate survived across five separate copies of
the same fit with nothing forcing them to agree.

Everything league-specific arrives as configuration. The function does not know
which league it is fitting.

Auxiliary matches (D42) come in through ``fit_production_model(auxiliary=...)``
already selected by ``models.auxiliary.select``, and only there. The training
frame, its window, its as-of date and its Dixon-Coles weights stay the primary
league's alone, and the fit reports its clubs and their evidence from primary
matches only: a club seen only in the other division is not priced, and the other
division never counts towards `min_team_evidence`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from penaltyblog.models import dixon_coles_weights

from betmodel.config.schema import LeagueConfig
from betmodel.dates import parse_date_only_series
from betmodel.models.auxiliary import AuxiliaryRows, select_all
from betmodel.models.poisson import DEFAULT, ContinuousPoissonModel, FitOptions

log = logging.getLogger(__name__)

#: Column pair the model is fitted against: the xG blend, not raw goals.
TARGET_COLUMNS = ("HExpG+", "AExpG+")

#: How far the Poisson score equation may drift from 1.0 before the fit is
#: rejected. This is the assertion whose absence let the truncation bug live.
SCORE_EQUATION_TOLERANCE = 1e-3


class FitError(RuntimeError):
    """The fit did not reach a usable optimum."""


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #

def prepare_training_frame(
    df: pd.DataFrame,
    config: LeagueConfig,
    *,
    target: tuple[str, str] = TARGET_COLUMNS,
    source: str = "",
) -> pd.DataFrame:
    """Window and clean a match history into a training frame.

    The window is anchored on the latest match in the data, not on today, so a
    refit is reproducible from a frozen file and does not quietly shrink because
    a season ended. The cutoff is recorded in ``attrs["window_cutoff"]`` so an
    auxiliary competition can be held to the same window (D42); it is taken
    before blank targets are dropped, so the returned frame cannot re-derive it.
    """
    df = df.copy()
    raw_dates = df["Date"].copy()
    df["Date"] = parse_date_only_series(df["Date"])
    unparseable = df["Date"].isna()
    if unparseable.any():
        examples = raw_dates.loc[unparseable].astype(str).head(10).tolist()
        raise FitError(f"unparseable Date values in {source or 'match history'}: {examples}")

    df = df.dropna(subset=["Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)

    cutoff = df["Date"].max() - pd.DateOffset(months=config.model.lookback_months)
    df = df[df["Date"] >= cutoff]

    home_target, away_target = target
    before = len(df)
    df = df.dropna(subset=[home_target, away_target])
    dropped = before - len(df)
    if dropped:
        log.warning("dropped %d training rows with missing %s/%s",
                    dropped, home_target, away_target)

    if len(df) < config.model.min_train:
        raise FitError(
            f"only {len(df)} training matches after the "
            f"{config.model.lookback_months}-month window; "
            f"min_train is {config.model.min_train}"
        )
    df.attrs["window_cutoff"] = cutoff
    return df


# --------------------------------------------------------------------------- #
# shrinkage
# --------------------------------------------------------------------------- #

def effective_n(teams, weights, home_series, away_series) -> np.ndarray:
    """Dixon-Coles-weighted match count per team, in ``teams`` order."""
    counts = {t: 0.0 for t in teams}
    for w, home, away in zip(np.asarray(weights, dtype=float), home_series, away_series):
        if home in counts:
            counts[home] += w
        if away in counts:
            counts[away] += w
    return np.array([counts[t] for t in teams])


def shrink_ratings(values: np.ndarray, eff_n: np.ndarray, k: float) -> np.ndarray:
    """Pull per-team ratings toward the sample-size-weighted league mean.

    A club with little evidence moves a long way, a well-sampled one barely
    moves. The weighted mean is restored afterwards so only the spread is
    regularised and the league's overall scoring level is untouched.
    """
    if eff_n.sum() <= 0:
        return values
    target = np.average(values, weights=eff_n)
    weight = eff_n / (eff_n + k)
    shrunk = target + weight * (values - target)
    shrunk += target - np.average(shrunk, weights=eff_n)
    return shrunk


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AuxiliaryUse:
    """What one auxiliary competition contributed to a fit."""

    id: str
    rows: int
    weight: float
    own_level: bool
    #: Clubs with matches in both this competition and the primary's window.
    #: With ``own_level`` they are the only thing identifying the level shift.
    bridges: tuple[str, ...]
    #: Log-rate shifts relative to the primary; ``None`` without ``own_level``.
    #: The level shift is the intercept's, *given* the clubs' ratings, which are
    #: centred over both divisions together. It is not by itself the difference
    #: in scoring rate between them, and its sign need not say which scores more.
    level_shift: float | None = None
    home_shift: float | None = None


@dataclass
class ProductionFit:
    """A fitted model plus what it took to fit it.

    ``weights`` and ``n_matches`` are the primary training frame's, row for row.
    ``teams`` are the clubs with at least one primary match, and ``attack``,
    ``defence`` and the evidence arrays are in that order. The model underneath
    may know more clubs when auxiliary matches were pooled; ``raw_attack`` and
    ``raw_defence`` are in ``model.teams`` order.
    """

    model: ContinuousPoissonModel
    weights: np.ndarray
    n_matches: int
    shrunk: bool
    raw_attack: np.ndarray
    raw_defence: np.ndarray
    primary_teams: np.ndarray | None = None
    #: ``(counts, weighted)`` from primary matches, at their undiluted weight.
    primary_evidence: tuple[np.ndarray, np.ndarray] | None = None
    #: ``(counts, weighted)`` from auxiliary matches, at the weight they fitted
    #: with (Dixon-Coles times the source's ``weight``).
    auxiliary_evidence: tuple[np.ndarray, np.ndarray] | None = None
    auxiliary: tuple[AuxiliaryUse, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.primary_teams is None:
            self.primary_teams = self.model.teams
        if self.primary_evidence is None:
            self.primary_evidence = self.model.appearances()
        if self.auxiliary_evidence is None:
            zeros = np.zeros(len(self.primary_teams))
            self.auxiliary_evidence = (zeros, zeros.copy())
        self._index = np.array(
            [self.model._team_index[t] for t in self.primary_teams], dtype=np.int64)

    @property
    def teams(self):
        return self.primary_teams

    @property
    def attack(self) -> np.ndarray:
        return np.asarray(self.model.attack)[self._index]

    @property
    def defence(self) -> np.ndarray:
        return np.asarray(self.model.defence)[self._index]

    def _require_primary(self, *clubs: str) -> None:
        # The model underneath knows auxiliary-only clubs too; they are not priced.
        for club in clubs:
            if club not in self._index_of:
                raise KeyError(f"team not in the primary training window: {club!r}")

    @property
    def _index_of(self) -> set[str]:
        return set(str(t) for t in self.primary_teams)

    def predict(self, home_team: str, away_team: str):
        self._require_primary(home_team, away_team)
        return self.model.predict(home_team, away_team)

    def lambdas(self, home_team: str, away_team: str):
        self._require_primary(home_team, away_team)
        return self.model.lambdas(home_team, away_team)


def fit_production_model(
    train: pd.DataFrame,
    config: LeagueConfig,
    *,
    auxiliary: Sequence[AuxiliaryRows] | None = None,
    target: tuple[str, str] = TARGET_COLUMNS,
    options: FitOptions = DEFAULT,
    max_goals: int | None = None,
    shrink: bool | None = None,
) -> ProductionFit:
    """Fit on an already-prepared training frame.

    ``options`` and ``max_goals`` are overridable so the merge can reproduce
    either pre-merge configuration exactly and measure the difference, rather
    than assert that a convex problem must land in the same place.

    ``auxiliary`` is matches from other competitions, already chosen for this
    fit by ``models.auxiliary.select``. Empty or absent, the fit is the plain one
    to the last bit. Each auxiliary row is weighted by its Dixon-Coles weight
    against the *primary's* latest match, times its source's ``weight``, so the
    primary rows' own weights are untouched by its presence.
    """
    home_target, away_target = target
    weights = np.asarray(dixon_coles_weights(train["Date"], xi=config.model.xi), dtype=float)
    pooled = [rows for rows in (auxiliary or ()) if len(rows.frame)]
    max_goals = config.model.max_goals if max_goals is None else max_goals

    if not pooled:
        model = ContinuousPoissonModel(
            train[home_target].to_numpy(dtype=float),
            train[away_target].to_numpy(dtype=float),
            train["Home"],
            train["Away"],
            weights,
            max_goals=max_goals,
            options=options,
        )
        all_weights, all_home, all_away = weights, train["Home"], train["Away"]
        primary = np.ones(len(train), dtype=bool)
    else:
        reference = train["Date"].max()
        parts = [train[["Home", "Away", home_target, away_target]].astype(
            {"Home": str, "Away": str})]
        part_weights, part_groups = [weights], [np.zeros(len(train), dtype=np.int64)]
        for label, rows in enumerate(pooled, start=1):
            frame = rows.frame
            age = (reference - frame["Date"]).dt.days.to_numpy(dtype=float)
            part_weights.append(np.exp(-config.model.xi * age) * rows.source.weight)
            part_groups.append(np.full(len(frame), label if rows.source.own_level else 0,
                                       dtype=np.int64))
            parts.append(frame[["Home", "Away", home_target, away_target]].astype(
                {"Home": str, "Away": str}))
        combined = pd.concat(parts, ignore_index=True)
        all_weights = np.concatenate(part_weights)
        all_home, all_away = combined["Home"], combined["Away"]
        primary = np.zeros(len(combined), dtype=bool)
        primary[: len(train)] = True
        model = ContinuousPoissonModel(
            combined[home_target].to_numpy(dtype=float),
            combined[away_target].to_numpy(dtype=float),
            all_home,
            all_away,
            all_weights,
            max_goals=max_goals,
            options=options,
            groups=np.concatenate(part_groups),
            pooled=True,
        )
    model.fit()

    # The Poisson score equation must hold at the optimum. Cheap, and it is the
    # check whose absence let a 27% scale error ship. With auxiliary groups on
    # their own level it holds within each group, so each is checked.
    for group in (0, *model.group_labels):
        ratio_home, ratio_away = model.score_equation_ratio(group)
        if (abs(ratio_home - 1.0) > SCORE_EQUATION_TOLERANCE
                or abs(ratio_away - 1.0) > SCORE_EQUATION_TOLERANCE):
            raise FitError(
                "Poisson score equation violated"
                f"{f' in auxiliary group {group}' if group else ''}: "
                f"sum(w*lambda)/sum(w*y) = ({ratio_home:.6f}, {ratio_away:.6f}), "
                "expected 1.0. Either the fit is not at its optimum, or goal targets "
                "are being coerced somewhere."
            )
    ratio_home, ratio_away = model.score_equation_ratio()

    raw_attack = np.array(model.attack, copy=True)
    raw_defence = np.array(model.defence, copy=True)

    do_shrink = config.model.shrinkage.enabled if shrink is None else shrink
    if do_shrink:
        eff = effective_n(model.teams, all_weights, all_home, all_away)
        k = config.model.shrinkage.k
        model.attack = shrink_ratings(model.attack, eff, k)
        model.defence = shrink_ratings(model.defence, eff, k)

    log.info(
        "fitted %d matches: mean lambda %.3f/%.3f, score eq (%.6f, %.6f), shrink=%s",
        len(train), model.lambda_home[primary].mean(), model.lambda_away[primary].mean(),
        ratio_home, ratio_away, do_shrink,
    )
    if not pooled:
        return ProductionFit(
            model=model, weights=weights, n_matches=len(train), shrunk=bool(do_shrink),
            raw_attack=raw_attack, raw_defence=raw_defence,
        )

    primary_teams = np.sort(np.unique(np.concatenate([
        np.asarray(train["Home"], dtype=str), np.asarray(train["Away"], dtype=str)])))
    index = np.array([model._team_index[t] for t in primary_teams], dtype=np.int64)
    counts, weighted = model.appearances(primary)
    aux_counts, aux_weighted = model.appearances(~primary)
    shifts = model.group_shifts()
    uses = []
    for label, rows in enumerate(pooled, start=1):
        seen = set(rows.frame["Home"].astype(str)) | set(rows.frame["Away"].astype(str))
        bridges = tuple(t for t in primary_teams if t in seen)
        level, home = shifts.get(label, (None, None)) if rows.source.own_level else (None, None)
        uses.append(AuxiliaryUse(
            id=rows.source.id, rows=len(rows.frame), weight=rows.source.weight,
            own_level=rows.source.own_level, bridges=bridges,
            level_shift=level, home_shift=home,
        ))
        if rows.source.own_level and not bridges:
            log.warning(
                "%s: no club has matches in both %s and the primary window, so its "
                "level shift is unidentified and it informs no primary rating",
                config.id, rows.source.id,
            )
        log.info(
            "pooled %d %s matches at weight %.2f, %s, %d bridge clubs%s",
            len(rows.frame), rows.source.id, rows.source.weight,
            "own level" if rows.source.own_level else "primary level", len(bridges),
            f": level {level:+.3f}, home advantage {home:+.3f}" if level is not None else "",
        )
    return ProductionFit(
        model=model, weights=weights, n_matches=len(train), shrunk=bool(do_shrink),
        raw_attack=raw_attack, raw_defence=raw_defence,
        primary_teams=primary_teams,
        primary_evidence=(counts[index], weighted[index]),
        auxiliary_evidence=(aux_counts[index], aux_weighted[index]),
        auxiliary=tuple(uses),
    )


def fit_from_csv(
    path: str, config: LeagueConfig, **kwargs
) -> tuple[ProductionFit, pd.DataFrame]:
    """Load, prepare and fit. Returns the fit and the training frame used.

    Primary-only unless ``auxiliary`` -- the loaded, unselected rows from
    ``models.auxiliary.load_enabled`` -- is passed explicitly. Gate G1 calls this
    with the frozen pre-merge history, and the question it asks is about the
    pre-merge fitter, which never saw League One.
    """
    df = pd.read_csv(path)
    train = prepare_training_frame(df, config, source=path)
    loaded = kwargs.pop("auxiliary", None)
    selected = select_all(loaded, train, config,
                          target=kwargs.get("target", TARGET_COLUMNS)) if loaded else None
    return fit_production_model(train, config, auxiliary=selected, **kwargs), train
