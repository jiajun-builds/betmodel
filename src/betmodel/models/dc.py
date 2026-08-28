"""The production fit recipe, in one place.

Both production and every research harness go through here, so a change to the
fit cannot silently apply to one and not the other. That matters more than it
sounds: the recipe used to be re-declared in eight backtest modules, which is how
a 27% error in the fitted scoring rate survived across five separate copies of
the same fit with nothing forcing them to agree.

Everything league-specific arrives as configuration. The function does not know
which league it is fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from penaltyblog.models import dixon_coles_weights

from betmodel.config.schema import LeagueConfig
from betmodel.dates import parse_date_only_series
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
    a season ended.
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

@dataclass
class ProductionFit:
    """A fitted model plus what it took to fit it."""

    model: ContinuousPoissonModel
    weights: np.ndarray
    n_matches: int
    shrunk: bool
    raw_attack: np.ndarray
    raw_defence: np.ndarray

    @property
    def teams(self):
        return self.model.teams

    def predict(self, home_team: str, away_team: str):
        return self.model.predict(home_team, away_team)

    def lambdas(self, home_team: str, away_team: str):
        return self.model.lambdas(home_team, away_team)


def fit_production_model(
    train: pd.DataFrame,
    config: LeagueConfig,
    *,
    target: tuple[str, str] = TARGET_COLUMNS,
    options: FitOptions = DEFAULT,
    max_goals: int | None = None,
    shrink: bool | None = None,
) -> ProductionFit:
    """Fit on an already-prepared training frame.

    ``options`` and ``max_goals`` are overridable so the merge can reproduce
    either pre-merge configuration exactly and measure the difference, rather
    than assert that a convex problem must land in the same place.
    """
    home_target, away_target = target
    weights = np.asarray(dixon_coles_weights(train["Date"], xi=config.model.xi), dtype=float)

    model = ContinuousPoissonModel(
        train[home_target].to_numpy(dtype=float),
        train[away_target].to_numpy(dtype=float),
        train["Home"],
        train["Away"],
        weights,
        max_goals=config.model.max_goals if max_goals is None else max_goals,
        options=options,
    )
    model.fit()

    # The Poisson score equation must hold at the optimum. Cheap, and it is the
    # check whose absence let a 27% scale error ship.
    ratio_home, ratio_away = model.score_equation_ratio()
    if (abs(ratio_home - 1.0) > SCORE_EQUATION_TOLERANCE
            or abs(ratio_away - 1.0) > SCORE_EQUATION_TOLERANCE):
        raise FitError(
            "Poisson score equation violated: sum(w*lambda)/sum(w*y) = "
            f"({ratio_home:.6f}, {ratio_away:.6f}), expected 1.0. Either the fit "
            "is not at its optimum, or goal targets are being coerced somewhere."
        )

    raw_attack = np.array(model.attack, copy=True)
    raw_defence = np.array(model.defence, copy=True)

    do_shrink = config.model.shrinkage.enabled if shrink is None else shrink
    if do_shrink:
        eff = effective_n(model.teams, weights, train["Home"], train["Away"])
        k = config.model.shrinkage.k
        model.attack = shrink_ratings(model.attack, eff, k)
        model.defence = shrink_ratings(model.defence, eff, k)

    log.info(
        "fitted %d matches: mean lambda %.3f/%.3f, score eq (%.6f, %.6f), shrink=%s",
        len(train), model.lambda_home.mean(), model.lambda_away.mean(),
        ratio_home, ratio_away, do_shrink,
    )
    return ProductionFit(
        model=model, weights=weights, n_matches=len(train), shrunk=bool(do_shrink),
        raw_attack=raw_attack, raw_defence=raw_defence,
    )


def fit_from_csv(
    path: str, config: LeagueConfig, **kwargs
) -> tuple[ProductionFit, pd.DataFrame]:
    """Load, prepare and fit. Returns the fit and the training frame used."""
    df = pd.read_csv(path)
    train = prepare_training_frame(df, config, source=path)
    return fit_production_model(train, config, **kwargs), train
