import numpy as np
import pandas as pd
from penaltyblog.models import FootballProbabilityGrid, dixon_coles_weights
from scipy.optimize import minimize_scalar
import os
from datetime import datetime

from csl.date_utils import parse_date_only_series
from csl.models.continuous_poisson import ContinuousPoissonGoalModel

# The production fit recipe, in one place. These used to be re-declared in eight
# backtest modules, which is how a 27% lambda error (see continuous_poisson) went
# unnoticed across five separate copies of the fit: nothing forced the backtests
# and production to agree. Import them; do not re-declare them.
PROD_XI = 0.001  # Dixon-Coles time decay
PROD_LOOKBACK_MONTHS = 18  # training window
# Minimum training fixtures before a backtest scores a match. The dataset starts
# 2023-04-15, so early-2023 matches have almost no history (median 26 fixtures for
# the 2023 rows that carry opening lines — the model there is noise). 100 cleanly
# excludes them; 2024+ always has >=240.
MIN_TRAIN = 100

# Draw de-bias (backtest/backtest.md §12.4): a market-free calibration that scales
# the scoreline-grid diagonal by a factor delta fit on the training window, then
# renormalizes.
#
# RETIRED — delta existed to patch the truncation bug, and does not survive it.
# While goal targets were being truncated (see continuous_poisson), lambda was
# ~27% too low, the grid piled up on low scores and the model OVER-priced the draw
# by +4.20pp; delta fit around 0.90 and suppressed it. On the corrected fit that
# excess is gone, and what remains does not generalize:
#
#   season       2023     2024     2025     2026    pooled
#   draw bias  +0.81pp  -0.23pp  -4.37pp  -3.56pp  -1.97pp
#   best shrink   0.00     0.00     1.00     0.50      0.15
#
# The residual draw bias changes SIGN across seasons, so it is not a structural
# low-score dependence that a diagonal scale could fix — it is season noise, and
# the in-sample delta MLE chases it. Applying the fitted delta (mean 1.258) made
# out-of-sample calibration worse, not better: draw bias -1.97pp -> +2.19pp and
# log-loss 0.9616 -> 0.9649, i.e. it overshot zero. The pooled out-of-sample
# optimum is a shrink of 0.15, i.e. delta ~= 1.04 — indistinguishable from off.
#
# So delta is computed and logged for diagnostics but NOT applied. Set
# DRAW_DELTA_SHRINK above 0.0 only with fresh cross-season evidence that the
# residual has acquired a stable sign; a value tuned on pooled data alone is
# fitting noise. Predictions with fixtures that lack a market anchor now ship the
# raw grid, which measured better on every axis.
DRAW_DELTA_SHRINK = 0.0
DRAW_DELTA_BOUNDS = (0.3, 2.0)
DRAW_DELTA_MIN_ROWS = 20  # below this, calibration is noise: fall back to 1.0


def fit_draw_delta(clf, train: pd.DataFrame, weights) -> float:
    """Fit the diagonal scale delta by maximizing the Dixon-Coles-weighted 1X2
    log-likelihood of the training fixtures. Needs no market data.

    With raw (pH, pD, pA) and z = 1 - pD + delta*pD, the adjusted outcome probs
    are (pH/z, delta*pD/z, pA/z) — identical to scaling the grid diagonal by
    delta and renormalizing, so this scalar fit matches what
    ``DrawCalibratedModel.predict`` applies to the full grid.
    """
    hg = pd.to_numeric(train["HG"], errors="coerce")
    ag = pd.to_numeric(train["AG"], errors="coerce")
    outcome = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    valid = hg.notna().to_numpy() & ag.notna().to_numpy()

    cache: dict = {}
    P, ks, ws = [], [], []
    w_arr = np.asarray(weights)
    n_grid = None
    for i, r in enumerate(train.itertuples(index=False)):
        if not valid[i]:
            continue
        key = (r.Home, r.Away)
        if key not in cache:
            try:
                grid = np.asarray(clf.predict(r.Home, r.Away).grid)
                n_grid = grid.shape[0]
                diff = np.subtract.outer(np.arange(n_grid), np.arange(n_grid))
                v = np.array([grid[diff > 0].sum(), grid[diff == 0].sum(), grid[diff < 0].sum()])
                cache[key] = v / v.sum()
            except Exception:
                cache[key] = None
        if cache[key] is None:
            continue
        P.append(cache[key])
        ks.append(int(outcome[i]))
        ws.append(w_arr[i])
    if len(P) < DRAW_DELTA_MIN_ROWS:
        return 1.0

    P = np.asarray(P)
    k = np.asarray(ks)
    w = np.asarray(ws)
    d = P[:, 1]
    raw = P[np.arange(len(k)), k]

    def nll(delta):
        z = 1.0 - d + delta * d
        pk = np.where(k == 1, delta * d, raw) / z
        return -(w * np.log(np.clip(pk, 1e-12, None))).sum()

    return float(minimize_scalar(nll, bounds=DRAW_DELTA_BOUNDS, method="bounded").x)


class DrawCalibratedModel:
    """A fitted goals model with the §12.4 draw de-bias applied to every prediction.

    Wraps the underlying penaltyblog model; ``predict`` scales the scoreline-grid
    diagonal by ``draw_delta`` and renormalizes, returning a regular
    ``FootballProbabilityGrid`` so downstream consumers (1X2 aggregation,
    ``asian_handicap_probs``) are unaffected by the wrapping.
    """

    def __init__(self, clf, draw_delta: float):
        self._clf = clf
        self.draw_delta = float(draw_delta)

    @property
    def teams(self):
        return self._clf.teams

    @property
    def _params(self):
        return self._clf._params

    def get_params(self):
        return self._clf.get_params()

    def predict(self, home_team: str, away_team: str) -> FootballProbabilityGrid:
        pred = self._clf.predict(home_team, away_team)
        grid = np.asarray(pred.grid, dtype=float).copy()
        idx = np.arange(grid.shape[0])
        grid[idx, idx] *= self.draw_delta
        grid /= grid.sum()
        return FootballProbabilityGrid(
            grid, pred.home_goal_expectation, pred.away_goal_expectation
        )

    def predict_raw(self, home_team: str, away_team: str) -> FootballProbabilityGrid:
        """Prediction WITHOUT the δ draw calibration.

        The market-anchored de-bias (backtest.md §12, roadmap #10) must start
        from the un-calibrated grid — anchoring on top of δ would shrink the
        draw twice. Anchorless consumers should keep using ``predict``.
        """
        return self._clf.predict(home_team, away_team)


def fit_production_model(train: pd.DataFrame, xi: float = PROD_XI) -> "DrawCalibratedModel":
    """Fit the production model on an already-filtered training frame.

    **The single definition of the production recipe.** Both production and every
    backtest must go through here, so a change to the fit cannot silently apply to
    one and not the other.

    Expects ``train`` to carry ``Date``, ``Home``, ``Away``, ``HExpG+``, ``AExpG+``
    (for the fit) and ``HG``/``AG`` (for the draw-delta calibration), already
    windowed and NaN-filtered — see ``fit_dixon_coles_model_from_csv`` for the
    canonical preparation. Returns a ``DrawCalibratedModel``.
    """
    # Time-decay weights so recent matches matter more.
    weights = dixon_coles_weights(train["Date"], xi=xi)

    # ContinuousPoissonGoalModel, not a penaltyblog family: the targets are xG
    # blends (non-integer) and penaltyblog's base class truncates them to int
    # before the likelihood. See csl.models.continuous_poisson for the evidence.
    clf = ContinuousPoissonGoalModel(
        train["HExpG+"],
        train["AExpG+"],
        train["Home"],
        train["Away"],
        weights,
    )
    clf.fit()

    # Guard the fix: the Poisson score equation must hold. This is what the old
    # fit violated (ratio 0.733), and it is cheap enough to assert on every run.
    ratio_home, ratio_away = clf.score_equation_ratio()
    if not (abs(ratio_home - 1.0) < 1e-3 and abs(ratio_away - 1.0) < 1e-3):
        raise RuntimeError(
            "Poisson score equation violated — sum(w*lambda)/sum(w*y) = "
            f"({ratio_home:.6f}, {ratio_away:.6f}), expected 1.0. The fit is not at "
            "its optimum, or goal targets are being coerced somewhere."
        )

    # delta is still fitted so regressions in the residual draw bias stay visible
    # in the logs, but DRAW_DELTA_SHRINK gates how much of it is applied. It is 0.0
    # because the fitted value does not generalize — see the constant's comment.
    delta_fit = fit_draw_delta(clf, train, weights)
    delta_applied = 1.0 + DRAW_DELTA_SHRINK * (delta_fit - 1.0)
    print(
        f"Fitted {len(train)} matches: mean lambda "
        f"{clf.lambda_home.mean():.3f}/{clf.lambda_away.mean():.3f}, "
        f"score eq ({ratio_home:.6f}, {ratio_away:.6f})"
    )
    print(
        f"Draw de-bias delta: fitted {delta_fit:.3f}, applied {delta_applied:.3f} "
        f"(shrink {DRAW_DELTA_SHRINK:.2f})"
    )
    return DrawCalibratedModel(clf, delta_applied)


def fit_dixon_coles_model_from_csv(input_csv_path, xi=PROD_XI):
    """
    Load league data, apply the standard 18-month filter and fit the project
    model via ``fit_production_model``: a weighted continuous-target Poisson fit
    on xG targets with Dixon-Coles time-decay weights, wrapped in the §12.4 draw
    de-bias calibration. Returns a DrawCalibratedModel.
    """
    df = pd.read_csv(input_csv_path)
    raw_dates = df["Date"].copy()

    # Accept both legacy slash dates and the canonical YYYY-MM-DD format.
    df["Date"] = parse_date_only_series(df["Date"])
    bad_dates = df["Date"].isna()
    if bad_dates.any():
        bad_values = raw_dates.loc[bad_dates].astype(str).head(10).tolist()
        raise ValueError(f"Found unparseable Date values in {input_csv_path}: {bad_values}")

    # Drop rows where 'Home' or 'Away' teams are missing
    df = df.dropna(subset=["Home", "Away"])

    # Ensure team names are strings
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)

    # The model is trained on expected-goal targets, so rows without a full
    # HExpG+/AExpG+ pair cannot be used yet (for example, when xG is delayed).
    df["HExpG+"] = pd.to_numeric(df["HExpG+"], errors="coerce")
    df["AExpG+"] = pd.to_numeric(df["AExpG+"], errors="coerce")

    # Filter to the standard training window, measured back from the latest match
    # date in the dataset rather than from today.
    cutoff_date = df["Date"].max() - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)
    df = df[df["Date"] >= cutoff_date]

    df = df.dropna(subset=["HExpG+", "AExpG+"]).copy()
    if df.empty:
        raise ValueError("No training rows remain after dropping missing HExpG+/AExpG+ values")

    return fit_production_model(df, xi=xi)


def run_dixon_coles_model(input_csv_path, output_csv_path, xi=0.001):
    """
    Automates the model process: reading data, extracting teams, applying weights, fitting the model,
    extracting parameters, and saving results to a CSV file.

    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file.
        xi (float): Decay factor for time weighting. Higher values down-weight older matches more
                    aggressively. Default is 0.001. Typical range: 0.0001 (slow decay) to 0.01 (fast decay).
    """
    # Step 1: Fit the shared model used across exports
    clf = fit_dixon_coles_model_from_csv(input_csv_path, xi=xi)

    # Step 2: Extract Parameters
    # Use clf.teams to guarantee team order matches the internal parameter array
    teams = clf.teams
    params = clf._params
    attack = params[:len(teams)]        # Attack values
    defense = params[len(teams):len(teams)*2]  # Defense values

    # Step 3: Create DataFrame for Team Statistics
    team_stats = pd.DataFrame({
        "Team": teams,
        "Attack": attack,
        "Defense": defense
    })
    team_stats["Date"] = datetime.now().strftime("%Y-%m-%d")

    # Step 4: Simulate Matches Between All Teams
    simulation_results = []
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team:
                probs = clf.predict(home_team, away_team)
                results = {
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Home Win Probability": probs.asian_handicap("home", 0),
                    "Draw Probability": 1 - probs.asian_handicap("home", 0) - probs.asian_handicap("away", 0),
                    "Away Win Probability": probs.asian_handicap("away", 0),
                    "Home -1 Handicap": probs.asian_handicap("home", -1),
                    "Home -2 Handicap": probs.asian_handicap("home", -2),
                    "Away -1 Handicap": probs.asian_handicap("away", -1),
                    "Away -2 Handicap": probs.asian_handicap("away", -2),
                }
                simulation_results.append(results)

    match_simulations_df = pd.DataFrame(simulation_results)
    match_simulations_df["Date"] = datetime.now().strftime("%Y-%m-%d")

    # Reorder columns
    match_simulations_df = match_simulations_df[["Date", "Home Team", "Away Team", "Home Win Probability",
                                                 "Draw Probability", "Away Win Probability", "Home -1 Handicap",
                                                 "Home -2 Handicap", "Away -1 Handicap", "Away -2 Handicap"]]

    # Step 5: Save DataFrames to CSV
    team_stats.to_csv(output_csv_path, index=False)
    match_simulations_df.to_csv(output_csv_path.replace(".csv", "_match_simulations.csv"), index=False)

    print(f"Team stats and match simulation results successfully saved to: {output_csv_path}")
