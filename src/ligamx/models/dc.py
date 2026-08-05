"""
Goals model for Liga MX: weighted Poisson fit on blended expected goals
(HExpG+/AExpG+) with Dixon-Coles recency weights and empirical-Bayes shrinkage.

Outputs team strengths (MEX_team_stats.csv) and per-fixture 1X2 + Asian-Handicap
cover probabilities (MEX_team_stats_match_simulations.csv).

fit_production_model() is the single source of truth for "how the model is
fitted" -- the eval harnesses (rps_backtest, clv_backtest, clv_signals,
hyperparam_sweep) all go through it, so a backtest can never silently drift from
what production does.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from penaltyblog.models import dixon_coles_weights

from ligamx import paths
from ligamx.models.continuous_poisson import fit_continuous_poisson
from ligamx.date_utils import parse_date_only_series

# Dixon-Coles time-decay parameter (higher => older matches down-weighted faster).
# Centralized here so every fit uses the same value.
MODEL_XI = 0.0015

# Trailing training window.
TRAINING_WINDOW_MONTHS = 24

# Post-hoc draw recalibration: deflate the 1X2 draw by this factor and renormalize
# onto home/away. DISABLED (1.0) as of the continuous-target fix.
#
# The old ALPHA=0.85 was compensating for a bug, not for the sport: penaltyblog
# truncated the continuous xG target to integers, which deflated every scoring
# rate ~30% and inflated the draw to ~0.30 against a realized ~0.25. With the rate
# scale fixed the raw model draw lands at 0.241 vs realized 0.239 (calibration,
# n=439) and 0.238 vs 0.244 (out-of-sample, n=250) -- deflating it again would
# re-introduce a ~2pp bias in the other direction. RPS is nearly flat in alpha
# (0.0003 across 0.80-1.00 on calibration, which nominally prefers 0.90); the
# out-of-sample window prefers 1.00 outright, and the calibration evidence picks
# it too. Re-validate with `python -m ligamx.eval.draw_calibration`.
# Only 1X2 outputs are recalibrated; Asian-handicap cover probabilities come from
# the full scoreline grid and are left untouched.
DRAW_CALIBRATION_ALPHA = 1.0

# Asian Handicap lines to generate (mirrored home/away).
AH_LINES = [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

# Training target columns: the blended-xG pair. Overridable so the sweep can try
# other blends (and pure goals) through the same fit path.
DEFAULT_TARGET = ("HExpG+", "AExpG+")

# Empirical-Bayes shrinkage of team ratings toward the (weighted) league mean.
# Teams with few matches in the window (e.g. promoted sides) get unstable, extreme
# MLE strengths; shrinkage pulls them to the mean. Per-team weight w = n/(n+K),
# where n is the Dixon-Coles-weighted match count: n >> K keeps a team's own
# estimate, n << K pulls it to the mean. K is a pseudo-count, loosely "how many
# recent matches before the model half-trusts a team's own rating". Tunable —
# re-validate in the backtest.
ENABLE_SHRINKAGE = True
SHRINKAGE_K = 6.0


def calibrate_1x2(home_win: float, draw: float, away_win: float,
                  alpha: float = DRAW_CALIBRATION_ALPHA) -> tuple[float, float, float]:
    """Deflate the draw probability by ``alpha`` and renormalize onto home/away.

    Corrects the model's systematic draw over-pricing (see DRAW_CALIBRATION_ALPHA).
    The home/away split is preserved; only mass is shifted off the draw. alpha=1.0
    is a no-op.
    """
    d = alpha * draw
    rest = home_win + away_win
    scale = (1.0 - d) / rest if rest > 0 else 1.0
    return home_win * scale, d, away_win * scale


def _shrink_ratings(values: np.ndarray, eff_n: np.ndarray, k: float) -> np.ndarray:
    """Shrink per-team ratings toward the sample-size-weighted league mean.

    Low-sample teams are pulled strongly to the mean; well-sampled teams barely
    move. The weighted mean is preserved (only the spread is regularized) so the
    league's overall scoring level is unchanged.
    """
    if eff_n.sum() <= 0:
        return values
    target = np.average(values, weights=eff_n)
    w = eff_n / (eff_n + k)
    shrunk = target + w * (values - target)
    # Re-center so the weighted mean is unchanged (regularize spread only).
    shrunk += target - np.average(shrunk, weights=eff_n)
    return shrunk


def _effective_n(teams, weights, home_series, away_series) -> np.ndarray:
    """Dixon-Coles-weighted match count per team, in ``teams`` order."""
    eff = {t: 0.0 for t in teams}
    for w, home, away in zip(np.asarray(weights, dtype=float), home_series, away_series):
        if home in eff:
            eff[home] += w
        if away in eff:
            eff[away] += w
    return np.array([eff[t] for t in teams])


def fit_production_model(train, target=DEFAULT_TARGET, xi=MODEL_XI,
                         shrinkage_k=SHRINKAGE_K, shrink=ENABLE_SHRINKAGE):
    """Fit the production goals model on a training frame.

    ``train`` needs Date, Home, Away and the two ``target`` columns. Returns a
    ContinuousPoissonFit whose ratings have already been shrunk, so callers just
    call .predict(home, away).
    """
    gh, ga = target
    n_before = len(train)
    train = train.dropna(subset=[gh, ga, "Home", "Away", "Date"])
    dropped = n_before - len(train)
    if dropped:
        print(f"WARNING: dropped {dropped} training rows with missing {gh}/{ga}")
    weights = np.asarray(dixon_coles_weights(train["Date"], xi=xi), dtype=float)
    fit = fit_continuous_poisson(
        train[gh].to_numpy(dtype=float), train[ga].to_numpy(dtype=float),
        train["Home"], train["Away"], weights)
    if shrink:
        eff_n = _effective_n(fit.teams, weights, train["Home"], train["Away"])
        fit = replace(fit,
                      attack=_shrink_ratings(fit.attack, eff_n, shrinkage_k),
                      defence=_shrink_ratings(fit.defence, eff_n, shrinkage_k))
    return fit


def run_goals_model(input_csv_path, output_csv_path):
    """
    Fit the goals model and write team stats + per-fixture simulations.

    Parameters:
        input_csv_path (str): Path to the input match-history CSV.
        output_csv_path (str): Path to write the team-stats CSV (the match
            simulations CSV is derived from this name).
    """
    # Step 1: Load data
    df = pd.read_csv(input_csv_path)

    # Convert Date explicitly using the known format (more reliable than
    # parse_dates in read_csv, which can silently fall back to strings).
    df["Date"] = parse_date_only_series(df["Date"])

    # Drop rows where 'Home' or 'Away' teams are missing
    df = df.dropna(subset=["Home", "Away"])

    # Ensure team names are strings and extract unique names
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    teams = sorted(set(df["Home"].unique()).union(set(df["Away"].unique())))

    print(f"Found {len(teams)} teams: {teams}")

    # Filter to the trailing training window, anchored on the latest match date.
    cutoff_date = df["Date"].max() - pd.DateOffset(months=TRAINING_WINDOW_MONTHS)
    df = df[df["Date"] >= cutoff_date]

    # Step 2: Fit, with Dixon-Coles recency weights and shrinkage, through the
    # shared entry point the eval harnesses also use.
    raw = fit_production_model(df, shrink=False)
    clf = fit_production_model(df)
    teams_ordered = list(clf.teams)
    attack = clf.attack
    defense = clf.defence
    print(f"Home advantage: {clf.home_advantage:.4f}, baseline log-rate: {clf.intercept:.4f}")

    if ENABLE_SHRINKAGE:
        eff_n = _effective_n(clf.teams, dixon_coles_weights(df["Date"], xi=MODEL_XI),
                             df["Home"], df["Away"])
        movers = sorted(
            ((t, eff_n[i], raw.attack[i], attack[i]) for i, t in enumerate(teams_ordered)),
            key=lambda r: abs(r[3] - r[2]), reverse=True,
        )
        print(f"Shrinkage applied (K={SHRINKAGE_K}); top movers by |Δattack|:")
        for t, n, a0, a1 in movers[:5]:
            print(f"  {t:<20} eff_n={n:5.1f}  attack {a0:+.3f} -> {a1:+.3f}")

    # Step 4: Team stats DataFrame (post-shrinkage ratings)
    team_stats = pd.DataFrame({
        "Team": teams_ordered,
        "Attack": attack,
        "Defense": defense,
    })

    # Tag every row with the latest match date used in the fit.
    latest_match_date = df["Date"].max().strftime("%Y/%m/%d")
    team_stats["Date"] = latest_match_date

    print("Team stats created successfully")
    print(team_stats.head())

    # Step 5: Simulate matches between all ordered team pairs.
    simulation_results = []

    print("Starting match simulations...")
    for home_team in teams_ordered:
        for away_team in teams_ordered:
            if home_team == away_team:
                continue
            try:
                probs = clf.predict(home_team, away_team)

                home_win_prob = probs.asian_handicap("home", 0)
                away_win_prob = probs.asian_handicap("away", 0)
                draw_prob = 1 - home_win_prob - away_win_prob

                # Deflate the over-priced draw and renormalize onto home/away.
                home_win_prob, draw_prob, away_win_prob = calibrate_1x2(
                    home_win_prob, draw_prob, away_win_prob)

                results = {
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Home Win Probability": home_win_prob,
                    "Draw Probability": draw_prob,
                    "Away Win Probability": away_win_prob,
                }

                # Home-side Asian Handicaps (negative = home gives advantage).
                for line in AH_LINES:
                    results[f"Home {line}"] = probs.asian_handicap("home", line)

                # Away-side Asian Handicaps.
                for line in AH_LINES:
                    results[f"Away {line}"] = probs.asian_handicap("away", line)

                simulation_results.append(results)

            except Exception as pred_error:
                print(f"Error predicting {home_team} vs {away_team}: {pred_error}")
                continue

    print(f"Generated {len(simulation_results)} match simulations")

    if simulation_results:
        match_simulations_df = pd.DataFrame(simulation_results)
        match_simulations_df["Date"] = latest_match_date

        # Reorder columns: Date, team names, 1X2 probabilities, then AH columns.
        base_cols = ["Date", "Home Team", "Away Team", "Home Win Probability", "Draw Probability", "Away Win Probability"]
        ah_cols = [col for col in match_simulations_df.columns if col not in base_cols]
        match_simulations_df = match_simulations_df[base_cols + ah_cols]

        team_stats.to_csv(output_csv_path, index=False, mode="w")
        sim_path = output_csv_path.replace(".csv", "_match_simulations.csv")
        match_simulations_df.to_csv(sim_path, index=False, mode="w")

        print(f"Team stats saved to: {output_csv_path}")
        print(f"Match simulation results saved to: {sim_path}")
    else:
        print("No simulation results generated. Only saving team stats.")
        team_stats.to_csv(output_csv_path, index=False, mode="w")
        print(f"Team stats saved to: {output_csv_path}")


def main():
    run_goals_model(paths.ligamx_data_csv(), paths.team_stats_csv())


if __name__ == "__main__":
    main()
