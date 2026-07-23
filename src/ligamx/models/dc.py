"""
Goals model for Liga MX: penaltyblog NegativeBinomialGoalModel fit on blended
expected goals (HExpG+/AExpG+) with Dixon-Coles recency weights.

Outputs team strengths (MEX_team_stats.csv) and per-fixture 1X2 + Asian-Handicap
cover probabilities (MEX_team_stats_match_simulations.csv).
"""

import numpy as np
import pandas as pd
import penaltyblog as pb
from penaltyblog.models import dixon_coles_weights

from ligamx import paths

# Dixon-Coles time-decay parameter (higher => older matches down-weighted faster).
# Centralized here so every fit uses the same value.
MODEL_XI = 0.0015

# Trailing training window.
TRAINING_WINDOW_MONTHS = 24

# Asian Handicap lines to generate (mirrored home/away).
AH_LINES = [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

# Empirical-Bayes shrinkage of team ratings toward the (weighted) league mean.
# Teams with few matches in the window (e.g. promoted sides) get unstable, extreme
# MLE strengths; shrinkage pulls them to the mean. Per-team weight w = n/(n+K),
# where n is the Dixon-Coles-weighted match count: n >> K keeps a team's own
# estimate, n << K pulls it to the mean. K is a pseudo-count, loosely "how many
# recent matches before the model half-trusts a team's own rating". Tunable —
# re-validate in the backtest.
ENABLE_SHRINKAGE = True
SHRINKAGE_K = 6.0


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


def run_negative_binomial_model(input_csv_path, output_csv_path):
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
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")

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

    # Step 2: Fit the NegativeBinomial model with Dixon-Coles recency weights
    weights = dixon_coles_weights(df["Date"], xi=MODEL_XI)
    clf = pb.models.NegativeBinomialGoalModel(df["HExpG+"], df["AExpG+"], df["Home"], df["Away"], weights,)
    clf.fit()
    clf.get_params()

    # Step 3: Extract fitted parameters. penaltyblog order is
    # [attack x n, defence x n, home_advantage, dispersion]; use clf.teams as the
    # authoritative team ordering aligned with those parameter slots.
    teams_ordered = list(clf.teams)
    n_teams = len(teams_ordered)
    params = clf._params
    attack = np.asarray(params[:n_teams], dtype=float)
    defense = np.asarray(params[n_teams:2 * n_teams], dtype=float)
    home_advantage = float(params[-2])
    dispersion = float(params[-1])
    print(f"Home advantage: {home_advantage:.4f}, Dispersion: {dispersion:.4f}")

    # Step 3b: Empirical-Bayes shrinkage toward the league mean. Effective sample
    # size per team = sum of Dixon-Coles weights over its matches in the window.
    # Injected back into clf._params so clf.predict() uses the shrunk strengths.
    if ENABLE_SHRINKAGE:
        eff_n = {t: 0.0 for t in teams_ordered}
        for w_row, home, away in zip(np.asarray(weights, dtype=float), df["Home"], df["Away"]):
            if home in eff_n:
                eff_n[home] += w_row
            if away in eff_n:
                eff_n[away] += w_row
        n_arr = np.array([eff_n[t] for t in teams_ordered])
        attack_raw, defense_raw = attack.copy(), defense.copy()
        attack = _shrink_ratings(attack, n_arr, SHRINKAGE_K)
        defense = _shrink_ratings(defense, n_arr, SHRINKAGE_K)

        clf._params[:n_teams] = attack
        clf._params[n_teams:2 * n_teams] = defense

        movers = sorted(
            ((t, n_arr[i], attack_raw[i], attack[i]) for i, t in enumerate(teams_ordered)),
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
    run_negative_binomial_model(paths.ligamx_data_csv(), paths.team_stats_csv())


if __name__ == "__main__":
    main()
