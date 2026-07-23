"""
Goals model for Liga MX: penaltyblog NegativeBinomialGoalModel fit on blended
expected goals (HExpG+/AExpG+) with Dixon-Coles recency weights.

Outputs team strengths (MEX_team_stats.csv) and per-fixture 1X2 + Asian-Handicap
cover probabilities (MEX_team_stats_match_simulations.csv).
"""

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

    # Step 3: Extract parameters
    params = clf._params
    print(f"Total parameters: {len(params)}")
    print(f"Number of teams: {len(teams)}")

    # Compute parameter indices dynamically rather than hardcoding.
    n_teams = len(teams)
    attack = params[:n_teams]            # first n_teams values: Attack
    defense = params[n_teams:2 * n_teams]  # next n_teams values: Defense

    # Guard against a parameter-length mismatch.
    if len(attack) != n_teams or len(defense) != n_teams:
        print("Warning: Parameter length mismatch!")
        print(f"Attack params: {len(attack)}, Defense params: {len(defense)}, Teams: {n_teams}")

        attack = attack[:n_teams] if len(attack) >= n_teams else list(attack) + [0.0] * (n_teams - len(attack))
        defense = defense[:n_teams] if len(defense) >= n_teams else list(defense) + [0.0] * (n_teams - len(defense))

    # Home advantage and rho (correlation) parameters.
    home_advantage = params[2 * n_teams] if len(params) > 2 * n_teams else 0.0
    rho = params[2 * n_teams + 1] if len(params) > 2 * n_teams + 1 else 0.0

    print(f"Home advantage: {home_advantage}, Rho: {rho}")

    # Step 4: Team stats DataFrame
    team_stats = pd.DataFrame({
        "Team": teams,
        "Attack": attack,
        "Defense": defense
    })

    # Tag every row with the latest match date used in the fit.
    latest_match_date = df["Date"].max().strftime("%Y/%m/%d")
    team_stats["Date"] = latest_match_date

    print("Team stats created successfully")
    print(team_stats.head())

    # Step 5: Simulate matches between all ordered team pairs.
    simulation_results = []

    print("Starting match simulations...")
    for home_team in teams:
        for away_team in teams:
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
