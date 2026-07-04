import pandas as pd
import penaltyblog as pb
from penaltyblog.models import dixon_coles_weights
import os

def run_negative_binomial_model(input_csv_path, output_csv_path):
    """
    Automates the Negative Binomial model process: reading data, applying weights, fitting the model,
    extracting parameters, and saving results to a CSV file.

    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file.
    """
    try:
        # Step 1: Load Data
        df = pd.read_csv(input_csv_path)

        # Convert Date column to datetime64 explicitly using the known format
        # This is more reliable than parse_dates in read_csv, which can silently fall back to strings
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
        
        # Drop rows where 'Home' or 'Away' teams are missing
        df = df.dropna(subset=["Home", "Away"])
        
        # Ensure team names are strings and extract unique names
        df["Home"] = df["Home"].astype(str)
        df["Away"] = df["Away"].astype(str)
        teams = sorted(set(df["Home"].unique()).union(set(df["Away"].unique())))
        
        print(f"Found {len(teams)} teams: {teams}")
        
        # Filter to most recent 2 years of data only
        # Uses the latest match date in the dataset as the reference point
        cutoff_date = df["Date"].max() - pd.DateOffset(months=24)
        df = df[df["Date"] >= cutoff_date]
        
        # Step 2: Fit NegativeBinomial Model
        xi = 0.0015
        weights = dixon_coles_weights(df["Date"], xi=xi)
        clf = pb.models.NegativeBinomialGoalModel(df["HExpG+"], df["AExpG+"], df["Home"], df["Away"], weights,)
        clf.fit()
        clf.get_params()
        
        # Step 4: Extract Parameters
        params = clf._params
        print(f"Total parameters: {len(params)}")
        print(f"Number of teams: {len(teams)}")
        
        # 动态计算参数索引，而不是硬编码
        n_teams = len(teams)
        attack = params[:n_teams]  # 前 n_teams 个值：Attack
        defense = params[n_teams:2*n_teams]  # 接下来 n_teams 个值：Defense
        
        # 检查参数长度是否匹配
        if len(attack) != n_teams or len(defense) != n_teams:
            print(f"Warning: Parameter length mismatch!")
            print(f"Attack params: {len(attack)}, Defense params: {len(defense)}, Teams: {n_teams}")
            
            # 调整参数长度以匹配球队数量
            attack = attack[:n_teams] if len(attack) >= n_teams else list(attack) + [0.0] * (n_teams - len(attack))
            defense = defense[:n_teams] if len(defense) >= n_teams else list(defense) + [0.0] * (n_teams - len(defense))
        
        # 获取 home_advantage 和 rho
        home_advantage = params[2*n_teams] if len(params) > 2*n_teams else 0.0
        rho = params[2*n_teams + 1] if len(params) > 2*n_teams + 1 else 0.0
        
        print(f"Home advantage: {home_advantage}, Rho: {rho}")
        
        # Step 5: Create DataFrame for Teams
        team_stats = pd.DataFrame({
            "Team": teams,
            "Attack": attack,
            "Defense": defense
        })
        
        # Add the last game date to the team stats DataFrame
        latest_match_date = df["Date"].max().strftime("%Y/%m/%d")
        team_stats["Date"] = latest_match_date

        print("Team stats created successfully")
        print(team_stats.head())

        # Simulate Matches Between All Teams
        simulation_results = []

        # Asian Handicap lines to generate
        ah_lines_home = [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
        ah_lines_away = [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

        print("Starting match simulations...")
        for i, home_team in enumerate(teams):
            for j, away_team in enumerate(teams):
                if home_team != away_team:  # Skip matches where home team equals away team
                    try:
                        probs = clf.predict(home_team, away_team)

                        # Calculate the Draw Probability
                        home_win_prob = probs.asian_handicap("home", 0)
                        away_win_prob = probs.asian_handicap("away", 0)
                        draw_prob = 1 - home_win_prob - away_win_prob  # Draw probability
                        
                        # Collect probabilities for different Asian handicaps
                        results = {
                            "Home Team": home_team,
                            "Away Team": away_team,
                            "Home Win Probability": home_win_prob,
                            "Draw Probability": draw_prob,
                            "Away Win Probability": away_win_prob,
                        }
                        
                        # Home side Asian Handicaps (negative = home gives advantage)
                        for line in ah_lines_home:
                            col_name = f"Home {line}".replace("-", " minus ").replace("+", " plus ")
                            col_name = col_name.replace(" minus ", "-").replace(" plus ", "+")
                            results[col_name] = probs.asian_handicap("home", line)
                        
                        # Away side Asian Handicaps
                        for line in ah_lines_away:
                            col_name = f"Away {line}".replace("-", " minus ").replace("+", " plus ")
                            col_name = col_name.replace(" minus ", "-").replace(" plus ", "+")
                            results[col_name] = probs.asian_handicap("away", line)
                        
                        simulation_results.append(results)
                        
                    except Exception as pred_error:
                        print(f"Error predicting {home_team} vs {away_team}: {pred_error}")
                        continue

        print(f"Generated {len(simulation_results)} match simulations")

        # Create a DataFrame for the simulation results
        if simulation_results:
            match_simulations_df = pd.DataFrame(simulation_results)
            
            # Add the last game date for every row in the DataFrame
            match_simulations_df['Date'] = latest_match_date
            
            # Reorder columns - put Date first, then team names, then probabilities
            base_cols = ['Date', 'Home Team', 'Away Team', 'Home Win Probability', 'Draw Probability', 'Away Win Probability']
            ah_cols = [col for col in match_simulations_df.columns if col not in base_cols]
            match_simulations_df = match_simulations_df[base_cols + ah_cols]

            # Save Team Stats and Simulation to CSV
            team_stats.to_csv(output_csv_path, index=False, mode='w')
            match_simulations_df.to_csv(output_csv_path.replace(".csv", "_match_simulations.csv"), index=False, mode='w')

            print(f"Team stats saved to: {output_csv_path}")
            print(f"Match simulation results saved to: {output_csv_path.replace('.csv', '_match_simulations.csv')}")
        else:
            print("No simulation results generated. Only saving team stats.")
            team_stats.to_csv(output_csv_path, index=False, mode='w')
            print(f"Team stats saved to: {output_csv_path}")

        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Example Usage
input_csv_path = "data/MEX_ligamx.csv"  # Path to input CSV file
output_csv_path = "models/MEX_team_stats.csv"  # Path to save the results

run_negative_binomial_model(input_csv_path, output_csv_path)
