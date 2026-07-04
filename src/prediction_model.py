"""
Prediction Model - uses pre-computed match_simulations.csv for probabilities.
"""

import pandas as pd
from typing import Dict, Optional, List
import config
import os


class PredictionModel:
    MODEL_NAME_MAP = {
        "Club América": "Club America",
        "Club America": "Club America",
        "Leon": "Leon",
        "León": "Leon",
        "Queretaro": "Queretaro",
        "Querétaro": "Queretaro",
        "FC Juarez": "FC Juarez",
        "FC Juárez": "FC Juarez",
        "Mazatlan": "Mazatlan",
        "Mazatlán": "Mazatlan",
    }
    
    def __init__(self):
        self.simulations_file = "models/MEX_team_stats_match_simulations.csv"
        self.team_stats_file = "models/MEX_team_stats.csv"
        self._simulations_df = None
        self._team_stats_df = None

    def _map_team_name(self, team_name: str) -> str:
        """Map Odds API team names to model team names."""
        # First try the standard mapping from config
        standard = config.ODDS_TO_STANDARD.get(team_name, team_name)
        
        # Then apply manual fixes for model encoding issues
        return self.MODEL_NAME_MAP.get(standard, standard)

    def load(self):
        """Load pre-computed model outputs."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        sim_path = os.path.join(base_dir, self.simulations_file)
        stats_path = os.path.join(base_dir, self.team_stats_file)
        
        self._simulations_df = pd.read_csv(sim_path)
        self._team_stats_df = pd.read_csv(stats_path)

    def get_probabilities(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Get probabilities for a specific matchup."""
        if self._simulations_df is None:
            self.load()
        
        mapped_home = self._map_team_name(home_team)
        mapped_away = self._map_team_name(away_team)
        
        match = self._simulations_df[
            (self._simulations_df["Home Team"] == mapped_home) &
            (self._simulations_df["Away Team"] == mapped_away)
        ]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        
        # Build dictionary with all AH columns
        ah_cols = {}
        for col in self._simulations_df.columns:
            if col.startswith("Home ") or col.startswith("Away "):
                if col not in ['Home Team', 'Away Team']:
                    ah_cols[col] = row[col]
        
        return {
            "home_win": row["Home Win Probability"],
            "draw": row["Draw Probability"],
            "away_win": row["Away Win Probability"],
            **ah_cols,
        }

    def get_ah_probability(self, home_team: str, away_team: str, spread: float) -> Optional[Dict]:
        """Get the correct AH probability for a given spread.
        
        Args:
            home_team: Home team name (Odds API format)
            away_team: Away team name (Odds API format)
            spread: Asian Handicap line (e.g., -1.5, 0, +0.5)
        
        Returns:
            Dict with 'probability' and 'line' keys
        """
        probs = self.get_probabilities(home_team, away_team)
        if not probs:
            return None
        
        # Map spread to column name
        # Spread negative: home gives handicap (e.g., -1.5 means home -1.5)
        # Spread positive: away gives handicap (e.g., +0.5 means home +0.5 = away -0.5)
        
        if spread == 0:
            # Draw no bet - use 1X2 probability
            return {"probability": probs.get("Home Win Probability", 0), "line": 0, "type": "1x2"}
        
        # Determine which column to use
        col_name = None
        
        if spread < 0:
            # Home gives handicap (e.g., -1.5, -1, -0.5)
            col_name = f"Home {spread}"
        else:
            # Home receives handicap (e.g., +0.5 means home +0.5)
            col_name = f"Home +{spread}"
        
        # Try exact match first
        if col_name in probs:
            return {"probability": probs[col_name], "line": spread, "type": "ah"}
        
        # For lines not in the model, find closest available
        # Map to nearest available line
        available_home = [col for col in probs.keys() if col.startswith("Home ") and col not in ['Home Team', 'Away Team']]
        available_home.sort(key=lambda x: abs(self._parse_line(x) - spread))
        
        if available_home:
            closest = available_home[0]
            return {"probability": probs[closest], "line": self._parse_line(closest), "type": "ah_approximate"}
        
        # Fallback to 1X2
        return {"probability": probs.get("Home Win Probability", 0), "line": 0, "type": "1x2_fallback"}

    def _parse_line(self, col_name: str) -> float:
        """Parse line value from column name like 'Home -1.5'"""
        try:
            if col_name.startswith("Home "):
                val = col_name.replace("Home ", "")
                return float(val)
            elif col_name.startswith("Away "):
                val = col_name.replace("Away ", "")
                return float(val)
        except:
            pass
        return 0.0

    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """Get team attack/defense stats."""
        if self._team_stats_df is None:
            self.load()
        
        team = self._team_stats_df[self._team_stats_df["Team"] == team_name]
        
        if team.empty:
            return None
        
        row = team.iloc[0]
        return {
            "attack": row["Attack"],
            "defense": row["Defense"],
        }

    def get_all_team_pairings(self) -> List[Dict]:
        """Get all possible team pairings with probabilities."""
        if self._simulations_df is None:
            self.load()
        
        return self._simulations_df.to_dict("records")