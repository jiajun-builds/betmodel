"""
Prediction Model - reads the pre-computed model simulations for probabilities.
"""

import pandas as pd
from typing import Dict, Optional, List

from ligamx import config, paths


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
        self._simulations_df = None
        self._team_stats_df = None

    def _map_team_name(self, team_name: str) -> str:
        """Map Odds API team names to model team names."""
        # First the standard mapping from config, then manual encoding fixes.
        standard = config.ODDS_TO_STANDARD.get(team_name, team_name)
        return self.MODEL_NAME_MAP.get(standard, standard)

    def load(self):
        """Load pre-computed model outputs."""
        self._simulations_df = pd.read_csv(paths.match_simulations_csv())
        self._team_stats_df = pd.read_csv(paths.team_stats_csv())

    def get_probabilities(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Get probabilities for a specific matchup (team names in Odds API format)."""
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

        ah_cols = {}
        for col in self._simulations_df.columns:
            if col.startswith("Home ") or col.startswith("Away "):
                if col not in ["Home Team", "Away Team"]:
                    ah_cols[col] = row[col]

        return {
            "home_win": row["Home Win Probability"],
            "draw": row["Draw Probability"],
            "away_win": row["Away Win Probability"],
            **ah_cols,
        }

    def get_ah_probability(self, home_team: str, away_team: str, spread: float) -> Optional[Dict]:
        """Get the model cover probability for a given Asian-Handicap spread."""
        probs = self.get_probabilities(home_team, away_team)
        if not probs:
            return None

        if spread == 0:
            return {"probability": probs.get("Home Win Probability", 0), "line": 0, "type": "1x2"}

        if spread < 0:
            col_name = f"Home {spread}"
        else:
            col_name = f"Home +{spread}"

        if col_name in probs:
            return {"probability": probs[col_name], "line": spread, "type": "ah"}

        available_home = [c for c in probs.keys() if c.startswith("Home ") and c not in ["Home Team", "Away Team"]]
        available_home.sort(key=lambda x: abs(self._parse_line(x) - spread))

        if available_home:
            closest = available_home[0]
            return {"probability": probs[closest], "line": self._parse_line(closest), "type": "ah_approximate"}

        return {"probability": probs.get("Home Win Probability", 0), "line": 0, "type": "1x2_fallback"}

    def _parse_line(self, col_name: str) -> float:
        """Parse the line value from a column name like 'Home -1.5'."""
        try:
            if col_name.startswith("Home "):
                return float(col_name.replace("Home ", ""))
            elif col_name.startswith("Away "):
                return float(col_name.replace("Away ", ""))
        except ValueError:
            pass
        return 0.0

    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """Get team attack/defense stats (team name in model/standard format)."""
        if self._team_stats_df is None:
            self.load()

        team = self._team_stats_df[self._team_stats_df["Team"] == team_name]
        if team.empty:
            return None

        row = team.iloc[0]
        return {"attack": row["Attack"], "defense": row["Defense"]}

    def get_all_team_pairings(self) -> List[Dict]:
        """Get all team pairings with probabilities."""
        if self._simulations_df is None:
            self.load()
        return self._simulations_df.to_dict("records")
