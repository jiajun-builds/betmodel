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

        return {
            "home_win": row["Home Win Probability"],
            "draw": row["Draw Probability"],
            "away_win": row["Away Win Probability"],
        }

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
