"""
Configuration for LigaMX Betting Tool
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_API_KEY = os.getenv("API_FOOTBALL_API_KEY", "")

THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

LIGA_MX_LEAGUE_ID = 262
ODDS_SPORT_KEY = "soccer_mexico_ligamx"
BOOKMAKER = "pinnacle"
MARKET = "spreads"

DATA_FILE = "data/MEX_ligamx.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "output"

CSV_DATE_FORMAT = "%Y/%m/%d"
XG_REFRESH_DAYS = 30
XG_COMPARE_EPSILON = 1e-6

# Sofascore season IDs
SOFASCORE_APERTURA = 11621
SOFASCORE_CLASURA = 11620

# Load team mapping from CSV
def _load_team_mapping():
    """Load team mapping from CSV file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "ligamx_team_name_mapping.csv")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Create mapping dictionaries
        odds_to_sofascore = dict(zip(df["odds_team"], df["sofascore_team"]))
        odds_to_standard = dict(zip(df["odds_team"], df["standard_team"]))
        standard_to_odds = dict(zip(df["standard_team"], df["odds_team"]))
        
        # Create model to odds mapping (model uses standard names but may have encoding issues)
        # Map from model CSV names to odds names
        model_to_odds = {}
        for _, row in df.iterrows():
            standard = row["standard_team"]
            odds = row["odds_team"]
            # Handle the encoding issues in model file
            model_to_odds[standard] = odds
            # Also add the encoded versions
            model_to_odds[standard.replace("ó", "_").replace("á", "_").replace("é", "_").replace("í", "_").replace("ú", "_")] = odds
            if standard == "Atl. San Luis":
                model_to_odds["Atl_tico San Luis"] = odds
        
        sofascore_to_standard = dict(zip(df["sofascore_team"], df["standard_team"]))
        
        return {
            "odds_to_sofascore": odds_to_sofascore,
            "odds_to_standard": odds_to_standard,
            "standard_to_odds": standard_to_odds,
            "model_to_odds": model_to_odds,
            "sofascore_to_standard": sofascore_to_standard,
        }
    except Exception as e:
        print(f"Warning: Could not load team mapping CSV: {e}")
        return {
            "odds_to_sofascore": {},
            "odds_to_standard": {},
            "standard_to_odds": {},
            "model_to_odds": {},
        }

TEAM_MAPPING = _load_team_mapping()
ODDS_TO_SOFASCORE = TEAM_MAPPING["odds_to_sofascore"]
ODDS_TO_STANDARD = TEAM_MAPPING["odds_to_standard"]
STANDARD_TO_ODDS = TEAM_MAPPING["standard_to_odds"]
MODEL_TO_ODDS = TEAM_MAPPING["model_to_odds"]
SOFASCORE_TO_STANDARD = TEAM_MAPPING["sofascore_to_standard"]

# Legacy mappings (keep for backward compatibility)
TEAM_NAME_MAPPING = ODDS_TO_SOFASCORE
REVERSE_TEAM_MAPPING = {v: k for k, v in TEAM_NAME_MAPPING.items()}
