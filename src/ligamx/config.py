"""
Configuration for the Liga MX Terminal pipeline.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from ligamx import paths

load_dotenv()

THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

ODDS_SPORT_KEY = "soccer_mexico_ligamx"
BOOKMAKER = "pinnacle"
MARKET = "h2h"

# DATA_FILE is repo-relative; consumers resolve it against paths.project_root()
# (kept as a relative string so tests can redirect it to a temp path).
DATA_FILE = "data/MEX_ligamx.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "output"

CSV_DATE_FORMAT = "%Y/%m/%d"
XG_REFRESH_DAYS = 30
XG_COMPARE_EPSILON = 1e-6

# Odds columns of the MEX_ligamx.csv schema, populated by the odds/capture layer.
# Pinnacle closing 1X2 is the model-vs-market benchmark; opens + the exchange
# venues (Betfair, Polymarket) drive the CLV work. Asian-handicap columns stay
# reserved. `open` = earliest captured price at a fixed lead time before kickoff;
# `close` = last price before kickoff. Keep in sync with data/MEX_ligamx.csv.
ODDS_COLUMNS = [
    "pinnacle_open_h", "pinnacle_open_d", "pinnacle_open_a",
    "pinnacle_close_h", "pinnacle_close_d", "pinnacle_close_a",
    "pinnacle_open_ah", "pinnacle_open_ah_h", "pinnacle_open_ah_a",
    "pinnacle_close_ah", "pinnacle_close_ah_h", "pinnacle_close_ah_a",
    "betfair_open_h", "betfair_open_d", "betfair_open_a",
    "betfair_close_h", "betfair_close_d", "betfair_close_a",
    "polymarket_open_h", "polymarket_open_d", "polymarket_open_a",
    "polymarket_close_h", "polymarket_close_d", "polymarket_close_a",
    "betano_open_h", "betano_open_d", "betano_open_a",
    "duel_open_h", "duel_open_d", "duel_open_a",
]

# --- Odds capture (poller) --------------------------------------------------
# The Odds API books to snapshot. matchbook is captured into the store even
# though it has no MEX_ligamx column yet (reducible later). betfair_ex_eu is
# the exchange venue surfaced as betfair_* in the schema.
ODDS_API_REGIONS = "eu,uk"
ODDS_API_BOOKMAKERS = ["pinnacle", "betfair_ex_eu", "matchbook"]
# venue key (as stored) -> MEX_ligamx column prefix (None = store-only, no column)
# betano/duel arrive from odds-api.io (see odds/oddsapi_io.CAPTURE_BOOKS) and are
# opening-line sources only -- the plan sells no closes for them, so there are no
# {prefix}_close_* columns to fill.
VENUE_TO_SCHEMA_PREFIX = {
    "pinnacle": "pinnacle",
    "betfair_ex_eu": "betfair",
    "matchbook": None,
    "polymarket": "polymarket",
    "betano": "betano",
    "duel": "duel",
}

# Polymarket read-only Gamma catalogue API (unauthenticated). Match markets are
# a bundle of 3 Yes/No sub-markets (home win / draw / away win); the Yes price is
# the implied probability, so decimal odds = 1 / yes_price.
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
# CLOB read API: /prices-history?market={token_id} returns a free, per-outcome
# price time-series from market open to settlement (works for closed markets too),
# so Polymarket early prices can be backfilled retroactively at no quota cost.
POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)

# SofaScore unique-tournament ids (Liga MX runs two tournaments per year).
SOFASCORE_APERTURA = 11621
SOFASCORE_CLAUSURA = 11620


# Load team mapping from CSV
def _load_team_mapping():
    """Load team mapping from CSV file."""
    csv_path = paths.team_mapping_csv()

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
            "sofascore_to_standard": {},
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
