"""
Fetch Chinese Super League Pinnacle 1X2 (h2h) odds from The Odds API.

Roadmap #10: the Asian-handicap route was falsified in backtest (winner's curse,
backtest.md §9), so this fetch requests the ``h2h`` market — home/draw/away prices
— instead of ``spreads``. The stored ``market`` label is "h2h" (aligned with the API
market key). Module and output filenames keep their historical "spreads" names so
workflows and downstream paths stay stable.

This fetch is restricted to pre-match upcoming fixtures only. Live matches are
excluded by requesting odds for the league sport key and applying
`commenceTimeFrom` at the current UTC timestamp.

Usage (仓库根目录，PYTHONPATH=src):
    export THE_ODDS_API_KEY=...
    python -m csl.odds.fetch_pinnacle_spreads

Default output:
    data/raw_data/CHN_pinnacle_spreads.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from csl.paths import data_output_dir, data_raw_dir

THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "soccer_china_superleague"
BOOKMAKER = "pinnacle"
# The Odds API market key for 1X2 prices. Stored rows use the same "h2h" label so
# the history CSV's ``market`` column aligns with the API's market key (roadmap #10
# follow-up: switched from the earlier "moneyline" label per user request).
MARKET = "h2h"
MARKET_LABEL = "h2h"
DEFAULT_REGIONS = "us"

# Books stored at every open-window capture (roadmap #8 recon). Pinnacle is the
# reference/anchor; the rest are the survey's sub-5%-overround candidates —
# the only prices cheap enough for the model's ~1.2-1.4pp excess CLV to clear the
# p x R bar (backtest.md §11.7/§12).
#
# Cost: ZERO extra. The Odds API bills `markets x regions` and counts each 10
# bookmakers as one region, and `bookmakers` takes precedence over `regions` —
# measured 2026-07-16: this 5-book list = 1 credit, same as the Pinnacle-only call,
# and it reaches eu/uk-only books that `regions=us` alone would miss. Keep the list
# at <= 10 entries or the cost doubles.
#
# `onexbet` was REMOVED 2026-08-02: 1xBet's opening line now comes from odds-api.io
# (`csl.odds.fetch_oddsapiio_opens`), whose ~500-requests/DAY free tier can poll without a
# predicted window — the window this module's captures depend on is what lost the
# Shandong Taishan vs Tianjin Jinmen Tiger open. Dropping it costs nothing here (still
# 1 credit) and, via REQUIRED_OPEN_BOOKS in capture_scheduler, actively saves The Odds
# API quota by letting a fixture stop ticking as soon as Pinnacle opens.
CAPTURE_BOOKMAKERS = (BOOKMAKER, "betfair_ex_eu", "betfair_ex_uk", "matchbook")
API_KEY_ENV = "THE_ODDS_API_KEY"

DEFAULT_OUTPUT_CSV = os.path.join(data_raw_dir(), "CHN_pinnacle_spreads.csv")
# All-book "Now" snapshot written alongside the Pinnacle-only CSV: same 1-credit
# fetch (bookmakers filter is free), every CAPTURE_BOOKMAKERS row retained. This is
# the source the zero-quota fallback (backfill_open) reads to record a missed 1xBet
# (or Pinnacle) open, so the bet-price book is no longer Pinnacle-only downstream.
DEFAULT_ALL_BOOKS_CSV = os.path.join(data_raw_dir(), "CHN_now_all_books.csv")
TEAM_MAPPING_CSV = os.path.join(data_output_dir(), "CHN_team_name_mapping.csv")

OUTPUT_COLUMNS = [
    "event_id",
    "commence_time",
    "api_home_team",
    "api_away_team",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "bookmaker",
    "market",
    "regions",
    "last_update",
    "fetched_at",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamMapping:
    odds_to_standard: dict[str, str]
    standard_to_standard: dict[str, str]
    match_to_standard: dict[str, str]
    # odds-api.io's own club spellings ("Shandong Taishan FC", "Henan", "Zhejiang FC").
    # Optional and defaulted so older mapping files — and any caller constructing a
    # TeamMapping directly — keep working; it is simply empty when the column is absent.
    oddsapiio_to_standard: dict[str, str] = field(default_factory=dict)


def _clean_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _build_mapping_column(df: pd.DataFrame, source_col: str, target_col: str) -> dict[str, str]:
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Mapping file missing required columns: {source_col}, {target_col}")

    sub = df[[source_col, target_col]].copy()
    sub[source_col] = sub[source_col].map(_clean_name)
    sub[target_col] = sub[target_col].map(_clean_name)
    sub = sub.dropna(subset=[source_col, target_col])

    dupes = sub[source_col].duplicated(keep=False)
    if dupes.any():
        names = sorted(sub.loc[dupes, source_col].unique().tolist())
        log.warning("Duplicate %s values in mapping file; using last row: %s", source_col, names)

    sub = sub.drop_duplicates(subset=[source_col], keep="last")
    return dict(zip(sub[source_col], sub[target_col]))


def load_team_mapping(path: str = TEAM_MAPPING_CSV) -> TeamMapping:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Team mapping file not found: {path}")

    df = pd.read_csv(path)
    for col in ("odds_team", "standard_team", "match_team"):
        if col not in df.columns:
            raise ValueError(f"Mapping file missing required column: {col}")

    mapping = TeamMapping(
        odds_to_standard=_build_mapping_column(df, "odds_team", "standard_team"),
        standard_to_standard=_build_mapping_column(df, "standard_team", "standard_team"),
        match_to_standard=_build_mapping_column(df, "match_team", "standard_team"),
        # Deliberately NOT in the required-column check above: odds-api.io is a
        # supplementary source (1xBet opens only), so a mapping file without the column
        # must still load for the Pinnacle path rather than failing the whole pipeline.
        oddsapiio_to_standard=(
            _build_mapping_column(df, "oddsapiio_team", "standard_team")
            if "oddsapiio_team" in df.columns
            else {}
        ),
    )
    log.info("Loaded team mapping from %s", path)
    return mapping


def normalize_team_name(api_name: str, mapping: TeamMapping) -> str | None:
    name = _clean_name(api_name)
    if not name:
        return None
    if name in mapping.odds_to_standard:
        return mapping.odds_to_standard[name]
    if name in mapping.standard_to_standard:
        return mapping.standard_to_standard[name]
    if name in mapping.match_to_standard:
        return mapping.match_to_standard[name]
    # Checked last so adding odds-api.io's vocabulary cannot change how any existing
    # The-Odds-API / results-CSV name already resolves.
    if name in mapping.oddsapiio_to_standard:
        return mapping.oddsapiio_to_standard[name]
    return None


def get_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {API_KEY_ENV}")
    return api_key


def fetch_odds_response(
    api_key: str, regions: str, *, bookmakers: str | None = BOOKMAKER
) -> requests.Response:
    """Request pre-match 1X2 odds and return the raw HTTP response.

    Kept separate from ``fetch_odds_payload`` so callers that need the quota
    headers (``x-requests-remaining`` etc.) can read them off the response.

    ``bookmakers=None`` drops the bookmaker filter so every book in ``regions``
    comes back. **This does not cost extra quota**: The Odds API bills
    `markets × regions` per /odds call and the ``bookmakers`` filter is free —
    which is what lets the capture path collect the whole book slate for the
    price of the Pinnacle-only call it already made (roadmap #8).
    """
    url = f"{THE_ODDS_API_BASE_URL}/sports/{ODDS_SPORT_KEY}/odds"
    commence_time_from = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": MARKET,
        "oddsFormat": "decimal",
        "commenceTimeFrom": commence_time_from,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    log.info(
        "Requested %s odds (regions=%s, bookmakers=%s) with commenceTimeFrom=%s",
        ODDS_SPORT_KEY, regions, bookmakers or "<all>", commence_time_from,
    )
    return response


def fetch_odds_payload(
    api_key: str, regions: str, *, bookmakers: str | None = BOOKMAKER
) -> list[dict[str, Any]]:
    response = fetch_odds_response(api_key, regions, bookmakers=bookmakers)
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected list response from The Odds API, got: {type(data)}")
    return data


def _event_bookmakers(event: dict[str, Any]) -> list[dict[str, Any]]:
    bookmakers = event.get("bookmakers")
    if not isinstance(bookmakers, list):
        return []
    return [b for b in bookmakers if isinstance(b, dict) and b.get("key")]


def _find_market(bookmaker: dict[str, Any]) -> dict[str, Any] | None:
    markets = bookmaker.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, dict) and market.get("key") == MARKET:
            return market
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _book_prices(
    bookmaker: dict[str, Any], api_home_team: str, api_away_team: str
) -> tuple[float, float, float] | None:
    """(home, draw, away) decimal prices from one book's h2h market, or None.

    None means this book has no usable 3-outcome h2h for the event — a normal,
    per-book condition once the whole slate is parsed (some books post no draw),
    so callers skip rather than warn.
    """
    market = _find_market(bookmaker)
    if market is None:
        return None
    outcomes = market.get("outcomes")
    if not isinstance(outcomes, list):
        return None

    home_outcome = draw_outcome = away_outcome = None
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = _clean_name(outcome.get("name"))
        if name == api_home_team:
            home_outcome = outcome
        elif name == api_away_team:
            away_outcome = outcome
        elif name and name.casefold() == "draw":
            draw_outcome = outcome
    if home_outcome is None or draw_outcome is None or away_outcome is None:
        return None

    prices = (
        _coerce_float(home_outcome.get("price")),
        _coerce_float(draw_outcome.get("price")),
        _coerce_float(away_outcome.get("price")),
    )
    if any(p is None for p in prices):
        return None
    return prices  # type: ignore[return-value]


def extract_rows(
    events: list[dict[str, Any]],
    mapping: TeamMapping,
    *,
    regions: str,
    fetched_at: str,
    bookmaker_keys: set[str] | None = frozenset({BOOKMAKER}),
) -> list[dict[str, Any]]:
    """One row per (event, bookmaker) with a usable h2h market.

    ``bookmaker_keys=None`` keeps every book in the payload (the roadmap-#8
    survey/capture path); the default keeps Pinnacle only, which is what the
    single-snapshot Now-line fetch and the dashboard comparison expect.
    """
    rows: list[dict[str, Any]] = []
    unmapped_names: set[str] = set()

    for event in events:
        if not isinstance(event, dict):
            log.warning("Skipping malformed event payload: %r", event)
            continue

        api_home_team = _clean_name(event.get("home_team"))
        api_away_team = _clean_name(event.get("away_team"))
        if not api_home_team or not api_away_team:
            log.warning("Skipping event with missing home/away team: %r", event.get("id"))
            continue

        home_team = normalize_team_name(api_home_team, mapping)
        away_team = normalize_team_name(api_away_team, mapping)
        if home_team is None:
            unmapped_names.add(api_home_team)
        if away_team is None:
            unmapped_names.add(api_away_team)
        if home_team is None or away_team is None:
            continue

        books = _event_bookmakers(event)
        if bookmaker_keys is not None:
            books = [b for b in books if b.get("key") in bookmaker_keys]
        if not books:
            log.warning(
                "Skipping event %s: no requested bookmaker present (wanted %s)",
                event.get("id"), sorted(bookmaker_keys) if bookmaker_keys else "<all>",
            )
            continue

        event_rows = 0
        for bookmaker in books:
            prices = _book_prices(bookmaker, api_home_team, api_away_team)
            if prices is None:
                continue
            home_odds, draw_odds, away_odds = prices
            market = _find_market(bookmaker)
            rows.append(
                {
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "api_home_team": api_home_team,
                    "api_away_team": api_away_team,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "bookmaker": bookmaker.get("key"),
                    "market": MARKET_LABEL,
                    "regions": regions,
                    "last_update": (market or {}).get("last_update") or bookmaker.get("last_update"),
                    "fetched_at": fetched_at,
                }
            )
            event_rows += 1
        if event_rows == 0:
            log.warning("Skipping event %s: no book had a usable 3-outcome h2h market", event.get("id"))

    if unmapped_names:
        names = ", ".join(sorted(unmapped_names))
        raise ValueError(
            "Unmapped The Odds API team names found in response. "
            f"Please populate odds_team or existing standard/match mappings first: {names}"
        )

    return rows


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run(
    *, out_path: str, regions: str, all_books_out_path: str = DEFAULT_ALL_BOOKS_CSV
) -> pd.DataFrame:
    api_key = get_api_key()
    mapping = load_team_mapping()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # Fetch every capture book in one request (bookmakers filter is free, so this is
    # the same 1 credit as the old Pinnacle-only call) and keep them all for the
    # all-book Now snapshot; the Pinnacle-only slice keeps the downstream contract.
    events = fetch_odds_payload(api_key, regions, bookmakers=",".join(CAPTURE_BOOKMAKERS))
    log.info("Fetched %d events from The Odds API", len(events))
    all_rows = extract_rows(events, mapping, regions=regions, fetched_at=fetched_at, bookmaker_keys=None)
    all_frame = rows_to_frame(all_rows)

    all_out_dir = os.path.dirname(os.path.abspath(all_books_out_path))
    os.makedirs(all_out_dir, exist_ok=True)
    all_frame.to_csv(all_books_out_path, index=False, encoding="utf-8")
    log.info(
        "Wrote %s (%d rows across %d book(s))",
        all_books_out_path, len(all_frame), all_frame["bookmaker"].nunique() if not all_frame.empty else 0,
    )

    frame = all_frame[all_frame["bookmaker"] == BOOKMAKER].copy() if not all_frame.empty else all_frame
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8")
    log.info("Wrote %s (%d rows)", out_path, len(frame))
    if frame.empty:
        log.info("API fetch succeeded but returned zero valid Pinnacle 1X2 rows.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CSL Pinnacle 1X2 (h2h) odds from The Odds API and export CSV"
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--regions",
        default=DEFAULT_REGIONS,
        help="Regions parameter for The Odds API (default: us)",
    )
    args = parser.parse_args()

    try:
        run(out_path=args.out, regions=args.regions)
    except requests.HTTPError as exc:
        log.error("The Odds API HTTP error: %s", exc)
        sys.exit(1)
    except requests.RequestException as exc:
        log.error("The Odds API request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
