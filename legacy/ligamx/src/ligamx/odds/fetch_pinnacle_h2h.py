"""
Fetch the current ('Now') Pinnacle 1X2 ('h2h') line for upcoming Liga MX
matches from The Odds API, and cache it to MEX_pinnacle_h2h.csv.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

import requests

from ligamx import config, paths

COLUMNS = [
    "event_id", "commence_time", "home_team", "away_team",
    "home_odds", "draw_odds", "away_odds", "bookmaker", "last_update", "fetched_at",
]


def _get(endpoint: str, params: dict) -> list:
    url = f"{config.THE_ODDS_API_BASE_URL}{endpoint}"
    params = dict(params)
    params["apiKey"] = config.THE_ODDS_API_KEY
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_odds(regions: str = "eu", markets: str = "h2h") -> list:
    """Return the current Pinnacle 1X2 (home/draw/away) line per upcoming fixture."""
    params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "bookmakers": config.BOOKMAKER,
    }
    data = _get(f"/sports/{config.ODDS_SPORT_KEY}/odds", params)

    results = []
    for event in data:
        bms = event.get("bookmakers", [])
        if not bms:
            continue
        bm = bms[0]
        mkts = bm.get("markets", [])
        if not mkts:
            continue
        outcomes = mkts[0].get("outcomes", [])

        home_team = event.get("home_team")
        away_team = event.get("away_team")
        home_odds = draw_odds = away_odds = None
        for o in outcomes:
            name = o.get("name")
            if name == home_team:
                home_odds = o.get("price")
            elif name == away_team:
                away_odds = o.get("price")
            elif name == "Draw":
                draw_odds = o.get("price")

        if home_odds and draw_odds and away_odds:
            results.append({
                "event_id": event.get("id"),
                "commence_time": event.get("commence_time"),
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "bookmaker": config.BOOKMAKER,
                "last_update": bm.get("last_update"),
            })
    return results


def fetch_and_save() -> list:
    odds = get_odds()
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = paths.pinnacle_h2h_csv()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for o in odds:
            row = dict(o)
            row["fetched_at"] = fetched_at
            w.writerow(row)
    print(f"Fetched {len(odds)} Pinnacle 1X2 lines -> {out}")
    return odds


if __name__ == "__main__":
    fetch_and_save()
