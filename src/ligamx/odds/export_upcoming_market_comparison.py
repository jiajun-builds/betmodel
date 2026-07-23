"""
Join model probabilities to the current Pinnacle Asian-Handicap line, compute
per-side EV, and emit the market-comparison dataset that drives the dashboard's
EV view. Reads the cached Now-line CSV if present (no extra API spend).

Note: EV here is Asian-Handicap based (the current strategy). The 1X2 + draw
de-bias path is deferred until the Liga MX backtest decides 1X2-vs-AH.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import pandas as pd

from ligamx import config, paths
from ligamx.odds.fetch_pinnacle_spreads import get_odds
from ligamx.odds.prediction_model import PredictionModel
from ligamx.odds.ev_calculator import EVCalculator

COLUMNS = [
    "home_team", "away_team", "round", "match_date", "match_time", "kickoff_utc",
    "home_win_prob", "draw_prob", "away_win_prob", "home_minus_1", "away_minus_1",
    "spread", "home_odds", "away_odds", "home_ev", "away_ev", "home_is_ev", "away_is_ev",
    "signal_pick", "signal_state", "bookmaker", "last_update", "fetched_at",
]


def _load_cached_odds() -> list:
    try:
        df = pd.read_csv(paths.pinnacle_spreads_csv())
        return df.to_dict("records")
    except Exception:
        return []


def _load_upcoming_lookup() -> dict:
    try:
        df = pd.read_csv(paths.upcoming_fixtures_csv())
    except Exception:
        return {}
    return {(str(r["Home"]), str(r["Away"])): r for _, r in df.iterrows()}


def run(odds=None) -> list:
    if odds is None:
        odds = _load_cached_odds() or get_odds()

    model = PredictionModel()
    model.load()
    upcoming = _load_upcoming_lookup()
    ev_calc = EVCalculator()
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    unmatched = []
    for od in odds:
        oh, oa = od.get("home_team"), od.get("away_team")
        probs = model.get_probabilities(oh, oa)
        if not probs:
            unmatched.append((oh, oa))
            continue

        std_home = config.ODDS_TO_STANDARD.get(oh, oh)
        std_away = config.ODDS_TO_STANDARD.get(oa, oa)
        spread = od.get("spread", 0) or 0
        ho, ao = od.get("home_odds"), od.get("away_odds")

        home_win = probs.get("home_win", 0)
        draw = probs.get("draw", 0)
        away_win = probs.get("away_win", 0)
        h1 = probs.get("Home -1", 0)
        a1 = probs.get("Away -1", 0)

        evs = ev_calc.calculate_match_evs(home_win, draw, away_win, h1, a1, ho, ao, spread)
        he, ae = evs["home_ev"], evs["away_ev"]

        signal_pick = "none"
        if he > 0 or ae > 0:
            signal_pick = "home" if he >= ae else "away"
        signal_state = "bet" if signal_pick != "none" else "none"

        uf = upcoming.get((std_home, std_away))
        commence = str(od.get("commence_time", ""))
        rows.append({
            "home_team": std_home,
            "away_team": std_away,
            "round": (uf["round"] if uf is not None else ""),
            "match_date": (uf["Date"] if uf is not None else commence[:10]),
            "match_time": (uf["Time"] if uf is not None else commence[11:16]),
            "kickoff_utc": (uf["kickoff_utc"] if uf is not None else commence),
            "home_win_prob": home_win,
            "draw_prob": draw,
            "away_win_prob": away_win,
            "home_minus_1": h1,
            "away_minus_1": a1,
            "spread": spread,
            "home_odds": ho,
            "away_odds": ao,
            "home_ev": round(he * 100, 3),
            "away_ev": round(ae * 100, 3),
            "home_is_ev": he > 0,
            "away_is_ev": ae > 0,
            "signal_pick": signal_pick,
            "signal_state": signal_state,
            "bookmaker": od.get("bookmaker", config.BOOKMAKER),
            "last_update": od.get("last_update", ""),
            "fetched_at": fetched_at,
        })

    rows.sort(key=lambda r: str(r["kickoff_utc"]))
    os.makedirs(paths.data_output_dir(), exist_ok=True)
    out = paths.market_comparison_csv()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} market-comparison rows -> {out}")
    if unmatched:
        print("Unmatched odds fixtures (no model row / name mismatch):")
        for h, a in unmatched:
            print(f"   - {h} vs {a}")
    return rows


if __name__ == "__main__":
    run()
