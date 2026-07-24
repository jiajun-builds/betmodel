"""
Join model probabilities to the current Pinnacle 1X2 (h2h) line, compute
per-outcome EV, and emit the market-comparison dataset that drives the
dashboard's EV view. Reads the cached Now-line CSV if present (no extra API spend).

EV here is 1X2 based: for each outcome EV = model_prob * decimal_odds - 1.
The firing signal is the highest-EV outcome among home/draw/away with EV > 0.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import pandas as pd

from ligamx import config, paths
from ligamx.odds.fetch_pinnacle_h2h import get_odds
from ligamx.odds.prediction_model import PredictionModel
from ligamx.odds.ev_calculator import EVCalculator

COLUMNS = [
    "home_team", "away_team", "round", "match_date", "match_time", "kickoff_utc",
    "home_win_prob", "draw_prob", "away_win_prob",
    "home_odds", "draw_odds", "away_odds",
    "home_ev", "draw_ev", "away_ev",
    "home_is_ev", "draw_is_ev", "away_is_ev",
    "signal_pick", "signal_state", "bookmaker", "last_update", "fetched_at",
]


def _load_cached_odds() -> list:
    try:
        df = pd.read_csv(paths.pinnacle_h2h_csv())
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
        ho, do, ao = od.get("home_odds"), od.get("draw_odds"), od.get("away_odds")

        home_win = probs.get("home_win", 0)
        draw = probs.get("draw", 0)
        away_win = probs.get("away_win", 0)

        evs = ev_calc.calculate_1x2_evs(home_win, draw, away_win, ho, do, ao)
        he, de, ae = evs["home_ev"], evs["draw_ev"], evs["away_ev"]

        # Firing signal = best +EV outcome among home / draw / away.
        candidates = {"home": he, "draw": de, "away": ae}
        best_pick = max(candidates, key=candidates.get)
        signal_pick = best_pick if candidates[best_pick] > 0 else "none"
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
            "home_odds": ho,
            "draw_odds": do,
            "away_odds": ao,
            "home_ev": round(he * 100, 3),
            "draw_ev": round(de * 100, 3),
            "away_ev": round(ae * 100, 3),
            "home_is_ev": he > 0,
            "draw_is_ev": de > 0,
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
