"""
Join model probabilities to the price you can actually bet, compute per-outcome
EV, and emit the market-comparison dataset that drives the dashboard's EV view.

The bet price is the captured OPENING line from the soft books (Betano UK, Duel),
composited per outcome. Pinnacle rides along as a reference column only.

Why not Pinnacle for EV
-----------------------
EV = model_prob * decimal_odds - 1, and `odds` has to be a price you can get.
Pinnacle is a low-vig fair-odds anchor -- useful for telling whether an edge is
the model disagreeing with the market or just a soft book being soft -- but it is
not where these bets get placed. Pricing EV off it overstates nothing and
understates a lot: the whole surviving thesis here is "pay less, don't predict
better", which only shows up against the book you actually pay.

How the composite works
-----------------------
Per outcome, the best of the two books, and a record of which one it was:

    home_odds = max(betano_home_odds, duel_home_odds)   [and home_book names it]

That is line shopping written down. Averaging would produce a number matching no
bettable price, and would understate the edge by construction. Coverage is
uneven -- measured 2026-08-12, Betano priced 6 of 10 upcoming fixtures and Duel 9
of 10 -- so a composite made of one book is normal, not a defect.

The composite's overround is lower than either book's, which is correct: shopping
two books IS a cheaper synthetic book. Do not read it as a real market price.

A note on staleness
-------------------
These are opening prices, captured once and deliberately never refreshed (see the
`missing` gate in fetch_oddsapiio_opens -- overwriting a banked opener with a
mid-market price is the one thing that capture must never do). So the price here
can be days old and may no longer be available. `price_age_h` is exported for
exactly that reason; the dashboard should show it.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import pandas as pd

from ligamx import config, paths
from ligamx.odds.capture_store import load_history
from ligamx.odds.ev_calculator import EVCalculator
from ligamx.odds.prediction_model import PredictionModel

# Books composited into the bet price, in preference order. The order only breaks
# ties, so it is cosmetic -- but keeping it fixed makes `*_book` deterministic,
# which matters if a signal ever drives an alert that dedups on the book name.
BET_BOOKS = ("betano", "duel")

# Displayed for contrast, never priced against.
REFERENCE_BOOK = "pinnacle"

SIDES = ("home", "draw", "away")

COLUMNS = [
    "home_team", "away_team", "round", "match_date", "match_time", "kickoff_utc",
    "home_win_prob", "draw_prob", "away_win_prob",
    "home_odds", "draw_odds", "away_odds",
    "home_book", "draw_book", "away_book",
    "home_ev", "draw_ev", "away_ev",
    "home_is_ev", "draw_is_ev", "away_is_ev",
    "signal_pick", "signal_state",
    "betano_home_odds", "betano_draw_odds", "betano_away_odds",
    "duel_home_odds", "duel_draw_odds", "duel_away_odds",
    "pinnacle_home_odds", "pinnacle_draw_odds", "pinnacle_away_odds",
    "pinnacle_last_update", "pinnacle_captured_at",
    "bookmaker", "price_captured_at", "price_age_h", "last_update", "fetched_at",
]


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _text(value):
    """A timestamp column as a string, with pandas' missing values flattened to "".

    read_csv turns a blank cell into NaN, which would otherwise reach the export as the
    literal "NaN" in the CSV and a float in the JSON -- a value a consumer would try to
    parse as a date. The other absent-timestamp fields here are already "", so this
    keeps missing meaning the same thing everywhere.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def load_captured_opens() -> dict:
    """(home, away) -> {book: {home/draw/away odds, last_update, captured_at}}.

    Earliest fetched_at per (fixture, book) -- the first price we ever saw, which
    is what "opening line" means here.

    Deliberately NO trust gate, unlike reduce_capture_history. That gate exists to
    keep prices we may have caught late out of the permanent record in
    MEX_ligamx.csv, where they would contaminate the backtest series. The
    dashboard is asking a different question -- "what can I bet right now" -- and
    for that a price is a price whatever its provenance.
    """
    history = load_history()
    if history.empty:
        return {}
    opens = history[(history["snapshot_type"] == "open")
                    & (history["bookmaker"].isin(BET_BOOKS))].copy()
    if opens.empty:
        return {}
    # No leading underscore: itertuples renames such columns to positional _N.
    opens["captured_at"] = pd.to_datetime(opens["fetched_at"], utc=True, errors="coerce")
    opens = opens.dropna(subset=["captured_at"]).sort_values("captured_at")

    out: dict = {}
    for row in opens.itertuples(index=False):
        key = (row.home_team, row.away_team)
        book = out.setdefault(key, {})
        if row.bookmaker in book:
            continue  # already have this book's earliest, which is the opener
        book[row.bookmaker] = {
            "home": _num(row.home_odds), "draw": _num(row.draw_odds),
            "away": _num(row.away_odds), "last_update": row.last_update,
            "captured_at": row.captured_at,
        }
    return out


def composite(books: dict) -> dict:
    """Best price per outcome across the books, plus which book supplied it."""
    best: dict = {}
    for side in SIDES:
        winner, price = None, None
        for name in BET_BOOKS:
            odds = (books.get(name) or {}).get(side)
            if odds is not None and (price is None or odds > price):
                winner, price = name, odds
        best[side] = price
        best[f"{side}_book"] = winner
    return best


def load_pinnacle_reference() -> dict:
    """(home, away) -> current Pinnacle 1X2 and its clocks, standard names.

    Carries `last_update` (Pinnacle's own, as the odds API reports it) and `fetched_at`
    (when we pulled it) alongside the prices. Both are already columns in the CSV, so
    reading them costs no API call. Without them a consumer can say what the anchor is
    but not when it was true -- which is the whole question for a signal that opened
    against Pinnacle before any soft book had put a line up.

    Unlike the captured openers these are *rolling*: fetch_pinnacle_h2h overwrites this
    file on every run, so both clocks move. Anything presenting them as a historical
    event has to freeze them at the moment it cared about. Empty if unavailable.
    """
    try:
        df = pd.read_csv(paths.pinnacle_h2h_csv())
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    out = {}
    for row in df.to_dict("records"):
        home = config.ODDS_TO_STANDARD.get(row.get("home_team"), row.get("home_team"))
        away = config.ODDS_TO_STANDARD.get(row.get("away_team"), row.get("away_team"))
        out[(home, away)] = {
            "home": _num(row.get("home_odds")), "draw": _num(row.get("draw_odds")),
            "away": _num(row.get("away_odds")),
            "last_update": row.get("last_update"), "fetched_at": row.get("fetched_at"),
        }
    return out


def load_upcoming() -> list:
    """Upcoming fixtures, soonest first. The universe this export iterates."""
    try:
        df = pd.read_csv(paths.upcoming_fixtures_csv())
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []
    return df.to_dict("records")


def run() -> list:
    model = PredictionModel()
    model.load()
    ev_calc = EVCalculator()
    opens = load_captured_opens()
    pinnacle = load_pinnacle_reference()
    now = datetime.now(timezone.utc)
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows, no_price, no_model = [], [], []
    for fx in load_upcoming():
        home, away = str(fx.get("Home", "")), str(fx.get("Away", ""))
        kickoff = pd.to_datetime(fx.get("kickoff_utc"), utc=True, errors="coerce")
        if pd.isna(kickoff) or kickoff <= now:
            continue

        books = opens.get((home, away))
        if not books:
            no_price.append((home, away))
            continue
        best = composite(books)
        if best["home"] is None and best["draw"] is None and best["away"] is None:
            no_price.append((home, away))
            continue

        probs = model.get_probabilities(home, away)
        if not probs:
            no_model.append((home, away))
            continue

        hw, dr, aw = probs["home_win"], probs["draw"], probs["away_win"]
        evs = ev_calc.calculate_1x2_evs(hw, dr, aw,
                                        best["home"], best["draw"], best["away"])
        he, de, ae = evs["home_ev"], evs["draw_ev"], evs["away_ev"]

        # Firing signal = best +EV outcome. An outcome with no price cannot fire:
        # EVCalculator returns 0.0 for missing odds, which is not > 0, but being
        # explicit here keeps a future change to that default from leaking a bet.
        candidates = {s: ev for s, ev in zip(SIDES, (he, de, ae))
                      if best[s] is not None}
        best_pick = max(candidates, key=candidates.get) if candidates else None
        signal_pick = best_pick if best_pick and candidates[best_pick] > 0 else "none"

        captured = min((b["captured_at"] for b in books.values()), default=None)
        ref = pinnacle.get((home, away), {})
        per_book = {name: (books.get(name) or {}) for name in BET_BOOKS}
        signal_book = best[f"{signal_pick}_book"] if signal_pick != "none" else ""

        rows.append({
            "home_team": home, "away_team": away,
            "round": fx.get("round", ""), "match_date": fx.get("Date", ""),
            "match_time": fx.get("Time", ""), "kickoff_utc": fx.get("kickoff_utc", ""),
            "home_win_prob": hw, "draw_prob": dr, "away_win_prob": aw,
            "home_odds": best["home"], "draw_odds": best["draw"], "away_odds": best["away"],
            "home_book": best["home_book"] or "", "draw_book": best["draw_book"] or "",
            "away_book": best["away_book"] or "",
            "home_ev": round(he * 100, 3), "draw_ev": round(de * 100, 3),
            "away_ev": round(ae * 100, 3),
            "home_is_ev": he > 0, "draw_is_ev": de > 0, "away_is_ev": ae > 0,
            "signal_pick": signal_pick,
            "signal_state": "bet" if signal_pick != "none" else "none",
            "betano_home_odds": per_book["betano"].get("home"),
            "betano_draw_odds": per_book["betano"].get("draw"),
            "betano_away_odds": per_book["betano"].get("away"),
            "duel_home_odds": per_book["duel"].get("home"),
            "duel_draw_odds": per_book["duel"].get("draw"),
            "duel_away_odds": per_book["duel"].get("away"),
            "pinnacle_home_odds": ref.get("home"),
            "pinnacle_draw_odds": ref.get("draw"),
            "pinnacle_away_odds": ref.get("away"),
            # The anchor's own clocks, so a consumer can date the reference price the
            # same way it dates the bettable one. Rolling, not banked -- see
            # load_pinnacle_reference.
            "pinnacle_last_update": _text(ref.get("last_update")),
            "pinnacle_captured_at": _text(ref.get("fetched_at")),
            "bookmaker": signal_book or "+".join(sorted(books)),
            "price_captured_at": captured.strftime("%Y-%m-%dT%H:%M:%SZ") if captured is not None else "",
            # How stale the quoted price is. These are openers, captured once and
            # never refreshed, so this is the number that says whether the price
            # on screen is still gettable.
            "price_age_h": (round((now - captured).total_seconds() / 3600.0, 1)
                            if captured is not None else None),
            "last_update": (books.get(signal_book) or {}).get("last_update", ""),
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
    print(f"  priced off {'/'.join(BET_BOOKS)} openers; {REFERENCE_BOOK} shown for reference")
    if no_price:
        print(f"  {len(no_price)} upcoming fixture(s) have no captured opener yet "
              f"(they appear once a book posts a price)")
    if no_model:
        print("  no model row / name mismatch:")
        for h, a in no_model:
            print(f"   - {h} vs {a}")
    return rows


if __name__ == "__main__":
    run()
