"""
Append-only, long-format store of captured odds snapshots.

One row per (poll, venue, match): the raw time-series that the reducer later
collapses into open/close columns. Never overwritten — every poll appends, so
the full price curve is preserved and any lead-time (Delta) can be re-derived.

Columns:
    captured_at    UTC ISO8601 timestamp of the poll
    source         "oddsapi" | "polymarket"
    venue          pinnacle | betfair_ex_eu | matchbook | polymarket
    event_id       provider event id (Odds API id, or Polymarket event slug)
    commence_time  kickoff, UTC ISO8601
    home_team      fixture home name (shared across venues for a given match)
    away_team      fixture away name
    home_odds      decimal odds (1 / implied prob)
    draw_odds      decimal odds
    away_odds      decimal odds
    liquidity      venue-reported liquidity/volume if any (else blank)
    last_update    book last_update if any (else blank)
"""

from __future__ import annotations

import csv
import os

import pandas as pd

from ligamx import paths

COLUMNS = [
    "captured_at", "source", "venue", "event_id", "commence_time",
    "home_team", "away_team", "home_odds", "draw_odds", "away_odds",
    "liquidity", "last_update",
]


def append(rows: list[dict]) -> int:
    """Append snapshot rows to the store, writing the header if it's new."""
    if not rows:
        return 0
    path = paths.odds_snapshots_csv()
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return len(rows)


def load() -> pd.DataFrame:
    """Load the full store (empty frame with the right columns if absent)."""
    path = paths.odds_snapshots_csv()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str)
