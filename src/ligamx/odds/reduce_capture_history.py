"""Collapse the capture history into MEX_ligamx.csv's open/close columns.

Companion to reduce_open_close, which does the same job for the dense poll store.
The two are separate because the inputs differ in kind: that store is a price
curve that has to be searched for the right moment, while every row here was
captured at a deliberate moment already. So there is no "snapshot nearest
kickoff - Delta" search, and no in-play guard -- the capture window did that work.

    open   earliest fetched_at per (fixture, book)  -- the first price we ever saw
    close  latest fetched_at per (fixture, book), if within MAX_CLOSE_LEAD_HOURS

Which opens can be trusted
--------------------------
Not all of them, and the difference matters more than anything else in this file:
the 461 hand-collected Betano openers are the only positive-EV result the project
has, and quietly mixing mid-market prices into that series would corrupt it.

A captured "open" is a real opening line only if we were *watching before the book
priced it*. Two independent things can establish that, and either is sufficient:

  OBSERVED  we saw this (fixture, book) carrying no price at a moment before we
            captured one, so what we captured is by definition the first price
            that book posted. Recorded live by the capture loop -- see
            capture_watch, and note the evidence cannot be reconstructed after
            the fact because an unpriced fixture leaves no odds row.

  WINDOW    (kickoff - lookahead_days) >= the first capture we ever made. The
            fixture only became observable after capture began, so nothing about
            its price can have escaped us.

WINDOW alone was the original rule. It is sound but conservative in a way that
costs real data: it rejects a fixture that was already inside the lookahead when
capture began even when no book had priced it yet. On 2026-08-12 that was the
entire 8/22 round -- nine fixtures, both books unpriced, every one destined to
produce a genuine opener, all of which WINDOW would have thrown away. OBSERVED is
a proof where WINDOW is a proxy, so it decides those cases.

Both self-calibrate: neither needs an assumption about when any book opens, and on
day one, with no evidence of either kind, everything is correctly rejected. Rows
that fail stay in the history -- they are perfectly good price observations -- they
just never reach MEX_ligamx.csv. `open_proof` records which test passed.

Nothing here overwrites. pinnacle_close_* already holds 856 values assembled from
three sources and repaired by hand; a reducer that clobbered them would undo work
that cannot be redone.

    python -m ligamx.odds.reduce_capture_history [--dry-run]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ligamx import config, paths
from ligamx.date_utils import parse_date_only_series
from ligamx.odds.capture_store import load_history
from ligamx.odds.capture_watch import opener_proof, watched_before
from ligamx.odds.fetch_oddsapiio_opens import DEFAULT_LOOKAHEAD_DAYS
from ligamx.odds.reduce_open_close import MAX_CLOSE_LEAD_HOURS, _find_row

log = logging.getLogger(__name__)

ODDS_FIELDS = ("home_odds", "draw_odds", "away_odds")


def _prepared(history_path: str | None = None) -> pd.DataFrame:
    """The history with timestamps parsed, odds numeric and unusable rows dropped."""
    df = load_history(history_path)
    if df.empty:
        return df
    df = df.copy()
    for col in ODDS_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_ko"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    df["_prefix"] = df["bookmaker"].map(config.VENUE_TO_SCHEMA_PREFIX)
    return df.dropna(subset=["_ko", "_at", "_prefix", *ODDS_FIELDS])


def build_records(history_path: str | None = None,
                  lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
                  max_close_lead: float = MAX_CLOSE_LEAD_HOURS,
                  watch_path: str | None = None) -> list[dict]:
    """One record per (fixture, book) carrying whichever of open/close is usable."""
    df = _prepared(history_path)
    if df.empty:
        return []

    # The moment capture began. Everything kicking off less than a lookahead after
    # this could have opened unobserved -- see the module docstring.
    watched = watched_before(watch_path)
    watching_since = df["_at"].min()
    horizon = pd.Timedelta(days=lookahead_days)

    records = []
    for (home, away, prefix), g in df.groupby(["home_team", "away_team", "_prefix"],
                                              dropna=False):
        kickoff = g["_ko"].max()
        rec = {
            "home": home, "away": away, "prefix": prefix,
            "date": kickoff.strftime("%Y/%m/%d"),
            "open_h": None, "open_d": None, "open_a": None, "open_lead_h": None,
            "close_h": None, "close_d": None, "close_a": None, "close_lead_h": None,
            "open_trusted": False, "open_proof": "",
        }

        opens = g[g["snapshot_type"] == "open"]
        if not opens.empty:
            first = opens.loc[opens["_at"].idxmin()]
            rec["open_lead_h"] = (kickoff - first["_at"]).total_seconds() / 3600.0

            # Two independent proofs, either sufficient; see capture_watch.
            proof = opener_proof(home=home, away=away, bookmaker=first.bookmaker,
                                 captured_at=first["_at"], kickoff=kickoff,
                                 watched=watched, watching_since=watching_since,
                                 horizon=horizon)
            rec["open_trusted"] = bool(proof)
            rec["open_proof"] = proof
            if rec["open_trusted"]:
                rec.update(open_h=first.home_odds, open_d=first.draw_odds,
                           open_a=first.away_odds)

        closes = g[g["snapshot_type"] == "close"]
        if not closes.empty:
            last = closes.loc[closes["_at"].idxmax()]
            lead = (kickoff - last["_at"]).total_seconds() / 3600.0
            rec["close_lead_h"] = lead
            # Negative lead is an in-play print; too-large is not a close at all.
            if 0 <= lead <= max_close_lead:
                rec.update(close_h=last.home_odds, close_d=last.draw_odds,
                           close_a=last.away_odds)

        records.append(rec)
    return records


def merge(records: list[dict], dry_run: bool = False) -> dict:
    """Write records into MEX_ligamx.csv, filling blanks only.

    A cell that already holds a value is left alone and counted in ``kept``: this
    file is the one place where a wrong number is expensive and a missing one is
    merely inconvenient.
    """
    stats = {"records": len(records), "no_row": 0, "written": 0, "kept": 0,
             "no_column": 0}
    path = paths.ligamx_data_csv()
    df = pd.read_csv(path, dtype=str)
    dates = parse_date_only_series(df["Date"])  # parsed once, not per record
    changed = False

    for rec in records:
        idx = _find_row(df, rec["date"], rec["home"], rec["away"], dates)
        if idx is None:
            stats["no_row"] += 1  # fixture not played yet, or not in the table
            continue
        prefix = rec["prefix"]
        cells = {
            f"{prefix}_open_h": rec["open_h"], f"{prefix}_open_d": rec["open_d"],
            f"{prefix}_open_a": rec["open_a"], f"{prefix}_close_h": rec["close_h"],
            f"{prefix}_close_d": rec["close_d"], f"{prefix}_close_a": rec["close_a"],
        }
        for col, val in cells.items():
            if val is None or pd.isna(val):
                continue
            if col not in df.columns:
                stats["no_column"] += 1
                continue
            if str(df.at[idx, col]).strip() not in ("", "nan"):
                stats["kept"] += 1  # never overwrite
                continue
            df.at[idx, col] = str(round(float(val), 4))
            stats["written"] += 1
            changed = True

    if changed and not dry_run:
        df.to_csv(path, index=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce the odds-capture history into MEX_ligamx open/close columns.")
    parser.add_argument("--lookahead-days", type=int, default=DEFAULT_LOOKAHEAD_DAYS,
                        help="Must match the capture setting: it defines when a "
                             "fixture became observable (default: %(default)s)")
    parser.add_argument("--max-close-lead", type=float, default=MAX_CLOSE_LEAD_HOURS,
                        help="Drop the close if captured staler than this (hours)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the reduce table; write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s  %(message)s")

    records = build_records(lookahead_days=args.lookahead_days,
                            max_close_lead=args.max_close_lead)
    untrusted = [r for r in records if r["open_lead_h"] is not None and not r["open_trusted"]]
    print(f"Reduced {len(records)} (fixture x book) record(s)")
    if untrusted:
        print(f"  {len(untrusted)} open(s) held back: the book may have priced them "
              f"before capture began, so they are observations, not openers")
    if args.dry_run and records:
        show = pd.DataFrame(records)[
            ["date", "home", "away", "prefix", "open_trusted", "open_lead_h",
             "open_h", "open_d", "open_a", "close_lead_h", "close_h", "close_d", "close_a"]]
        print(show.to_string(index=False))

    stats = merge(records, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}merge: {stats}")


if __name__ == "__main__":
    main()
