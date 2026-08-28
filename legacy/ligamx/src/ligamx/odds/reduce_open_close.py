"""
Reduce the append-only snapshot store into open/close odds on MEX_ligamx.csv.

For each (venue, match) the store holds a price time-series. This collapses it to:
  open  = snapshot nearest to (kickoff - Delta), among pre-kickoff snapshots
  close = last snapshot before kickoff, but only if it is within
          MAX_CLOSE_LEAD_HOURS of kickoff (else left blank)
and writes {prefix}_open_/{prefix}_close_ columns for the venues that have a
schema column (pinnacle, betfair, polymarket). matchbook stays store-only.

Because the store keeps the whole curve, Delta is a analysis-time choice
(--delta-hours), not baked into capture. Only rows already present in
MEX_ligamx.csv (i.e. played matches) are filled; future matches wait in the store.

    python -m ligamx.odds.reduce_open_close [--delta-hours 48] [--dry-run]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd

from ligamx import config, paths
from ligamx.date_utils import parse_date_only_series
from ligamx.odds import snapshot_store
from ligamx.odds.capture_odds import _parse_iso

# A "close" is only a close if we actually captured near kickoff. Taking the last
# pre-kickoff snapshot unconditionally is what wrote 13 Apertura-2026 rows of
# Pinnacle prices captured a median 35h out (range 11-57h) into pinnacle_close_*;
# they carried 105.5% overround against 103.4% for real Pinnacle closes, because
# a book's margin is still converging that far from kickoff. Beyond this gate the
# close is left blank -- a missing benchmark is recoverable, a silently wrong one
# poisons every CLV number computed against it.
MAX_CLOSE_LEAD_HOURS = 6.0

# A 3-way price this lopsided is in-play: the match has started. Used to find the
# post-kickoff tail of a curve independently of any recorded kickoff, because the
# recorded kickoff is exactly what cannot be trusted -- MEX_ligamx's Time mixes
# local and UTC, and Polymarket's own gameStartTime is wrong on some events too
# (Tijuana-Mazatlan 2026-02-22 carries gameStartTime 02-23T03:06Z while its book
# had already settled at 02-22T12:00Z, which is impossible). The prices are the
# one source that cannot lie about whether the match is under way.
#
# The level is measured, not chosen to taste: across 29,155 stored observations at
# T-48h or deeper -- which cannot contain in-play information -- the most lopsided
# Liga MX 1X2 book ever reached is 0.836, and NONE exceeds 0.85. Among in-play
# prints 33.9% do. So a "close" above this is contamination, not a strong favourite.
RESOLVED_PROB = 0.85


def _std(name: str) -> str:
    return config.ODDS_TO_STANDARD.get(name, name)


def _drop_post_match(g: pd.DataFrame) -> pd.DataFrame:
    """Cut a fixture's curve at the first resolved observation.

    Everything from that point on is post-match by definition, so it can never be
    part of an open or a close no matter what kickoff the row is stamped with. This
    is the guard that does not depend on a trustworthy timestamp; the kickoff-based
    filter still runs afterwards and does the ordinary work.
    """
    inv = 1.0 / g[["home_odds", "draw_odds", "away_odds"]].to_numpy(dtype=float)
    p = inv / inv.sum(axis=1, keepdims=True)
    resolved = p.max(axis=1) >= RESOLVED_PROB
    if not resolved.any():
        return g
    first = g["_cap"].to_numpy()[resolved].min()
    return g[g["_cap"] < first]


def reduce(delta_hours: int = 48,
           max_close_lead: float = MAX_CLOSE_LEAD_HOURS) -> list[dict]:
    """Collapse the store to one open/close record per (match, venue).

    ``close_*`` is None when the freshest pre-kickoff snapshot is staler than
    ``max_close_lead`` hours; ``close_lead_h`` always reports what was available
    so the exclusion is auditable.
    """
    df = snapshot_store.load()
    if df.empty:
        return []
    for c in ("home_odds", "draw_odds", "away_odds"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_cap"] = df["captured_at"].map(_parse_iso)
    df["_ko"] = df["commence_time"].map(_parse_iso)
    df["home_std"] = df["home_team"].map(_std)
    df["away_std"] = df["away_team"].map(_std)

    records = []
    grp = df.groupby(["home_std", "away_std", "commence_time", "venue"], dropna=False)
    for (home, away, commence, venue), g in grp:
        prefix = config.VENUE_TO_SCHEMA_PREFIX.get(venue)
        if prefix is None:
            continue  # store-only venue (matchbook)
        ko = _parse_iso(commence)
        g = g.dropna(subset=["_cap", "home_odds", "draw_odds", "away_odds"])
        if g.empty:
            continue
        g = _drop_post_match(g)
        if g.empty:
            continue
        pre = g[g["_cap"] <= ko] if ko is not None else g
        pre = pre if not pre.empty else g
        close_row = pre.loc[pre["_cap"].idxmax()]
        if ko is not None:
            target = ko - timedelta(hours=delta_hours)
            open_row = pre.loc[(pre["_cap"] - target).abs().idxmin()]
            close_lead = (ko - close_row["_cap"]).total_seconds() / 3600.0
        else:
            open_row = pre.loc[pre["_cap"].idxmin()]
            close_lead = None
        is_close = close_lead is not None and close_lead <= max_close_lead
        records.append({
            "home_std": home, "away_std": away, "commence": commence,
            "date": ko.strftime("%Y/%m/%d") if ko else "", "venue": venue, "prefix": prefix,
            "n": len(g), "close_lead_h": close_lead,
            "open_h": open_row.home_odds, "open_d": open_row.draw_odds, "open_a": open_row.away_odds,
            "close_h": close_row.home_odds if is_close else None,
            "close_d": close_row.draw_odds if is_close else None,
            "close_a": close_row.away_odds if is_close else None,
        })
    return records


def _find_row(df: pd.DataFrame, date: str, home: str, away: str,
              dates: pd.Series | None = None) -> int | None:
    """Match a MEX_ligamx row by (Home, Away) with the nearest Date (+/-1 day).

    A candidate whose date cannot be read is skipped, never accepted. The previous
    version scored an unparseable date as ``diff = 0`` -- a perfect match -- so when
    MEX_ligamx came back from a spreadsheet as DD/MM/YYYY every date failed to parse
    and every fixture bound to the FIRST meeting of that pair, writing 2026 prices
    onto 2023 rows. Dates now go through the tolerant shared parser, so both shapes
    work and anything genuinely unreadable is dropped rather than silently matched.
    """
    cand = df[(df["Home"] == home) & (df["Away"] == away)]
    if cand.empty:
        return None
    target = parse_date_only_series(pd.Series([date])).iloc[0]
    if pd.isna(target):
        return None
    if dates is None:
        dates = parse_date_only_series(df["Date"])
    best, best_diff = None, None
    for idx in cand.index:
        d = dates.loc[idx]
        if pd.isna(d):
            continue
        diff = abs((d - target).days)
        if diff <= 1 and (best_diff is None or diff < best_diff):
            best, best_diff = idx, diff
    return best


def merge(records: list[dict], dry_run: bool = False) -> dict:
    stats = {"records": len(records), "matched": 0, "no_row": 0, "written": 0}
    df = pd.read_csv(paths.ligamx_data_csv(), dtype=str)
    dates = parse_date_only_series(df["Date"])  # parsed once, not per record
    changed = False
    for rec in records:
        idx = _find_row(df, rec["date"], rec["home_std"], rec["away_std"], dates)
        if idx is None:
            stats["no_row"] += 1
            continue
        stats["matched"] += 1
        p = rec["prefix"]
        cols = {f"{p}_open_h": rec["open_h"], f"{p}_open_d": rec["open_d"], f"{p}_open_a": rec["open_a"],
                f"{p}_close_h": rec["close_h"], f"{p}_close_d": rec["close_d"], f"{p}_close_a": rec["close_a"]}
        for c, v in cols.items():
            if c in df.columns and pd.notna(v):
                df.at[idx, c] = str(round(float(v), 4))  # CSV is str-dtype
        stats["written"] += 1
        changed = True
    if changed and not dry_run:
        df.to_csv(paths.ligamx_data_csv(), index=False)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Reduce odds snapshots into MEX_ligamx open/close.")
    ap.add_argument("--delta-hours", type=int, default=48, help="lead time before kickoff for 'open'")
    ap.add_argument("--max-close-lead", type=float, default=MAX_CLOSE_LEAD_HOURS,
                    help="drop the close if the last snapshot is staler than this (hours)")
    ap.add_argument("--dry-run", action="store_true", help="print the reduce table; don't write the CSV")
    args = ap.parse_args()

    records = reduce(args.delta_hours, args.max_close_lead)
    dropped = sum(r["close_h"] is None for r in records)
    print(f"Reduced {len(records)} (match x venue) open/close records "
          f"(Delta={args.delta_hours}h, max close lead={args.max_close_lead}h)")
    print(f"  {dropped} closes dropped as too stale to be a close\n")
    if args.dry_run:
        show = pd.DataFrame(records)
        if not show.empty:
            show = show[["date", "home_std", "away_std", "venue", "n", "close_lead_h",
                         "open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]]
            print(show.to_string(index=False))
    stats = merge(records, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}merge: {stats}")


if __name__ == "__main__":
    main()
