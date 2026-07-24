"""
Reduce the append-only snapshot store into open/close odds on MEX_ligamx.csv.

For each (venue, match) the store holds a price time-series. This collapses it to:
  open  = snapshot nearest to (kickoff - Delta), among pre-kickoff snapshots
  close = last snapshot before kickoff
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
from ligamx.odds import snapshot_store
from ligamx.odds.capture_odds import _parse_iso


def _std(name: str) -> str:
    return config.ODDS_TO_STANDARD.get(name, name)


def reduce(delta_hours: int = 48) -> list[dict]:
    """Collapse the store to one open/close record per (match, venue)."""
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
        pre = g[g["_cap"] <= ko] if ko is not None else g
        pre = pre if not pre.empty else g
        close_row = pre.loc[pre["_cap"].idxmax()]
        if ko is not None:
            target = ko - timedelta(hours=delta_hours)
            open_row = pre.loc[(pre["_cap"] - target).abs().idxmin()]
        else:
            open_row = pre.loc[pre["_cap"].idxmin()]
        records.append({
            "home_std": home, "away_std": away, "commence": commence,
            "date": ko.strftime("%Y/%m/%d") if ko else "", "venue": venue, "prefix": prefix,
            "n": len(g),
            "open_h": open_row.home_odds, "open_d": open_row.draw_odds, "open_a": open_row.away_odds,
            "close_h": close_row.home_odds, "close_d": close_row.draw_odds, "close_a": close_row.away_odds,
        })
    return records


def _find_row(df: pd.DataFrame, date: str, home: str, away: str) -> int | None:
    """Match a MEX_ligamx row by (Home, Away) with the nearest Date (+/-1 day)."""
    cand = df[(df["Home"] == home) & (df["Away"] == away)]
    if cand.empty:
        return None
    target = _parse_iso(date.replace("/", "-") + "T00:00:00Z")
    best, best_diff = None, None
    for idx, r in cand.iterrows():
        d = _parse_iso(str(r["Date"]).replace("/", "-") + "T00:00:00Z")
        diff = abs((d - target).days) if (d and target) else 0
        if diff <= 1 and (best_diff is None or diff < best_diff):
            best, best_diff = idx, diff
    return best


def merge(records: list[dict], dry_run: bool = False) -> dict:
    stats = {"records": len(records), "matched": 0, "no_row": 0, "written": 0}
    df = pd.read_csv(paths.ligamx_data_csv(), dtype=str)
    changed = False
    for rec in records:
        idx = _find_row(df, rec["date"], rec["home_std"], rec["away_std"])
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
    ap.add_argument("--dry-run", action="store_true", help="print the reduce table; don't write the CSV")
    args = ap.parse_args()

    records = reduce(args.delta_hours)
    print(f"Reduced {len(records)} (match x venue) open/close records "
          f"(Delta={args.delta_hours}h)\n")
    if args.dry_run:
        show = pd.DataFrame(records)
        if not show.empty:
            show = show[["date", "home_std", "away_std", "venue", "n",
                         "open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]]
            print(show.to_string(index=False))
    stats = merge(records, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}merge: {stats}")


if __name__ == "__main__":
    main()
