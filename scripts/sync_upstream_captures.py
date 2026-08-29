#!/usr/bin/env python3
"""Bring across capture rows the source repositories collected after the merge began.

Only the two irreplaceable stores: the capture history and the unpriced-observation
log. An opening line exists only while a book shows it and no provider sells opener
history, so these are the rows that cannot be recovered by rerunning anything.

**Deliberately not the match table.** That file has since gained an explicit
kickoff column, reduced odds columns and corrected xG, none of which exist
upstream, and results are already current here because the fixture and xG stages
pull them straight from the providers. Copying it wholesale would trade a
recoverable file for an unrecoverable regression.

Reads from ``origin/main`` without touching the source working trees. Idempotent:
the store's dedup key means a second run appends nothing.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pandas as pd

from betmodel import paths
from betmodel.odds import capture_store, capture_watch

SOURCES = {
    "csl": ("/Users/jordan/Developer/python/cslmonitor", {
        "history": "data/raw_data/CHN_pinnacle_spreads_history.csv",
        "watch": None,  # this pipeline never recorded unpriced observations
    }),
    "ligamx": ("/Users/jordan/Developer/python/ligamxterminal", {
        "history": "data/MEX_odds_capture_history.csv",
        "watch": "data/MEX_capture_watch.csv",
    }),
}


def _at_origin(repo: str, path: str) -> pd.DataFrame | None:
    subprocess.run(["git", "-C", repo, "fetch", "-q", "origin", "main"], check=False)
    result = subprocess.run(
        ["git", "-C", repo, "show", f"origin/main:{path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return pd.read_csv(io.StringIO(result.stdout), dtype=str, keep_default_na=False)


def main() -> int:
    dry = "--check" in sys.argv
    problems = []
    for league, (repo, files) in SOURCES.items():
        lp = paths.for_league(league)

        upstream = _at_origin(repo, files["history"])
        if upstream is None:
            problems.append(f"{league}: cannot read {files['history']}")
            continue
        before = len(capture_store.load_history(lp.capture_history_csv))
        if not dry:
            _append_preserving_type(upstream, lp.capture_history_csv)
        after = len(capture_store.load_history(lp.capture_history_csv))
        print(f"  {league:7s} history  upstream {len(upstream):4d}  "
              f"local {before:4d} -> {after:4d}  (+{after - before})")

        if files["watch"]:
            watch = _at_origin(repo, files["watch"])
            if watch is not None and not dry:
                added = capture_watch.record_unpriced(
                    [(r.home_team, r.away_team, r.bookmaker) for r in watch.itertuples()],
                    path=lp.capture_watch_csv,
                    observed_at="",  # each row carries its own, restored below
                )
                _restore_watch_times(watch, lp.capture_watch_csv)
                print(f"  {league:7s} watch    upstream {len(watch):4d}  (+{added})")

    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}", file=sys.stderr)
        return 1
    return 0


def _append_preserving_type(upstream: pd.DataFrame, path: str) -> int:
    """Append rows keeping each one's own snapshot_type.

    The store stamps a single type per call, so upstream rows are grouped by the
    type they already carry rather than relabelled.
    """
    total = 0
    for snapshot_type, group in upstream.groupby("snapshot_type"):
        if snapshot_type not in capture_store.VALID_SNAPSHOT_TYPES:
            continue
        # preserve_meta: these rows were captured elsewhere and already carry
        # their own target_round and capture_reason. capture_reason is the only
        # record of how a price was obtained, and blanking it would drop the row
        # out of the opening-line series.
        _, appended = capture_store.append_snapshots(
            group, path=path, snapshot_type=str(snapshot_type), preserve_meta=True,
        )
        total += appended
    return total


def _restore_watch_times(upstream: pd.DataFrame, path: str) -> None:
    """Keep each observation's original timestamp.

    The moment we first saw a fixture unpriced is the evidence itself; stamping
    it with now would claim we started watching later than we did and silently
    downgrade every opener proof that depends on it.
    """
    local = capture_watch.load_watch(path)
    if local.empty:
        return
    known = {
        (r.home_team, r.away_team, r.bookmaker): r.first_seen_unpriced_at
        for r in upstream.itertuples()
    }
    local["first_seen_unpriced_at"] = [
        known.get((r.home_team, r.away_team, r.bookmaker), r.first_seen_unpriced_at)
        or r.first_seen_unpriced_at
        for r in local.itertuples()
    ]
    local.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
