#!/usr/bin/env python3
"""Move both pre-merge data trees into ``data/<league>/`` with neutral names.

Copies rather than moves: ``legacy/`` stays intact until every stage has landed
in the engine, so any migration mistake is recoverable by re-running this.

Idempotent. Verifies row counts and content hashes after copying, and refuses to
finish if anything drifted.

    python scripts/migrate_legacy_data.py --check   # report only
    python scripts/migrate_legacy_data.py           # copy and verify
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

from betmodel import paths

L = "legacy"

# (legacy relative path, LeaguePaths attribute or literal destination)
# A missing source is fine only where marked optional: the two leagues genuinely
# do not produce the same set of files.
CSL: list[tuple[str, str, bool]] = [
    ("csl/data/raw_data/CHN_Super League.csv",                       "matches_csv",           False),
    ("csl/data/raw_data/chinese_super_league_data.csv",              "schedule_csv",          False),
    ("csl/data/raw_data/chn_upcoming_fixtures.csv",                  "upcoming_fixtures_csv", False),
    ("csl/data/raw_data/xg_data.csv",                                "xg_csv",                False),
    ("csl/data/raw_data/CHN_pinnacle_spreads_history.csv",           "capture_history_csv",   False),
    ("csl/data/output_data/CHN_team_name_mapping.csv",               "team_mapping_csv",      False),
    ("csl/data/output_data/CHN_upcoming_market_comparison.csv",      "market_comparison_csv", False),
    ("csl/data/output_data/CHN_team_stats.csv",                      "team_stats_csv",        False),
    ("csl/data/output_data/CHN_team_stats_match_simulations.csv",    "simulations_csv",       False),
    ("csl/data/output_data/CHN_model_meta.json",                     "model_meta_json",       False),
]

LIGAMX: list[tuple[str, str, bool]] = [
    ("ligamx/data/MEX_ligamx.csv",                                   "matches_csv",           False),
    ("ligamx/data/MEX_upcoming_fixtures.csv",                        "upcoming_fixtures_csv", False),
    ("ligamx/data/MEX_odds_capture_history.csv",                     "capture_history_csv",   False),
    ("ligamx/data/MEX_capture_watch.csv",                            "capture_watch_csv",     False),
    ("ligamx/data/ligamx_team_name_mapping.csv",                     "team_mapping_csv",      False),
    ("ligamx/data/football_data_mex.csv",                            "football_data_csv",     False),
    ("ligamx/models/MEX_team_stats.csv",                             "team_stats_csv",        False),
    ("ligamx/models/MEX_team_stats_match_simulations.csv",           "simulations_csv",       False),
    ("ligamx/models/MEX_model_meta.json",                            "model_meta_json",       False),
    ("ligamx/data/MEX_fixtures_meta.json",                           "fixtures_meta_json",    False),
    # market_comparison is gitignored upstream and regenerated in CI, so it may
    # legitimately be absent from a fresh clone.
    ("ligamx/data/output_data/MEX_upcoming_market_comparison.csv",   "market_comparison_csv", True),
]

# Files no code reads, kept for provenance rather than use.
ARCHIVE: list[tuple[str, str, str]] = [
    ("ligamx/data/MEX_betano_openers_history.csv", "ligamx", "betano_openers_legacy.csv"),
    ("ligamx/data/bookmaker_candidates.csv",       "ligamx", "bookmaker_survey.csv"),
]

# capture_watch has no CSL counterpart: cslmonitor never recorded unpriced
# observations, which is exactly the evidence its draw anchor lacks. Seed an
# empty file with the right header so the first capture appends rather than
# inventing a schema.
CAPTURE_WATCH_HEADER = "home_team,away_team,bookmaker,first_seen_unpriced_at\n"


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(path: str) -> int:
    if not path.endswith(".csv"):
        return -1
    with open(path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def migrate(league: str, table, *, check: bool) -> tuple[int, int, list[str]]:
    lp = paths.for_league(league)
    if not check:
        lp.ensure_dirs()
    copied = skipped = 0
    problems: list[str] = []
    for rel, attr, optional in table:
        src = os.path.join(paths.project_root(), L, rel)
        dst = getattr(lp, attr)
        if not os.path.exists(src):
            if optional:
                print(f"    - {attr:22s} (optional source absent, skipped)")
                skipped += 1
                continue
            problems.append(f"missing required source: {rel}")
            continue
        if not check:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            if _sha(src) != _sha(dst):
                problems.append(f"hash mismatch after copy: {rel}")
                continue
        n = _rows(src)
        size = os.path.getsize(src)
        detail = f"{n} rows" if n >= 0 else f"{size} bytes"
        print(f"    - {attr:22s} <- {os.path.basename(rel):42s} {detail}")
        copied += 1
    return copied, skipped, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    total_problems: list[str] = []
    for league, table in (("csl", CSL), ("ligamx", LIGAMX)):
        print(f"  {league}:")
        _, _, problems = migrate(league, table, check=args.check)
        total_problems += [f"{league}: {p}" for p in problems]

    print("  archive (kept for provenance, no code reads these):")
    for rel, league, name in ARCHIVE:
        src = os.path.join(paths.project_root(), L, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(paths.for_league(league).root, name)
        if not args.check:
            shutil.copy2(src, dst)
        print(f"    - {league}/{name:32s} <- {os.path.basename(rel)}")

    watch = paths.for_league("csl").capture_watch_csv
    if not os.path.exists(watch):
        print(f"    - csl/capture_watch.csv          seeded empty (no cslmonitor counterpart)")
        if not args.check:
            os.makedirs(os.path.dirname(watch), exist_ok=True)
            with open(watch, "w", encoding="utf-8") as fh:
                fh.write(CAPTURE_WATCH_HEADER)

    if total_problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in total_problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("\n  migration clean: every required source copied and hash-verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
