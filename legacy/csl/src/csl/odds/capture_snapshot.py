"""Single-shot Pinnacle snapshot capture -> append to the history store.

Phase 2 of the scheduled odds-capture pipeline (AGENTS.md roadmap #2). One run
polls The Odds API once (the whole CSL slate comes back in a single request),
tags the rows with a ``snapshot_type`` and appends them to the append-only history
CSV via ``snapshot_store``.

Reuses the fetch/parse machinery in ``fetch_pinnacle_spreads`` (team-name mapping,
row extraction) so the capture path and the one-shot path stay schema-identical.

Quota guard: the free Odds-API plan is capped (~500 requests/month). Before spending
the paid ``/odds`` request we read the remaining quota from the *free* ``/sports``
endpoint (0 quota cost) and abort if it is below ``--min-remaining``.

Usage (repo root, PYTHONPATH=src, THE_ODDS_API_KEY set):
    python -m csl.odds.capture_snapshot --snapshot-type open --target-round 18 \
        --capture-reason "open-window: Shanghai Port vs Chengdu"
    python -m csl.odds.capture_snapshot --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import requests

from csl.odds.fetch_pinnacle_spreads import (
    DEFAULT_REGIONS,
    ODDS_SPORT_KEY,
    THE_ODDS_API_BASE_URL,
    extract_rows,
    fetch_odds_response,
    get_api_key,
    load_team_mapping,
    rows_to_frame,
)
from csl.odds.snapshot_store import HISTORY_CSV, append_snapshots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MIN_REMAINING = 50


def read_quota(api_key: str) -> tuple[int | None, int | None, bool]:
    """Hit the free ``/sports`` endpoint to read quota + confirm the CSL key is live.

    ``/sports`` costs 0 quota, so this is a safe pre-spend check. Returns
    ``(remaining, used, csl_available)``; the counts are None if the headers are
    absent (older API behaviour) so callers can decide how to proceed.
    """
    url = f"{THE_ODDS_API_BASE_URL}/sports"
    response = requests.get(url, params={"apiKey": api_key}, timeout=30)
    response.raise_for_status()

    def _as_int(name: str) -> int | None:
        raw = response.headers.get(name)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    remaining = _as_int("x-requests-remaining")
    used = _as_int("x-requests-used")

    sports = response.json()
    csl_available = isinstance(sports, list) and any(
        isinstance(s, dict) and s.get("key") == ODDS_SPORT_KEY for s in sports
    )
    return remaining, used, csl_available


def capture(
    *,
    snapshot_type: str,
    target_round: str,
    capture_reason: str,
    regions: str,
    min_remaining: int,
    dry_run: bool,
    history_path: str = HISTORY_CSV,
) -> int:
    """Run one capture. Returns the number of rows appended (0 on dry-run/guard/no-op)."""
    api_key = get_api_key()

    remaining, used, csl_available = read_quota(api_key)
    log.info(
        "Quota: remaining=%s used=%s | %s in slate: %s",
        remaining, used, ODDS_SPORT_KEY, csl_available,
    )

    # Pre-spend guard: bail before the paid /odds request if quota is too low.
    if remaining is not None and remaining < min_remaining:
        log.warning(
            "Aborting: quota remaining=%d below threshold=%d; not spending a request.",
            remaining, min_remaining,
        )
        return 0

    if dry_run:
        log.info("Dry run: connectivity OK, no /odds request spent, nothing written.")
        return 0

    mapping = load_team_mapping()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    response = fetch_odds_response(api_key, regions)  # 1 paid request
    events = response.json()
    if not isinstance(events, list):
        raise ValueError(f"Expected list response from The Odds API, got: {type(events)}")
    log.info("Fetched %d events from The Odds API", len(events))

    rows = extract_rows(events, mapping, regions=regions, fetched_at=fetched_at)
    frame = rows_to_frame(rows)
    if frame.empty:
        log.info("No valid Pinnacle 1X2 rows in response; nothing appended.")
        return 0

    _, appended = append_snapshots(
        frame,
        snapshot_type=snapshot_type,
        target_round=target_round,
        capture_reason=capture_reason,
        path=history_path,
    )
    return appended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a single Pinnacle spreads snapshot and append to the history store."
    )
    parser.add_argument(
        "--snapshot-type",
        default="ad_hoc",
        choices=["open", "close", "ad_hoc"],
        help="Why this capture fired (default: ad_hoc)",
    )
    parser.add_argument("--target-round", default="", help="Round this capture targets")
    parser.add_argument("--capture-reason", default="", help="Free-text audit label")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Odds API regions (default: us)")
    parser.add_argument(
        "--min-remaining",
        type=int,
        default=DEFAULT_MIN_REMAINING,
        help="Abort before spending a request if quota remaining is below this (default: 50)",
    )
    parser.add_argument("--out", default=HISTORY_CSV, help="History CSV path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check connectivity/quota via the free /sports endpoint; spend nothing, write nothing",
    )
    args = parser.parse_args()

    try:
        capture(
            snapshot_type=args.snapshot_type,
            target_round=args.target_round,
            capture_reason=args.capture_reason,
            regions=args.regions,
            min_remaining=args.min_remaining,
            dry_run=args.dry_run,
            history_path=args.out,
        )
    except requests.HTTPError as exc:
        log.error("The Odds API HTTP error: %s", exc)
        sys.exit(1)
    except requests.RequestException as exc:
        log.error("The Odds API request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
