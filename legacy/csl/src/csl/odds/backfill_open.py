"""Fallback opening-line capture from the 12h Now-line refresh (roadmap #6/#8 gap fix).

The 10-min ``capture_scheduler`` grabs a fixture's opening line only while "now" is
inside its predicted window ``[anchor, min(anchor + capture-window, kickoff)]``. Three
things can still leave a fixture with a Now line but **no** captured open:

  1. Pinnacle posts the line later than the capture window (feed-lag beyond 12h);
  2. the 10-min capture workflow was throttled/skipped by GitHub cron across the
     whole window;
  3. the fixture has no schedulable anchor (a team's previous match is missing from
     the schedule), so it never gets a window at all.

All three used to surface as "dashboard shows current odds but no opening odds".

This module is the safety net. It runs inside the every-12h Now-line refresh — which
already fetches the whole capture slate (all CAPTURE_BOOKMAKERS), so it costs **zero
extra quota** — and, for each of ``REQUIRED_OPEN_BOOKS``, records the current line as a
fallback ``open`` for any fixture that has a Now line but no captured open for that book
**and whose primary capture window has already closed** (or never existed). The
window-closed guard means the 10-min capture keeps first crack at a fresher open while
the window is still live; only genuine misses are backfilled.

Scope note (2026-08-02): ``REQUIRED_OPEN_BOOKS`` is now Pinnacle alone, so this fallback
covers the λ anchor only. 1xBet's open is no longer sourced from The Odds API at all —
``csl.odds.fetch_oddsapiio_opens`` polls odds-api.io for it without any window, so there is
no window to miss and nothing here to back-fill for that book.

The recorded price is the line as it stands at this refresh — the best opening proxy
available once the true-open window is gone — and ``capture_reason`` marks it as a
fallback so it is never mistaken for a window-fresh open.

Usage (invoked by ``scripts/csl.sh`` after the Now-line fetch; also runnable directly):
    python -m csl.odds.backfill_open
    python -m csl.odds.backfill_open --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from csl.odds.capture_scheduler import (
    DEFAULT_CAPTURE_WINDOW_HOURS,
    REQUIRED_OPEN_BOOKS,
    _norm,
    captured_open_books,
)
from csl.odds.fetch_pinnacle_spreads import DEFAULT_ALL_BOOKS_CSV, DEFAULT_OUTPUT_CSV
from csl.odds.opening_calendar import (
    DEFAULT_SCHEDULE_CSV,
    DEFAULT_TARGET_CSV,
    build_open_windows,
)
from csl.odds.snapshot_store import HISTORY_CSV, append_snapshots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CAPTURE_REASON = "now-refresh fallback (open window missed)"


def _window_close_by_fixture(schedule_path: str, target_path: str, window_hours: float):
    """Map (norm_home, norm_away) -> (open_to, round) from the predicted calendar.

    ``open_to`` is None for a fixture with no anchor; such fixtures are always
    eligible for the fallback (they never get a live capture window).
    """
    out = {}
    for w in build_open_windows(schedule_path, target_path, window_hours):
        out[(_norm(w.home), _norm(w.away))] = (w.open_to, w.round)
    return out


def find_missed(
    now_df: pd.DataFrame,
    *,
    book: str,
    now: datetime,
    captured: dict[tuple[str, str], set[str]],
    windows: dict[tuple[str, str], tuple[datetime | None, str]],
) -> pd.DataFrame:
    """Now-line rows to store as fallback opens for ``book``: uncaptured, window closed."""
    if now_df.empty:
        return now_df
    ref = now_df[now_df["bookmaker"] == book].copy()
    if ref.empty:
        return ref

    def _eligible(row) -> bool:
        key = (_norm(str(row["home_team"])), _norm(str(row["away_team"])))
        if book in captured.get(key, set()):
            return False  # already have this book's open for the fixture
        open_to, _round = windows.get(key, (None, ""))
        # No window (no anchor) -> always eligible. Window still live -> defer to the
        # 10-min capture for a fresher open. Window closed -> this is a genuine miss.
        if open_to is not None and now <= open_to:
            return False
        return True

    return ref[ref.apply(_eligible, axis=1)]


def run(
    *,
    now: datetime | None = None,
    now_csv: str = DEFAULT_ALL_BOOKS_CSV,
    pinnacle_csv: str = DEFAULT_OUTPUT_CSV,
    history_path: str = HISTORY_CSV,
    schedule_path: str = DEFAULT_SCHEDULE_CSV,
    target_path: str = DEFAULT_TARGET_CSV,
    window_hours: float | None = None,
    dry_run: bool = False,
) -> int:
    """Append fallback opens for missed fixtures across REQUIRED_OPEN_BOOKS.

    Reads the all-book Now snapshot (``now_csv``) so BOTH the Pinnacle anchor and the
    1xBet bet price can be backfilled; falls back to the Pinnacle-only Now CSV when the
    all-book file is absent (older run / partial deploy), covering Pinnacle only then.
    Returns the total number of fallback ``open`` rows appended.
    """
    now = now or datetime.now(timezone.utc)
    if window_hours is None:
        window_hours = DEFAULT_CAPTURE_WINDOW_HOURS

    source_csv = now_csv if os.path.isfile(now_csv) else pinnacle_csv
    if not os.path.isfile(source_csv):
        log.info("No Now-line CSV at %s or %s; nothing to backfill.", now_csv, pinnacle_csv)
        return 0
    now_df = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    if now_df.empty or "bookmaker" not in now_df.columns:
        log.info("Now-line CSV %s empty or malformed; nothing to backfill.", source_csv)
        return 0
    if source_csv == pinnacle_csv:
        log.info("All-book Now CSV %s absent; backfilling from Pinnacle-only %s "
                 "(1xBet fallback unavailable this run).", now_csv, pinnacle_csv)

    captured = captured_open_books(history_path)
    windows = _window_close_by_fixture(schedule_path, target_path, window_hours)

    total_appended = 0
    for book in REQUIRED_OPEN_BOOKS:
        missed = find_missed(now_df, book=book, now=now, captured=captured, windows=windows)
        if missed.empty:
            log.info("No missed fixtures for %s: each Now-line fixture already has an "
                     "open or is still inside its capture window.", book)
            continue

        labels = ", ".join(f"{r.home_team} vs {r.away_team}" for r in missed.itertuples(index=False))
        rounds = sorted({windows.get((_norm(str(r.home_team)), _norm(str(r.away_team))), (None, ""))[1]
                         for r in missed.itertuples(index=False)} - {""})
        log.info("Fallback %s open for %d missed fixture(s): %s", book, len(missed), labels)

        if dry_run:
            log.info("Dry run: would append %d fallback %s open row(s); writing nothing.",
                     len(missed), book)
            continue

        _, appended = append_snapshots(
            missed,
            snapshot_type="open",
            target_round=",".join(rounds),
            capture_reason=CAPTURE_REASON,
            path=history_path,
        )
        # Re-read captured after each book so a fixture backfilled for one book is not
        # counted as still-missing for the next (independent keys here, but cheap safety).
        captured = captured_open_books(history_path)
        total_appended += appended

    return total_appended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill fallback opening lines from the current Now-line CSV (zero quota)."
    )
    parser.add_argument("--now-csv", default=DEFAULT_ALL_BOOKS_CSV,
                        help="All-book Now-line CSV path (source for the fallback opens)")
    parser.add_argument("--pinnacle", default=DEFAULT_OUTPUT_CSV,
                        help="Pinnacle-only Now-line CSV (fallback source if --now-csv is absent)")
    parser.add_argument("--history", default=HISTORY_CSV, help="Capture history CSV path")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE_CSV, help="Full-season schedule CSV")
    parser.add_argument("--target", default=DEFAULT_TARGET_CSV, help="Upcoming fixtures CSV")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = parser.parse_args()

    try:
        run(
            now_csv=args.now_csv,
            pinnacle_csv=args.pinnacle,
            history_path=args.history,
            schedule_path=args.schedule,
            target_path=args.target,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
