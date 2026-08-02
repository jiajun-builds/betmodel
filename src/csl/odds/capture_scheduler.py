"""Scheduler "tick": capture opening lines only when a fixture is in its window.

Phase 3 of the scheduled odds-capture pipeline (AGENTS.md roadmap #2). Meant to be
run frequently (e.g. every ~10 min by launchd/cron — Phase 4). Each tick:

  1. Builds predicted open windows from ``opening_calendar`` (local CSVs only, no network).
  2. Finds fixtures whose window ``[open_from, open_to]`` contains "now" AND that have
     no ``snapshot_type=open`` row in the history yet ("pending").
  3. If none are pending, exits doing nothing — an idle tick costs zero API quota.
  4. Otherwise spends ONE ``/odds`` request (the whole slate comes back at once),
     keeps only the pending fixtures' rows, and appends them as ``open``.

Because one request returns every fixture, several simultaneously-in-window matches
are covered by a single call. Non-in-window fixtures in that same response are
DISCARDED — each fixture's ``open`` row must be its own true opening line, so a
fixture is only ever stored as ``open`` while it is inside its own window.

Quota guard: the free plan is capped, so before spending we read remaining quota
from the free ``/sports`` endpoint and abort below ``--min-remaining``.

Usage (repo root, PYTHONPATH=src, THE_ODDS_API_KEY set):
    python -m csl.odds.capture_scheduler
    python -m csl.odds.capture_scheduler --dry-run   # decide only, spend/write nothing
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import requests

from csl.odds.capture_snapshot import DEFAULT_MIN_REMAINING, read_quota
from csl.odds.fetch_pinnacle_spreads import (
    BOOKMAKER,
    CAPTURE_BOOKMAKERS,
    DEFAULT_REGIONS,
    extract_rows,
    fetch_odds_response,
    get_api_key,
    load_team_mapping,
    rows_to_frame,
)
from csl.odds.opening_calendar import (
    DEFAULT_SCHEDULE_CSV,
    DEFAULT_TARGET_CSV,
    build_open_windows,
)
from csl.odds.snapshot_store import DEDUP_KEY, HISTORY_CSV, append_snapshots, load_history

# The scheduler's *capture* window is deliberately wider than the ~1h *display*
# window (opening_calendar.DEFAULT_WINDOW_HOURS) it is derived from. The Odds API
# lists fixtures in waves, so a fixture's feed entry (or Pinnacle's posted line) can
# appear only AFTER the validated 1h open window has closed; with a 1h capture bound
# such a fixture is never seen inside its window and its opening line is lost forever
# (observed on round 18: Shanghai Port vs Dalian Yingbo). Widening the *capture* bound
# to [anchor, min(anchor + this, kickoff)] lets a still-uncaptured fixture be grabbed
# on first feed availability after its window, at 10-min freshness.
#
# 6h → 12h (2026-07-16): 6h still missed opens that posted later (user-reported). 12h
# is essentially free in the normal case — a fixture is captured ~1h after its anchor
# and drops out immediately, so the wider bound only keeps polling for genuinely-late
# opens, which is the point. A late open costs ~1 credit per 10-min tick until it
# posts, so the bound stays finite rather than running to kickoff; anything later than
# 12h (or missed while the 10-min workflow was cron-throttled, or a fixture with no
# schedulable anchor) is caught quota-free by the 12h Now-refresh fallback
# (csl.odds.backfill_open). opening_calendar caps open_to at kickoff either way.
DEFAULT_CAPTURE_WINDOW_HOURS = 12.0

# The book whose opening line defines "captured" (roadmap #8). Since that roadmap
# item the tick STORES every book in CAPTURE_BOOKMAKERS (same 1-credit cost). It is
# kept as the *anchor* book (Pinnacle's open feeds the λ de-bias) and is one of the
# two REQUIRED_OPEN_BOOKS below.
REFERENCE_BOOKMAKER = BOOKMAKER

# The books a fixture must have an ``open`` row for before it stops being pending.
#
# Pinnacle only, since 2026-08-02. 1xBet used to be required here too (P0-1), because
# its open is the dashboard's bet price / EV basis and a line posting later than
# Pinnacle's would otherwise be lost. That whole problem now belongs to a different
# provider: 1xBet's open comes from odds-api.io via ``csl.odds.fetch_onexbet_open``,
# which polls without any predicted window (~500 requests/DAY there vs ~500/MONTH here).
#
# Reverting to Pinnacle-only is a quota *win*, not a regression: a fixture now stops
# ticking the moment Pinnacle's anchor is stored instead of continuing to spend 1 credit
# per 10-min tick waiting for a book this module no longer even fetches.
#
# Recon-only books (betfair/matchbook) are deliberately NOT required: they have patchy
# coverage (matchbook 4/8 fixtures) and would burn a fixture to window-close every time.
REQUIRED_OPEN_BOOKS = (BOOKMAKER,)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _norm(name: str) -> str:
    """Case/space-insensitive key for matching team names across sources."""
    return " ".join((name or "").split()).casefold()


def captured_open_books(history_path: str) -> dict[tuple[str, str], set[str]]:
    """Map (norm_home, norm_away) -> set of bookmaker keys that have an ``open`` row.

    The per-book view the dual-book pending rule and the append-once filter are
    keyed on: a fixture is done when this set covers every REQUIRED_OPEN_BOOKS.
    """
    hist = load_history(history_path)
    if hist.empty:
        return {}
    opens = hist[hist["snapshot_type"] == "open"]
    out: dict[tuple[str, str], set[str]] = {}
    for home, away, book in zip(opens["home_team"], opens["away_team"], opens["bookmaker"]):
        out.setdefault((_norm(home), _norm(away)), set()).add(book)
    return out


def already_captured_open(
    history_path: str, bookmaker: str = REFERENCE_BOOKMAKER
) -> set[tuple[str, str]]:
    """Set of (home, away) normalized keys with an ``open`` row for ``bookmaker``."""
    return {key for key, books in captured_open_books(history_path).items() if bookmaker in books}


def missing_required_books(captured: dict[tuple[str, str], set[str]], key: tuple[str, str]) -> set[str]:
    """REQUIRED_OPEN_BOOKS still without an ``open`` row for fixture ``key``."""
    return set(REQUIRED_OPEN_BOOKS) - captured.get(key, set())


def pending_open_fixtures(now: datetime, *, schedule_path, target_path, window_hours, history_path):
    """Fixtures whose open window contains ``now`` and still miss a REQUIRED_OPEN_BOOKS open.

    Returns a list of (home, away, round) as they appear in the calendar
    (standard names). A fixture stays pending until *every* required book's opening
    line is stored, so a book (e.g. 1xBet) that posts later than Pinnacle is still
    chased on subsequent ticks instead of being lost.
    """
    captured = captured_open_books(history_path)
    pending = []
    for w in build_open_windows(schedule_path, target_path, window_hours):
        if w.open_from is None or w.open_to is None:
            continue
        if not (w.open_from <= now <= w.open_to):
            continue
        if not missing_required_books(captured, (_norm(w.home), _norm(w.away))):
            continue
        pending.append((w.home, w.away, w.round))
    return pending


def tick(
    *,
    now: datetime | None = None,
    schedule_path: str = DEFAULT_SCHEDULE_CSV,
    target_path: str = DEFAULT_TARGET_CSV,
    window_hours: float = DEFAULT_CAPTURE_WINDOW_HOURS,
    regions: str = DEFAULT_REGIONS,
    min_remaining: int = DEFAULT_MIN_REMAINING,
    history_path: str = HISTORY_CSV,
    dry_run: bool = False,
) -> int:
    """Run one scheduler tick. Returns the number of ``open`` rows appended."""
    now = now or datetime.now(timezone.utc)

    pending = pending_open_fixtures(
        now,
        schedule_path=schedule_path,
        target_path=target_path,
        window_hours=window_hours,
        history_path=history_path,
    )
    if not pending:
        log.info("Tick %s: no fixtures in an uncaptured open window; nothing to do.", now.isoformat())
        return 0

    pending_keys = {(_norm(h), _norm(a)) for h, a, _ in pending}
    pending_rounds = sorted({r for _, _, r in pending if r})
    log.info(
        "Tick %s: %d fixture(s) in open window: %s",
        now.isoformat(), len(pending), ", ".join(f"{h} vs {a}" for h, a, _ in pending),
    )

    api_key = get_api_key()
    remaining, used, csl_available = read_quota(api_key)
    log.info("Quota: remaining=%s used=%s | CSL in slate: %s", remaining, used, csl_available)
    if remaining is not None and remaining < min_remaining:
        log.warning("Aborting: quota remaining=%d below threshold=%d.", remaining, min_remaining)
        return 0

    if dry_run:
        log.info("Dry run: would capture %d fixture(s) as open; spending/writing nothing.", len(pending))
        return 0

    mapping = load_team_mapping()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # CAPTURE_BOOKMAKERS instead of Pinnacle alone: still 1 credit (<=10 books = 1
    # region), and it reaches the eu/uk-only candidate books.
    response = fetch_odds_response(  # 1 paid request
        api_key, regions, bookmakers=",".join(CAPTURE_BOOKMAKERS)
    )
    events = response.json()
    if not isinstance(events, list):
        raise ValueError(f"Expected list response from The Odds API, got: {type(events)}")
    log.info("Fetched %d events from The Odds API", len(events))

    rows = extract_rows(events, mapping, regions=regions, fetched_at=fetched_at, bookmaker_keys=None)
    frame = rows_to_frame(rows)

    # Keep ONLY the fixtures currently in their own open window; discard the rest.
    if not frame.empty:
        keep = frame.apply(lambda r: (_norm(r["home_team"]), _norm(r["away_team"])) in pending_keys, axis=1)
        frame = frame[keep]
    if frame.empty:
        log.info("None of the in-window fixtures were present in the odds response; nothing appended.")
        return 0

    # Append each (fixture, book) open EXACTLY once: drop rows for a book that already
    # has an open for this fixture. Without this, a fixture kept pending for a missing
    # book (e.g. 1xBet) would, on each subsequent tick, re-append Pinnacle's *current*
    # line as another snapshot_type=open — a later line mislabelled as the open. (The
    # read side takes the earliest fetched_at, so the anchor stays correct, but the
    # history would accrue mislabelled rows.)
    captured = captured_open_books(history_path)
    if not frame.empty:
        needed = frame.apply(
            lambda r: r["bookmaker"] not in captured.get((_norm(r["home_team"]), _norm(r["away_team"])), set()),
            axis=1,
        )
        frame = frame[needed]
    if frame.empty:
        log.info("All in-window books already have an open on file; nothing new to append.")
        return 0

    books = sorted(frame["bookmaker"].unique())
    has_ref = REFERENCE_BOOKMAKER in books
    log.info(
        "New in-window open rows: %d across %d book(s): %s | %s present: %s",
        len(frame), len(books), ", ".join(books), REFERENCE_BOOKMAKER, has_ref,
    )
    if not has_ref:
        # Other books' rows are still worth storing (they prove those books opened
        # first), and the fixture stays pending so a later tick grabs Pinnacle's open.
        log.info(
            "%s not among the new rows this tick; storing the other books' opens and "
            "leaving any book still missing pending.", REFERENCE_BOOKMAKER,
        )

    _, appended = append_snapshots(
        frame,
        snapshot_type="open",
        target_round=",".join(pending_rounds),
        capture_reason=f"scheduler open-window tick @ {now.isoformat()}",
        path=history_path,
    )
    return appended


def main() -> None:
    parser = argparse.ArgumentParser(description="Odds-capture scheduler tick (open windows).")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE_CSV, help="Full-season schedule CSV")
    parser.add_argument("--target", default=DEFAULT_TARGET_CSV, help="Upcoming fixtures CSV")
    parser.add_argument("--window-hours", type=float, default=DEFAULT_CAPTURE_WINDOW_HOURS,
                        help="Capture window width in hours after the anchor kickoff: a fixture is "
                             "captured while now is in [anchor, anchor+this]. Wider than the ~1h "
                             "predicted-open window so feed-lagged fixtures aren't missed.")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Odds API regions (default: us)")
    parser.add_argument("--min-remaining", type=int, default=DEFAULT_MIN_REMAINING,
                        help="Abort before spending if quota remaining is below this")
    parser.add_argument("--out", default=HISTORY_CSV, help="History CSV path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Decide only: report in-window fixtures, spend/write nothing")
    args = parser.parse_args()

    try:
        tick(
            schedule_path=args.schedule,
            target_path=args.target,
            window_hours=args.window_hours,
            regions=args.regions,
            min_remaining=args.min_remaining,
            history_path=args.out,
            dry_run=args.dry_run,
        )
    except requests.RequestException as exc:
        log.error("The Odds API request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
