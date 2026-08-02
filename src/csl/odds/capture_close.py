"""Capture Pinnacle **closing** lines — the last price before kickoff.

The third leg of the capture pipeline (AGENTS.md roadmap #3), alongside:

    capture_scheduler     Pinnacle OPEN, The Odds API, predicted open window.
    fetch_onexbet_open    1xBet   OPEN, odds-api.io, no window.
    this module           Pinnacle CLOSE, The Odds API, pre-kickoff window.

Why the close matters: it is the CLV benchmark. Every edge claim in
``backtest/backtest.md`` is graded as "did the price we took beat Pinnacle's close",
and until now the only close lines on file were manually entered ones (2023-24 for
AH, nothing live). This closes that gap for fixtures going forward.

The 5-minute target, and why the window is an hour
--------------------------------------------------
The line we *want* is the one ~5 minutes before kickoff — that is the sharpest,
most information-complete price and the one the CLV literature grades against.
A naive implementation would therefore only fire inside ``[kickoff-5m, kickoff]``.

That does not survive contact with this repo's actual tick cadence. Measured over
299 real ``capture-odds.yml`` runs the gap between landed ticks has a median of
10min but a p75 of 74min, a p90 of 137min and a max of 232min — the external
dispatch timer dies intermittently and GitHub's cron fallback is heavily throttled
(same measurement that killed the minute-band gate on the 1xBet capture). A 5-minute
window would be *missed* on most fixtures, and a missed close is unrecoverable: the
fixture leaves the pre-match feed at kickoff and the price is gone forever.

So the target is separated from the window:

  * ``DEFAULT_CLOSE_WINDOW_MINUTES`` (60) — how early we *start* chasing the close.
    Every tick inside the window captures, so the stored close is always the latest
    price seen before kickoff. Under a healthy 10-min cadence the final capture
    lands within ~10 minutes of kickoff; under a degraded one we still get *a*
    close instead of none.
  * ``CLOSE_TARGET_MINUTES`` (5) — once a Pinnacle close is on file that was fetched
    within this many minutes of kickoff, the fixture is **final** and stops being
    pending. That is the "we got the real close, stop spending" rule; it is what
    keeps the repeated capture from running all the way to kickoff.

Read side: a fixture can therefore accumulate more than one ``close`` row as the
line moves. **The close is the row with the latest ``fetched_at``** — mirroring how
the open is read as the *earliest*. ``latest_close_rows`` implements that so
consumers don't each reinvent it.

Cost
----
The Odds API's free plan is ~500 requests/month and one ``/odds`` call is 1 request
covering the whole slate, so cost is per *tick*, not per fixture: a tick with any
pending fixture spends 1, an idle tick spends 0. CSL kickoffs cluster into waves,
so the realistic bill is ~1-6 credits per kickoff wave (fewer when the first capture
already lands near kickoff) — order ~25-60/month against the 500 cap, on top of the
open capture. ``--min-remaining`` aborts before spending if that estimate is ever wrong.

Usage (repo root, PYTHONPATH=src, THE_ODDS_API_KEY set):
    python -m csl.odds.capture_close
    python -m csl.odds.capture_close --dry-run   # decide only, spend/write nothing
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from csl.odds.capture_scheduler import _norm
from csl.odds.capture_snapshot import DEFAULT_MIN_REMAINING, read_quota
from csl.odds.fetch_onexbet_open import Fixture, load_upcoming
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
from csl.odds.opening_calendar import DEFAULT_TARGET_CSV
from csl.odds.snapshot_store import HISTORY_CSV, append_snapshots, load_history

# Start chasing the close this many minutes before kickoff. Wide on purpose — see
# the module docstring: the tick cadence is not reliable enough for a narrow window.
DEFAULT_CLOSE_WINDOW_MINUTES = 60.0

# A close captured within this many minutes of kickoff IS the close: the fixture goes
# final and stops spending quota. This is the user-specified target (2026-08-02).
CLOSE_TARGET_MINUTES = 5.0

# The book whose close defines "done". Mirrors capture_scheduler.REFERENCE_BOOKMAKER:
# other books in CAPTURE_BOOKMAKERS ride along for free (the bookmakers filter costs
# no extra quota) but their coverage is patchy, so requiring them would keep fixtures
# pending — and spending — all the way to kickoff.
REFERENCE_BOOKMAKER = BOOKMAKER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO-8601 ``fetched_at`` (``...Z``) into a UTC-aware datetime."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def captured_close_times(
    history_path: str = HISTORY_CSV, bookmaker: str = REFERENCE_BOOKMAKER
) -> dict[tuple[str, str], datetime]:
    """Map (norm_home, norm_away) -> the LATEST ``fetched_at`` of a stored close.

    Only ``bookmaker``'s rows count, since that is the book the final-check is on.
    """
    hist = load_history(history_path)
    if hist.empty:
        return {}
    rows = hist[(hist["snapshot_type"] == "close") & (hist["bookmaker"] == bookmaker)]
    out: dict[tuple[str, str], datetime] = {}
    for home, away, fetched in zip(rows["home_team"], rows["away_team"], rows["fetched_at"]):
        stamp = _parse_ts(fetched)
        if stamp is None:
            continue
        key = (_norm(home), _norm(away))
        if key not in out or stamp > out[key]:
            out[key] = stamp
    return out


def is_final(fixture: Fixture, closes: dict[tuple[str, str], datetime]) -> bool:
    """True when this fixture's stored close is already inside the 5-min target band."""
    stamp = closes.get(fixture.key)
    if stamp is None:
        return False
    return stamp >= fixture.kickoff - timedelta(minutes=CLOSE_TARGET_MINUTES)


def pending_close_fixtures(
    now: datetime,
    *,
    target_path: str = DEFAULT_TARGET_CSV,
    history_path: str = HISTORY_CSV,
    window_minutes: float = DEFAULT_CLOSE_WINDOW_MINUTES,
) -> list[Fixture]:
    """Fixtures inside the pre-kickoff window whose close is not yet final.

    Three conditions:
      * ``now`` is at or past ``kickoff - window_minutes`` (close chasing has started);
      * ``kickoff`` is still ahead — after kickoff the pre-match line is gone and any
        price we could still fetch would not be a closing line;
      * no Pinnacle close on file within ``CLOSE_TARGET_MINUTES`` of kickoff.
    """
    closes = captured_close_times(history_path)
    window = timedelta(minutes=window_minutes)
    return [
        f for f in load_upcoming(target_path)
        if f.kickoff - window <= now < f.kickoff and not is_final(f, closes)
    ]


def latest_close_rows(history_path: str = HISTORY_CSV) -> pd.DataFrame:
    """One row per (fixture, book): the LATEST stored close — i.e. the closing line.

    The capture re-fires across the pre-kickoff window, so a fixture whose line moved
    has several ``close`` rows. The closing line is the last one fetched; this is the
    canonical reader so consumers (CLV grading, any future dashboard column) agree.
    """
    hist = load_history(history_path)
    if hist.empty:
        return hist
    closes = hist[hist["snapshot_type"] == "close"].copy()
    if closes.empty:
        return closes
    closes["_fetched"] = closes["fetched_at"].map(_parse_ts)
    closes = closes.sort_values("_fetched", kind="stable")
    closes = closes.drop_duplicates(subset=["event_id", "bookmaker"], keep="last")
    return closes.drop(columns="_fetched")


def tick(
    *,
    now: datetime | None = None,
    target_path: str = DEFAULT_TARGET_CSV,
    history_path: str = HISTORY_CSV,
    window_minutes: float = DEFAULT_CLOSE_WINDOW_MINUTES,
    regions: str = DEFAULT_REGIONS,
    min_remaining: int = DEFAULT_MIN_REMAINING,
    dry_run: bool = False,
) -> int:
    """Run one closing-line tick. Returns the number of ``close`` rows appended."""
    now = now or datetime.now(timezone.utc)

    pending = pending_close_fixtures(
        now,
        target_path=target_path,
        history_path=history_path,
        window_minutes=window_minutes,
    )
    if not pending:
        log.info("Tick %s: no fixture inside its pre-kickoff close window; nothing to do.",
                 now.isoformat())
        return 0

    log.info(
        "Tick %s: %d fixture(s) in the close window: %s",
        now.isoformat(), len(pending),
        ", ".join(f"{f.label} (T-{(f.kickoff - now).total_seconds() / 60:.0f}m)" for f in pending),
    )

    # Dry run before the key lookup so the pending set can be inspected without a key.
    if dry_run:
        log.info("Dry run: would spend 1 /odds request for %d fixture(s); writing nothing.",
                 len(pending))
        return 0

    api_key = get_api_key()
    remaining, used, csl_available = read_quota(api_key)  # free /sports read
    log.info("Quota: remaining=%s used=%s | CSL in slate: %s", remaining, used, csl_available)
    if remaining is not None and remaining < min_remaining:
        log.warning("Aborting: quota remaining=%d below threshold=%d.", remaining, min_remaining)
        return 0

    mapping = load_team_mapping()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # Same slate-wide call the open capture makes: the bookmakers filter is free, so
    # every CAPTURE_BOOKMAKERS close rides along on the one credit we are spending.
    response = fetch_odds_response(  # 1 paid request
        api_key, regions, bookmakers=",".join(CAPTURE_BOOKMAKERS)
    )
    events = response.json()
    if not isinstance(events, list):
        raise ValueError(f"Expected list response from The Odds API, got: {type(events)}")
    log.info("Fetched %d events from The Odds API", len(events))

    rows = extract_rows(events, mapping, regions=regions, fetched_at=fetched_at, bookmaker_keys=None)
    frame = rows_to_frame(rows)

    # Keep ONLY fixtures currently inside their own close window. A fixture that is
    # still hours from kickoff is in the same response; storing its price as `close`
    # would corrupt the CLV benchmark with a mid-week line.
    pending_keys = {f.key for f in pending}
    if not frame.empty:
        keep = frame.apply(
            lambda r: (_norm(r["home_team"]), _norm(r["away_team"])) in pending_keys, axis=1
        )
        frame = frame[keep]
    if frame.empty:
        log.info("None of the in-window fixtures were present in the odds response; "
                 "nothing appended.")
        return 0

    # No append-once filter here (unlike the open capture): re-capturing IS the point,
    # because a later close is a better close. snapshot_store dedups on the book's own
    # `last_update`, so a line that has not moved since the previous tick is silently
    # skipped and only genuine movement adds a row.
    books = sorted(frame["bookmaker"].unique())
    rounds = sorted({f.round for f in pending if f.round})
    log.info(
        "Close rows this tick: %d across %d book(s): %s | %s present: %s",
        len(frame), len(books), ", ".join(books),
        REFERENCE_BOOKMAKER, REFERENCE_BOOKMAKER in books,
    )

    _, appended = append_snapshots(
        frame,
        snapshot_type="close",
        target_round=",".join(rounds),
        capture_reason=f"pre-kickoff close tick @ {now.isoformat()}",
        path=history_path,
    )

    # Report which fixtures reached the 5-min target on this tick — they go final and
    # stop costing quota, which is the signal that the window logic is behaving.
    finalized = [
        f.label for f in pending
        if now >= f.kickoff - timedelta(minutes=CLOSE_TARGET_MINUTES)
    ]
    if finalized:
        log.info("Reached the T-%.0fm close target (no further captures): %s",
                 CLOSE_TARGET_MINUTES, ", ".join(finalized))
    return appended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Pinnacle closing lines in the pre-kickoff window."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_CSV, help="Upcoming fixtures CSV")
    parser.add_argument("--out", default=HISTORY_CSV, help="History CSV path")
    parser.add_argument("--window-minutes", type=float, default=DEFAULT_CLOSE_WINDOW_MINUTES,
                        help="Start chasing the close this many minutes before kickoff. Wider "
                             f"than the T-{CLOSE_TARGET_MINUTES:.0f}m target on purpose, so an "
                             "unreliable tick cadence still yields a usable closing line.")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Odds API regions (default: us)")
    parser.add_argument("--min-remaining", type=int, default=DEFAULT_MIN_REMAINING,
                        help="Abort before spending if quota remaining is below this")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report in-window fixtures; spend/write nothing")
    args = parser.parse_args()

    try:
        tick(
            target_path=args.target,
            history_path=args.out,
            window_minutes=args.window_minutes,
            regions=args.regions,
            min_remaining=args.min_remaining,
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
