"""Capture 1xBet opening lines from odds-api.io — no predicted window, no expiry.

This replaces the 1xBet half of ``capture_scheduler``. The difference that matters:

    capture_scheduler   captures a book's open only while "now" is inside a *predicted*
                        window ``[anchor, anchor + 12h]``. Miss the window (feed lag,
                        cron throttling, no anchor) and the opening line is lost forever.

    this module         has no window at all. A fixture is pending from the moment it
                        appears in the upcoming-fixtures CSV until either its 1xBet open
                        is stored or it kicks off. The first 1X2 price odds-api.io reports
                        for it *is* the opening line, and polling every ~15 min means
                        "first reported" is within a tick of "first posted".

That is only affordable because the two providers have very different budgets:
The Odds API's free plan is ~500 requests/**month**, odds-api.io's is ~500/**day**
(100/hour) with ``/odds/multi`` returning 10 events per request. See ``oddsapi_io``.

Quota discipline
----------------
  * An idle tick — every fixture already has its 1xBet open — spends **zero** requests,
    exactly like ``capture_scheduler``'s idle path. Most ticks are idle.
  * A busy tick spends ``1`` (``/events``) + ``ceil(pending / 10)`` (``/odds/multi``),
    hard-capped by ``--max-requests``. The capture workflow passes ``2`` — one events
    call plus one batch of up to 10 fixtures — so even at a fully-recovered 144
    runs/day the ceiling is 288/day. Overflow past 10 pending fixtures waits for the
    next tick rather than costing more. The default of 4 here is for manual runs, where
    draining the whole pending set in one go is usually what you want.
  * A fixture that never gets a 1xBet price drops out of pending at kickoff, so it can
    never burn requests indefinitely.

Usage (repo root, PYTHONPATH=src, ODDS_API_IO_KEY set):
    python -m csl.odds.fetch_onexbet_open
    python -m csl.odds.fetch_onexbet_open --dry-run   # decide only, spend/write nothing
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import requests

from csl.odds.capture_scheduler import _norm, captured_open_books
from csl.odds.fetch_pinnacle_spreads import (
    load_team_mapping,
    normalize_team_name,
    rows_to_frame,
)
from csl.odds.oddsapi_io import (
    MULTI_BATCH_SIZE,
    TARGET_BOOKMAKER_KEY,
    event_to_row,
    extract_ml,
    fetch_multi_odds,
    get_api_key,
    list_events,
)
from csl.odds.opening_calendar import DEFAULT_TARGET_CSV, _parse_kickoff
from csl.odds.snapshot_store import HISTORY_CSV, append_snapshots

# How far ahead to ask odds-api.io for fixtures. The upcoming-fixtures CSV carries
# roughly two rounds; 21 days covers it with margin so a fixture is pending (and
# therefore capturable) from the earliest moment 1xBet might post a price.
DEFAULT_LOOKAHEAD_DAYS = 21

# Total requests one run may spend, INCLUDING the /events call. 4 => 1 events call plus
# up to 3 batched odds calls = 30 fixtures, far more than a CSL slate ever needs, while
# keeping the 15-min-cadence worst case at 384/day. Raise only with the daily cap in mind.
DEFAULT_MAX_REQUESTS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class Fixture(NamedTuple):
    """An upcoming fixture in the repo's own vocabulary (standard team names, UTC)."""

    home: str
    away: str
    kickoff: datetime
    round: str

    @property
    def key(self) -> tuple[str, str]:
        return _norm(self.home), _norm(self.away)

    @property
    def label(self) -> str:
        return f"{self.home} vs {self.away}"


def load_upcoming(target_path: str) -> list[Fixture]:
    """Every upcoming fixture from the target CSV, standard team names.

    Reuses ``opening_calendar._parse_kickoff`` so the UTC interpretation of the CSV's
    Date/Time columns stays identical to the rest of the odds pipeline (the source
    stores kickoffs in UTC, not UK local — a long-standing trap documented in AGENTS.md).
    """
    fixtures: list[Fixture] = []
    with open(target_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            home = (row.get("Home") or "").strip()
            away = (row.get("Away") or "").strip()
            kickoff = _parse_kickoff((row.get("Date") or "").strip(), row.get("Time", ""))
            round_str = (row.get("Wk") or row.get("Round") or "").strip()
            if home and away and kickoff is not None:
                fixtures.append(Fixture(home, away, kickoff, round_str))
    return fixtures


def pending_fixtures(now: datetime, *, target_path: str, history_path: str) -> list[Fixture]:
    """Upcoming fixtures with no stored 1xBet open yet.

    Two conditions only — no window arithmetic:
      * kickoff is still ahead (a started match's pre-match line is gone, and keeping it
        pending would burn a request every tick for nothing);
      * ``TARGET_BOOKMAKER_KEY`` has no ``snapshot_type=open`` row in the history.
    """
    captured = captured_open_books(history_path)
    return [
        f for f in load_upcoming(target_path)
        if f.kickoff > now and TARGET_BOOKMAKER_KEY not in captured.get(f.key, set())
    ]


def match_events_to_pending(events: list[dict], pending: list[Fixture], mapping):
    """``(event, fixture)`` pairs for every odds-api.io event that is a pending fixture.

    odds-api.io spells clubs its own way ("Shandong Taishan FC", "Henan", "Zhejiang FC"),
    so every event is put through the shared mapping before matching. Unmappable names
    are logged and skipped rather than raised on — see ``oddsapi_io.event_to_row``.
    """
    by_key = {f.key: f for f in pending}
    matched, unmapped = [], set()
    for event in events:
        home = normalize_team_name(str(event.get("home") or ""), mapping)
        away = normalize_team_name(str(event.get("away") or ""), mapping)
        if home is None:
            unmapped.add(str(event.get("home") or ""))
        if away is None:
            unmapped.add(str(event.get("away") or ""))
        if home is None or away is None:
            continue
        fixture = by_key.get((_norm(home), _norm(away)))
        if fixture is not None:
            matched.append((event, fixture))
    if unmapped:
        log.warning(
            "Unmapped odds-api.io team name(s), fixtures skipped: %s — add an "
            "`oddsapiio_team` row to CHN_team_name_mapping.csv",
            ", ".join(sorted(unmapped)),
        )
    return matched


def run(
    *,
    now: datetime | None = None,
    target_path: str = DEFAULT_TARGET_CSV,
    history_path: str = HISTORY_CSV,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    dry_run: bool = False,
) -> int:
    """One capture pass. Returns the number of ``open`` rows appended."""
    now = now or datetime.now(timezone.utc)

    pending = pending_fixtures(now, target_path=target_path, history_path=history_path)
    if not pending:
        log.info("Tick %s: every upcoming fixture already has a 1xBet open; nothing to do.",
                 now.isoformat())
        return 0

    log.info("Tick %s: %d fixture(s) without a 1xBet open: %s",
             now.isoformat(), len(pending), ", ".join(f.label for f in pending))

    # Before the key lookup: a dry run spends nothing, so it must work without one
    # (useful for validating the pending set in CI or on a machine with no key).
    if dry_run:
        batches = math.ceil(len(pending) / MULTI_BATCH_SIZE)
        log.info("Dry run: would spend up to %d request(s) (1 events + %d odds batch(es), "
                 "capped at %d); writing nothing.",
                 min(1 + batches, max_requests), batches, max_requests)
        return 0

    api_key = get_api_key()
    mapping = load_team_mapping()

    events = list_events(api_key, from_dt=now,
                         to_dt=now + timedelta(days=lookahead_days))
    matched = match_events_to_pending(events, pending, mapping)
    if not matched:
        log.info("None of the %d pending fixture(s) are listed by odds-api.io yet; "
                 "nothing appended.", len(pending))
        return 0

    # One request already spent on /events; the rest of the budget goes to odds batches.
    budget = max(max_requests - 1, 0)
    batches = [matched[i:i + MULTI_BATCH_SIZE] for i in range(0, len(matched), MULTI_BATCH_SIZE)]
    if len(batches) > budget:
        log.warning("Request cap: %d batch(es) needed but only %d affordable this run; "
                    "the remainder stays pending for the next tick.", len(batches), budget)
        batches = batches[:budget]

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows: list[dict] = []
    priced: set[tuple[str, str]] = set()
    for batch in batches:
        by_id = {str(event.get("id")): (event, fixture) for event, fixture in batch}
        for quoted in fetch_multi_odds(api_key, list(by_id)):
            event, fixture = by_id.get(str(quoted.get("id")), (quoted, None))
            prices = extract_ml(quoted)
            if prices is None:
                continue
            row = event_to_row(event, prices, mapping, fetched_at=fetched_at)
            if row is None:
                log.warning("Could not map teams for %s; skipped.",
                            fixture.label if fixture else quoted.get("id"))
                continue
            if fixture is not None:
                row["_round"] = fixture.round
                priced.add(fixture.key)
            rows.append(row)

    # /odds/multi omits an event entirely when the selected book has no market for it,
    # so "requested but absent from the response" and "present but no ML" are the same
    # state: 1xBet has not posted a price yet. Report it either way — silently dropping
    # these is how a coverage gap would go unnoticed.
    unpriced = [f.label for batch in batches for _, f in batch if f.key not in priced]
    if unpriced:
        log.info("1xBet has no 1X2 price yet for %d fixture(s): %s (they stay pending)",
                 len(unpriced), ", ".join(unpriced))
    if not rows:
        log.info("No new 1xBet opening prices this tick; nothing appended.")
        return 0

    log.info("New 1xBet opening line(s): %s",
             ", ".join(f"{r['home_team']} vs {r['away_team']} "
                       f"({r['home_odds']}/{r['draw_odds']}/{r['away_odds']})" for r in rows))

    rounds = sorted({r.pop("_round", "") for r in rows} - {""})
    _, appended = append_snapshots(
        rows_to_frame(rows),
        snapshot_type="open",
        target_round=",".join(rounds),
        capture_reason=f"odds-api.io first-seen 1xBet price @ {now.isoformat()}",
        path=history_path,
    )
    return appended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture 1xBet opening lines from odds-api.io (no predicted window)."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_CSV, help="Upcoming fixtures CSV")
    parser.add_argument("--out", default=HISTORY_CSV, help="Capture history CSV path")
    parser.add_argument("--lookahead-days", type=int, default=DEFAULT_LOOKAHEAD_DAYS,
                        help="How far ahead to request fixtures from odds-api.io")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                        help="Total requests this run may spend, including the /events call. "
                             "Keep low: the free tier allows ~500/day across all runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report pending fixtures and projected cost; spend/write nothing")
    args = parser.parse_args()

    try:
        run(
            target_path=args.target,
            history_path=args.out,
            lookahead_days=args.lookahead_days,
            max_requests=args.max_requests,
            dry_run=args.dry_run,
        )
    except requests.RequestException as exc:
        log.error("odds-api.io request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
