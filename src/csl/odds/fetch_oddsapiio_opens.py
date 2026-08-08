"""Capture opening lines from odds-api.io — every entitled book, no predicted window.

This replaces the recreational-book half of ``capture_scheduler``. The difference that
matters:

    capture_scheduler   captures a book's open only while "now" is inside a *predicted*
                        window ``[anchor, anchor + 12h]``. Miss the window (feed lag,
                        cron throttling, no anchor) and the opening line is lost forever.

    this module         has no window at all. A (fixture, book) pair is pending from the
                        moment the fixture appears in the upcoming-fixtures CSV until
                        either that book's open is stored or the match kicks off. The
                        first 1X2 price odds-api.io reports for it *is* the opening line,
                        and polling every ~15 min means "first reported" is within a tick
                        of "first posted".

That is only affordable because the two providers have very different budgets:
The Odds API's free plan is ~500 requests/**month**, odds-api.io's is ~500/**day**
(100/hour) with ``/odds/multi`` returning 10 events per request. See ``oddsapi_io``.

Two books, one tick, one request (2026-08-08)
---------------------------------------------
``oddsapi_io.CAPTURE_BOOKS`` is ``(1xbet, Duel)`` and **both are captured at equal
priority on the same tick** — neither anchors or backs up the other. This costs nothing:
``/odds/multi`` takes a comma-separated ``bookmakers`` list and bills the request once,
so the cadence, the ``--max-requests`` cap and the daily budget are all unchanged from
when this module fetched 1xBet alone.

Pending is tracked **per (fixture, book)**, which is the one thing that must not be
simplified back to per-fixture. Only a book that has no stored open for the fixture is
eligible to write one; if 1xBet's open is already banked and Duel's is not, the request
still returns 1xBet's *current* price, and storing that as ``snapshot_type="open"`` would
silently overwrite an opening line with a mid-market one. ``PendingFixture.missing`` is
what prevents it.

Quota discipline
----------------
  * An idle tick — every fixture already has an open from **every** book — spends
    **zero** requests, exactly like ``capture_scheduler``'s idle path. Most ticks are idle.
  * A busy tick spends ``1`` (``/events``) + ``ceil(pending / 10)`` (``/odds/multi``),
    hard-capped by ``--max-requests``. The capture workflow passes ``2`` — one events
    call plus one batch of up to 10 fixtures — so even at a fully-recovered 144
    runs/day the ceiling is 288/day. Overflow past 10 pending fixtures waits for the
    next tick rather than costing more. The default of 4 here is for manual runs, where
    draining the whole pending set in one go is usually what you want.
  * Pending is ordered **soonest kickoff first**, so when the set overflows one batch it
    is the fixtures whose lines are opening now that get the requests, not ones weeks out.
    Adding a second book widened the pending set (a fixture stays pending until *both*
    books have opened), which is exactly when that ordering starts to matter.
  * A fixture no book ever prices drops out of pending at kickoff, so it can never burn
    requests indefinitely.

Usage (repo root, PYTHONPATH=src, ODDS_API_IO_KEY set):
    python -m csl.odds.fetch_oddsapiio_opens
    python -m csl.odds.fetch_oddsapiio_opens --dry-run   # decide only, spend/write nothing
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
    CAPTURE_BOOKS,
    MULTI_BATCH_SIZE,
    Book,
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


class PendingFixture(NamedTuple):
    """A fixture still owed an opening line, plus exactly which books still owe it."""

    fixture: Fixture
    missing: tuple[Book, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.fixture.key

    @property
    def label(self) -> str:
        return self.fixture.label


def pending_fixtures(
    now: datetime,
    *,
    target_path: str,
    history_path: str,
    books: tuple[Book, ...] = CAPTURE_BOOKS,
) -> list[PendingFixture]:
    """Upcoming fixtures still missing an open from at least one of ``books``.

    Two conditions only — no window arithmetic:
      * kickoff is still ahead (a started match's pre-match line is gone, and keeping it
        pending would burn a request every tick for nothing);
      * at least one book in ``books`` has no ``snapshot_type=open`` row in the history.

    Carrying ``missing`` on each entry rather than recomputing it later is what keeps the
    per-book gate honest: a fixture is requested because *some* book still owes an open,
    but only the books in ``missing`` may write one. See the module docstring.

    Sorted by kickoff so an overflowing pending set spends its batch on the fixtures
    whose lines are opening soonest.
    """
    captured = captured_open_books(history_path)
    pending: list[PendingFixture] = []
    for fixture in load_upcoming(target_path):
        if fixture.kickoff <= now:
            continue
        have = captured.get(fixture.key, set())
        missing = tuple(b for b in books if b.stored_key not in have)
        if missing:
            pending.append(PendingFixture(fixture, missing))
    return sorted(pending, key=lambda p: p.fixture.kickoff)


def match_events_to_pending(events: list[dict], pending: list[PendingFixture], mapping):
    """``(event, pending_fixture)`` pairs for every odds-api.io event that is pending.

    odds-api.io spells clubs its own way ("Shandong Taishan FC", "Henan", "Zhejiang FC"),
    so every event is put through the shared mapping before matching. Unmappable names
    are logged and skipped rather than raised on — see ``oddsapi_io.event_to_row``.

    Preserves ``pending``'s kickoff ordering: the result is built by walking ``events``,
    so it is re-sorted before returning to keep the soonest-first batching guarantee.
    """
    by_key = {p.key: p for p in pending}
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
        entry = by_key.get((_norm(home), _norm(away)))
        if entry is not None:
            matched.append((event, entry))
    if unmapped:
        log.warning(
            "Unmapped odds-api.io team name(s), fixtures skipped: %s — add an "
            "`oddsapiio_team` row to CHN_team_name_mapping.csv",
            ", ".join(sorted(unmapped)),
        )
    return sorted(matched, key=lambda pair: pair[1].fixture.kickoff)


def run(
    *,
    now: datetime | None = None,
    target_path: str = DEFAULT_TARGET_CSV,
    history_path: str = HISTORY_CSV,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    dry_run: bool = False,
    books: tuple[Book, ...] = CAPTURE_BOOKS,
) -> int:
    """One capture pass over every book in ``books``. Returns ``open`` rows appended."""
    now = now or datetime.now(timezone.utc)

    pending = pending_fixtures(now, target_path=target_path, history_path=history_path,
                               books=books)
    if not pending:
        log.info("Tick %s: every upcoming fixture already has an open from every book "
                 "(%s); nothing to do.",
                 now.isoformat(), ", ".join(b.provider_name for b in books))
        return 0

    log.info("Tick %s: %d fixture(s) still owed an open: %s",
             now.isoformat(), len(pending),
             ", ".join(f"{p.label} [{'+'.join(b.provider_name for b in p.missing)}]"
                       for p in pending))

    # Before the key lookup: a dry run spends nothing, so it must work without one
    # (useful for validating the pending set in CI or on a machine with no key).
    if dry_run:
        batches = math.ceil(len(pending) / MULTI_BATCH_SIZE)
        log.info("Dry run: would spend up to %d request(s) (1 events + %d odds batch(es), "
                 "capped at %d) covering %d book(s) in the same call(s); writing nothing.",
                 min(1 + batches, max_requests), batches, max_requests, len(books))
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
    priced: set[tuple[tuple[str, str], str]] = set()
    for batch in batches:
        by_id = {str(event.get("id")): (event, entry) for event, entry in batch}
        # One request covers this batch's fixtures AND every book in `books`.
        for quoted in fetch_multi_odds(api_key, list(by_id), books=books):
            event, entry = by_id.get(str(quoted.get("id")), (quoted, None))
            # THE per-book gate: iterate `entry.missing`, never `books`. A book that
            # already has this fixture's open is skipped, so its current price can never
            # be written over the opening line we banked earlier.
            for book in (entry.missing if entry is not None else books):
                prices = extract_ml(quoted, book)
                if prices is None:
                    continue
                row = event_to_row(event, prices, mapping, fetched_at=fetched_at, book=book)
                if row is None:
                    log.warning("Could not map teams for %s (%s); skipped.",
                                entry.label if entry else quoted.get("id"), book.provider_name)
                    continue
                if entry is not None:
                    row["_round"] = entry.fixture.round
                    priced.add((entry.key, book.stored_key))
                rows.append(row)

    # /odds/multi omits a book (or the whole event) when it has no market for it, so
    # "absent from the response" and "present but no ML" are the same state: that book
    # has not posted a price yet. Report it per book — silently dropping these is how a
    # coverage gap would go unnoticed, and the two books do NOT open together.
    for book in books:
        unpriced = [entry.label for batch in batches for _, entry in batch
                    if book in entry.missing and (entry.key, book.stored_key) not in priced]
        if unpriced:
            log.info("%s has no 1X2 price yet for %d fixture(s): %s (they stay pending)",
                     book.provider_name, len(unpriced), ", ".join(unpriced))
    if not rows:
        log.info("No new opening prices this tick; nothing appended.")
        return 0

    log.info("New opening line(s): %s",
             ", ".join(f"{r['home_team']} vs {r['away_team']} [{r['bookmaker']}] "
                       f"({r['home_odds']}/{r['draw_odds']}/{r['away_odds']})" for r in rows))

    rounds = sorted({r.pop("_round", "") for r in rows} - {""})
    _, appended = append_snapshots(
        rows_to_frame(rows),
        snapshot_type="open",
        target_round=",".join(rounds),
        capture_reason=f"odds-api.io first-seen open price @ {now.isoformat()}",
        path=history_path,
    )
    return appended


def _select_books(spec: str | None) -> tuple[Book, ...]:
    """Resolve a ``--books`` string to ``Book``s, rejecting names the plan cannot fetch.

    Fails loudly on an unknown name rather than silently capturing fewer books: a typo
    would otherwise look exactly like "that book never posts a price".
    """
    if not spec:
        return CAPTURE_BOOKS
    by_name = {b.provider_name.casefold(): b for b in CAPTURE_BOOKS}
    chosen: list[Book] = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        book = by_name.get(name.casefold())
        if book is None:
            raise ValueError(
                f"Unknown book {name!r}. Capturable: "
                f"{', '.join(b.provider_name for b in CAPTURE_BOOKS)}"
            )
        chosen.append(book)
    return tuple(chosen) or CAPTURE_BOOKS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture opening lines from odds-api.io for every entitled book "
                    f"({', '.join(b.provider_name for b in CAPTURE_BOOKS)}) — no predicted window."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_CSV, help="Upcoming fixtures CSV")
    parser.add_argument("--out", default=HISTORY_CSV, help="Capture history CSV path")
    parser.add_argument("--lookahead-days", type=int, default=DEFAULT_LOOKAHEAD_DAYS,
                        help="How far ahead to request fixtures from odds-api.io")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                        help="Total requests this run may spend, including the /events call. "
                             "Keep low: the free tier allows ~500/day across all runs. "
                             "Independent of how many books are captured — they share one call.")
    parser.add_argument("--books", default=None,
                        help="Comma-separated odds-api.io book names to capture "
                             f"(default: {','.join(b.provider_name for b in CAPTURE_BOOKS)}). "
                             "Useful to chase one book's open in isolation.")
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
            books=_select_books(args.books),
        )
    except requests.RequestException as exc:
        log.error("odds-api.io request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
