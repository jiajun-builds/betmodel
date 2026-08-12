"""Capture opening 1X2 lines from odds-api.io -- every entitled book, no window.

Betano's opener is the only positive-EV result this project has (n=355, EV>10%
-> +2.56pp, t=4.64), and all 461 rows of it were entered by hand. This is the
loop that stops that being true.

There is no predicted opening window here, deliberately. A (fixture, book) pair
is pending from the moment the fixture enters the lookahead until either that
book's open is stored or the match kicks off; the first 1X2 price odds-api.io
reports for it *is* the opening line. Polling every ~15 min means "first
reported" is within a tick of "first posted". A window-based design has the
opposite failure mode -- miss the window and the opener is gone for good.

That is affordable only because of the budget asymmetry: The Odds API's free plan
is ~500 requests per MONTH, odds-api.io's is ~500 per DAY (100/hour), and
/odds/multi returns 10 events per request.

The per-(fixture, book) gate
----------------------------
``PendingFixture.missing`` names exactly which books still owe this fixture an
open, and only those may write one. This is the one thing that must not be
simplified back to per-fixture. Betano UK and Duel do not open together (measured
2026-08-12: 6/10 and 9/10 of the same slate), so a request made because Duel is
still missing will also return Betano's *current* price -- and storing that as
snapshot_type="open" would overwrite an opening line banked days earlier with a
mid-market one. Dedup cannot save us here: Duel stamps every fixture with one
shared feed-refresh ``updatedAt``, so its ``last_update`` moves whether or not
its price did. The gate is the whole defence.

Quota discipline
----------------
  * An idle tick -- every fixture in the lookahead has an open from every book --
    spends ZERO requests. Most ticks are idle.
  * A busy tick spends 1 (/events) + ceil(pending / 10) (/odds/multi), hard-capped
    by --max-requests. At --max-requests 3 and a 15-minute cadence the ceiling is
    288/day against a ~500/day budget, and 12/hour against a 100/hour limit.
  * --lookahead-days is what makes idle ticks possible at all. Liga MX publishes
    its whole season, so without it all 126 future fixtures are pending forever
    and every tick spends the cap on matches three months out that no book has
    priced. 14 days sits just outside Betano's measured median open of T-293h.
  * Pending is ordered soonest-kickoff-first, so an overflowing set spends its
    batch on the lines opening now, not the ones weeks away.

Usage (repo root, PYTHONPATH=src, ODDS_API_IO_KEY set):
    python -m ligamx.odds.fetch_oddsapiio_opens
    python -m ligamx.odds.fetch_oddsapiio_opens --dry-run   # decide only
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pandas as pd
import requests

from ligamx import paths
from ligamx.odds import oddsapi_io
from ligamx.odds.capture_store import append_snapshots, load_history
from ligamx.odds.oddsapi_io import CAPTURE_BOOKS, MULTI_BATCH_SIZE, Book

# Far enough ahead to be pending before any book opens (Betano's median is
# T-293h ~ 12.2d), close enough that the pending set drains and ticks go idle.
DEFAULT_LOOKAHEAD_DAYS = 14

# Total requests one run may spend, INCLUDING the /events call. 3 => 1 events call
# plus 2 batches = 20 fixtures, which covers a full 14-day Liga MX lookahead
# (measured 18). Independent of how many books are captured -- they share the call.
DEFAULT_MAX_REQUESTS = 3

log = logging.getLogger(__name__)


def _norm(name: str) -> str:
    return " ".join(str(name).split()).casefold()


class Fixture(NamedTuple):
    """An upcoming fixture in the repo's vocabulary: standard names, UTC kickoff."""

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


def load_upcoming(target_path: str | None = None) -> list[Fixture]:
    """Every upcoming fixture, read from the ``kickoff_utc`` column.

    Uses kickoff_utc and never the Date/Time pair: MEX_ligamx.csv's Time column
    mixes local and UTC, and that ambiguity has already fabricated a result once.
    kickoff_utc is written by the SofaScore fixture fetcher and is unambiguous.
    """
    target_path = target_path or paths.upcoming_fixtures_csv()
    fixtures: list[Fixture] = []
    with open(target_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            home = (row.get("Home") or "").strip()
            away = (row.get("Away") or "").strip()
            raw = (row.get("kickoff_utc") or "").strip()
            if not (home and away and raw):
                continue
            try:
                kickoff = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                log.warning("unparseable kickoff_utc %r for %s vs %s; skipped",
                            raw, home, away)
                continue
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            fixtures.append(
                Fixture(home, away, kickoff, str(row.get("round") or "").strip()))
    return fixtures


def captured_open_books(history_path: str | None = None) -> dict[tuple[str, str], set[str]]:
    """(home, away) -> the set of bookmakers that already have a stored open."""
    history = load_history(history_path)
    if history.empty:
        return {}
    opens = history[history["snapshot_type"] == "open"]
    out: dict[tuple[str, str], set[str]] = {}
    for home, away, book in opens[["home_team", "away_team", "bookmaker"]].itertuples(index=False):
        out.setdefault((_norm(home), _norm(away)), set()).add(book)
    return out


class PendingFixture(NamedTuple):
    """A fixture still owed an open, and exactly which books still owe it."""

    fixture: Fixture
    missing: tuple[Book, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.fixture.key

    @property
    def label(self) -> str:
        return self.fixture.label


def pending_fixtures(now: datetime, *, target_path: str | None = None,
                     history_path: str | None = None,
                     lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
                     books: tuple[Book, ...] = CAPTURE_BOOKS) -> list[PendingFixture]:
    """Fixtures inside the lookahead still missing an open from at least one book.

    Three conditions, no window arithmetic:
      * kickoff is still ahead -- a started match's pre-match line is gone, and
        keeping it pending would burn requests every tick for nothing;
      * kickoff is inside the lookahead -- see the module docstring on why;
      * some book in ``books`` has no snapshot_type=open row for it.

    Carrying ``missing`` on the entry rather than recomputing it later is what keeps
    the per-book gate honest downstream.
    """
    horizon = now + timedelta(days=lookahead_days)
    captured = captured_open_books(history_path)
    pending: list[PendingFixture] = []
    for fixture in load_upcoming(target_path):
        if not (now < fixture.kickoff <= horizon):
            continue
        have = captured.get(fixture.key, set())
        missing = tuple(b for b in books if b.stored_key not in have)
        if missing:
            pending.append(PendingFixture(fixture, missing))
    return sorted(pending, key=lambda p: p.fixture.kickoff)


def match_events_to_pending(events: list[dict],
                            pending: list[PendingFixture]) -> list[tuple[dict, PendingFixture]]:
    """(event, pending) pairs for every odds-api.io event that is pending.

    odds-api.io spells clubs its own way ("CF America", "Pumas UNAM"), so names go
    through API_TO_STANDARD first. Unmappable ones are logged and skipped, never
    raised on -- see oddsapi_io.standard_name.
    """
    by_key = {p.key: p for p in pending}
    matched = []
    for event in events:
        home = oddsapi_io.standard_name(str(event.get("home") or ""))
        away = oddsapi_io.standard_name(str(event.get("away") or ""))
        if home is None or away is None:
            continue
        entry = by_key.get((_norm(home), _norm(away)))
        if entry is not None:
            matched.append((event, entry))
    return sorted(matched, key=lambda pair: pair[1].fixture.kickoff)


def run(*, now: datetime | None = None, target_path: str | None = None,
        history_path: str | None = None,
        lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
        max_requests: int = DEFAULT_MAX_REQUESTS, dry_run: bool = False,
        books: tuple[Book, ...] = CAPTURE_BOOKS) -> int:
    """One capture pass. Returns the number of ``open`` rows appended."""
    now = now or datetime.now(timezone.utc)

    pending = pending_fixtures(now, target_path=target_path, history_path=history_path,
                               lookahead_days=lookahead_days, books=books)
    if not pending:
        log.info("Tick %s: every fixture inside %dd has an open from every book (%s); "
                 "nothing to do.", now.isoformat(), lookahead_days,
                 ", ".join(b.provider_name for b in books))
        return 0

    log.info("Tick %s: %d fixture(s) still owed an open: %s", now.isoformat(), len(pending),
             ", ".join(f"{p.label} [{'+'.join(b.provider_name for b in p.missing)}]"
                       for p in pending))

    # Before the key lookup: a dry run spends nothing, so it must work without one.
    if dry_run:
        batches = math.ceil(len(pending) / MULTI_BATCH_SIZE)
        log.info("Dry run: would spend up to %d request(s) (1 events + %d batch(es), "
                 "capped at %d) covering %d book(s) in the same call(s); writing nothing.",
                 min(1 + batches, max_requests), batches, max_requests, len(books))
        return 0

    client = oddsapi_io._Client(oddsapi_io._api_key())
    events = list(oddsapi_io.list_upcoming_events(
        client, pd.Timestamp(now), pd.Timestamp(now + timedelta(days=lookahead_days))))
    matched = match_events_to_pending(events, pending)
    if not matched:
        log.info("None of the %d pending fixture(s) are listed by odds-api.io yet; "
                 "nothing appended.", len(pending))
        return 0

    # One request already went on /events; the rest of the budget buys odds batches.
    budget = max(max_requests - 1, 0)
    batches = [matched[i:i + MULTI_BATCH_SIZE]
               for i in range(0, len(matched), MULTI_BATCH_SIZE)]
    if len(batches) > budget:
        log.warning("Request cap: %d batch(es) needed but only %d affordable; the "
                    "remainder stays pending for the next tick.", len(batches), budget)
        batches = batches[:budget]

    fetched_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows: list[dict] = []
    priced: set[tuple[tuple[str, str], str]] = set()
    for batch in batches:
        by_id = {str(event.get("id")): (event, entry) for event, entry in batch}
        for quoted in oddsapi_io.fetch_multi_odds(client, list(by_id), books=books):
            found = by_id.get(str(quoted.get("id")))
            if found is None:
                continue
            event, entry = found
            # THE per-book gate: iterate entry.missing, never books. A book that
            # already banked this fixture's open is skipped, so its current price
            # can never be written over the opening line.
            for book in entry.missing:
                prices = oddsapi_io.extract_ml(quoted, book)
                if prices is None:
                    continue
                row = oddsapi_io.event_to_row(event, prices, book, fetched_at)
                if row is None:
                    continue
                row["_round"] = entry.fixture.round
                priced.add((entry.key, book.stored_key))
                rows.append(row)

    # Absent from the response and present-but-unpriced are the same state: that
    # book has not posted yet. Report per book -- silently dropping these is how a
    # coverage gap goes unnoticed, and the two books do NOT open together.
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
             ", ".join(f"{r['home_team']} v {r['away_team']} [{r['bookmaker']}] "
                       f"({r['home_odds']}/{r['draw_odds']}/{r['away_odds']})" for r in rows))

    rounds = sorted({r.pop("_round", "") for r in rows} - {""})
    _, appended = append_snapshots(
        pd.DataFrame(rows), snapshot_type="open", target_round=",".join(rounds),
        capture_reason=f"odds-api.io first-seen open price @ {now.isoformat()}",
        path=history_path)
    return appended


def _select_books(spec: str | None) -> tuple[Book, ...]:
    """Resolve a --books string, failing loudly on a name the plan cannot fetch.

    A typo would otherwise look exactly like "that book never posts a price".
    """
    if not spec:
        return CAPTURE_BOOKS
    by_name = {b.provider_name.casefold(): b for b in CAPTURE_BOOKS}
    chosen = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        book = by_name.get(name.casefold())
        if book is None:
            raise ValueError(f"Unknown book {name!r}. Capturable: "
                             f"{', '.join(b.provider_name for b in CAPTURE_BOOKS)}")
        chosen.append(book)
    return tuple(chosen) or CAPTURE_BOOKS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture opening 1X2 lines from odds-api.io for every entitled "
                    f"book ({', '.join(b.provider_name for b in CAPTURE_BOOKS)}).")
    parser.add_argument("--target", default=None, help="Upcoming fixtures CSV")
    parser.add_argument("--out", default=None, help="Capture history CSV path")
    parser.add_argument("--lookahead-days", type=int, default=DEFAULT_LOOKAHEAD_DAYS,
                        help="How far ahead a fixture becomes pending (default: %(default)s)")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                        help="Total requests this run may spend, including /events. "
                             "Keep low: the free tier allows ~500/day across all runs. "
                             "Independent of book count -- they share one call.")
    parser.add_argument("--books", default=None,
                        help="Comma-separated book names to capture (default: all "
                             "entitled). Useful to chase one book's open in isolation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report pending fixtures and projected cost; write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    try:
        run(target_path=args.target, history_path=args.out,
            lookahead_days=args.lookahead_days, max_requests=args.max_requests,
            dry_run=args.dry_run, books=_select_books(args.books))
    except requests.RequestException as exc:
        log.error("odds-api.io request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
