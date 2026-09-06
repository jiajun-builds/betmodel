"""Capture opening 1X2 lines. No predicted window, for any book.

A ``(fixture, book)`` pair is pending from the moment the fixture enters the
lookahead until that book's open is stored or the match kicks off. The first
price the provider reports for a pending pair **is** the opening line. Polling
often enough makes "first reported" within a tick of "first posted", and unlike a
predicted window the failure mode is a slightly late capture rather than no
capture at all. The window it replaces missed 17% of opens outright and
backfilled them with a mid-market price.

**The per-(fixture, book) gate is the whole defence and must never be
simplified back to per-fixture.** Two books do not open together: measured on one
slate, one priced six of ten fixtures and the other nine of ten. So a request
made because one book is still missing will also return the other book's
*current* price, and storing that as an open would overwrite an opening line
banked days earlier with a mid-market one. Dedup cannot save us: at least one
book stamps every fixture with a single shared feed-refresh timestamp, so its
``last_update`` moves whether or not its price did.

**Quota discipline.** An idle tick, where every fixture in the lookahead has an
open from every book, spends nothing, and most ticks are idle. The lookahead is
what makes that possible: a league that publishes its whole season would
otherwise keep every fixture pending forever and spend the cap every tick on
matches months away that no book has priced. Pending is ordered
soonest-kickoff-first, so an overflowing set spends its budget on the lines
opening now.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from betmodel import paths, teams
from betmodel.dates import stamp
from betmodel.config.schema import BookConfig, LeagueConfig
from betmodel.fixtures.upcoming import Fixture, load_upcoming
from betmodel.odds import capture_store, capture_watch
from betmodel.providers import oddsapiio, theoddsapi

log = logging.getLogger(__name__)


class QuotaRefused(RuntimeError):
    """A metered provider was not called because the account is under its floor.

    Its own exception because the caller has to be able to tell it apart from a
    tick that legitimately found nothing: one is the pipeline working and the
    other is the pipeline quietly not running, and they look identical in the
    numbers a tick returns.
    """


@dataclass(frozen=True)
class Pending:
    """A fixture still owed an open, and exactly which books still owe it.

    Carrying ``missing`` on the entry rather than recomputing it downstream is
    what keeps the per-book gate honest.
    """

    fixture: Fixture
    missing: tuple[BookConfig, ...]


def captured_open_books(league: str, *, history_path: str | None = None) -> dict[tuple[str, str], set[str]]:
    """``(home, away) -> books that already have an open row``."""
    history = capture_store.load_history(
        history_path or paths.for_league(league).capture_history_csv
    )
    if history.empty:
        return {}
    opens = history[history["snapshot_type"] == "open"]
    out: dict[tuple[str, str], set[str]] = {}
    for home, away, book in zip(opens["home_team"], opens["away_team"], opens["bookmaker"]):
        out.setdefault((str(home), str(away)), set()).add(str(book))
    return out


def pending_fixtures(
    league: str,
    config: LeagueConfig,
    *,
    books: tuple[BookConfig, ...],
    now: datetime | None = None,
    lookahead_days: int | None = None,
    history_path: str | None = None,
    fixtures_path: str | None = None,
) -> list[Pending]:
    """Fixtures inside the lookahead still missing an open from at least one book.

    Three conditions and no window arithmetic. Kickoff must still be ahead, since
    a started match's pre-match line is gone and keeping it pending burns
    requests forever. Kickoff must be inside the lookahead. And some book must
    still owe an open.

    **The lookahead is per book**, because the books do not publish at the same
    time and their accounts do not bill at the same rate. Asking a book about a
    fixture it will not price for another fortnight is a request with a known
    answer, and on a monthly allowance those are what exhaust it.
    """
    now = now or datetime.now(timezone.utc)
    default_days = (
        config.odds.open.lookahead_days if lookahead_days is None else lookahead_days
    )
    horizons = {
        b.key: now + timedelta(
            days=default_days if lookahead_days is not None or b.lookahead_days is None
            else b.lookahead_days
        )
        for b in books
    }
    captured = captured_open_books(league, history_path=history_path)

    fixtures = fixtures_path or paths.for_league(league).upcoming_fixtures_csv
    pending: list[Pending] = []
    for fixture in load_upcoming(fixtures):
        if fixture.kickoff <= now:
            continue
        have = captured.get(fixture.key, set())
        missing = tuple(
            b for b in books
            if b.key not in have and fixture.kickoff <= horizons[b.key]
        )
        if missing:
            pending.append(Pending(fixture, missing))
    return sorted(pending, key=lambda p: p.fixture.kickoff)


#: The timer's dispatch grid. Must match the Worker's cron in
#: ``tools/capture-timer/wrangler.toml``: the pacing below snaps the clock back to
#: this grid to recover which tick it is running for, and a mismatch would snap to
#: slots the timer never fires on.
TIMER_TICK_MINUTES = 5


def books_for(config: LeagueConfig, provider: str) -> tuple[BookConfig, ...]:
    """Polled books belonging to one provider, in declared order."""
    return tuple(b for b in config.odds.polled_books if b.provider == provider)


def books_due(
    config: LeagueConfig, provider: str, now: datetime
) -> tuple[BookConfig, ...]:
    """Those of ``provider``'s books whose own interval has come round.

    ``poll_interval_minutes`` was carried into the config for every book and then
    never read: `polled_books` tests it for truthiness and the number itself was
    ignored, so a book declared at 180 minutes was polled exactly as often as one
    declared at 10. The intervals in the YAML described an intent the code did
    not implement, which is worse than not having them.

    Paced off the clock rather than off stored state. The timer fires on fixed
    five-minute boundaries, so "minutes since midnight is a multiple of the
    interval" needs nothing remembered between runs, cannot drift, and is
    trivially predictable when reading a log. Intervals must divide the day
    evenly for that to hold, which the schema enforces.

    **The clock is snapped back to the timer's grid first, and that is not a
    detail.** The first version tested ``now`` directly, and ``now`` is when this
    code runs, not when the tick fired: the workflow has to be queued, a runner
    started and dependencies installed, which lands the check 60 to 90 seconds
    late. A tick dispatched at :40 evaluated at :41, ``41 % 10`` was never zero,
    and every provider was declined on every tick. Opening-line capture stopped
    dead for twenty-two hours and reported success the whole time, because a tick
    that declines to spend looks exactly like a tick with nothing pending.

    Snapping means the question asked is "which tick am I", which is the question
    that was always meant. It tolerates any startup lag shorter than one tick; a
    run delayed by more than five minutes lands in the next slot and skips, which
    for a ten-minute book costs ten minutes.

    A missed tick skips that slot rather than delaying it. That is acceptable
    where it applies: the anchor's opening line appears a median of 165 hours
    before kickoff, so losing one three-hour slot costs nothing. It would not be
    acceptable for closes, which is why those are gated on kickoff proximity by
    the timer instead, and never on a modulus.
    """
    minutes = now.hour * 60 + now.minute
    slot = minutes - (minutes % TIMER_TICK_MINUTES)
    due = tuple(
        b for b in books_for(config, provider)
        if slot % b.poll_interval_minutes == 0
    )
    log.debug("%s: %02d:%02d resolves to slot %02d:%02d, due: %s",
              provider, now.hour, now.minute, slot // 60, slot % 60,
              [b.key for b in due] or "nothing")
    return due


def _row(
    *, fixture: Fixture, book: BookConfig, prices: dict, event: dict, fetched_at: str
) -> dict:
    return {
        # Namespaced per the provider's own convention: the id is part of the
        # dedup key, so a row written under the wrong form is a different row.
        "event_id": capture_store.stored_event_id(
            book.provider, event.get("id", "")),
        "commence_time": stamp(fixture.kickoff),
        "api_home_team": str(event.get("home", event.get("home_team", ""))),
        "api_away_team": str(event.get("away", event.get("away_team", ""))),
        "home_team": fixture.home,
        "away_team": fixture.away,
        "home_odds": prices["home_odds"],
        "draw_odds": prices["draw_odds"],
        "away_odds": prices["away_odds"],
        "bookmaker": book.key,
        "market": "h2h",
        "regions": book.provider,
        "last_update": prices.get("last_update", ""),
        "fetched_at": fetched_at,
    }


# --------------------------------------------------------------------------- #
# odds-api.io: the bet books
# --------------------------------------------------------------------------- #

def _capture_oddsapiio(
    league: str,
    config: LeagueConfig,
    pending: list[Pending],
    *,
    now: datetime,
    dry_run: bool,
) -> tuple[list[dict], list[tuple[str, str, str]], int]:
    if dry_run:
        # "Decide only" has to mean spending nothing, and this provider bills per
        # request including the event listing. The pending set is the decision.
        return [], [], 0

    provider = config.odds.providers["oddsapiio"]
    client = oddsapiio.OddsApiIoClient(
        provider.require("league_slugs"),
        sport=provider.get("sport", "football"),
        base_url=provider.get("base_url", oddsapiio.BASE_URL),
        credential=provider.get("credential", "default"),
    )
    events = client.upcoming_events(config.odds.open.lookahead_days)
    requests_used = 1

    mapping = teams.for_league(league)
    by_key: dict[tuple[str, str], dict] = {}
    unmapped: list[str] = []
    for event in events:
        raw_home = str(event.get("home") or "")
        raw_away = str(event.get("away") or "")
        home = mapping.to_standard(raw_home)
        away = mapping.to_standard(raw_away)
        if home and away:
            by_key[(home, away)] = event
        else:
            unmapped.append(f"{raw_home} v {raw_away}")

    matched = [(p, by_key[p.fixture.key]) for p in pending if p.fixture.key in by_key]
    if not matched:
        # Say so, and say it with the join that failed. This return spends the
        # listing request, records no unpriced sighting and appends nothing, so
        # without a line here it is indistinguishable from a tick that had
        # nothing to do -- and it repeats every poll interval for as long as the
        # condition lasts. CSL's soft books went six days like this, green the
        # whole way, on `captured: 0` and nothing else.
        #
        # The two names are the diagnosis. A fixture absent from the listing is
        # the provider's to fix; one present but unmapped is a row in
        # team_name_mapping.csv.
        log.warning(
            "%s/oddsapiio: listing has %d event(s), none of the %d pending "
            "fixture(s) among them; pending=%s; unmapped in listing=%s",
            league, len(events), len(pending),
            [p.fixture.label for p in pending], unmapped or "none",
        )
        return [], [], requests_used
    if len(matched) < len(pending):
        log.info("%s/oddsapiio: %d of %d pending fixture(s) in the listing; absent=%s",
                 league, len(matched), len(pending),
                 [p.fixture.label for p in pending if p.fixture.key not in by_key])

    fetched_at = stamp(now)
    rows: list[dict] = []
    unpriced: list[tuple[str, str, str]] = []
    budget = config.odds.open.max_requests

    ids = [str(event.get("id")) for _, event in matched]
    #: Event ids we actually paid to ask about. A batch skipped for budget was
    #: never asked, so its silence says nothing and must not be reported as if
    #: the provider had withheld an answer.
    asked: set[str] = set()
    answered: set[str] = set()
    for batch in client.iter_batches(ids):
        if requests_used >= budget:
            log.info("request budget %d reached; %d pending left for the next tick",
                     budget, len(ids) - len(batch))
            break
        quoted = client.multi_odds(batch, oddsapiio.books_from_config(config.odds.books))
        requests_used += 1
        asked.update(batch)
        quoted_by_id = {str(q.get("id", q.get("eventId", ""))): q for q in quoted}
        answered.update(quoted_by_id)
        for pend, event in matched:
            quote = quoted_by_id.get(str(event.get("id")))
            if quote is None:
                continue
            for book in pend.missing:
                if book.provider != "oddsapiio":
                    continue
                prices = oddsapiio.extract_ml(
                    quote, oddsapiio.Book(book.provider_name or book.key, book.key)
                )
                if prices is None:
                    # Absent is not an error: it is what "has not opened yet"
                    # looks like, and it is the evidence the opener proof needs.
                    unpriced.append((pend.fixture.home, pend.fixture.away, book.key))
                    continue
                rows.append(_row(fixture=pend.fixture, book=book, prices=prices,
                                 event=event, fetched_at=fetched_at))

    # An event we paid to ask about and got no entry back for. That is not the
    # same as "the book has not opened yet": that case comes back as an entry
    # without the book's market, lands in `unpriced`, and is the evidence the
    # opener proof runs on. Here we learn nothing and record nothing, having paid
    # for the call. Liga MX was spending two of these a tick in silence.
    silent = [
        p.fixture.label for p, e in matched
        if str(e.get("id")) in asked and str(e.get("id")) not in answered
    ]
    if silent:
        log.warning("%s/oddsapiio: odds/multi returned no entry for %d of the %d "
                    "event(s) asked about: %s", league, len(silent), len(asked), silent)
    return rows, unpriced, requests_used


# --------------------------------------------------------------------------- #
# The Odds API: the anchor book
# --------------------------------------------------------------------------- #

def _capture_theoddsapi(
    league: str,
    config: LeagueConfig,
    pending: list[Pending],
    books: tuple[BookConfig, ...],
    *,
    now: datetime,
    dry_run: bool,
) -> tuple[list[dict], list[tuple[str, str, str]], int]:
    """One billed request returns the whole slate, so pending only decides
    whether to spend at all."""
    provider = config.odds.providers["theoddsapi"]
    credential = books[0].credential
    client = theoddsapi.TheOddsApiClient(
        provider.require("sport_key"),
        credential=credential,
        base_url=provider.get("base_url", theoddsapi.BASE_URL),
        market=provider.get("market", "h2h"),
    )

    quota = client.quota()
    if quota.below(config.odds.quota_floor):
        # Raised, not just logged. A refusal here is indistinguishable in the
        # returned stats from an honest "nothing new to capture" -- same zero
        # rows, same zero requests -- and the workflow step still exits green, so
        # the league silently stops anchoring and nothing downstream notices.
        # That has already happened once: the CSL account sat under its floor for
        # three days and only a human reading the config against the output
        # caught it.
        log.warning("skipping: %s has %s requests left, floor is %d",
                    theoddsapi.key_env_for(credential), quota.remaining,
                    config.odds.quota_floor)
        raise QuotaRefused(
            f"{theoddsapi.key_env_for(credential)} has {quota.remaining} "
            f"requests left, floor is {config.odds.quota_floor}"
        )
    if dry_run:
        return [], [], 0

    events = client.odds(bookmakers=[b.key for b in books])
    mapping = teams.for_league(league)
    fetched_at = stamp(now)

    wanted = {p.fixture.key: p for p in pending}
    priced: set[tuple[str, str, str]] = set()
    rows: list[dict] = []

    for record in theoddsapi.iter_prices(events):
        home = mapping.to_standard(str(record["api_home_team"]))
        away = mapping.to_standard(str(record["api_away_team"]))
        pend = wanted.get((home, away)) if home and away else None
        if pend is None:
            continue
        book = next((b for b in pend.missing if b.key == record["bookmaker"]), None)
        if book is None:
            continue  # this book already has an open: the per-book gate
        outcomes = record["outcomes"]
        prices = {
            "home_odds": outcomes.get(record["api_home_team"]),
            "draw_odds": outcomes.get("Draw"),
            "away_odds": outcomes.get(record["api_away_team"]),
            "last_update": record["last_update"],
        }
        if not all(prices[k] for k in ("home_odds", "draw_odds", "away_odds")):
            continue
        priced.add((pend.fixture.home, pend.fixture.away, book.key))
        rows.append(_row(fixture=pend.fixture, book=book, prices=prices,
                         event={"id": record["event_id"],
                                "home": record["api_home_team"],
                                "away": record["api_away_team"]},
                         fetched_at=fetched_at))

    unpriced = [
        (p.fixture.home, p.fixture.away, b.key)
        for p in pending for b in p.missing
        if b in books and (p.fixture.home, p.fixture.away, b.key) not in priced
    ]
    return rows, unpriced, 1


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def capture_opens(
    league: str,
    config: LeagueConfig,
    *,
    providers: tuple[str, ...] = ("oddsapiio", "theoddsapi"),
    ignore_schedule: bool = False,
    now: datetime | None = None,
    dry_run: bool = False,
    history_path: str | None = None,
    watch_path: str | None = None,
    fixtures_path: str | None = None,
) -> dict[str, int]:
    """One capture tick. Returns what it did.

    ``providers`` selects which books to poll, because they run on different
    cadences: the bet books are cheap enough to poll every few minutes while the
    anchor is metered by the month. The schedule itself lives in the workflow,
    not here.
    """
    now = now or datetime.now(timezone.utc)
    lp = paths.for_league(league)
    history_path = history_path or lp.capture_history_csv
    watch_path = watch_path or lp.capture_watch_csv
    stats = {"pending": 0, "captured": 0, "appended": 0, "unpriced": 0,
             "requests": 0, "refused": 0}

    all_rows: list[dict] = []
    all_unpriced: list[tuple[str, str, str]] = []
    refusals: list[str] = []

    for provider in providers:
        books = (books_for(config, provider) if ignore_schedule
                 else books_due(config, provider, now))
        if not books:
            log.info("%s/%s: not due this tick, spending nothing", league, provider)
            continue
        pending = pending_fixtures(
            league, config, books=books, now=now,
            history_path=history_path, fixtures_path=fixtures_path,
        )
        stats["pending"] += len(pending)
        if not pending:
            log.info("%s/%s: nothing pending, spending nothing", league, provider)
            continue
        try:
            if provider == "oddsapiio":
                rows, unpriced, used = _capture_oddsapiio(
                    league, config, pending, now=now, dry_run=dry_run)
            else:
                rows, unpriced, used = _capture_theoddsapi(
                    league, config, pending, books, now=now, dry_run=dry_run)
        except QuotaRefused as exc:
            # One provider being out of credit must not stop the other: the soft
            # books and the anchor are different accounts, and losing both
            # because one is spent would turn a degraded tick into a dead one.
            log.warning("%s/%s: refused, %s", league, provider, exc)
            stats["refused"] += 1
            refusals.append(f"{provider}: {exc}")
            continue
        all_rows += rows
        all_unpriced += unpriced
        stats["requests"] += used

    stats["captured"] = len(all_rows)
    stats["unpriced"] = len(all_unpriced)

    if dry_run:
        return stats

    if all_unpriced:
        capture_watch.record_unpriced(
            all_unpriced, path=watch_path,
            observed_at=stamp(now),
        )
    if all_rows:
        _, appended = capture_store.append_snapshots(
            pd.DataFrame(all_rows),
            path=history_path,
            snapshot_type="open",
            capture_reason=f"first-seen open price @ {stamp(now)}",
        )
        stats["appended"] = appended
    return stats
