#!/usr/bin/env python3
"""What odds-api.io's event listing actually contains, against what we asked for.

The capture path decides everything on one join: our pending fixtures, keyed by
standard team names, against the provider's listing mapped through
``teams.for_league``. When that join comes out empty the capture returns having
spent the listing request and having logged nothing, so a soft-book blackout and
a healthy idle tick print the same line. This prints the join.

It answers two questions the log cannot separate:

    is the fixture missing from the provider's listing entirely
    or is it there under a name our mapping does not resolve

The distinction decides the fix. The first is the provider's problem and nothing
in this repo corrects it; the second is one row in
``data/<league>/team_name_mapping.csv``.

**One request per run** -- the listing only, the same call the capture makes
first. ``--odds`` adds one more to ask what the books actually quote for the
fixtures that did match; it is off by default because this provider bills per
request and the question above does not need it.

The key is read from the environment exactly as the pipeline reads it
(``ODDS_API_IO_KEY_<CREDENTIAL>``, falling back to ``ODDS_API_IO_KEY``) and is
never printed.

    python scripts/check_event_listing.py csl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from betmodel import teams
from betmodel.config import load_league
from betmodel.odds import capture_open
from betmodel.providers import oddsapiio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("league")
    parser.add_argument("--odds", action="store_true",
                        help="spend one more request on odds/multi for matched events")
    parser.add_argument("--lookback", type=int, default=0, metavar="DAYS",
                        help="also start the window this many days in the PAST. A "
                             "postponed match replayed weeks later may still be "
                             "listed under the date it was originally due, which "
                             "a window starting at now would never see.")
    parser.add_argument("--lookahead", type=int, default=None, metavar="DAYS",
                        help="override the league's own open lookahead")
    args = parser.parse_args(argv)

    config = load_league(args.league)
    provider = config.odds.providers["oddsapiio"]
    client = oddsapiio.OddsApiIoClient(
        provider.require("league_slugs"),
        sport=provider.get("sport", "football"),
        base_url=provider.get("base_url", oddsapiio.BASE_URL),
        credential=provider.get("credential", "default"),
    )

    lookahead = args.lookahead or config.odds.open.lookahead_days
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.lookback)
    end = now + timedelta(days=lookahead)
    slugs = provider.require("league_slugs")
    print(f"query: sport={provider.get('sport', 'football')} slugs={slugs}")
    print(f"       from {start.isoformat()} to {end.isoformat()}")
    events = client.list_events(start, end)
    print(f"listing: {len(events)} event(s)\n")

    if events:
        # The id is what `odds/multi` is asked about, and `str(None)` is a
        # perfectly well-formed way to ask about nothing. Printing the raw keys
        # of one event is how you find out the field is not called what the code
        # thinks it is called -- which would look exactly like a book that has
        # not opened.
        print(f"first event's raw keys: {sorted(events[0])}")
        print(f"first event's id: {events[0].get('id')!r}\n")

    mapping = teams.for_league(args.league)
    by_key: dict[tuple[str, str], dict] = {}
    print(f"{'provider name':52s} {'-> standard':52s} kickoff")
    for event in sorted(events, key=lambda e: str(e.get("date") or e.get("commence_time") or "")):
        # Exactly the expression `_capture_oddsapiio` builds its own index from.
        # A looser one here would report a fixture as present that the capture
        # cannot see, which is the one answer this script must never give.
        raw_home = str(event.get("home") or "")
        raw_away = str(event.get("away") or "")
        home = mapping.to_standard(raw_home)
        away = mapping.to_standard(raw_away)
        when = str(event.get("date") or event.get("commence_time") or "?")
        shown = f"{raw_home or '(no home field)'} v {raw_away or '(no away field)'}"
        arrow = f"{home} v {away}" if home and away else "UNMAPPED"
        print(f"{shown:52s} {arrow:52s} {when}")
        if home and away:
            by_key[(home, away)] = event

    books = capture_open.books_for(config, "oddsapiio")
    pending = capture_open.pending_fixtures(args.league, config, books=books)
    now = datetime.now(timezone.utc)
    print(f"\npending ({len(pending)}), and whether the listing carries it:")
    matched = []
    for pend in pending:
        days = (pend.fixture.kickoff - now).total_seconds() / 86400
        event = by_key.get(pend.fixture.key)
        if event is not None:
            matched.append((pend, event))
        verdict = "in listing" if event is not None else "NOT IN LISTING"
        print(f"  {pend.fixture.home:26s} v {pend.fixture.away:26s} "
              f"{days:5.1f}d  {verdict:15s} missing={[b.key for b in pend.missing]}")

    if not matched:
        print("\nNothing matched. This is the silent early return in "
              "capture_open._capture_oddsapiio: the listing request is spent and "
              "no unpriced sighting is recorded, so the opener proof gets nothing "
              "either.")
        return 0

    if not args.odds:
        print(f"\n{len(matched)} matched. Re-run with --odds to spend one more "
              "request on what the books quote for them.")
        return 0

    wanted = oddsapiio.books_from_config(config.odds.books)
    quoted = client.multi_odds([str(e.get("id")) for _, e in matched][:10], wanted)
    quoted_by_id = {str(q.get("id", q.get("eventId", ""))): q for q in quoted}
    print(f"\nodds/multi returned {len(quoted)} entr(y/ies):")
    for pend, event in matched:
        quote = quoted_by_id.get(str(event.get("id")))
        if quote is None:
            print(f"  {pend.fixture.label}: no entry returned")
            continue
        for book in pend.missing:
            if book.provider != "oddsapiio":
                continue
            prices = oddsapiio.extract_ml(
                quote, oddsapiio.Book(book.provider_name or book.key, book.key)
            )
            print(f"  {pend.fixture.label}: {book.key} -> "
                  f"{prices if prices else 'not priced yet (would record unpriced)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
