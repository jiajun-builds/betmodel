"""Capture closing 1X2 lines in the minutes before kickoff.

The close is the benchmark every closing-line-value claim is graded against, and
for one league there is no free source for it at all since the public archive
stopped publishing it.

**Unlike the opener, this has a hard deadline.** A fixture leaves the pre-match
feed at kickoff, so a missed close is unrecoverable at any price. There is no
backfill. That asymmetry is why the close tick runs several times more often than
the open tick.

**The window is a budget decision.** Cost is per tick, not per fixture: one call
covers the whole slate. So the number of ticks a fixture is eligible for is what
spends the monthly allowance.

| window | ticks per fixture at a 5-minute cadence |
|---|---|
| 60 min | 12 |
| 20 min | 4 |

A league whose trigger is unreliable needs the wider window; one firing on a
dependable schedule can afford the tight one and should take it. Both are
configuration.

A fixture goes final, and stops spending, as soon as the first book listed in
``close.books`` lands a close inside the target band. Until then it is
re-captured, which keeps the latest price rather than the first.

Naming several books alongside the first is free: billing counts markets by
region and up to ten bookmakers fall inside one region. Not naming them leaves
columns permanently empty for nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from betmodel import paths, teams
from betmodel.dates import local_matchday, stamp
from betmodel.config.schema import LeagueConfig
from betmodel.fixtures.upcoming import Fixture, load_upcoming
from betmodel.odds import capture_store
from betmodel.providers import theoddsapi

log = logging.getLogger(__name__)

PROVIDER_TAG = "theoddsapi"


def finalisation_book(config: LeagueConfig) -> str:
    """The book whose close ends a fixture's eligibility to spend.

    The first entry in ``close.books``. Order is load-bearing there for exactly
    this reason: a fixture is done when the book that matters has closed, not
    when any book has.
    """
    return config.odds.close.books[0]


def finalised_fixtures(league: str, config: LeagueConfig, *, history_path: str | None = None
                       ) -> set[tuple[str, str, str]]:
    """Fixtures whose close already landed inside the target band.

    Keyed on team names rather than the provider's event id, because the point is
    to stop spending on the *fixture* however it happens to be identified -- and
    on the local matchday alongside them, because a pair of names is not a
    fixture. Without the matchday one meeting's close finalises the other, and
    the second of two meetings would never be captured at all.
    """
    path = history_path or paths.for_league(league).capture_history_csv
    history = capture_store.load_history(path)
    if history.empty:
        return set()
    book = finalisation_book(config)
    closes = history[
        (history["snapshot_type"] == "close") & (history["bookmaker"] == book)
    ]
    target = config.odds.close.target_minutes * 60
    done: set[tuple[str, str, str]] = set()
    for row in closes.itertuples(index=False):
        kickoff = pd.to_datetime(row.commence_time, utc=True, errors="coerce")
        fetched = pd.to_datetime(row.fetched_at, utc=True, errors="coerce")
        if pd.isna(kickoff) or pd.isna(fetched):
            continue
        if 0 <= (kickoff - fetched).total_seconds() <= target:
            done.add((
                str(row.home_team), str(row.away_team),
                local_matchday(kickoff.to_pydatetime(), config.timezone),
            ))
    return done


def fixtures_in_window(
    league: str,
    config: LeagueConfig,
    *,
    now: datetime | None = None,
    history_path: str | None = None,
    fixtures_path: str | None = None,
) -> list[Fixture]:
    """Fixtures inside the window that are not already final.

    Decided entirely from local files, so an idle tick costs nothing at all.
    """
    now = now or datetime.now(timezone.utc)
    done = finalised_fixtures(league, config, history_path=history_path)
    window = timedelta(minutes=config.odds.close.window_minutes)
    path = fixtures_path or paths.for_league(league).upcoming_fixtures_csv
    return sorted(
        (
            f for f in load_upcoming(path)
            if f.kickoff - window <= now < f.kickoff
            and (*f.key, local_matchday(f.kickoff, config.timezone)) not in done
        ),
        key=lambda f: f.kickoff,
    )


def extract_rows(
    league: str, config: LeagueConfig, events: list[dict],
    wanted: set[tuple[str, str, str]], fetched_at: str,
) -> list[dict]:
    """Rows for the fixtures we are actually in-window for, and no others.

    The call returns the whole slate, and storing a fixture's price hours before
    kickoff as its close is exactly the mistake that once put thirteen lines
    captured a median of thirty-five hours out into the closing columns, carrying
    a 105.5% overround against 103.4% for real closes. Everything outside the
    window is discarded here rather than filtered later.

    **`wanted` carries the matchday, and it is the whole guard.** On team names
    alone the discard above does not discard: when two clubs meet twice, the
    slate's entry for the *other* meeting matches the pair and is stored as this
    one's close. That is the same thirty-five-hours-out mistake arriving by a
    different door -- and it happened. UNAM Pumas v Leon kicked off on
    2026-09-06, and what went into the store as its close was the price of their
    2026-09-11 meeting, five days from kickoff, filed under 2026-09-11. The
    fixture that actually closed has no close at all, and its closing-line value
    cannot be recovered at any price.
    """
    mapping = teams.for_league(league)
    rows: list[dict] = []
    for record in theoddsapi.iter_prices(events):
        api_home = str(record["api_home_team"])
        api_away = str(record["api_away_team"])
        home = mapping.to_standard(api_home)
        away = mapping.to_standard(api_away)
        if home is None or away is None:
            log.warning(
                "unmapped team in %r vs %r; add it to the league's team mapping "
                "(fixture skipped)", api_home, api_away,
            )
            continue
        kickoff = pd.to_datetime(record["commence_time"], utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        # The provider's own kickoff decides which meeting this price belongs to.
        # A day rather than an instant, because the two sources disagree by
        # minutes on the same match -- see `local_matchday`.
        day = local_matchday(kickoff.to_pydatetime(), config.timezone)
        if (home, away, day) not in wanted:
            continue
        outcomes = record["outcomes"]
        odds = (outcomes.get(api_home), outcomes.get("Draw"), outcomes.get(api_away))
        if not all(odds):  # a two-way or partial book is not a 1X2 close
            continue
        rows.append({
            "event_id": record["event_id"],
            "commence_time": record["commence_time"],
            "api_home_team": api_home, "api_away_team": api_away,
            "home_team": home, "away_team": away,
            "home_odds": odds[0], "draw_odds": odds[1], "away_odds": odds[2],
            "bookmaker": record["bookmaker"], "market": "h2h",
            "regions": PROVIDER_TAG,
            "last_update": record["last_update"],
            "fetched_at": fetched_at,
        })
    return rows


def capture_closes(
    league: str,
    config: LeagueConfig,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    history_path: str | None = None,
    fixtures_path: str | None = None,
) -> dict[str, int]:
    """One close tick."""
    now = now or datetime.now(timezone.utc)
    stats = {"in_window": 0, "captured": 0, "appended": 0, "requests": 0}

    due = fixtures_in_window(
        league, config, now=now, history_path=history_path, fixtures_path=fixtures_path
    )
    stats["in_window"] = len(due)
    if not due:
        log.info("%s: no fixture in the close window, spending nothing", league)
        return stats

    close = config.odds.close
    provider = config.odds.providers["theoddsapi"]
    client = theoddsapi.TheOddsApiClient(
        provider.require("sport_key"),
        credential=close.credential,
        base_url=provider.get("base_url", theoddsapi.BASE_URL),
        market=provider.get("market", "h2h"),
    )

    quota = client.quota()  # free
    if quota.below(close.min_remaining):
        log.warning(
            "%s: skipping, %s has %s requests left and the floor is %d",
            league, theoddsapi.key_env_for(close.credential),
            quota.remaining, close.min_remaining,
        )
        return stats
    if dry_run:
        log.info("%s: would capture %d fixture(s): %s",
                 league, len(due), ", ".join(f.label for f in due))
        return stats

    # No regions parameter: billing counts markets by region, and leaving it
    # unset keeps one call unambiguously one region's worth.
    events = client.odds(bookmakers=list(close.books))
    stats["requests"] = 1

    fetched_at = stamp(now)
    rows = extract_rows(
        league, config, events,
        {(*f.key, local_matchday(f.kickoff, config.timezone)) for f in due},
        fetched_at,
    )
    stats["captured"] = len(rows)
    if not rows:
        return stats

    lead = min((f.kickoff - now).total_seconds() / 60 for f in due)
    _, appended = capture_store.append_snapshots(
        pd.DataFrame(rows),
        path=history_path or paths.for_league(league).capture_history_csv,
        snapshot_type="close",
        capture_reason=f"pre-kickoff close tick @ {stamp(now)} (T-{lead:.0f}m)",
    )
    stats["appended"] = appended
    return stats
