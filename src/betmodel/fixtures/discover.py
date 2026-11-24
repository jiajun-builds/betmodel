"""Add newly scheduled fixtures to the upcoming list between daily refreshes.

**Why this exists.** Opening prices are polled only for fixtures on the upcoming
list, and the list is rewritten once a day by the full fixture sync. A regular
season is scheduled months ahead, so the daily cadence costs nothing there. A
knockout round is not: its pairings exist only once the previous round is played,
the bookmakers price them within hours, and the list learned of them up to a day
later. Liga MX's Apertura 2026 quarterfinals were set on 2026-11-23 and no
opening price was polled for any of them that day. A fixture first seen already
priced can never be proven an opener, so the anchor gate refuses it and the
fixture cannot fire.

**What it may do, and what it may not.** It only *adds*: a future fixture the
provider schedules that is not on the list under that pairing and local matchday
is appended. It never removes a row, never rewrites one, never touches the match
history and never stamps the fixtures' freshness. Everything else stays the full
sync's, which remains the authority and replaces whatever this added on its next
run. That is what lets this run every half hour on the capture tick without
being able to damage the schedule: the worst it can do is add a row the next
refresh drops.

Cheap on purpose: SofaScore's ``events/next`` only, one season listing and about
one page per tournament, through the same residential path as the full sync.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd

from betmodel import paths
from betmodel.config.schema import LeagueConfig
from betmodel.fixtures import sync
from betmodel.fixtures.sync import UPCOMING_COLUMNS, upcoming_row

log = logging.getLogger(__name__)

#: The capture timer's grid. Discovery is paced off it exactly as the polled
#: books are, so "which tick am I" is answered the same way in both places.
TIMER_TICK_MINUTES = 5

#: Where a league opts in, under ``sources.fixtures``.
INTERVAL_KEY = "discover_interval_minutes"


def interval(config: LeagueConfig) -> int | None:
    """The league's discovery interval in minutes, or None when it has none.

    Only a SofaScore-backed schedule can be asked for its next events alone; a
    league on another provider has no cheap call to make here, and none needed
    so far: the CSL has no knockout stage.
    """
    source = config.sources.get("fixtures")
    if source is None or source.provider != "sofascore":
        return None
    value = source.get(INTERVAL_KEY)
    if value in (None, "", 0):
        return None
    minutes = int(value)
    if minutes % TIMER_TICK_MINUTES or (24 * 60) % minutes:
        raise ValueError(
            f"{config.id}: sources.fixtures.{INTERVAL_KEY} must be a multiple of "
            f"{TIMER_TICK_MINUTES} that divides the day, got {minutes}"
        )
    return minutes


def due(config: LeagueConfig, now: datetime) -> bool:
    """Whether this tick is a discovery slot. See `capture_open.books_due` for
    why the clock is snapped back to the timer's grid before the modulus."""
    minutes_every = interval(config)
    if minutes_every is None:
        return False
    minutes = now.hour * 60 + now.minute
    slot = minutes - (minutes % TIMER_TICK_MINUTES)
    return slot % minutes_every == 0


def discover(
    league: str,
    config: LeagueConfig,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    ignore_schedule: bool = False,
    upcoming_path: str | None = None,
) -> dict[str, int]:
    """Append the provider's newly scheduled fixtures. Returns counts."""
    now = now or datetime.now(timezone.utc)
    stats = {"scheduled": 0, "added": 0}
    if interval(config) is None:
        log.info("%s: no fixture discovery configured", league)
        return stats
    if not ignore_schedule and not due(config, now):
        log.info("%s: not a discovery slot (every %d min)", league, interval(config))
        return stats

    path = upcoming_path or paths.for_league(league).upcoming_fixtures_csv
    source = config.sources["fixtures"]
    scheduled = [
        m for m in sync._from_sofascore(config, source, kinds=("next",))
        if not m.played and m.kickoff > now
    ]
    stats["scheduled"] = len(scheduled)

    if os.path.exists(path):
        current = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        current = pd.DataFrame(columns=UPCOMING_COLUMNS)
    known = set(zip(current["Home"], current["Away"], current["Date"]))

    new = []
    for m in scheduled:
        key = (m.home, m.away, m.local_date(config.timezone))
        if key not in known:
            known.add(key)
            new.append(m)
    stats["added"] = len(new)

    if not new:
        # Said out loud: this spent provider requests, and a tick that found
        # nothing must not look like one that never asked.
        log.info("%s: discovery found %d scheduled fixture(s), none new",
                 league, len(scheduled))
        return stats

    labels = [f"{m.round} {m.home} v {m.away} ({sync._iso(m.kickoff)})" for m in new]
    if dry_run:
        log.info("%s: discovery would add %d fixture(s): %s", league, len(new), labels)
        return stats

    added = pd.DataFrame([upcoming_row(m, config) for m in new], columns=UPCOMING_COLUMNS)
    combined = pd.concat([current[UPCOMING_COLUMNS], added], ignore_index=True)
    combined = combined.sort_values("kickoff_utc", kind="stable")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False)
    log.info("%s: discovery added %d fixture(s) to the upcoming list: %s",
             league, len(new), labels)
    return stats
