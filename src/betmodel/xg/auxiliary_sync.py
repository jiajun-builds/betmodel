"""Keep an auxiliary competition's match table current (D42).

The primary xG sync fills in a match table someone else (the fixture sync) owns.
An auxiliary competition has no fixture sync and no odds, so this one owns its
file outright: it lists the current season's finished events and upserts them
by SofaScore event id.

It inherits the primary's two lessons about xG, for the same reasons. **xG
arrives twice**, so every match inside the same review window is refetched, not
only the ones with nothing stored. **A zero is not a measurement**: an all-zero
pair is the provider's not-yet-computed placeholder and is stored blank, which
keeps the match out of the fit until the real figure lands.

And it has one rule of its own. **It may never cost the primary anything.** It
runs after the primary sync has written its result, and any failure -- the
provider unreachable, a malformed listing -- is logged with what was not done and
returns. A League One outage is a reason for a slightly staler promoted-club
rating, never for a stale CSL.

The file is only ever added to: a row not in this season's listing (every past
season's, for a start) is kept as it is.

**A blank is asked about for a month, not for ever.** SofaScore publishes xG
within minutes of full time and revises it within days, or does not publish it
at all: League One 2026 rounds 1-7 have none, nor does one 2024 match, and that
did not change in the eight months the season ran. Asking about all fifty every
day spent about three minutes of each refresh on answers known in advance. A
blank is refetched until its match is ``GIVE_UP_DAYS`` old, then counted as
``abandoned`` and left; ``give_up_days=None`` asks about every blank again. A row
new to the file is always asked once, whatever its age.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd

from betmodel import paths
from betmodel.config.schema import AuxiliarySource, LeagueConfig
from betmodel.dates import stamp
from betmodel.fixtures.sync import status_finished
from betmodel.models.auxiliary import COLUMNS
from betmodel.xg.sync import REVIEW_WINDOW_DAYS

log = logging.getLogger(__name__)

#: How long after kickoff a match with no xG is still asked about. Thirty days is
#: several times the longest wait for a first figure seen, and costs one request
#: a day per blank inside it.
GIVE_UP_DAYS = 30


def _number(value) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell(value) -> str:
    """How a number is stored: an integer without a trailing ``.0``."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def _event_row(event: dict, goals: tuple[int, int], kickoff: datetime) -> dict[str, str]:
    from betmodel.providers.sofascore import round_label

    home, away = event.get("homeTeam") or {}, event.get("awayTeam") or {}
    season = (event.get("season") or {}).get("year") or kickoff.year
    return {
        "event_id": str(int(event["id"])),
        "Season": str(season),
        "Round": round_label(event),
        "Date": kickoff.strftime("%Y-%m-%d"),
        "Time": kickoff.strftime("%H:%M"),
        "kickoff_utc": stamp(kickoff),
        "home_sofascore": str(home.get("name") or ""),
        "away_sofascore": str(away.get("name") or ""),
        "home_team_id": _cell(home.get("id")),
        "away_team_id": _cell(away.get("id")),
        "HG": str(goals[0]),
        "AG": str(goals[1]),
        "HxG": "",
        "AxG": "",
    }


def _client(config: LeagueConfig):
    from betmodel.providers.sofascore import SofascoreClient

    return SofascoreClient(
        xg_strategy=config.sources["xg"].get("period_strategy", "halves_sum"))


def sync(
    league: str,
    config: LeagueConfig,
    source: AuxiliarySource,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    review_days: int = REVIEW_WINDOW_DAYS,
    give_up_days: int | None = GIVE_UP_DAYS,
    now: datetime | None = None,
    client=None,
) -> dict[str, int]:
    """Upsert the current season's finished matches and refresh their xG.

    Never raises for a provider failure; ``stats["failed"]`` is 1 and the file is
    untouched. xG is read with the league's own ``period_strategy``.
    """
    stats = {"listed": 0, "added": 0, "fetched": 0, "written": 0, "revised": 0,
             "unpublished": 0, "cleared": 0, "abandoned": 0, "failed": 0}
    if source.source is None or source.source.provider != "sofascore":
        log.warning("%s: auxiliary %s has no sofascore source; not synced",
                    league, source.id)
        return stats

    path = paths.for_league(league).auxiliary_csv(source.file)
    try:
        return _sync(league, config, source, path, stats, dry_run=dry_run, limit=limit,
                     review_days=review_days, give_up_days=give_up_days,
                     now=now or datetime.now(timezone.utc), client=client)
    except Exception as exc:  # noqa: BLE001 -- see the module note
        stats["failed"] = 1
        log.error(
            "%s: auxiliary %s sync failed (%s: %s); no new %s matches were added and "
            "no xG was refreshed, %s is unchanged. The primary xG sync stands.",
            league, source.id, type(exc).__name__, exc, source.name, path,
            exc_info=not _is_provider_failure(exc),
        )
        return stats


def _is_provider_failure(exc: Exception) -> bool:
    from betmodel.providers import http
    from betmodel.providers.sofascore import SofascoreUnavailable

    return isinstance(exc, (SofascoreUnavailable, http.HttpError))


def _sync(league, config, source, path, stats, *, dry_run, limit, review_days,
          give_up_days, now, client):
    if os.path.exists(path):
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        table = pd.DataFrame(columns=list(COLUMNS))
    for column in COLUMNS:
        if column not in table.columns:
            table[column] = ""
    known = {str(int(_number(e))): i for i, e in zip(table.index, table["event_id"])
             if _number(e) is not None}

    client = client or _client(config)
    tournament = int(source.source.require("unique_tournament_id"))
    season_id = client.current_season_id(tournament)
    if season_id is None:
        log.warning("%s: auxiliary %s: SofaScore lists no season for tournament %d; "
                    "nothing synced", league, source.id, tournament)
        return stats
    events = client.season_events(tournament, season_id, "last")
    stats["listed"] = len(events)

    from betmodel.providers.sofascore import event_goals

    new_rows: list[dict[str, str]] = []
    added: set[str] = set()
    changed = False
    for event in events:
        if status_finished((event.get("status") or {}).get("type")) is False:
            continue
        goals = event_goals(event)
        start = event.get("startTimestamp")
        if goals[0] is None or goals[1] is None or not start or event.get("id") is None:
            continue
        key = str(int(event["id"]))
        if key in added:
            continue
        if key in known:
            position = known[key]
            for column, value in (("HG", goals[0]), ("AG", goals[1])):
                if _number(table.at[position, column]) != float(value):
                    table.at[position, column] = str(value)
                    changed = True
            continue
        kickoff = datetime.fromtimestamp(int(start), tz=timezone.utc)
        new_rows.append(_event_row(event, goals, kickoff))
        added.add(key)
    if new_rows:
        stats["added"] = len(new_rows)
        table = pd.concat([table, pd.DataFrame(new_rows)], ignore_index=True).fillna("")
        changed = True

    # The same rule as the primary: anything without xG, and anything inside the
    # review window of the latest match whether it has a figure or not.
    dates = pd.to_datetime(table["Date"], errors="coerce")
    home_xg = table["HxG"].map(_number)
    away_xg = table["AxG"].map(_number)
    missing = home_xg.isna() | away_xg.isna() | ((home_xg == 0.0) & (away_xg == 0.0))
    latest = dates.max()
    recent = (dates >= latest - pd.Timedelta(days=review_days)) if pd.notna(latest) \
        else pd.Series(False, index=table.index)
    # A blank is asked about only while SofaScore might still publish it.
    if give_up_days is None:
        abandoned = pd.Series(False, index=table.index)
    else:
        kickoff = pd.to_datetime(table["kickoff_utc"], utc=True, errors="coerce")
        kickoff = kickoff.fillna(pd.to_datetime(table["Date"], utc=True, errors="coerce"))
        # A row this run added has never been asked about, however old its
        # match: a first sync, or a backfill, still gets one try at every row.
        new = table["event_id"].isin(added)
        abandoned = missing & ~recent & ~new & (
            kickoff < pd.Timestamp(now) - pd.Timedelta(days=give_up_days))
    stats["abandoned"] = int((abandoned & (table["event_id"] != "")).sum())
    wanted = table.index[(missing | recent) & ~abandoned & (table["event_id"] != "")]

    for position in wanted:
        if limit is not None and stats["fetched"] >= limit:
            break
        home, away = client.event_xg(int(float(table.at[position, "event_id"])))
        stats["fetched"] += 1
        if home is None or away is None:
            continue
        if home == 0.0 and away == 0.0:
            stats["unpublished"] += 1
            if table.at[position, "HxG"] != "" or table.at[position, "AxG"] != "":
                table.at[position, "HxG"] = table.at[position, "AxG"] = ""
                stats["cleared"] += 1
                changed = True
            continue
        was_home, was_away = home_xg[position], away_xg[position]
        if pd.notna(was_home) and pd.notna(was_away):
            if abs(was_home - home) < 1e-9 and abs(was_away - away) < 1e-9:
                continue
            if not (was_home == 0.0 and was_away == 0.0):
                stats["revised"] += 1
        table.at[position, "HxG"] = _cell(home)
        table.at[position, "AxG"] = _cell(away)
        stats["written"] += 1
        changed = True

    if not changed:
        log.info("%s: auxiliary %s up to date %s", league, source.id, stats)
        return stats
    order = list(COLUMNS) + [c for c in table.columns if c not in COLUMNS]
    sort_id = pd.to_numeric(table["event_id"], errors="coerce")
    table = (table.assign(_id=sort_id)
             .sort_values(["kickoff_utc", "_id"], kind="stable")
             .drop(columns="_id")[order])
    if dry_run:
        log.info("%s: auxiliary %s would write %s", league, source.id, stats)
        return stats
    table.to_csv(path, index=False)
    log.info("%s: auxiliary %s %s", league, source.id, stats)
    return stats
