"""Fill in per-match xG, then recompute the model's training target.

xG comes from a provider that blocks datacenter IPs, so this is the stage that
pins a league to a residential egress path. It is also the stage that decides
what the model learns from, since the fitted target is a blend of xG and goals.

The match history does not store the provider's event id, so an event has to be
located before its statistics can be fetched: the season's events are walked once
and matched to history rows by team pairing within a date tolerance, the same
rule the fixture sync uses and for the same reason, that the stored date does not
reliably mean any particular timezone.

Only played matches missing xG are fetched. A statistics call per match is the
expensive part, and refetching a match whose xG is already stored buys nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from betmodel import paths, teams
from betmodel.config.schema import LeagueConfig
from betmodel.dates import parse_date_only_series
from betmodel.fixtures.sync import DATE_TOLERANCE_DAYS, KICKOFF_COLUMN
from betmodel.xg.blend import recompute

log = logging.getLogger(__name__)


def _events(config: LeagueConfig):
    """Every finished event of the current season, with its provider id."""
    from betmodel.providers.sofascore import SofascoreClient, event_goals

    source = config.sources["xg"]
    if source.provider != "sofascore":
        raise ValueError(
            f"{config.id}: xG provider {source.provider!r} is not supported; "
            "only sofascore publishes per-match xG here"
        )
    client = SofascoreClient(xg_strategy=source.get("period_strategy", "halves_sum"))
    mapping = teams.for_league(config.id)

    tournaments = source.get("tournaments") or {
        "season": source.require("unique_tournament_id")
    }
    out = []
    for tournament in tournaments.values():
        season_id = client.current_season_id(int(tournament))
        if season_id is None:
            continue
        for event in client.season_events(int(tournament), season_id, "last"):
            home = mapping.to_standard(str((event.get("homeTeam") or {}).get("name") or ""))
            away = mapping.to_standard(str((event.get("awayTeam") or {}).get("name") or ""))
            start = event.get("startTimestamp")
            if not (home and away and start):
                continue
            out.append({
                "id": event["id"], "home": home, "away": away,
                "kickoff": datetime.fromtimestamp(int(start), tz=timezone.utc),
            })
    return client, out


def sync(
    league: str, config: LeagueConfig, *, dry_run: bool = False, limit: int | None = None
) -> dict[str, int]:
    """Fetch missing xG, then recompute the training target."""
    lp = paths.for_league(league)
    history = pd.read_csv(lp.matches_csv)
    history["MatchDate"] = parse_date_only_series(history["Date"])

    played = pd.to_numeric(history["HG"], errors="coerce").notna()
    missing = played & pd.to_numeric(history["HxG"], errors="coerce").isna()
    stats = {"missing": int(missing.sum()), "fetched": 0, "written": 0, "blended": 0}
    log.info("%s: %d played match(es) without xG", league, stats["missing"])

    if stats["missing"]:
        client, events = _events(config)
        zone = ZoneInfo(config.timezone)
        tolerance = pd.Timedelta(days=DATE_TOLERANCE_DAYS)

        by_pairing: dict[tuple[str, str], list[dict]] = {}
        for event in events:
            by_pairing.setdefault((event["home"], event["away"]), []).append(event)

        for position in history.index[missing]:
            row = history.loc[position]
            candidates = by_pairing.get((str(row.Home), str(row.Away)))
            if not candidates:
                continue
            target = row["MatchDate"]
            event = min(
                candidates,
                key=lambda e: abs(pd.Timestamp(e["kickoff"].astimezone(zone).date()) - target),
            )
            gap = abs(pd.Timestamp(event["kickoff"].astimezone(zone).date()) - target)
            if gap > tolerance:
                continue
            if limit is not None and stats["fetched"] >= limit:
                break
            home_xg, away_xg = client.event_xg(event["id"])
            stats["fetched"] += 1
            if home_xg is None or away_xg is None:
                continue
            history.at[position, "HxG"] = home_xg
            history.at[position, "AxG"] = away_xg
            if pd.isna(history.at[position, KICKOFF_COLUMN]):
                history.at[position, KICKOFF_COLUMN] = (
                    event["kickoff"].isoformat().replace("+00:00", "Z")
                )
            stats["written"] += 1

    history, stats["blended"] = recompute(history, config)

    if dry_run:
        log.info("%s: would write %s", league, stats)
        return stats
    history.drop(columns=["MatchDate"]).to_csv(lp.matches_csv, index=False)
    log.info("%s: xg %s", league, stats)
    return stats
