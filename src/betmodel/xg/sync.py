"""Fill in per-match xG, then recompute the model's training target.

xG comes from a provider that blocks datacenter IPs, so this is the stage that
pins a league to a residential egress path. It is also the stage that decides
what the model learns from, since the fitted target is a blend of xG and goals.

The match history does not store the provider's event id, so an event has to be
located before its statistics can be fetched: the season's events are walked once
and matched to history rows by team pairing within a date tolerance, the same
rule the fixture sync uses and for the same reason, that the stored date does not
reliably mean any particular timezone.

**xG arrives twice.** The provider publishes a figure within minutes of full
time and revises it after review, and the revision is not cosmetic: measured
across the two leagues, 8 of 14 recent matches changed, by as much as 0.41 on a
mean around 1.5. A pipeline that fetches once and never looks again trains on the
provisional number forever. So every match inside a review window is refetched,
not only the ones with nothing stored.

**A zero is not a measurement.** Immediately after full time the provider serves
xG as 0.0 before computing it, and 0.0 is a plausible-looking number that a
fitter will happily accept as a real observation. An all-zero figure for a played
match is treated as "not published yet" and left blank, which is what a blank
means everywhere else in this pipeline.
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


#: How far back a played match is still refetched, to pick up the provider's
#: revision. Measured: revisions land within a few days of the match, so a week
#: is comfortable and costs a handful of calls per run.
REVIEW_WINDOW_DAYS = 7


def sync(
    league: str,
    config: LeagueConfig,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    review_days: int = REVIEW_WINDOW_DAYS,
) -> dict[str, int]:
    """Fetch missing xG, refresh recent xG, then recompute the training target."""
    lp = paths.for_league(league)
    history = pd.read_csv(lp.matches_csv)
    history["MatchDate"] = parse_date_only_series(history["Date"])

    played = pd.to_numeric(history["HG"], errors="coerce").notna()
    stored_home = pd.to_numeric(history["HxG"], errors="coerce")
    stored_away = pd.to_numeric(history["AxG"], errors="coerce")
    # An all-zero pair is the provider's not-yet-computed placeholder, not a
    # goalless chance count, and some were stored before that was understood.
    # Treated as missing so they are refetched and cleared if still unpublished.
    placeholder = played & (stored_home == 0.0) & (stored_away == 0.0)
    missing = played & (stored_home.isna() | placeholder)
    # Recently played matches are refetched even with a value stored, because the
    # first figure the provider publishes is provisional.
    cutoff = history.loc[played, "MatchDate"].max() - pd.Timedelta(days=review_days)
    recent = played & (history["MatchDate"] >= cutoff)
    wanted = missing | recent

    stats = {
        "missing": int(missing.sum()), "in_review": int((recent & ~missing).sum()),
        "fetched": 0, "written": 0, "revised": 0, "unpublished": 0,
        "cleared": 0, "blended": 0,
    }
    log.info("%s: %d without xG, %d inside the %d-day review window",
             league, stats["missing"], stats["in_review"], review_days)

    if wanted.any():
        client, events = _events(config)
        zone = ZoneInfo(config.timezone)
        tolerance = pd.Timedelta(days=DATE_TOLERANCE_DAYS)

        by_pairing: dict[tuple[str, str], list[dict]] = {}
        for event in events:
            by_pairing.setdefault((event["home"], event["away"]), []).append(event)

        for position in history.index[wanted]:
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
            # Not yet computed. The provider serves 0.0 in the minutes after full
            # time, and storing it would put a fabricated observation in the
            # training target.
            if home_xg == 0.0 and away_xg == 0.0:
                stats["unpublished"] += 1
                # Clear a placeholder that was stored before this was understood.
                if pd.notna(history.at[position, "HxG"]):
                    history.at[position, "HxG"] = pd.NA
                    history.at[position, "AxG"] = pd.NA
                    stats["cleared"] += 1
                continue
            was = pd.to_numeric(pd.Series([history.at[position, "HxG"]]),
                                errors="coerce").iloc[0]
            if pd.notna(was):
                if abs(float(was) - home_xg) < 1e-9 and abs(
                    float(pd.to_numeric(pd.Series([history.at[position, "AxG"]]),
                                        errors="coerce").iloc[0]) - away_xg) < 1e-9:
                    continue
                stats["revised"] += 1
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
