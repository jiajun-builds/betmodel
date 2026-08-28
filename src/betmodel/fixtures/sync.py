"""Refresh the match history and the upcoming-fixture list.

Provider-agnostic: which upstream a league uses is configuration, and the two
shipped leagues use different ones because only one of them has a schedule source
that a cloud runner can reach.

Three rules, each protecting against a way this stage can destroy data rather
than merely fail.

**Never erase.** A played match already in the history is updated in place, never
removed. An upstream that briefly forgets a season would otherwise delete years of
results, and a model refit immediately afterwards would train on the remains.

**Never blank the schedule.** An empty upcoming list is refused unless explicitly
allowed. Every timing decision downstream reads that file, so an outage that
emptied it would silently stop opening-line capture rather than raise.

**Forward-only.** Results are merged for matches the upstream now reports as
played; a match that reverts to unplayed is left as it is.

A match is **not** keyed on its stored date, because that date cannot be trusted
to mean anything in particular. Classified against provider truth, one league's
history carries at least three timezone conventions: league-local for one season,
UK-local for the stretch backfilled from a public archive that stamps in UK time,
and UTC for a handful. Keying on it reported 98 additions where the answer was
none, and writing them would have duplicated a third of a season.

So the key is the team pairing within a small date tolerance. A pairing does not
recur within days, which makes it unambiguous where the date is not, and the
matched row then gets an explicit ``kickoff_utc`` written into it so the
ambiguity stops propagating.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from betmodel import paths, teams
from betmodel.config.schema import LeagueConfig, SourceConfig
from betmodel.dates import parse_date_only_series

log = logging.getLogger(__name__)

#: Columns a fixture row contributes to the match history.
MATCH_COLUMNS = [
    "Country", "League", "Season", "Round", "Date", "Time",
    "Home", "Away", "HG", "AG", "Res",
]

#: The unambiguous kickoff, written into the match history as it is learned.
#: The Date and Time columns predate it and are left alone: they are what several
#: seasons of research already reference, and rewriting them would silently move
#: rows other work is keyed to.
KICKOFF_COLUMN = "kickoff_utc"

#: How far the stored date may sit from the true matchday and still be the same
#: match. Two days covers every timezone convention found in the histories, and a
#: pairing does not recur within two days.
DATE_TOLERANCE_DAYS = 2

UPCOMING_COLUMNS = ["Season", "Round", "Date", "Time", "Home", "Away", "kickoff_utc"]


@dataclass(frozen=True)
class Match:
    """One match as every provider must present it."""

    season: str
    round: str
    kickoff: datetime
    home: str
    away: str
    home_goals: int | None = None
    away_goals: int | None = None

    def local_date(self, timezone_name: str) -> str:
        """The matchday as the league experiences it, which is how it is stored."""
        return self.kickoff.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d")

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> str | None:
        if not self.played:
            return None
        if self.home_goals > self.away_goals:
            return "H"
        return "A" if self.home_goals < self.away_goals else "D"


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #

def _from_thesportsdb(config: LeagueConfig, source: SourceConfig) -> list[Match]:
    from betmodel.providers.thesportsdb import TheSportsDBClient

    client = TheSportsDBClient(
        league_id=str(source.require("league_id")),
        max_rounds=int(source.get("max_rounds", 35)),
    )
    mapping = teams.for_league(config.id)
    matches: list[Match] = []
    for label, candidates in source.require("seasons"):
        spelling = client.resolve_season(list(candidates))
        if spelling is None:
            log.warning("no season spelling answered for %s; skipping", label)
            continue
        for event in client.season_events(spelling):
            kickoff = _parse_kickoff(event.get("dateEvent"), event.get("strTime"))
            home = mapping.to_standard(str(event.get("strHomeTeam") or ""))
            away = mapping.to_standard(str(event.get("strAwayTeam") or ""))
            if kickoff is None or not home or not away:
                continue
            matches.append(Match(
                season=str(label), round=str(event.get("intRound") or ""),
                kickoff=kickoff, home=home, away=away,
                home_goals=_int(event.get("intHomeScore")),
                away_goals=_int(event.get("intAwayScore")),
            ))
    return matches


def _from_sofascore(config: LeagueConfig, source: SourceConfig) -> list[Match]:
    from betmodel.providers.sofascore import (
        SofascoreClient, event_goals, round_label,
    )

    client = SofascoreClient(
        xg_strategy=source.get("period_strategy", "halves_sum")
    )
    mapping = teams.for_league(config.id)
    matches: list[Match] = []
    for label, tournament in (source.require("tournaments") or {}).items():
        seasons = client.seasons(int(tournament))
        if not seasons:
            log.warning("no seasons for tournament %s", tournament)
            continue
        season_id = seasons[0]["id"]
        # "Apertura 2026", not "Apertura". The history stores the year, and a
        # label without it collapses every edition of the tournament into one.
        year = str(seasons[0].get("year") or "").strip()
        season_label = f"{str(label).title()} {year}".strip()
        for kind in ("last", "next"):
            for event in client.season_events(int(tournament), season_id, kind):
                kickoff = _from_timestamp(event.get("startTimestamp"))
                home = mapping.to_standard(
                    str((event.get("homeTeam") or {}).get("name") or "")
                )
                away = mapping.to_standard(
                    str((event.get("awayTeam") or {}).get("name") or "")
                )
                if kickoff is None or not home or not away:
                    continue
                home_goals, away_goals = event_goals(event)
                matches.append(Match(
                    season=season_label, round=round_label(event),
                    kickoff=kickoff, home=home, away=away,
                    home_goals=home_goals, away_goals=away_goals,
                ))
    return matches


PROVIDERS = {"thesportsdb": _from_thesportsdb, "sofascore": _from_sofascore}


def fetch(config: LeagueConfig) -> list[Match]:
    source = config.sources["fixtures"]
    if source.provider not in PROVIDERS:
        raise ValueError(
            f"{config.id}: no fixture provider named {source.provider!r}; "
            f"have {sorted(PROVIDERS)}"
        )
    return PROVIDERS[source.provider](config, source)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #

def sync(
    league: str,
    config: LeagueConfig,
    *,
    dry_run: bool = False,
    allow_empty_upcoming: bool = False,
    now: datetime | None = None,
) -> dict[str, int]:
    """Fetch, merge into the history, and rewrite the upcoming list."""
    now = now or datetime.now(timezone.utc)
    lp = paths.for_league(league)
    matches = fetch(config)
    stats = {"fetched": len(matches), "updated": 0, "added": 0, "upcoming": 0}
    if not matches:
        log.warning("%s: the fixture provider returned nothing; leaving files alone", league)
        return stats

    history = pd.read_csv(lp.matches_csv)
    history["MatchDate"] = parse_date_only_series(history["Date"])
    if KICKOFF_COLUMN not in history.columns:
        history[KICKOFF_COLUMN] = pd.NA
    stats["stamped"] = 0

    by_pairing: dict[tuple[str, str], list[int]] = {}
    for position, row in history.iterrows():
        by_pairing.setdefault((str(row.Home), str(row.Away)), []).append(position)
    tolerance = pd.Timedelta(days=DATE_TOLERANCE_DAYS)

    for match in matches:
        if not match.played:
            continue
        position = _locate(history, by_pairing, match, config.timezone, tolerance)
        if position is None:
            stats["added"] += 1
            history.loc[len(history)] = _row(config, match, history.columns)
            continue
        # Record the unambiguous kickoff on the row we just identified, so the
        # next run does not have to infer it again.
        if pd.isna(history.at[position, KICKOFF_COLUMN]):
            history.at[position, KICKOFF_COLUMN] = _iso(match.kickoff)
            stats["stamped"] += 1
        # Update in place. Never remove, and never revert a played match.
        if pd.isna(history.at[position, "HG"]):
            history.at[position, "HG"] = match.home_goals
            history.at[position, "AG"] = match.away_goals
            history.at[position, "Res"] = match.result
            stats["updated"] += 1

    upcoming = sorted(
        (m for m in matches if not m.played and m.kickoff > now),
        key=lambda m: m.kickoff,
    )
    stats["upcoming"] = len(upcoming)
    if not upcoming and not allow_empty_upcoming:
        raise RuntimeError(
            f"{league}: the provider returned no upcoming fixture. Every timing "
            "decision downstream reads that file, so it is not overwritten with "
            "nothing. Pass allow_empty_upcoming to override at season end."
        )

    if dry_run:
        log.info("%s: would write %s", league, stats)
        return stats

    history.drop(columns=["MatchDate"]).to_csv(lp.matches_csv, index=False)
    pd.DataFrame([{
        "Season": m.season, "Round": m.round,
        "Date": m.local_date(config.timezone),
        "Time": m.kickoff.strftime("%H:%M"),
        "Home": m.home, "Away": m.away,
        "kickoff_utc": m.kickoff.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    } for m in upcoming], columns=UPCOMING_COLUMNS).to_csv(
        lp.upcoming_fixtures_csv, index=False
    )
    _stamp(lp.fixtures_meta_json, now)
    log.info("%s: fixtures %s", league, stats)
    return stats


def _locate(history, by_pairing, match: Match, zone: str, tolerance) -> int | None:
    """The history row for this match, or None.

    Nearest date wins among the candidates for a pairing, which resolves the one
    case a pairing can repeat inside the tolerance: a two-legged tie.
    """
    candidates = by_pairing.get((match.home, match.away))
    if not candidates:
        return None
    target = pd.Timestamp(match.local_date(zone))
    best, best_gap = None, None
    for position in candidates:
        gap = abs(history.at[position, "MatchDate"] - target)
        if gap <= tolerance and (best_gap is None or gap < best_gap):
            best, best_gap = position, gap
    return best


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _row(config: LeagueConfig, match: Match, columns) -> list:
    values = {
        "Country": config.name, "League": config.name, "Season": match.season,
        "Round": match.round, "Date": match.local_date(config.timezone),
        "Time": match.kickoff.strftime("%H:%M"), "Home": match.home, "Away": match.away,
        "HG": match.home_goals, "AG": match.away_goals, "Res": match.result,
        "MatchDate": pd.Timestamp(match.local_date(config.timezone)),
        KICKOFF_COLUMN: _iso(match.kickoff),
    }
    return [values.get(column) for column in columns]


def _stamp(path: str, moment: datetime) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"fixtures_updated_at": moment.isoformat(timespec="seconds")}, handle)


def _int(value) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _parse_kickoff(date_value, time_value) -> datetime | None:
    if not date_value:
        return None
    stamp = pd.to_datetime(f"{date_value} {time_value or '00:00'}", utc=True, errors="coerce")
    return None if pd.isna(stamp) else stamp.to_pydatetime()


def _from_timestamp(value) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError):
        return None
