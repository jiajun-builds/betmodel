"""TheSportsDB: league schedule and results. Free tier, no key of our own.

An ordinary JSON API with an ordinary IP policy, so unlike SofaScore it runs from
a cloud runner unchanged. That is the whole reason a league on this provider gets
its full refresh in CI while a SofaScore-only league needs a residential egress.

The free tier is rate limited rather than quota limited, so retrying a 429 costs
patience but not allowance, and the walk paces itself between rounds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from betmodel.providers import http

log = logging.getLogger(__name__)

#: The documented free-tier key. Public, and part of the base URL upstream.
BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"

#: Rounds are walked rather than listed, so the walk needs a stop rule. A single
#: empty round is not the end of a season: providers leave gaps for postponed
#: rounds and for cup weeks, and stopping at the first gap truncates the schedule.
EMPTY_ROUNDS_BEFORE_STOP = 3

#: Between rounds. The free tier throttles aggressively and a whole-season walk
#: is thirty-odd requests, so pacing costs a minute and avoids a 429 storm.
ROUND_PAUSE_SECONDS = 2.0


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_event(event: dict, *, season_label: str, country: str, league: str) -> dict:
    """One upstream event as a row of the master match table.

    Country, league and season are passed in rather than read from the payload:
    they identify the competition we asked for, and taking them from the response
    would let a provider-side mislabelling rewrite our own history.

    xG and closing-odds columns are minted empty here. They are filled by the xG
    stage and the odds reducer respectively, and creating them now keeps the
    column set stable whichever stages have run.
    """
    home_goals = _int_or_none(event.get("intHomeScore"))
    away_goals = _int_or_none(event.get("intAwayScore"))
    result = None
    if home_goals is not None and away_goals is not None:
        result = "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D")

    kickoff = event.get("strTime") or ""
    return {
        "Country": country,
        "League": league,
        "Season": season_label,
        "Round": _int_or_none(event.get("intRound")),
        "Date": event.get("dateEvent"),
        "Time": kickoff[:5] if len(kickoff) >= 5 else None,
        "Home": event.get("strHomeTeam"),
        "Away": event.get("strAwayTeam"),
        "HG": home_goals,
        "AG": away_goals,
        "HxG": None,
        "AxG": None,
        "HExpG+": None,
        "AExpG+": None,
        "Res": result,
        "PSCH": None,
        "PSCD": None,
        "PSCA": None,
    }


@dataclass
class TheSportsDBClient:
    """Schedule and results for one league.

    ``season_formats`` exists because the provider spells a season differently
    per competition, sometimes "2026" and sometimes "2026-2027", and gets it wrong
    silently: an unknown spelling returns an empty event list rather than an
    error. So the spelling is resolved once by asking for round one under each
    candidate and keeping the first that answers.
    """

    league_id: str
    max_rounds: int = 35
    pause: float = ROUND_PAUSE_SECONDS
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self._http = http.client(
            "thesportsdb",
            retry=http.RetryPolicy(
                attempts=3,
                # Rate limited, not quota limited: a 429 costs patience only.
                retry_statuses=frozenset({408, 429, 500, 502, 503, 504}),
                backoff_base=5.0,
                backoff_max=60.0,
            ),
            timeout=self.timeout,
        )

    def _round(self, season: str, round_num: int) -> list[dict]:
        data = self._http.json(
            f"{BASE_URL}/eventsround.php",
            params={"id": self.league_id, "r": round_num, "s": season},
        )
        return (data or {}).get("events") or []

    def resolve_season(self, candidates: Sequence[str]) -> str | None:
        """The spelling this provider uses for the season, or None.

        Returning None rather than guessing matters: every downstream call takes
        the season string, and a wrong one yields empty rounds indistinguishable
        from an off-season.
        """
        for candidate in candidates:
            if self._round(candidate, 1):
                log.info("season resolved as %r", candidate)
                return candidate
            time.sleep(1)
        return None

    def season_events(self, season: str) -> list[dict]:
        """Every event of a season, walking rounds until the season runs out."""
        events: list[dict] = []
        empty_streak = 0
        for round_num in range(1, self.max_rounds + 1):
            found = self._round(season, round_num)
            if found:
                empty_streak = 0
                events.extend(found)
                played = sum(1 for e in found if e.get("intHomeScore") not in (None, ""))
                log.debug("round %2d: %2d matches (%d played)", round_num, len(found), played)
            else:
                empty_streak += 1
                if empty_streak >= EMPTY_ROUNDS_BEFORE_STOP:
                    log.debug("stopping at round %d after %d empty rounds",
                              round_num, empty_streak)
                    break
            time.sleep(self.pause)
        return events

    def season_rows(
        self, season: str, *, season_label: str, country: str, league: str
    ) -> list[dict]:
        """Season events already shaped as master-table rows."""
        return [
            parse_event(e, season_label=season_label, country=country, league=league)
            for e in self.season_events(season)
        ]
