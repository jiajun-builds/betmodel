"""SofaScore: fixtures, results and per-match xG. No API key.

Reaching it needs two things at once and neither is optional. The handshake must
look like a browser, which is what ``curl_cffi`` impersonation provides, and the
egress must not be a hosting provider. A plain client from a residential IP is
refused, and an impersonating client from a datacenter IP is refused. So this is
the one provider configured with a proxy, and with a retry policy that treats 403
as a statement about the current exit IP rather than about the request.

Everything below the transport is knowledge that was paid for once already and is
carried over verbatim: which score field is trustworthy, which xG period is, and
why a failed page must not look like an empty one.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from betmodel.providers import http

log = logging.getLogger(__name__)

API_BASE = "https://api.sofascore.com/api/v1"
PROXY_ENV = "SOFASCORE_PROXY_URL"

#: The two 45-minute halves as SofaScore keys them in /event/{id}/statistics.
#: Anything else it emits there is the ALL rollup or extra time (ET1/ET2).
REGULATION_PERIODS = ("1ST", "2ND")

#: How to turn published per-period xG into one number per side.
#:
#: ``halves_sum``
#:     Sum the two regulation halves. Preferred on evidence: SofaScore's ALL
#:     rollup is internally inconsistent with its own per-half figures and is
#:     sometimes plainly wrong (observed: an away ALL of 0.16 when the second half
#:     alone was 1.03). Measured over 696 team-innings the half-sum is also the
#:     better feature, MAE against goals 0.797 versus ALL's 0.815.
#:
#: ``all_first``
#:     Prefer the ALL rollup. Kept because one league's entire stored history was
#:     built this way, and switching would silently restate three seasons of model
#:     input. Migrate deliberately, with a refit, not as a side effect of a merge.
XG_STRATEGIES = ("halves_sum", "all_first")

_ALL_FIRST_ORDER = ("ALL", "REGULAR_TIME", "1ST", "2ND")


class SofascoreUnavailable(RuntimeError):
    """A request failed for reasons that say nothing about the data.

    Deliberately distinct from a 404. A 404 is SofaScore answering that the
    resource is not there, which during a paginated walk means "no more pages".
    An exhausted retry budget means we never got an answer, so a caller that
    would read it as "no more pages" would silently truncate a schedule.
    """


def _client(timeout: float = 25.0) -> http.HttpClient:
    return http.client(
        "sofascore",
        transport=http.BROWSER,
        retry=http.ROTATING_PROXY,
        timeout=timeout,
        proxy_env=PROXY_ENV,
    )


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# event field readers
# --------------------------------------------------------------------------- #

def round_label(event: dict) -> str:
    """Round number for league rounds, stage name for the post-season.

    SofaScore files playoff ties under non-sequential round numbers but names
    them in ``roundInfo.name``, and "Semifinals" says far more about a match's
    incentive structure than "28" does.

    Play-in ties are the awkward case: ``roundInfo`` is null for them and the only
    place the stage appears is the sub-tournament slug. Without that fallback
    three matches per tournament come back unlabelled, and they are exactly the
    matches where the incentives are most extreme.
    """
    info = event.get("roundInfo") or {}
    name = info.get("name")
    if name:
        return str(name)
    number = info.get("round")
    if number is not None:
        return str(number)
    slug = str(((event.get("tournament") or {}).get("slug")) or "")
    if "play-in" in slug or "play_in" in slug:
        return "Play-in"
    return ""


def event_goals(event: dict) -> tuple[int | None, int | None]:
    """Regulation-time goals for a finished event.

    Reads ``score.normaltime``, not ``score.current``. On knockout ties
    ``current`` is unusable as a goals figure: for a penalty decision it is
    regulation plus the shootout tally, so a 2-1 final reads as 11-9, and for an
    extra-time win it includes the extra-time goals, so 0-0 at ninety reads as
    3-0. The model forecasts a ninety-minute match.

    ``current`` is also occasionally self-inconsistent: SofaScore has served
    current=4 with normaltime=5 and per-half goals of 2+3 for the same match,
    where normaltime is the figure the halves corroborate.
    """

    def one(side: str) -> int | None:
        score = event.get(side) or {}
        for field in ("normaltime", "current"):
            value = score.get(field)
            if value is not None:
                return int(value)
        return None

    return one("homeScore"), one("awayScore")


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #

class SofascoreClient:
    """Fixtures, results and xG.

    ``xg_strategy`` is a per-league setting rather than a constant because the two
    merged pipelines genuinely disagreed, and their stored histories were built
    under the disagreement. See :data:`XG_STRATEGIES`.
    """

    def __init__(
        self,
        *,
        xg_strategy: str = "halves_sum",
        timeout: float = 25.0,
        pause: float = 0.35,
    ) -> None:
        if xg_strategy not in XG_STRATEGIES:
            raise ValueError(
                f"unknown xg strategy {xg_strategy!r}; expected one of {list(XG_STRATEGIES)}"
            )
        self.xg_strategy = xg_strategy
        self.pause = pause  # polite throttle between successful requests
        self._http = _client(timeout)

    @property
    def via_proxy(self) -> bool:
        return self._http.proxy_url is not None

    def _get(self, path: str, *, strict: bool = False) -> dict | None:
        """Parsed JSON, ``None`` for a 404, and a raise or ``None`` otherwise.

        With ``strict`` a failure raises :class:`SofascoreUnavailable` so a caller
        cannot mistake it for an empty result. A 404 still returns ``None`` under
        strict, because it is a real answer.
        """
        url = f"{API_BASE}{path}"
        try:
            data = self._http.json(url)
        except http.HttpError as exc:
            if exc.status == 404:
                return None
            if strict:
                raise SofascoreUnavailable(
                    f"SofaScore GET {path} failed: {exc}"
                ) from exc
            log.warning("SofaScore GET %s failed: %s", path, exc)
            return None
        time.sleep(self.pause)
        return data

    # --- tournaments and seasons -------------------------------------------
    def seasons(self, tournament_id: int) -> list[dict]:
        """Seasons for a tournament, newest first."""
        return (self._get(f"/unique-tournament/{tournament_id}/seasons") or {}).get(
            "seasons", []
        )

    def current_season_id(self, tournament_id: int) -> int | None:
        seasons = self.seasons(tournament_id)
        return seasons[0]["id"] if seasons else None

    # --- events -------------------------------------------------------------
    def round_events(self, tournament_id: int, season_id: int, round_num: int) -> list[dict]:
        """Events of one round, played and scheduled. Empty past the last round."""
        path = (
            f"/unique-tournament/{tournament_id}/season/{season_id}"
            f"/events/round/{round_num}"
        )
        return (self._get(path) or {}).get("events", [])

    def season_events(
        self,
        tournament_id: int,
        season_id: int,
        kind: str = "last",
        max_pages: int = 50,
    ) -> list[dict]:
        """Every event of a season, via the paginated endpoint.

        ``kind="last"`` returns finished events, ``"next"`` scheduled and
        in-progress ones. Preferred over walking rounds 1..N because it includes
        the knockout bracket: those ties sit under non-sequential round numbers,
        so a sequential walk stops at the first empty round and never sees them.

        A failed page raises rather than ending the walk. The caller writes this
        list straight to the fixture file, so a silent truncation to zero, or
        worse to a plausible-looking partial page, erases the schedule.
        """
        events: list[dict] = []
        for page in range(max_pages):
            path = (
                f"/unique-tournament/{tournament_id}/season/{season_id}"
                f"/events/{kind}/{page}"
            )
            data = self._get(path, strict=True)
            if not data:
                break
            events.extend(data.get("events", []))
            if not data.get("hasNextPage"):
                break
        return events

    # --- xG ------------------------------------------------------------------
    def event_xg_by_period(self, event_id: int) -> dict[str, tuple[float | None, float | None]]:
        """Every published 'Expected goals' figure, keyed by SofaScore's period."""
        data = self._get(f"/event/{event_id}/statistics")
        out: dict[str, tuple[float | None, float | None]] = {}
        if not data:
            return out
        for period in data.get("statistics", []):
            key = period.get("period")
            if not key:
                continue
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    if "xpected goals" in str(item.get("name", "")).lower():
                        out[key] = (
                            _to_float(item.get("home")),
                            _to_float(item.get("away")),
                        )
                        break
                if key in out:
                    break
        return out

    def event_xg(self, event_id: int) -> tuple[float | None, float | None]:
        """Canonical xG for one event, per the configured strategy.

        Extra time (ET1/ET2, which SofaScore does expose on knockout ties) is
        excluded either way, so xG stays on the same ninety-minute basis as
        :func:`event_goals`. Mixing extra-time xG with a regulation scoreline
        would put the model's two inputs on different clocks.
        """
        by_period = self.event_xg_by_period(event_id)
        if not by_period:
            return None, None

        if self.xg_strategy == "all_first":
            for key in _ALL_FIRST_ORDER:
                if key in by_period:
                    return by_period[key]
            return None, None

        halves = [v for k, v in by_period.items() if k in REGULATION_PERIODS]
        home = sum(h for h, _ in halves if h is not None)
        away = sum(a for _, a in halves if a is not None)
        rollup = by_period.get("ALL")

        # A league can publish the per-half keys and fill them with literal zeros
        # while carrying the real figure only in ALL. Chinese Super League does
        # exactly this. Zero is a plausible-looking number, so summing it yields a
        # silent all-zero xG feature and a model that has quietly become
        # goals-only. Treat an all-zero half-sum as "not published per half"
        # whenever the rollup disagrees.
        unpublished = not halves or (
            home == 0 and away == 0 and rollup is not None and any(rollup)
        )
        if unpublished:
            if rollup is not None:
                log.debug(
                    "event %s publishes no usable per-half xG; using the ALL rollup",
                    event_id,
                )
                return rollup
            return None, None
        return round(home, 2), round(away, 2)
