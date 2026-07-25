"""Direct SofaScore API client (curl_cffi browser impersonation).

SofaScore fingerprints the TLS handshake behind Cloudflare, so plain ``requests``
gets 403 from datacenter IPs. ``curl_cffi`` impersonates a real browser. No API key.

This is the single source for fixtures / results / upcoming (round events) and
per-event xG (event statistics).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from curl_cffi import requests as _creq

API_BASE = "https://api.sofascore.com/api/v1"
DEFAULT_IMPERSONATE = "chrome"

# The two 45-minute halves, as SofaScore keys them in /event/{id}/statistics.
# Everything else it may emit there is either the (untrusted) "ALL" rollup or
# extra time ("ET1"/"ET2"), both excluded from canonical xG -- see event_xg.
REGULATION_PERIODS = ("1ST", "2ND")


def round_label(event: dict) -> str:
    """Round label for an event: the round number for league rounds, the stage name
    ("Play-in", "Quarterfinals", "Semifinals", "Final") for the post-season.

    SofaScore files playoff ties under non-sequential round numbers but names them in
    ``roundInfo.name``, so the name is preferred where present -- "28" says much less
    than "Semifinals" about a match's incentive structure.

    Play-in (reclasificación) ties are the awkward case: ``roundInfo`` is null for
    them, and the only place SofaScore identifies the stage is the sub-tournament
    ("Liga MX, Apertura, Play In", slug ``liga-mx-apertura-play-in``). Without this
    fallback three matches per tournament come back unlabelled -- and they are exactly
    the matches where the incentive structure is most extreme.
    """
    ri = event.get("roundInfo") or {}
    name = ri.get("name")
    if name:
        return str(name)
    rnd = ri.get("round")
    if rnd is not None:
        return str(rnd)
    slug = str(((event.get("tournament") or {}).get("slug")) or "")
    if "play-in" in slug or "play_in" in slug:
        return "Play-in"
    return ""


def event_goals(event: dict) -> Tuple[Optional[int], Optional[int]]:
    """(home_goals, away_goals) over REGULATION TIME for a finished event.

    Reads ``score.normaltime``, not ``score.current``. ``current`` is unusable as a
    goals figure on knockout ties: for a penalty decision it is regulation + the
    shootout tally (a 2-1 final read as 11-9), and for an extra-time win it includes
    the extra-time goals (0-0 at 90' read as 3-0). The model forecasts a 90-minute
    match, so regulation is the quantity that belongs in the CSV -- and it is what
    the existing history already stores.

    ``current`` is also occasionally self-inconsistent: SofaScore has served
    ``current=4`` with ``normaltime=5`` and per-half goals of 2+3 for the same match,
    where normaltime is the figure corroborated by the halves.

    Falls back to ``current`` only when normaltime is absent.
    """
    def one(side: str) -> Optional[int]:
        score = event.get(side) or {}
        for field in ("normaltime", "current"):
            value = score.get(field)
            if value is not None:
                return int(value)
        return None
    return (one("homeScore"), one("awayScore"))


def _to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class SofascoreClient:
    def __init__(self, impersonate: str = DEFAULT_IMPERSONATE, max_retries: int = 3,
                 timeout: int = 25, pause: float = 0.35):
        self.impersonate = impersonate
        self.max_retries = max_retries
        self.timeout = timeout
        self.pause = pause  # polite throttle between successful requests

    def _get(self, path: str) -> Optional[dict]:
        url = f"{API_BASE}{path}"
        last_status = None
        for attempt in range(self.max_retries):
            try:
                r = _creq.get(url, impersonate=self.impersonate, timeout=self.timeout)
                last_status = r.status_code
                if r.status_code == 200:
                    time.sleep(self.pause)
                    return r.json()
                if r.status_code == 404:
                    return None
            except Exception:
                pass
            time.sleep(self.pause * (attempt + 1))
        if last_status is not None:
            print(f"  SofaScore GET {path} failed (last status {last_status})")
        return None

    def seasons(self, unique_tournament_id: int) -> list:
        """Seasons for a tournament, newest first."""
        data = self._get(f"/unique-tournament/{unique_tournament_id}/seasons")
        return (data or {}).get("seasons", [])

    def current_season_id(self, unique_tournament_id: int) -> Optional[int]:
        seasons = self.seasons(unique_tournament_id)
        return seasons[0]["id"] if seasons else None

    def round_events(self, unique_tournament_id: int, season_id: int, round_num: int) -> list:
        """Events for one round (played + scheduled). Empty list past the last round."""
        data = self._get(
            f"/unique-tournament/{unique_tournament_id}/season/{season_id}/events/round/{round_num}"
        )
        return (data or {}).get("events", [])

    def season_events(self, unique_tournament_id: int, season_id: int,
                      kind: str = "last", max_pages: int = 50) -> list:
        """Every event of a season via the paginated events endpoint.

        ``kind="last"`` returns finished events, ``kind="next"`` scheduled and
        in-progress ones. Unlike walking rounds 1..N, this enumeration includes the
        liguilla bracket -- SofaScore files Quarterfinals/Semifinals/Final under
        non-sequential round numbers (27/28/29), so a sequential walk stops at the
        first empty round and never sees the playoffs.
        """
        events: list = []
        for page in range(max_pages):
            data = self._get(
                f"/unique-tournament/{unique_tournament_id}/season/{season_id}"
                f"/events/{kind}/{page}"
            )
            if not data:
                break
            events.extend(data.get("events", []))
            if not data.get("hasNextPage"):
                break
        return events

    def event_xg_by_period(self, event_id: int) -> dict:
        """{period_key: (home_xg, away_xg)} for every 'Expected goals' stat published.

        Period keys are SofaScore's own ("ALL", "1ST", "2ND"). A period whose group
        set has no expected-goals item is omitted.
        """
        data = self._get(f"/event/{event_id}/statistics")
        out: dict = {}
        if not data:
            return out
        for period in data.get("statistics", []):
            key = period.get("period")
            if not key:
                continue
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    if "xpected goals" in str(item.get("name", "")).lower():
                        out[key] = (_to_float(item.get("home")), _to_float(item.get("away")))
                        break
                if key in out:
                    break
        return out

    def event_xg(self, event_id: int) -> Tuple[Optional[float], Optional[float]]:
        """(home_xg, away_xg) as the SUM OF THE REGULATION HALVES -- canonical xG.

        Deliberately not the ALL-period value. SofaScore's ALL is internally
        inconsistent with its own per-half figures and is sometimes plainly glitched
        (observed: an away ALL of 0.16 when the 2nd half alone was 1.03), and the
        CSV's 3-year history was built from the half-sum. Measured over 696
        team-innings the half-sum is also the marginally better feature (MAE vs goals
        0.797 against ALL's 0.815). See ligamx.xg.verify_xg for the audit.

        Extra-time periods (ET1/ET2, which SofaScore does expose on liguilla ties) are
        excluded on purpose, to stay on the same 90-minute basis as ``event_goals``.
        Mixing extra-time xG with a regulation scoreline would put the model's two
        inputs on different clocks.

        Falls back to ALL only when no per-half figure is published at all, so a match
        still yields xG rather than nothing.
        """
        by_period = self.event_xg_by_period(event_id)
        halves = [v for k, v in by_period.items() if k in REGULATION_PERIODS]
        if halves:
            home = sum(h for h, _ in halves if h is not None)
            away = sum(a for _, a in halves if a is not None)
            return (round(home, 2), round(away, 2))
        return by_period.get("ALL", (None, None))
