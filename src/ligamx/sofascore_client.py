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

    def event_xg(self, event_id: int) -> Tuple[Optional[float], Optional[float]]:
        """(home_xg, away_xg) from the event's ALL-period 'Expected goals' stat."""
        data = self._get(f"/event/{event_id}/statistics")
        if not data:
            return (None, None)
        for period in data.get("statistics", []):
            if period.get("period") != "ALL":
                continue
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    if "xpected goals" in str(item.get("name", "")).lower():
                        return (_to_float(item.get("home")), _to_float(item.get("away")))
        return (None, None)
