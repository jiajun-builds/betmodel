"""The Odds API: the reference and closing prices. Metered by the month.

The scarce resource here is requests, not bandwidth: the plan allows a few
hundred a month, so every call is deliberate and every refusal is an answer
rather than noise. Retries are limited to statuses that do not consume a request.

Two billing facts shape every caller, and both are easy to lose:

* Billing is ``markets x regions`` per ``/odds`` call. The ``bookmakers`` filter
  is **free**, so asking for the whole book slate costs exactly what asking for
  one book costs. Never split a slate across several calls to "save" quota.
* ``/sports`` costs nothing and carries the remaining-request headers, so a tick
  can check its allowance before spending any of it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from betmodel.providers import http

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

#: Environment variable holding the key for a named account. Two accounts are in
#: use, split by role rather than by league: opening polling is continuous while
#: closing capture is bursty, so pairing them on one key makes a busy matchday
#: able to starve the other job.
KEY_ENV_DEFAULT = "THE_ODDS_API_KEY"


def key_env_for(credential: str) -> str:
    """Environment variable name for a credential named in league config."""
    if not credential or credential == "default":
        return KEY_ENV_DEFAULT
    return f"{KEY_ENV_DEFAULT}_{credential.upper()}"


def api_key(credential: str = "default", *, required: bool = True) -> str:
    """The key for one credential, falling back to the shared one.

    The fallback exists so a single-account setup works with no configuration,
    and so a league that names a credential still runs on a machine where only
    the shared key is present.
    """
    env = key_env_for(credential)
    value = os.environ.get(env, "").strip()
    if not value and env != KEY_ENV_DEFAULT:
        value = os.environ.get(KEY_ENV_DEFAULT, "").strip()
        if value:
            log.debug("%s unset; using %s", env, KEY_ENV_DEFAULT)
    if not value and required:
        raise RuntimeError(
            f"no The Odds API key: set {env} (or {KEY_ENV_DEFAULT})"
        )
    return value


@dataclass(frozen=True)
class Quota:
    """What the free ``/sports`` probe reports."""

    remaining: int | None
    used: int | None
    sport_available: bool

    def below(self, floor: int) -> bool:
        """Whether to skip spending. Unknown remaining is treated as fine.

        Older API behaviour omits the headers, and refusing to run because a
        header is missing would turn an informational check into an outage.
        """
        return self.remaining is not None and self.remaining < floor


class TheOddsApiClient:
    """One league's view of The Odds API."""

    def __init__(
        self,
        sport_key: str,
        *,
        credential: str = "default",
        base_url: str = BASE_URL,
        market: str = "h2h",
        timeout: float = 30.0,
    ) -> None:
        self.sport_key = sport_key
        self.credential = credential
        self.base_url = base_url.rstrip("/")
        self.market = market
        self._http = http.client(
            "theoddsapi",
            # A 403 or 401 here is an authentication or entitlement answer, so it
            # is never retried. Only statuses that cost nothing are.
            retry=http.TRANSIENT_ONLY,
            timeout=timeout,
        )

    # --- free ---------------------------------------------------------------
    def quota(self) -> Quota:
        """Remaining allowance, and whether this sport is currently listed.

        Costs zero requests, so it is safe before every spend.
        """
        response = self._http.get(
            f"{self.base_url}/sports",
            params={"apiKey": api_key(self.credential)},
        )

        def header(name: str) -> int | None:
            raw = response.headers.get(name)
            try:
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None

        sports = response.json()
        available = isinstance(sports, list) and any(
            isinstance(s, dict) and s.get("key") == self.sport_key for s in sports
        )
        return Quota(
            remaining=header("x-requests-remaining"),
            used=header("x-requests-used"),
            sport_available=available,
        )

    # --- metered ------------------------------------------------------------
    def odds(
        self,
        *,
        bookmakers: Sequence[str] | None = None,
        regions: str | None = None,
        pre_match_only: bool = True,
    ) -> list[dict[str, Any]]:
        """One billed request for the whole slate.

        ``bookmakers`` filtering is free, so pass every book of interest at once.
        ``regions`` is optional and omitting it keeps billing unambiguous at one
        region per call; pass it only where the books needed are not reachable
        otherwise.

        ``pre_match_only`` pins ``commenceTimeFrom`` to now. An in-play price is
        not a closing price, and letting one through poisons every subsequent
        closing-line-value measurement.
        """
        params: dict[str, Any] = {
            "apiKey": api_key(self.credential),
            "markets": self.market,
            "oddsFormat": "decimal",
        }
        if regions:
            params["regions"] = regions
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        if pre_match_only:
            params["commenceTimeFrom"] = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        payload = self._http.json(
            f"{self.base_url}/sports/{self.sport_key}/odds", params=params
        )
        log.info(
            "the-odds-api: %s events for %s (books=%s, regions=%s)",
            len(payload) if isinstance(payload, list) else "?",
            self.sport_key,
            ",".join(bookmakers) if bookmakers else "<all>",
            regions or "<unset>",
        )
        return payload if isinstance(payload, list) else []


def iter_prices(events: Iterable[dict]) -> Iterable[dict]:
    """Flatten the nested payload into one record per (event, bookmaker).

    The upstream shape nests event -> bookmakers -> markets -> outcomes, and each
    caller reimplementing that walk is how ``last_update`` ends up read from the
    wrong level. ``last_update`` here is the bookmaker's own timestamp for the
    market, which is the one that means anything about price movement.
    """
    for event in events or []:
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                outcomes = {
                    str(o.get("name")): o.get("price")
                    for o in market.get("outcomes") or []
                }
                yield {
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "api_home_team": event.get("home_team"),
                    "api_away_team": event.get("away_team"),
                    "bookmaker": book.get("key"),
                    "market": market.get("key"),
                    "last_update": market.get("last_update") or book.get("last_update"),
                    "outcomes": outcomes,
                }
