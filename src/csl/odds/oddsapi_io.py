"""Thin odds-api.io client — the 1xBet opening-line source (replaces The Odds API for that book).

Why a second provider at all
----------------------------
1xBet's opening line used to come from the same The Odds API call as Pinnacle's,
captured by ``capture_scheduler`` only while "now" sat inside a *predicted* opening
window. That design loses a line whenever the feed publishes the fixture after the
window closes, and the window cannot simply be widened: The Odds API's free plan is
~500 requests **per month**, and a pending fixture costs one request per 10-min tick,
so a 48h window burns ~288 requests on a single stubborn fixture.

Observed failure (2026-08-01, Shandong Taishan vs Tianjin Jinmen Tiger): the scheduler
polled correctly for the full 12h window and The Odds API never listed the fixture; the
line posted ~25h after the anchor and the open was lost for good.

odds-api.io breaks the deadlock because its free tier is ~500 requests **per day**
(100/hour), and ``/odds/multi`` returns up to 10 events for a single request. That is
enough to poll the whole slate every 15 minutes all day, so there is no window to
expire — a late-posting line is picked up within one tick of appearing.

What this module is NOT
-----------------------
It does not fetch Pinnacle. odds-api.io's free tier carries recreational books only
(this account has exactly one book selected: ``1xbet``); sharp books are a paid add-on.
Pinnacle's open — the λ de-bias anchor — stays on The Odds API via
``fetch_pinnacle_spreads`` / ``capture_scheduler``.

``/odds/movements`` was the preferred design (it returns an ``opening`` object with a
real timestamp, which would make opening lines retrievable retroactively). Probed
2026-08-02 across four market/line combinations on live CSL fixtures: every call
returned **404 "No data found"**, not 403 — the movements store simply carries no data
for 1xbet on this league. Hence the polling approach below. Re-probe if the account is
ever upgraded; movements would let this module drop the polling entirely.

Vocabulary note
---------------
Rows are written with ``bookmaker="onexbet"`` — The Odds API's key for 1xBet, not
odds-api.io's ``"1xbet"``. This is deliberate: it keeps
``export_upcoming_market_comparison.BET_BOOKMAKER`` and every downstream consumer
(dashboard JSON fields, signal alerts, ``app.js``) working unchanged. The provider is
recorded in the ``regions`` column instead (see ``PROVIDER_TAG``).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from csl.odds.fetch_pinnacle_spreads import (
    MARKET_LABEL,
    TeamMapping,
    _clean_name,
    _coerce_float,
    normalize_team_name,
)

BASE_URL = "https://api.odds-api.io/v3"
API_KEY_ENV = "ODDS_API_IO_KEY"

# Verified by probe 2026-08-02: GET /v3/leagues?sport=football&all=true.
SPORT = "football"
LEAGUE_SLUG = "china-chinese-super-league"

# odds-api.io's own name for the book (its `bookmakers` params take display names,
# not slugs). Confirmed against GET /v3/bookmakers and /v3/bookmakers/selected,
# which reports this account's selection as {"bookmakers": ["1xbet"], "count": 1}.
BOOKMAKER_NAME = "1xbet"

# The key written to the history store. See the module docstring's vocabulary note.
TARGET_BOOKMAKER_KEY = "onexbet"

# Stored in the `regions` column, which is a The-Odds-API billing concept with no
# odds-api.io equivalent. Reusing it as a provenance tag means the history CSV keeps
# its 14-column shape while still recording which provider a row came from — needed
# because rows from both providers coexist for `bookmaker=onexbet` during the cutover.
PROVIDER_TAG = "oddsapiio"

# odds-api.io's market name for 1X2. Stored under the repo's existing "h2h" label so
# the `market` column stays consistent with every pre-existing row.
ML_MARKET = "ML"

# /odds/multi accepts at most 10 event ids per request, and that whole request bills
# as one. This is the single most important number for the quota budget.
MULTI_BATCH_SIZE = 10

# Docs' best-practice guidance for request timeouts.
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3

log = logging.getLogger(__name__)


def get_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {API_KEY_ENV}")
    return api_key


def read_rate_limit(response: requests.Response) -> tuple[int | None, int | None, str | None]:
    """``(limit, remaining, reset)`` from the ``x-ratelimit-*`` headers.

    Semantics mirror ``capture_snapshot.read_quota`` so callers can log a spend guard
    the same way. Note these headers describe the **hourly** window (100/hour on the
    free tier); the ~500/day cap is not exposed by any header, so the daily budget is
    enforced structurally instead — see ``fetch_onexbet_open``'s request cap.
    """
    def _int(name: str) -> int | None:
        raw = response.headers.get(name)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    return _int("x-ratelimit-limit"), _int("x-ratelimit-remaining"), response.headers.get(
        "x-ratelimit-reset"
    )


def _request(api_key: str, path: str, **params: Any) -> requests.Response:
    """GET ``{BASE_URL}{path}`` with the key injected, retrying on 429.

    The docs require honouring ``Retry-After`` on 429 with exponential backoff as the
    fallback. Retries are bounded so a rate-limited run fails loudly rather than
    silently stalling a scheduled workflow.
    """
    url = f"{BASE_URL}{path}"
    query = {k: v for k, v in params.items() if v is not None}
    query["apiKey"] = api_key

    for attempt in range(MAX_RETRIES):
        response = requests.get(url, params=query, timeout=REQUEST_TIMEOUT)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        if attempt == MAX_RETRIES - 1:
            break
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 2.0**attempt
        except ValueError:
            delay = 2.0**attempt
        log.warning("429 from %s; retrying in %.1fs (attempt %d/%d)",
                    path, delay, attempt + 1, MAX_RETRIES)
        time.sleep(delay)

    response.raise_for_status()
    return response


def list_events(api_key: str, *, from_dt: datetime, to_dt: datetime, limit: int = 50) -> list[dict]:
    """Upcoming CSL fixtures in ``[from_dt, to_dt]`` — one request, no odds.

    Returns the raw event dicts (``id``/``home``/``away``/``date``/``status``); team
    names are odds-api.io's own spellings and still need ``normalize_team_name``.
    """
    response = _request(
        api_key, "/events",
        sport=SPORT, league=LEAGUE_SLUG, limit=limit,
        **{"from": _rfc3339(from_dt), "to": _rfc3339(to_dt)},
    )
    limit_, remaining, reset = read_rate_limit(response)
    payload = response.json()
    events = payload if isinstance(payload, list) else []
    log.info("odds-api.io /events: %d CSL fixture(s) | rate limit %s/%s (resets %s)",
             len(events), remaining, limit_, reset)
    return events


def fetch_multi_odds(api_key: str, event_ids: list[str]) -> list[dict]:
    """Odds for up to ``MULTI_BATCH_SIZE`` events in ONE request.

    Raises ``ValueError`` rather than silently truncating: a caller that overfills the
    batch would otherwise lose fixtures without any signal.
    """
    if not event_ids:
        return []
    if len(event_ids) > MULTI_BATCH_SIZE:
        raise ValueError(
            f"/odds/multi accepts at most {MULTI_BATCH_SIZE} event ids, got {len(event_ids)}"
        )
    response = _request(
        api_key, "/odds/multi",
        eventIds=",".join(event_ids), bookmakers=BOOKMAKER_NAME,
    )
    payload = response.json()
    return payload if isinstance(payload, list) else []


def extract_ml(event: dict) -> tuple[float, float, float, str] | None:
    """``(home, draw, away, updated_at)`` from an event's 1xBet ML market, or None.

    Returns None — never raises — when the book has not posted a 1X2 price yet. That
    is the normal "line not open" state and the whole point of polling: the fixture
    simply stays pending until a price appears.
    """
    books = event.get("bookmakers")
    if not isinstance(books, dict):
        return None
    markets = books.get(BOOKMAKER_NAME)
    if not isinstance(markets, list):
        return None

    for market in markets:
        if not isinstance(market, dict) or market.get("name") != ML_MARKET:
            continue
        prices = market.get("odds")
        if not isinstance(prices, list) or not prices:
            continue
        first = prices[0]
        if not isinstance(first, dict):
            continue
        home = _coerce_float(first.get("home"))
        draw = _coerce_float(first.get("draw"))
        away = _coerce_float(first.get("away"))
        if home is None or draw is None or away is None:
            continue
        return home, draw, away, str(market.get("updatedAt") or "")
    return None


def event_to_row(
    event: dict,
    prices: tuple[float, float, float, str],
    mapping: TeamMapping,
    *,
    fetched_at: str,
) -> dict[str, Any] | None:
    """Build one history row (exactly ``OUTPUT_COLUMNS``' 14 keys), or None if unmappable.

    Deliberately returns None instead of raising on an unknown team name — unlike
    ``fetch_pinnacle_spreads.extract_rows``, which raises. This module is a supplementary
    source running alongside the main pipeline; one unrecognised club must not take the
    dashboard publish down with it. The caller logs the skip so the mapping gap is visible.
    """
    api_home = _clean_name(event.get("home"))
    api_away = _clean_name(event.get("away"))
    if not api_home or not api_away:
        return None

    home_team = normalize_team_name(api_home, mapping)
    away_team = normalize_team_name(api_away, mapping)
    if home_team is None or away_team is None:
        return None

    home_odds, draw_odds, away_odds, updated_at = prices
    return {
        # Namespaced so odds-api.io ids can never collide with The Odds API's in the
        # dedup key (event_id, bookmaker, last_update, snapshot_type).
        "event_id": f"{PROVIDER_TAG}:{event.get('id')}",
        "commence_time": event.get("date") or "",
        "api_home_team": api_home,
        "api_away_team": api_away,
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "bookmaker": TARGET_BOOKMAKER_KEY,
        "market": MARKET_LABEL,
        "regions": PROVIDER_TAG,
        "last_update": updated_at,
        "fetched_at": fetched_at,
    }


def _rfc3339(dt: datetime) -> str:
    """odds-api.io rejects bare YYYY-MM-DD dates; every date param must be RFC3339."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
