"""odds-api.io: the prices we actually bet. Metered by the day, and by the hour.

This is where the opening lines come from, so it is the provider whose timing
matters most: the price a signal fires on is whatever this returned the first
time a book posted a market.

Its limits are shaped differently from The Odds API's monthly allowance. There is
a generous daily budget and a tight per-window rate limit, and the window's reset
time comes back on every response. So the right response to a 429 is to park until
reset rather than back off blindly, because a full pass needs more calls than one
window allows and blind retries just burn the daily budget into refusals.

A 403 is never retried. It means an entitlement problem, the plan is limited to a
fixed set of bookmakers, and asking for one outside it fails the whole request.
The authoritative entitlement list is the body of that 403, not any catalogue
endpoint, which goes stale.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from betmodel.providers import http

log = logging.getLogger(__name__)

BASE_URL = "https://api.odds-api.io/v3"
KEY_ENV = "ODDS_API_IO_KEY"

#: Event ids per ``odds/multi`` call. Naming several bookmakers in one call costs
#: the same single request as naming one, which is what makes capturing every bet
#: book on every tick free.
MULTI_BATCH_SIZE = 10

#: Park when the window has this few requests left. Not zero: a concurrent job or
#: a retry would otherwise tip it into a 429 that costs the daily budget.
RATE_FLOOR = 2

ML_MARKET = "ML"


class RateLimited(RuntimeError):
    """The rate window is spent and waiting it out would outlive this tick.

    Its own exception, and deliberately not a hard failure. A tick that cannot
    reach the provider has spent nothing and decided nothing, which is the same
    shape as a quota refusal and is handled the same way: reported, counted, and
    survived, with the other provider left to run.

    What it replaces was a blind escalating backoff -- 60, 120, 180, 240, 300
    seconds -- ending in an unhandled error. On a five-minute tick sharing one
    concurrency group with the closing-line captures, that held the runner for
    fifteen minutes and cancelled everything queued behind it. Three of those on
    2026-09-06 cancelled fifteen ticks. None of them happened to be a close, and
    a close missed cannot be bought back at any price.
    """


class EntitlementError(RuntimeError):
    """A bookmaker outside the plan was requested, so the whole call failed.

    Raised distinctly because it is a configuration mistake, not a transient
    one: retrying spends the daily allowance to be refused identically.
    """


@dataclass(frozen=True)
class Book:
    """A bookmaker as this provider spells it, and as we store it.

    The two differ and the difference bites. The provider's spelling is what goes
    in a request, and a near-miss is refused rather than resolved: "Betano" is a
    403 where "Betano UK" is the entitled book.
    """

    provider_name: str
    key: str


def key_env_for(credential: str) -> str:
    """Environment variable holding the key for a named credential.

    A key is entitled to a fixed, small number of bookmakers, so leagues that bet
    different books cannot share one. Two leagues wanting three distinct books
    between them need two keys, and which key a league uses is therefore part of
    its configuration rather than a global.
    """
    if not credential or credential == "default":
        return KEY_ENV
    return f"{KEY_ENV}_{credential.upper()}"


def api_key(credential: str = "default", *, required: bool = True) -> str:
    env = key_env_for(credential)
    value = os.environ.get(env, "").strip()
    if not value and env != KEY_ENV:
        value = os.environ.get(KEY_ENV, "").strip()
        if value:
            log.debug("%s unset; falling back to %s", env, KEY_ENV)
    if not value and required:
        raise RuntimeError(f"no odds-api.io key: set {env} (or {KEY_ENV})")
    return value


def _rfc3339(moment: datetime) -> str:
    """The endpoint rejects a bare date, so timestamps are always full RFC3339."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class OddsApiIoClient:
    """Rate-limit-aware client for one league's slugs."""

    def __init__(
        self,
        league_slugs: Sequence[str],
        *,
        sport: str = "football",
        base_url: str = BASE_URL,
        credential: str = "default",
        timeout: float = 30.0,
        max_park_seconds: float = 60.0,
    ) -> None:
        if not league_slugs:
            raise ValueError("at least one league slug is required")
        self.league_slugs = tuple(league_slugs)
        self.sport = sport
        self.credential = credential
        self.base_url = base_url.rstrip("/")
        #: Total seconds one `get` may spend parked, across every attempt. A
        #: budget for the whole call rather than a cap per wait: five waits each
        #: inside a per-attempt cap still adds up to longer than the tick that
        #: contains them, which is how a rate limit turned into a blocked queue.
        self.max_park_seconds = max_park_seconds
        # Retries are handled here, not by the policy: a 429 needs the reset
        # header, and a 403 must never be retried at all.
        self._http = http.client("oddsapiio", retry=http.NO_RETRY, timeout=timeout)

    def _seconds_to_reset(self, headers, *, status: int | None = None) -> float | None:
        """Seconds until the rate window resets, or ``None`` if it cannot be read.

        ``None`` and ``0.0`` are different answers and the caller acts on the
        difference: zero means no need to wait, ``None`` means we do not know
        how long waiting would take. Guessing in that case is what produced the
        blind backoff.
        """
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if reset is None:
            return None
        try:
            if remaining is not None and int(remaining) > RATE_FLOOR:
                if status != 429:
                    return 0.0
        except (TypeError, ValueError):
            pass
        try:
            when = datetime.fromisoformat(str(reset).replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds() + 2)

    def _park(self, response) -> float | None:
        """Seconds to wait before this response's window resets."""
        return self._seconds_to_reset(response.headers,
                                      status=response.status_code)

    def get(self, path: str, **params: Any) -> Any:
        """One request, parking only as long as the window actually needs.

        **Nothing here may outlive the tick that contains it.** These run on a
        five-minute dispatch inside a concurrency group shared with the closing-
        line captures, so a call that parks longer than its slot does not merely
        arrive late -- it holds the group and cancels whatever queued behind it,
        which on the wrong tick is a close, and a close missed is unrecoverable.
        So `parked` is a budget for the whole call, and running out of it is a
        `RateLimited` refusal now rather than a longer wait for the same answer.
        """
        params["apiKey"] = api_key(self.credential)
        url = f"{self.base_url}/{path.lstrip('/')}"
        parked = 0.0
        for attempt in range(1, 6):
            try:
                response = self._http.get(url, params=params)
            except http.HttpError as exc:
                if exc.status == 403:
                    # The body names exactly which books this key may request,
                    # e.g. "You're allowed max 2 bookmakers. Allowed: 1xbet, Duel".
                    # It is the only authoritative list, so it goes in the message
                    # rather than being described in the abstract.
                    detail = getattr(exc, "body", "") or ""
                    raise EntitlementError(
                        f"odds-api.io refused {path} for credential "
                        f"{self.credential!r} ({key_env_for(self.credential)}): "
                        f"a requested bookmaker is outside this key's plan. "
                        f"{detail}".strip()
                    ) from exc
                if exc.status == 429:
                    # The reset moment travels on the error now, so this waits
                    # for the window rather than guessing at it. Unreadable is
                    # not a reason to guess: refuse, and let the next tick ask.
                    wait = self._seconds_to_reset(exc.headers, status=429)
                    if wait is None:
                        raise RateLimited(
                            f"odds-api.io rate limited on {path} and sent no "
                            f"readable reset; refusing rather than guessing"
                        ) from exc
                    if parked + wait > self.max_park_seconds:
                        # The allowance goes in the message. A reset an hour out
                        # is not a burst to ride out, it is a window we are over
                        # for the period, and the only way to tell those apart
                        # from a log is the size of the window and what is left
                        # of it.
                        limit = exc.headers.get("x-ratelimit-limit", "?")
                        left = exc.headers.get("x-ratelimit-remaining", "?")
                        raise RateLimited(
                            f"odds-api.io rate limited on {path} for credential "
                            f"{self.credential!r}; window resets in {wait:.0f}s "
                            f"(limit {limit}, remaining {left}), past this "
                            f"call's {self.max_park_seconds:.0f}s park budget"
                        ) from exc
                    parked += wait
                    log.info("odds-api.io rate limited, parking %.0fs for the "
                             "window reset (%.0fs of %.0fs budget used)",
                             wait, parked, self.max_park_seconds)
                    time.sleep(wait)
                    continue
                raise
            wait = self._park(response) or 0.0
            if wait:
                if parked + wait > self.max_park_seconds:
                    # The response is in hand and valid; the park was only to
                    # keep the *next* call off a spent window. Returning it and
                    # letting the next call meet the 429 is strictly better than
                    # sitting on a finished answer past the end of the tick.
                    log.info("odds-api.io window nearly spent and its reset is "
                             "past this call's park budget; returning now")
                    return response.json()
                parked += wait
                log.info("odds-api.io window nearly spent, parking %.0fs", wait)
                time.sleep(wait)
            return response.json()
        raise RateLimited(
            f"odds-api.io rate limited on every attempt at {path}"
        )

    # --- endpoints ----------------------------------------------------------
    def list_events(self, start: datetime, end: datetime) -> list[dict]:
        """Events in ``[start, end)``, trying each league slug in turn.

        Several slugs exist because some competitions split a calendar year into
        two tournaments, and only one of them answers at a given moment.
        """
        for slug in self.league_slugs:
            events = self.get(
                "events",
                sport=self.sport,
                league=slug,
                limit=100,
                **{"from": _rfc3339(start), "to": _rfc3339(end)},
            )
            if events:
                return list(events)
        return []

    def upcoming_events(self, lookahead_days: int) -> list[dict]:
        now = datetime.now(timezone.utc)
        return self.list_events(now, now + timedelta(days=lookahead_days))

    def multi_odds(self, event_ids: Sequence[Any], books: Sequence[Book]) -> list[dict]:
        """Current odds for up to :data:`MULTI_BATCH_SIZE` events, all books at once."""
        ids = [str(i) for i in event_ids]
        if len(ids) > MULTI_BATCH_SIZE:
            raise ValueError(
                f"at most {MULTI_BATCH_SIZE} event ids per call, got {len(ids)}"
            )
        if not ids:
            return []
        payload = self.get(
            "odds/multi",
            eventIds=",".join(ids),
            bookmakers=",".join(b.provider_name for b in books),
        )
        return list(payload or [])

    def iter_batches(self, event_ids: Sequence[Any]) -> Iterable[list[Any]]:
        for i in range(0, len(event_ids), MULTI_BATCH_SIZE):
            yield list(event_ids[i : i + MULTI_BATCH_SIZE])


def extract_ml(quoted: dict, book: Book) -> dict | None:
    """One book's 1X2 prices from an ``odds/multi`` entry, or ``None``.

    ``None`` is the normal state, not an error. The endpoint omits a book, or the
    whole event, until a market is posted, and "absent" is exactly what "has not
    opened yet" looks like. That absence is the signal the opening-line capture
    is built on, and it is common: measured on one round, one book priced six of
    ten upcoming fixtures and the other nine of ten.
    """
    markets = (quoted.get("bookmakers") or {}).get(book.provider_name) or []
    for market in markets:
        if market.get("name") != ML_MARKET:
            continue
        prices = market.get("odds") or []
        if not prices:
            continue
        first = prices[0]
        if not all(first.get(side) for side in ("home", "draw", "away")):
            continue
        return {
            "home_odds": first["home"],
            "draw_odds": first["draw"],
            "away_odds": first["away"],
            # The book's own timestamp for the market. Note that at least one
            # book stamps every fixture with a shared bulk feed-refresh time, so
            # this moves whether or not the price did; it is provenance, not
            # evidence of a price change.
            "last_update": market.get("updatedAt", ""),
        }
    return None


def books_from_config(book_configs: Iterable[Any]) -> tuple[Book, ...]:
    """Build the request-time book list from league config, order preserved.

    Order is load-bearing downstream: it decides published column order and
    breaks ties when two books quote the same best price.
    """
    return tuple(
        Book(provider_name=b.provider_name or b.key, key=b.key)
        for b in book_configs
        if b.provider == "oddsapiio"
    )
