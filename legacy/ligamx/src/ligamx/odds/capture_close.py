"""Capture closing 1X2 lines from The Odds API in the minutes before kickoff.

The close is the benchmark every CLV claim in this project is graded against, and
since football-data.co.uk stopped publishing PSC* for Liga MX after Oct-2025 there
has been no live source for it at all.

Unlike the opener, this has a hard deadline. A fixture leaves the pre-match feed
at kickoff, so a missed close is unrecoverable -- there is no backfill, at any
price. That asymmetry is why this runs on a 5-minute tick while the opener runs
on 15.

The window, and why it is 20 minutes
------------------------------------
cslmonitor opens its close window at KO-60m because its trigger was unreliable
(a requested */10 cron landed at p90 = 137 min). With a Cloudflare Worker firing
repository_dispatch every 5 minutes the window can be tight, and it has to be:

    window   ticks/fixture @5min   credits/month @39 fixtures
    60 min          12                    ~468   -- blows the 500/mo free cap
    20 min           4                    ~156   -- leaves headroom

A fixture goes FINAL -- and stops spending -- as soon as a Pinnacle close lands
inside CLOSE_TARGET_MINUTES of kickoff. Re-capturing until then keeps the latest
price rather than the first one.

Cost is per tick, not per fixture: one /odds call covers the whole slate, and The
Odds API bills markets x regions with each 10 bookmakers counting as one region.
So naming Betfair and Matchbook alongside Pinnacle is free, and worth doing --
betfair_close_* has been an empty column since the schema was written.

Usage (repo root, PYTHONPATH=src, THE_ODDS_API_KEY set):
    python -m ligamx.odds.capture_close
    python -m ligamx.odds.capture_close --dry-run   # decide only
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from ligamx import config
from ligamx.odds.capture_store import append_snapshots, load_history
from ligamx.odds.fetch_oddsapiio_opens import Fixture, _norm, load_upcoming

# How long before kickoff the window opens.
DEFAULT_WINDOW_MINUTES = 20.0

# Once a Pinnacle close is captured this close to kickoff, the fixture is done and
# stops spending quota. Wider than the 5-minute tick so a single landed tick
# finalises it; tighter than the window so we keep re-capturing until we get there.
CLOSE_TARGET_MINUTES = 10.0

# Abort before spending if the monthly allowance is nearly gone. Leaves enough for
# a full round's closes (~36) plus slack.
DEFAULT_MIN_REMAINING = 50

# Books to capture. Beyond Pinnacle these are free (10 bookmakers = 1 region), and
# the exchange closes are a second, lower-vig benchmark.
CLOSE_BOOKMAKERS = ("pinnacle", "betfair_ex_eu", "matchbook")

# The book whose close decides whether a fixture is finished. The exchanges are
# opportunistic extras; Pinnacle is the benchmark, so it alone gates the spend.
ANCHOR_BOOKMAKER = "pinnacle"

# Provenance tag written to the history's `regions` column, mirroring
# oddsapi_io.PROVIDER_TAG. The column has no meaning for either provider; this
# keeps "which API produced this row" answerable without a join.
PROVIDER_TAG = "theoddsapi"

log = logging.getLogger(__name__)


def read_quota(api_key: str) -> tuple[int | None, bool]:
    """(requests remaining, is Liga MX live) from the FREE /v4/sports endpoint.

    This call does not count against the allowance, so the guard costs nothing.
    """
    resp = requests.get(f"{config.THE_ODDS_API_BASE_URL}/sports",
                        params={"apiKey": api_key}, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    live = any(s.get("key") == config.ODDS_SPORT_KEY and s.get("active")
               for s in resp.json())
    try:
        return int(float(remaining)), live
    except (TypeError, ValueError):
        return None, live


def finalised_fixtures(history_path: str | None = None) -> set[tuple[str, str]]:
    """Fixtures whose Pinnacle close already landed inside the target band.

    Keyed on (home, away) rather than event id because the id is the provider's,
    and the point is to stop spending on the *fixture* however it is identified.
    """
    history = load_history(history_path)
    if history.empty:
        return set()
    closes = history[(history["snapshot_type"] == "close")
                     & (history["bookmaker"] == ANCHOR_BOOKMAKER)]
    done = set()
    for row in closes.itertuples(index=False):
        kickoff = pd.to_datetime(row.commence_time, utc=True, errors="coerce")
        fetched = pd.to_datetime(row.fetched_at, utc=True, errors="coerce")
        if pd.isna(kickoff) or pd.isna(fetched):
            continue
        if 0 <= (kickoff - fetched).total_seconds() <= CLOSE_TARGET_MINUTES * 60:
            done.add((_norm(row.home_team), _norm(row.away_team)))
    return done


def fixtures_in_window(now: datetime, *, target_path: str | None = None,
                       history_path: str | None = None,
                       window_minutes: float = DEFAULT_WINDOW_MINUTES) -> list[Fixture]:
    """Fixtures inside [kickoff - window, kickoff) that are not already finalised."""
    done = finalised_fixtures(history_path)
    window = timedelta(minutes=window_minutes)
    return sorted(
        (f for f in load_upcoming(target_path)
         if f.kickoff - window <= now < f.kickoff and f.key not in done),
        key=lambda f: f.kickoff)


def fetch_odds(api_key: str, now: datetime) -> tuple[list[dict], int | None]:
    """The current slate for CLOSE_BOOKMAKERS. Returns (events, credits remaining).

    `bookmakers` is passed without `regions` so the billed cost is unambiguous:
    one region, one credit, however many books are named.
    """
    resp = requests.get(
        f"{config.THE_ODDS_API_BASE_URL}/sports/{config.ODDS_SPORT_KEY}/odds",
        params={
            "apiKey": api_key,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": ",".join(CLOSE_BOOKMAKERS),
            # Pre-match only: an in-play price is not a close.
            "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, timeout=30)
    resp.raise_for_status()
    log.info("Spent %s credit(s); %s remaining",
             resp.headers.get("x-requests-last"), resp.headers.get("x-requests-remaining"))
    try:
        remaining = int(float(resp.headers.get("x-requests-remaining")))
    except (TypeError, ValueError):
        remaining = None
    return list(resp.json()), remaining


def extract_rows(events: list[dict], wanted: set[tuple[str, str]],
                 fetched_at: str) -> list[dict]:
    """History rows for the fixtures we are actually in-window for.

    Everything else in the response is discarded: the call returns the whole slate,
    but storing a fixture's price hours before kickoff as its `close` is exactly
    the mistake that once put 13 lines captured a median 35h out into
    pinnacle_close_* (105.5% overround against 103.4% for real closes).
    """
    rows = []
    for event in events:
        api_home, api_away = event.get("home_team", ""), event.get("away_team", "")
        home = config.ODDS_TO_STANDARD.get(api_home)
        away = config.ODDS_TO_STANDARD.get(api_away)
        if home is None or away is None:
            log.warning("unmapped Odds API team in %r vs %r; add it to "
                        "ligamx_team_name_mapping.csv (fixture skipped)", api_home, api_away)
            continue
        if (_norm(home), _norm(away)) not in wanted:
            continue
        for book in event.get("bookmakers", []):
            markets = book.get("markets") or []
            market = next((m for m in markets if m.get("key") == "h2h"), None)
            if market is None:
                continue
            prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
            odds = (prices.get(api_home), prices.get("Draw"), prices.get(api_away))
            if not all(odds):  # a two-way or partial book is not a 1X2 close
                continue
            rows.append({
                "event_id": event.get("id", ""),
                "commence_time": event.get("commence_time", ""),
                "api_home_team": api_home, "api_away_team": api_away,
                "home_team": home, "away_team": away,
                "home_odds": odds[0], "draw_odds": odds[1], "away_odds": odds[2],
                "bookmaker": book.get("key", ""), "market": "h2h",
                "regions": PROVIDER_TAG,
                "last_update": market.get("last_update") or book.get("last_update", ""),
                "fetched_at": fetched_at,
            })
    return rows


def run(*, now: datetime | None = None, target_path: str | None = None,
        history_path: str | None = None,
        window_minutes: float = DEFAULT_WINDOW_MINUTES,
        min_remaining: int = DEFAULT_MIN_REMAINING, dry_run: bool = False) -> int:
    """One close-capture pass. Returns the number of rows appended."""
    now = now or datetime.now(timezone.utc)

    in_window = fixtures_in_window(now, target_path=target_path,
                                   history_path=history_path,
                                   window_minutes=window_minutes)
    if not in_window:
        log.info("Tick %s: no fixture inside its %.0f-minute close window; "
                 "spending nothing.", now.isoformat(), window_minutes)
        return 0

    log.info("Tick %s: %d fixture(s) in the close window: %s", now.isoformat(),
             len(in_window),
             ", ".join(f"{f.label} (KO in {(f.kickoff - now).total_seconds() / 60:.0f}m)"
                       for f in in_window))

    if dry_run:
        log.info("Dry run: would spend 1 credit covering the whole slate "
                 "(%s); writing nothing.", ", ".join(CLOSE_BOOKMAKERS))
        return 0

    api_key = config.THE_ODDS_API_KEY
    if not api_key:
        raise RuntimeError("no API key: set THE_ODDS_API_KEY in .env or the environment")

    remaining, live = read_quota(api_key)
    if remaining is not None and remaining < min_remaining:
        log.error("Only %d request(s) left this month (floor %d); skipping the "
                  "capture rather than exhausting the allowance.", remaining, min_remaining)
        return 0
    if not live:
        log.warning("%s is not listed as active; the slate may be empty.",
                    config.ODDS_SPORT_KEY)

    events, _ = fetch_odds(api_key, now)
    fetched_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows = extract_rows(events, {f.key for f in in_window}, fetched_at)
    if not rows:
        log.info("No 1X2 prices for the in-window fixture(s); nothing appended.")
        return 0

    log.info("Closing line(s): %s",
             ", ".join(f"{r['home_team']} v {r['away_team']} [{r['bookmaker']}] "
                       f"({r['home_odds']}/{r['draw_odds']}/{r['away_odds']})" for r in rows))

    rounds = sorted({f.round for f in in_window} - {""})
    _, appended = append_snapshots(
        pd.DataFrame(rows), snapshot_type="close", target_round=",".join(rounds),
        capture_reason=f"pre-kickoff close tick @ {now.isoformat()}",
        path=history_path)
    return appended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture closing 1X2 lines in the minutes before kickoff.")
    parser.add_argument("--target", default=None, help="Upcoming fixtures CSV")
    parser.add_argument("--out", default=None, help="Capture history CSV path")
    parser.add_argument("--window-minutes", type=float, default=DEFAULT_WINDOW_MINUTES,
                        help="How long before kickoff the window opens "
                             "(default: %(default)s). Widening it multiplies the "
                             "monthly credit burn -- see the module docstring.")
    parser.add_argument("--min-remaining", type=int, default=DEFAULT_MIN_REMAINING,
                        help="Abort if fewer credits than this remain (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report in-window fixtures; spend and write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    try:
        run(target_path=args.target, history_path=args.out,
            window_minutes=args.window_minutes, min_remaining=args.min_remaining,
            dry_run=args.dry_run)
    except requests.RequestException as exc:
        log.error("The Odds API request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
