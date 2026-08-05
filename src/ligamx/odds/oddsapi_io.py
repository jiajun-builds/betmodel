"""Historical closing lines from odds-api.io.

Why this exists: football-data.co.uk stopped publishing Pinnacle closing 1X2
(PSC*) for Liga MX after Oct-2025, which left pinnacle_close_* empty from
2025-11 onward -- exactly the window the Polymarket CLV work runs on. This
module backfills a closing benchmark over that hole from odds-api.io.

**odds-api.io does not carry Pinnacle.** Its bookmaker list (273 books) has no
Pinnacle or PS3838 entry at any plan tier, so this cannot restore the
pinnacle_close_* column; it supplies a *different* book's close. On the free
plan the constraints are, measured:

  - sharps / exchanges / prediction markets (Betfair Exchange, Sbobet,
    Marathonbet, BetOnline, LowVig, Polymarket, Kalshi) are paid-only
  - at most 2 recreational bookmakers may ever be selected on the account
  - the bulk /historical/closing-lines endpoint is paid-only, so events are
    fetched one at a time
  - Liga MX history starts at 2026-01 (196 settled events through 2026-08)
  - 100 requests per rate-limit window, reported in x-ratelimit-* headers

/v3/historical/odds returns ONE snapshot per market, stamped `updatedAt`, which
lands ~1 minute before kickoff -- i.e. a true close, not a price series. There
is no opening-odds history here; openers can only be forward-captured.

Rows are cached long-format so a market's shape is preserved: ML fills
home/draw/away with an empty hdp, Spread fills hdp + home/away. Totals and BTTS
come back on the same request but have no home for them in the schema, so they
are dropped rather than stored half-used.

    python -m ligamx.odds.oddsapi_io --list                 # coverage report
    python -m ligamx.odds.oddsapi_io --fetch                # pull + cache
    python -m ligamx.odds.oddsapi_io --fetch --book Bet365  # a second book
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests

from ligamx import config, paths
from ligamx.date_utils import parse_date_only_series

BASE = "https://api.odds-api.io/v3"

# Both halves of the Liga MX season are separate league slugs upstream.
LEAGUES = ("mexico-liga-mx-clausura", "mexico-liga-mx-apertura")

# The event endpoints cap a query at 31 days, so walk the range in 30-day steps.
WINDOW_DAYS = 30

# Markets we have a schema home for. Totals/BTTS are returned by the same call
# but deliberately not stored.
MARKETS = ("ML", "Spread")

COLUMNS = [
    "event_id", "kickoff", "league", "home", "away", "hg", "ag",
    "bookmaker", "market", "hdp", "home_odds", "draw_odds", "away_odds",
    "updated_at", "lead_h",
]

# odds-api.io spelling -> MEX_ligamx.csv spelling. Complete for all 19 Liga MX
# sides seen across the 196 available events; a name outside this map is a hard
# error rather than a silent drop.
API_TO_STANDARD = {
    "Atlante FC": "Atlante",
    "Atlas FC": "Atlas",
    "Atletico San Luis": "Atletico San Luis",
    "CD Guadalajara": "Guadalajara",
    "CF America": "Club America",
    "CF Cruz Azul": "Cruz Azul",
    "CF Monterrey": "Monterrey",
    "CF Pachuca": "Pachuca",
    "Club Leon": "Leon",
    "Club Necaxa": "Necaxa",
    "Club Puebla": "Puebla",
    "Club Santos Laguna": "Santos Laguna",
    "Club Tijuana de Caliente": "Tijuana",
    "Deportivo Toluca FC": "Toluca",
    "FC Juarez": "FC Juarez",
    "Mazatlan FC": "Mazatlan",
    "Pumas UNAM": "UNAM Pumas",
    "Queretaro FC": "Queretaro",
    "Tigres UANL": "Tigres UANL",
}

# Keep one rate-limit slot in hand so a concurrent call doesn't 429 the run.
RATE_FLOOR = 2


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.getenv("ODDS_API_IO_KEY", "")
    if not key:
        raise SystemExit(
            "no API key: set ODDS_API_IO_KEY in .env or pass --api-key")
    return key


class _Client:
    """requests session that respects the x-ratelimit-* headers.

    The free plan allows 100 requests per window and reports the window's reset
    time on every response, so the run parks itself until reset rather than
    burning attempts into 429s -- a full backfill needs ~200 calls and so
    always spans more than one window.
    """

    def __init__(self, api_key: str):
        self.key = api_key
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.HTTP_USER_AGENT

    def get(self, path: str, **params):
        params["apiKey"] = self.key
        for attempt in range(5):
            resp = self.session.get(f"{BASE}/{path}", params=params, timeout=30)
            self._respect_limit(resp)
            if resp.status_code == 429:
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"rate limited five times on {path}")

    def _respect_limit(self, resp) -> None:
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if remaining is None or reset is None:
            return
        if int(remaining) > RATE_FLOOR and resp.status_code != 429:
            return
        wait = (pd.Timestamp(reset) - pd.Timestamp.now(tz="UTC")).total_seconds()
        if wait > 0:
            print(f"  rate limit reached, sleeping {wait:.0f}s until {reset}")
            time.sleep(wait + 2)


def list_events(client: _Client, start: str, end: str) -> pd.DataFrame:
    """Every settled Liga MX event upstream between ``start`` and ``end``.

    Both season slugs are walked in 30-day windows (the endpoint caps a query at
    31 days) and de-duplicated on event id.
    """
    seen: dict[int, dict] = {}
    for league in LEAGUES:
        cur, stop = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        while cur < stop:
            nxt = min(cur + pd.Timedelta(days=WINDOW_DAYS), stop)
            events = client.get(
                "historical/events", sport="football", league=league,
                **{"from": _iso(cur), "to": _iso(nxt)})
            for ev in events or []:
                seen[ev["id"]] = dict(ev, league=league)
            cur = nxt
    rows = []
    for ev in seen.values():
        scores = ev.get("scores") or {}
        rows.append({
            "event_id": ev["id"],
            "kickoff": ev["date"],
            "league": ev["league"],
            "home_api": ev["home"],
            "away_api": ev["away"],
            "hg": scores.get("home"),
            "ag": scores.get("away"),
            "status": ev.get("status"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("kickoff").reset_index(drop=True) if len(df) else df


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _standard(name: str) -> str:
    try:
        return API_TO_STANDARD[name]
    except KeyError as exc:
        raise KeyError(
            f"unmapped odds-api.io team {name!r}; add it to API_TO_STANDARD") from exc


def event_rows(client: _Client, event: pd.Series, book: str) -> list[dict]:
    """Closing ML + Spread rows for one event, or [] if the book didn't price it."""
    payload = client.get("historical/odds", eventId=int(event["event_id"]),
                         bookmakers=book)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    books = (payload or {}).get("bookmakers") or {}

    kickoff = pd.Timestamp(event["kickoff"])
    out = []
    for name, markets in books.items():
        for market in markets:
            if market["name"] not in MARKETS:
                continue
            updated = market.get("updatedAt")
            lead = ((kickoff - pd.Timestamp(updated)).total_seconds() / 3600.0
                    if updated else None)
            for odds in market["odds"]:
                out.append({
                    "event_id": event["event_id"],
                    "kickoff": event["kickoff"],
                    "league": event["league"],
                    "home": _standard(event["home_api"]),
                    "away": _standard(event["away_api"]),
                    "hg": event["hg"],
                    "ag": event["ag"],
                    "bookmaker": name,
                    "market": market["name"],
                    "hdp": odds.get("hdp"),
                    "home_odds": odds.get("home"),
                    "draw_odds": odds.get("draw"),
                    "away_odds": odds.get("away"),
                    "updated_at": updated,
                    "lead_h": lead,
                })
    return out


def load_cache() -> pd.DataFrame:
    path = paths.oddsapi_io_closes_csv()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def fetch(book: str, start: str, end: str, api_key: str | None = None,
          refresh: bool = False) -> pd.DataFrame:
    """Pull and cache closing lines for ``book``, resuming from the cache.

    Events already cached for this bookmaker are skipped unless ``refresh``, so
    an interrupted run (or a rate-limit window boundary) costs nothing to
    restart. The cache is rewritten after every 20 events for the same reason.
    """
    client = _Client(_api_key(api_key))
    events = list_events(client, start, end)
    if not len(events):
        print("no events upstream in that range")
        return load_cache()
    print(f"{len(events)} settled events upstream  "
          f"({events.kickoff.min()[:10]} -> {events.kickoff.max()[:10]})")

    cache = load_cache()
    done: set[int] = set()
    if len(cache) and not refresh:
        done = set(cache.loc[cache.bookmaker.str.lower() == book.lower(),
                             "event_id"].astype(int))
        print(f"cache already holds {len(done)} events for {book}")

    todo = events[~events.event_id.astype(int).isin(done)]
    print(f"fetching {len(todo)}")

    fresh: list[dict] = []
    missing = 0
    for n, (_, ev) in enumerate(todo.iterrows(), 1):
        try:
            rows = event_rows(client, ev, book)
        except Exception as exc:  # one bad event must not discard the run
            print(f"  ! {ev.event_id} {ev.home_api} v {ev.away_api}: {exc}")
            continue
        if not rows:
            missing += 1
        fresh += rows
        if n % 20 == 0:
            _checkpoint(cache, fresh, book, refresh)
            print(f"  {n}/{len(todo)} events, {len(fresh)} rows, {missing} unpriced")

    out = _checkpoint(cache, fresh, book, refresh)
    print(f"done: {len(fresh)} new rows, {missing} events {book} did not price")
    return out


def _checkpoint(cache: pd.DataFrame, fresh: list[dict], book: str,
                refresh: bool) -> pd.DataFrame:
    """Merge fresh rows into the cache and write it out."""
    base = cache
    if refresh and len(base):
        base = base[base.bookmaker.str.lower() != book.lower()]
    out = pd.concat([base, pd.DataFrame(fresh, columns=COLUMNS)],
                    ignore_index=True) if fresh else base
    out = out.drop_duplicates(
        subset=["event_id", "bookmaker", "market", "hdp"], keep="last")
    out.to_csv(paths.oddsapi_io_closes_csv(), index=False)
    return out


def _match_day(kickoff) -> pd.Series:
    """Local Mexican calendar day of a UTC kickoff.

    MEX_ligamx.csv keys matches by their local match day, and Liga MX kicks off
    late enough that most fixtures fall on the *next* UTC date -- a 19:06 local
    kickoff is 01:06Z. Normalising in UTC therefore shifts a match forward a day
    and can push it past the join's +/-1 day tolerance.
    """
    return (pd.to_datetime(kickoff, utc=True)
            .dt.tz_convert("America/Mexico_City").dt.tz_localize(None).dt.normalize())


def close_1x2(book: str | None = None) -> pd.DataFrame:
    """Cached ML closes as one row per (event, book): Date/Home/Away + decimals."""
    cache = load_cache()
    if not len(cache):
        return pd.DataFrame(columns=["Date", "Home", "Away", "bookmaker"])
    ml = cache[cache.market == "ML"].copy()
    if book:
        ml = ml[ml.bookmaker.str.lower() == book.lower()]
    ml = ml.rename(columns={"home": "Home", "away": "Away"})
    ml["Date"] = _match_day(ml.kickoff)
    for c in ("home_odds", "draw_odds", "away_odds"):
        ml[c] = pd.to_numeric(ml[c], errors="coerce")
    ml["overround"] = (1 / ml.home_odds + 1 / ml.draw_odds + 1 / ml.away_odds) * 100
    return ml[["Date", "Home", "Away", "bookmaker", "home_odds", "draw_odds",
               "away_odds", "overround", "lead_h", "hg", "ag", "event_id"]]


def close_ah(book: str | None = None) -> pd.DataFrame:
    """Cached Spread closes reduced to one main Asian-handicap line per event.

    The tape carries the whole ladder; the schema holds a single line. "Main" is
    taken as the rung whose two prices sit closest together, which is where the
    book has centred the match -- the same convention the pinnacle_*_ah columns
    already follow.
    """
    cache = load_cache()
    if not len(cache):
        return pd.DataFrame(columns=["Date", "Home", "Away", "bookmaker"])
    sp = cache[cache.market == "Spread"].copy()
    if book:
        sp = sp[sp.bookmaker.str.lower() == book.lower()]
    sp = sp.rename(columns={"home": "Home", "away": "Away"})
    for c in ("hdp", "home_odds", "away_odds"):
        sp[c] = pd.to_numeric(sp[c], errors="coerce")
    sp = sp.dropna(subset=["hdp", "home_odds", "away_odds"])
    sp["balance"] = (sp.home_odds - sp.away_odds).abs()
    sp = sp.sort_values("balance").groupby(["event_id", "bookmaker"], as_index=False).first()
    sp["Date"] = _match_day(sp.kickoff)
    return sp[["Date", "Home", "Away", "bookmaker", "hdp", "home_odds",
               "away_odds", "event_id"]]


def merge_into_history(book: str = "1xbet", prefix: str = "onexbet",
                       dry_run: bool = True) -> pd.DataFrame:
    """Write ``book``'s cached closes into MEX_ligamx.csv as ``prefix``_close_* columns.

    Joins on (Date, Home, Away) with a +/-1 day tolerance, the same tolerance
    eval/football_data.py uses, because providers disagree on which calendar day
    a late kickoff belongs to. Only blank cells are filled -- an existing value is
    never overwritten -- so re-running is safe.

    Returns the merged frame; writes it only when ``dry_run`` is False.
    """
    ml, ah = close_1x2(book), close_ah(book)
    if not len(ml):
        raise SystemExit(f"no cached closes for {book} -- run --fetch first")

    mx = pd.read_csv(paths.ligamx_data_csv())
    dates = parse_date_only_series(mx["Date"])

    ml_idx = {(r.Date, r.Home, r.Away): r for r in ml.itertuples()}
    ah_idx = {(r.Date, r.Home, r.Away): r for r in ah.itertuples()}

    cols = {f"{prefix}_close_h": [], f"{prefix}_close_d": [], f"{prefix}_close_a": [],
            f"{prefix}_close_ah": [], f"{prefix}_close_ah_h": [], f"{prefix}_close_ah_a": []}
    hits = 0
    for date, home, away in zip(dates, mx["Home"], mx["Away"]):
        m = a = None
        for off in (0, 1, -1):
            key = (date + pd.Timedelta(days=off), home, away)
            if m is None and key in ml_idx:
                m, a = ml_idx[key], ah_idx.get(key)
                break
        hits += m is not None
        cols[f"{prefix}_close_h"].append(m.home_odds if m else None)
        cols[f"{prefix}_close_d"].append(m.draw_odds if m else None)
        cols[f"{prefix}_close_a"].append(m.away_odds if m else None)
        cols[f"{prefix}_close_ah"].append(a.hdp if a else None)
        cols[f"{prefix}_close_ah_h"].append(a.home_odds if a else None)
        cols[f"{prefix}_close_ah_a"].append(a.away_odds if a else None)

    for name, values in cols.items():
        fresh = pd.Series(values, index=mx.index)
        mx[name] = fresh if name not in mx.columns else mx[name].combine_first(fresh)

    print(f"{book}: {len(ml)} cached closes -> matched {hits} of {len(mx)} history rows "
          f"({len(ml) - hits} cached closes had no match)")
    if dry_run:
        print("dry run -- pass --write to save")
    else:
        mx.to_csv(paths.ligamx_data_csv(), index=False)
        print(f"wrote {paths.ligamx_data_csv()}")
    return mx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", default="1xbet", help="bookmaker name (see /v3/bookmakers)")
    ap.add_argument("--start", default="2025-12-01", help="UTC date, inclusive")
    ap.add_argument("--end", default=None, help="UTC date, exclusive (default: tomorrow)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--fetch", action="store_true", help="pull and cache")
    ap.add_argument("--refresh", action="store_true", help="re-pull cached events")
    ap.add_argument("--list", action="store_true", help="report cache coverage only")
    ap.add_argument("--merge", action="store_true",
                    help="join cached closes into MEX_ligamx.csv (dry run by default)")
    ap.add_argument("--prefix", default="onexbet", help="column prefix for --merge")
    ap.add_argument("--write", action="store_true", help="let --merge save its result")
    args = ap.parse_args()

    end = args.end or (pd.Timestamp.now("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if args.fetch:
        fetch(args.book, args.start, end, args.api_key, args.refresh)

    if args.merge:
        merge_into_history(args.book, args.prefix, dry_run=not args.write)

    ml = close_1x2()
    if not len(ml):
        print("cache empty -- run with --fetch")
        return
    print("\ncached ML closes by bookmaker:")
    for book, g in ml.groupby("bookmaker"):
        print(f"  {book:<14} n={len(g):<4} overround {g.overround.mean():.2f}%  "
              f"median lead {g.lead_h.median():+.2f}h  "
              f"{g.Date.min().date()} -> {g.Date.max().date()}")


if __name__ == "__main__":
    main()
