"""
Forward odds poller (manual CLI). One invocation = one snapshot of the current
upcoming Liga MX slate, appended to the append-only snapshot store.

Sources:
  - The Odds API  : pinnacle, betfair_ex_eu, matchbook (h2h / 1X2).
  - Polymarket    : per-match event (3 linked Yes/No markets) via the public
                    Gamma catalogue API; Yes price = implied prob, so
                    decimal odds = 1 / yes_price. Discovered by searching the
                    Odds-API fixture's team names.

    python -m ligamx.odds.capture_odds [--no-oddsapi] [--no-polymarket]

Scheduling is intentionally external — run it by hand or from cron/CI later.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime, timezone

import requests

from ligamx import config, paths
from ligamx.odds import snapshot_store


# --- helpers ----------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fold(s: str) -> set[str]:
    """Lowercased, accent-stripped alnum tokens (for fuzzy team matching)."""
    if not s:
        return set()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = "".join(c if c.isalnum() else " " for c in s.lower())
    stop = {"fc", "cf", "cd", "club", "deportivo", "de", "la", "el", "los"}
    return {t for t in s.split() if t and t not in stop}


def _search_query(home: str, away: str) -> str:
    """Polymarket search query from meaningful tokens only.

    Noise words like 'Club'/'CF'/'FC' in a standard name (e.g. 'Club America')
    hurt Gamma's search ranking and can bury the per-match event under futures
    markets, so search on the folded tokens instead.
    """
    return " ".join(sorted(_fold(home)) + sorted(_fold(away)))


def _inv(price) -> float | str:
    """Decimal odds from an implied probability (Polymarket) or passthrough."""
    try:
        p = float(price)
        return round(1.0 / p, 4) if p > 0 else ""
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


# --- The Odds API -----------------------------------------------------------

def fetch_oddsapi(captured_at: str) -> tuple[list[dict], dict]:
    """Snapshot the configured books' 1X2 lines. Returns (rows, quota headers)."""
    url = f"{config.THE_ODDS_API_BASE_URL}/sports/{config.ODDS_SPORT_KEY}/odds"
    params = {
        "apiKey": config.THE_ODDS_API_KEY,
        "regions": config.ODDS_API_REGIONS,
        "markets": "h2h",
        "oddsFormat": "decimal",
        "bookmakers": ",".join(config.ODDS_API_BOOKMAKERS),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    quota = {
        "remaining": r.headers.get("x-requests-remaining"),
        "used": r.headers.get("x-requests-used"),
        "last_cost": r.headers.get("x-requests-last"),
    }
    rows = []
    for ev in r.json():
        home, away = ev.get("home_team"), ev.get("away_team")
        for bm in ev.get("bookmakers", []):
            mkts = bm.get("markets", [])
            if not mkts:
                continue
            odds = {"home": None, "draw": None, "away": None}
            for o in mkts[0].get("outcomes", []):
                if o.get("name") == home:
                    odds["home"] = o.get("price")
                elif o.get("name") == away:
                    odds["away"] = o.get("price")
                elif o.get("name") == "Draw":
                    odds["draw"] = o.get("price")
            if not all(odds.values()):
                continue
            rows.append({
                "captured_at": captured_at, "source": "oddsapi", "venue": bm.get("key"),
                "event_id": ev.get("id"), "commence_time": ev.get("commence_time"),
                "home_team": home, "away_team": away,
                "home_odds": odds["home"], "draw_odds": odds["draw"], "away_odds": odds["away"],
                "liquidity": "", "last_update": bm.get("last_update", ""),
            })
    return rows, quota


# --- Polymarket -------------------------------------------------------------

def _gamma(path: str, params: dict) -> object:
    r = requests.get(
        f"{config.POLYMARKET_GAMMA_BASE}{path}", params=params,
        headers={"User-Agent": config.HTTP_USER_AGENT, "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _parse_iso(s: str):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def _match_polymarket_event(events: list, home: str, away: str, commence: str,
                            window_days: int = 2, allow_closed: bool = False) -> dict | None:
    """Pick the event with both teams whose date is closest to kickoff.

    Providers can disagree on the exact datetime by up to a day this far out, so
    match on team tokens within a +/- window and take the nearest, rather than an
    exact same-day match. By default closed events are excluded (forward capture);
    backfill passes allow_closed=True to reach settled past meetings, and the
    nearest-date rule then disambiguates repeat fixtures across the season.
    """
    ht, at = _fold(home), _fold(away)
    kickoff = _parse_iso(commence)
    best, best_diff = None, None
    for ev in events:
        if ev.get("closed") and not allow_closed:
            continue
        if "more markets" in (ev.get("title", "")).lower():
            continue  # props sibling of the moneyline event; skip so 1X2 is picked
        tt = _fold(ev.get("title", ""))
        if len(ht & tt) < 1 or len(at & tt) < 1:  # both teams must appear
            continue
        end = _parse_iso(ev.get("endDate", ""))
        if kickoff and end:
            diff = abs((end - kickoff).total_seconds())
            if diff > window_days * 86400:
                continue
        else:
            diff = float("inf")
        if best_diff is None or diff < best_diff:
            best, best_diff = ev, diff
    return best


def _parse_polymarket_markets(ev: dict, home: str, away: str) -> dict | None:
    """Map the event's 3 Yes/No sub-markets to home/draw/away Yes-prices."""
    ht, at = _fold(home), _fold(away)
    yes = {"home": None, "draw": None, "away": None}
    for m in ev.get("markets", []):
        title = f"{m.get('groupItemTitle','')} {m.get('question','')}"
        toks = _fold(title)
        try:
            price = json.loads(m.get("outcomePrices") or "[]")[0]
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
        if "draw" in (m.get("groupItemTitle", "") + m.get("question", "")).lower():
            yes["draw"] = price
        elif len(ht & toks) > len(at & toks):
            yes["home"] = price
        elif len(at & toks) > len(ht & toks):
            yes["away"] = price
    if not all(yes.values()):
        return None
    return yes


def fetch_polymarket(fixtures: list[dict], captured_at: str) -> list[dict]:
    """Snapshot Polymarket per-match 1X2 for each given fixture (home/away/commence)."""
    rows, seen = [], set()
    for fx in fixtures:
        home, away, commence = fx["home_team"], fx["away_team"], fx["commence_time"]
        key = (home, away, commence)
        if key in seen:
            continue
        seen.add(key)
        try:
            res = _gamma("/public-search", {"q": _search_query(home, away), "limit_per_type": 8})
            ev = _match_polymarket_event(res.get("events", []), home, away, commence)
            if not ev:
                continue
            yes = _parse_polymarket_markets(ev, home, away)
            if not yes:
                continue
            rows.append({
                "captured_at": captured_at, "source": "polymarket", "venue": "polymarket",
                "event_id": ev.get("slug"), "commence_time": commence,
                "home_team": home, "away_team": away,
                "home_odds": _inv(yes["home"]), "draw_odds": _inv(yes["draw"]),
                "away_odds": _inv(yes["away"]), "liquidity": ev.get("liquidity", ""),
                "last_update": "",
            })
        except requests.RequestException as e:
            print(f"  polymarket lookup failed for {home} vs {away}: {e}")
    return rows


# --- orchestration ----------------------------------------------------------

def capture(use_oddsapi: bool = True, use_polymarket: bool = True) -> dict:
    captured_at = _now_iso()
    rows: list[dict] = []
    quota = {}

    if use_oddsapi:
        oa_rows, quota = fetch_oddsapi(captured_at)
        rows += oa_rows
        print(f"Odds API: {len(oa_rows)} rows "
              f"({len({r['event_id'] for r in oa_rows})} events x "
              f"{len({r['venue'] for r in oa_rows})} books) | "
              f"quota used={quota.get('used')} remaining={quota.get('remaining')} "
              f"cost={quota.get('last_cost')}")
        fixtures = list({(r["home_team"], r["away_team"], r["commence_time"]): r
                         for r in oa_rows}.values())
    else:
        fixtures = []

    if use_polymarket:
        if not fixtures:  # no Odds-API slate to seed from; derive from the store's latest
            store = snapshot_store.load()
            store = store[store["source"] == "oddsapi"]
            fixtures = [{"home_team": h, "away_team": a, "commence_time": c}
                        for h, a, c in store[["home_team", "away_team", "commence_time"]]
                        .drop_duplicates().itertuples(index=False)]
        pm_rows = fetch_polymarket(fixtures, captured_at)
        rows += pm_rows
        print(f"Polymarket: {len(pm_rows)}/{len(fixtures)} fixtures matched")

    n = snapshot_store.append(rows)
    print(f"Appended {n} rows -> {paths.odds_snapshots_csv()}")
    return {"appended": n, "quota": quota}


def main():
    ap = argparse.ArgumentParser(description="Snapshot Liga MX odds into the store.")
    ap.add_argument("--no-oddsapi", action="store_true")
    ap.add_argument("--no-polymarket", action="store_true")
    args = ap.parse_args()
    capture(use_oddsapi=not args.no_oddsapi, use_polymarket=not args.no_polymarket)


if __name__ == "__main__":
    main()
