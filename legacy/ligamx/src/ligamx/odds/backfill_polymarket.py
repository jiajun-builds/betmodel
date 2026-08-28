"""
Backfill Polymarket 1X2 price history into the snapshot store (free, retroactive).

Polymarket's CLOB /prices-history endpoint returns a per-outcome price series
from market open through settlement, for closed markets too. So unlike the paid
Odds API historical endpoint, Polymarket early prices can be reconstructed at no
cost. This walks played Liga MX fixtures, finds each settled Polymarket match
event, pulls the 3 outcome curves, aligns them into 1X2 snapshots, and appends
them to the same store the forward poller writes — where the normal reducer then
collapses them into polymarket_open_*/close_* columns.

    python -m ligamx.odds.backfill_polymarket [--since 2025-10-01] [--limit N] [--fidelity 60]

Snapshots are stamped with each candle's real timestamp (captured_at = trade time),
and commence_time is the fixture's true kickoff — so the reducer's pre-kickoff
split correctly excludes any in-play prints from the close.

Two properties of this data that are easy to get wrong:

ONLY THE YES TOKEN IS STORED, AND THAT LOSES NOTHING. Each outcome is a binary
market whose clobTokenIds are [Yes, No], but the pair shares one order book: 1 USDC
mints 1 Yes + 1 No, so buying No *is* selling Yes. Measured 2026-07-27 — across 312
historical observations Yes + No summed to 1.0000 with standard deviation 0.0, and
on live books NO_ask == 1 - YES_bid and NO_bid == 1 - YES_ask exactly. There is no
independent No price, hence no No-side mispricing to exploit: any "bet the No side
instead" strategy reduces algebraically to the overround. Do not re-test this.

EARLY PRICES ARE MOSTLY NOT PRICES. The three binary markets are seeded near 0.50
and are not linked to each other, so before anyone trades an event you get roughly
0.47/0.47/0.47 and Sum(YES) ~= 1.4. Beyond T-336h only 2% of stored snapshots are
real books. Anything measured against these placeholders is an artifact — filter
with odds/price_quality first.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse

import pandas as pd
import requests

from ligamx import config, paths
from ligamx.odds import snapshot_store
from ligamx.odds.capture_odds import _fold, _gamma, _inv, _match_polymarket_event, _search_query


def _true_kickoff_iso(ev: dict, fallback: str) -> str:
    """Authoritative UTC kickoff from Polymarket, not from MEX_ligamx's Time column.

    MEX_ligamx's Time mixes conventions -- some rows hold local Mexico time, some
    UTC -- and _kickoff_iso stamps a Z on whatever it finds. Measured against 178
    fixtures with a known kickoff, that put commence_time more than an hour off for
    126 of them, by -29h to +20h. When the stored kickoff runs LATE the reducer's
    "last snapshot before kickoff" reaches past the real one and takes an in-play or
    settled price as the close: 5 fixtures leaked that way, 2 of them fully resolved
    (Tijuana-Mazatlan 2026-02-22 at 0.999 on the draw it finished as).

    Polymarket publishes gameStartTime per market, so take that and never guess.
    The CSV-derived time is still fine for *matching* the event (it only has to be
    within a day), which is why it stays as the fallback.
    """
    for m in ev.get("markets", []):
        raw = m.get("gameStartTime") or m.get("endDate")
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = pd.to_datetime(ev.get("endDate"), utc=True, errors="coerce")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(ts) else fallback


def _kickoff_iso(date: str, tm: str) -> str:
    """Build a UTC ISO kickoff from MEX_ligamx Date (YYYY/MM/DD) + Time (HH:MM).

    Only used to *locate* the Polymarket event; the snapshots themselves are stamped
    with _true_kickoff_iso. A blank Time reads back from pandas as NaN, and str(NaN)
    is "nan" -- truthy, so a naive `or "00:00"` produced "...Tnan:00Z", an unparseable
    timestamp that any downstream to_datetime silently coerces to NaT and drops.
    Validate the shape.
    """
    d = str(date).replace("/", "-").strip()
    t = str(tm).strip()[:5]
    if len(t) != 5 or t[2] != ":" or not (t[:2] + t[3:]).isdigit():
        t = "00:00"
    return f"{d}T{t}:00Z"


def _outcome_tokens(ev: dict, home: str, away: str) -> dict | None:
    """Map the event's 3 Yes/No sub-markets to {home,draw,away} -> Yes token id."""
    ht, at = _fold(home), _fold(away)
    toks = {"home": None, "draw": None, "away": None}
    for m in ev.get("markets", []):
        try:
            yes_tok = json.loads(m.get("clobTokenIds") or "[]")[0]
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
        label = m.get("groupItemTitle", "") + " " + m.get("question", "")
        if "draw" in label.lower():
            toks["draw"] = yes_tok
            continue
        gt = _fold(m.get("groupItemTitle", "") or m.get("question", ""))
        if len(ht & gt) > len(at & gt):
            toks["home"] = yes_tok
        elif len(at & gt) > len(ht & gt):
            toks["away"] = yes_tok
    return toks if all(toks.values()) else None


# CLOB /prices-history drops sparse markets at fine resolution: an illiquid old
# match returns nothing at 60-360 min but data at 720/1440. Try fine first and
# fall back coarser so liquid (recent) markets keep precision while thin (older)
# ones still yield a daily curve.
FIDELITY_LADDER = [360, 720, 1440]


def _price_history(token_id: str, fidelity: int) -> pd.Series:
    """Yes-price time-series (index = unix seconds) for one outcome token."""
    r = requests.get(
        f"{config.POLYMARKET_CLOB_BASE}/prices-history",
        params={"market": token_id, "interval": "max", "fidelity": fidelity},
        headers={"User-Agent": config.HTTP_USER_AGENT, "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    hist = r.json().get("history", [])
    if not hist:
        return pd.Series(dtype=float)
    s = pd.Series({int(p["t"]): float(p["p"]) for p in hist})
    return s.sort_index()


def _fixture_snapshots(ev: dict, tokens: dict, home: str, away: str,
                       kickoff_iso: str, fidelities: list[int]) -> tuple[list[dict], int | None]:
    """Fetch + align the 3 outcome curves into 1X2 rows; return (rows, fidelity_used)."""
    cols, used = None, None
    for fid in fidelities:
        c = {k: _price_history(tokens[k], fid) for k in ("home", "draw", "away")}
        if all(not s.empty for s in c.values()):
            cols, used = c, fid
            break
    if cols is None:
        return [], None
    aligned = pd.DataFrame(cols).sort_index().ffill().dropna()  # last-known price per outcome
    rows = []
    for ts, r in aligned.iterrows():
        ho, do, ao = _inv(r["home"]), _inv(r["draw"]), _inv(r["away"])
        if "" in (ho, do, ao):
            continue
        rows.append({
            "captured_at": pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "polymarket", "venue": "polymarket", "event_id": ev.get("slug"),
            "commence_time": kickoff_iso, "home_team": home, "away_team": away,
            "home_odds": ho, "draw_odds": do, "away_odds": ao,
            "liquidity": "", "last_update": "",
        })
    return rows, used


def backfill(since: str | None = None, limit: int | None = None,
             fidelity: int = 360, pause: float = 0.3) -> dict:
    df = pd.read_csv(paths.ligamx_data_csv(), dtype=str)
    played = df[df["Res"].isin(["H", "D", "A"])].copy()
    played = played[played["Home"].notna() & played["Away"].notna()]
    if since:
        played = played[played["Date"].str.replace("/", "-") >= since]
    played = played.sort_values("Date")
    if limit:
        played = played.tail(limit)

    ladder = sorted({fidelity, *FIDELITY_LADDER})
    stats = {"fixtures": len(played), "matched": 0, "with_prices": 0,
             "no_event": 0, "no_tokens": 0, "no_history": 0, "rows": 0}
    all_rows: list[dict] = []
    for _, fx in played.iterrows():
        home, away = fx["Home"], fx["Away"]
        kickoff = _kickoff_iso(fx["Date"], fx.get("Time", ""))
        try:
            res = _gamma("/public-search", {"q": _search_query(home, away), "limit_per_type": 10})
            ev = _match_polymarket_event(res.get("events", []), home, away, kickoff, allow_closed=True)
            if not ev:
                stats["no_event"] += 1
                continue
            tokens = _outcome_tokens(ev, home, away)
            if not tokens:
                stats["no_tokens"] += 1
                continue
            stats["matched"] += 1
            rows, used = _fixture_snapshots(ev, tokens, home, away,
                                            _true_kickoff_iso(ev, kickoff), ladder)
            if not rows:
                stats["no_history"] += 1
                continue
            all_rows += rows
            stats["with_prices"] += 1
            stats["rows"] += len(rows)
            print(f"  {fx['Date']} {home:>14} v {away:<14} -> {len(rows):>4} pts @ {used}m  ({ev.get('slug')})")
        except requests.RequestException as e:
            print(f"  {fx['Date']} {home} v {away}: request failed: {e}")
        time.sleep(pause)

    n = snapshot_store.append(all_rows)
    print(f"\nAppended {n} backfilled Polymarket rows -> {paths.odds_snapshots_csv()}")
    print("stats:", stats)
    return stats


# Bound on how far a kickoff correction may move a fixture. MEX_ligamx's Time is
# wrong by at most ~20h (local-vs-UTC plus a date slip), so anything beyond this is
# far more likely to be _match_polymarket_event landing on the wrong fixture -- the
# same pairing recurs every season -- than a real correction. Those are reported
# rather than applied.
MAX_KICKOFF_SHIFT_H = 30.0


def repair_kickoffs(dry_run: bool = True, pause: float = 0.3,
                    max_shift: float = MAX_KICKOFF_SHIFT_H) -> pd.DataFrame:
    """Rewrite commence_time on stored Polymarket rows using Polymarket's own kickoff.

    The captured prices are correct -- only the kickoff they were stamped against was
    wrong (see _true_kickoff_iso) -- so this re-resolves each fixture's event and
    corrects the timestamp in place, rather than re-pulling every price curve. The
    reducer then re-derives open/close from the same curves against the real kickoff.
    """
    store = snapshot_store.load()
    pm = store[store["venue"] == "polymarket"]
    keys = pm[["home_team", "away_team", "commence_time"]].drop_duplicates()
    print(f"{len(keys)} stored Polymarket fixtures to check")

    fixes, rejected, unresolved = {}, [], 0
    for i, k in enumerate(keys.itertuples(index=False), 1):
        try:
            res = _gamma("/public-search",
                         {"q": _search_query(k.home_team, k.away_team), "limit_per_type": 10})
            ev = _match_polymarket_event(res.get("events", []), k.home_team, k.away_team,
                                         k.commence_time, allow_closed=True)
            if not ev:
                unresolved += 1
                continue
            true = _true_kickoff_iso(ev, k.commence_time)
            if true == k.commence_time:
                continue
            shift = abs((pd.Timestamp(k.commence_time) - pd.Timestamp(true)).total_seconds() / 3600)
            if shift > max_shift:
                rejected.append({"home": k.home_team, "away": k.away_team,
                                 "stored": k.commence_time, "true": true, "err_h": round(shift, 2)})
                continue
            fixes[(k.home_team, k.away_team, k.commence_time)] = true
        except requests.RequestException as e:
            print(f"  {k.home_team} v {k.away_team}: {e}")
            unresolved += 1
        if i % 25 == 0:
            print(f"  {i}/{len(keys)} checked, {len(fixes)} need correcting")
        time.sleep(pause)

    rows = [{"home": h, "away": a, "stored": c, "true": t,
             "err_h": round((pd.Timestamp(c) - pd.Timestamp(t)).total_seconds() / 3600, 2)}
            for (h, a, c), t in fixes.items()]
    report = pd.DataFrame(rows).sort_values("err_h") if rows else pd.DataFrame()
    print(f"\n{len(fixes)} fixtures need a corrected kickoff; {unresolved} unresolved; "
          f"{len(rejected)} rejected as beyond +/-{max_shift:g}h (probably a mis-matched event)")
    for r in rejected:
        print(f"  REJECTED {r['home']} v {r['away']}: {r['stored']} -> {r['true']} ({r['err_h']:+.1f}h)")

    if fixes and not dry_run:
        key = list(zip(store["home_team"], store["away_team"], store["commence_time"]))
        store["commence_time"] = [fixes.get(k, k[2]) for k in key]
        store.to_csv(paths.odds_snapshots_csv(), index=False)
        print(f"rewrote {paths.odds_snapshots_csv()}")
    elif fixes:
        print("dry run -- pass --write to apply")
    return report


def main():
    ap = argparse.ArgumentParser(description="Backfill Polymarket 1X2 history into the store.")
    ap.add_argument("--since", default="2025-10-01", help="only fixtures on/after this date (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=None, help="cap to the most recent N fixtures")
    ap.add_argument("--fidelity", type=int, default=60, help="prices-history candle size in minutes")
    ap.add_argument("--repair-kickoffs", action="store_true",
                    help="fix commence_time on existing stored rows (no price re-pull)")
    ap.add_argument("--write", action="store_true", help="let --repair-kickoffs save its result")
    args = ap.parse_args()
    if args.repair_kickoffs:
        rep = repair_kickoffs(dry_run=not args.write)
        if len(rep):
            print(rep.to_string(index=False))
        return
    backfill(since=args.since, limit=args.limit, fidelity=args.fidelity)


if __name__ == "__main__":
    main()
