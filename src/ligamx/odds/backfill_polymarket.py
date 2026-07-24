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


def _kickoff_iso(date: str, tm: str) -> str:
    """Build a UTC ISO kickoff from MEX_ligamx Date (YYYY/MM/DD) + Time (HH:MM)."""
    d = str(date).replace("/", "-").strip()
    t = (str(tm).strip() or "00:00")[:5]
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
            rows, used = _fixture_snapshots(ev, tokens, home, away, kickoff, ladder)
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


def main():
    ap = argparse.ArgumentParser(description="Backfill Polymarket 1X2 history into the store.")
    ap.add_argument("--since", default="2025-10-01", help="only fixtures on/after this date (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=None, help="cap to the most recent N fixtures")
    ap.add_argument("--fidelity", type=int, default=60, help="prices-history candle size in minutes")
    args = ap.parse_args()
    backfill(since=args.since, limit=args.limit, fidelity=args.fidelity)


if __name__ == "__main__":
    main()
