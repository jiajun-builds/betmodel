"""
Tick-level Polymarket trade tape, normalized to Yes-space aggressor terms.

WHY THIS EXISTS. eval/maker_backtest decides whether a resting quote was filled by
asking whether the *price curve* ever reached it. The curve is hourly, so it only
registers moves that persist for an hour or more -- exactly the moves that are bad
for a maker. The fills that pay a market maker are transient dips that revert inside
the hour, and they are invisible to it. That was the one caveat left standing over
the "passive quoting loses to adverse selection" result, so this module removes it:
the CLOB records every print, and data-api serves them retroactively for settled
markets at no cost.

    python -m ligamx.odds.polymarket_trades [--since 2025-10-01] [--limit N]

WHAT A PRINT MEANS, AND THE SIGN TRAP. Each outcome is a binary market whose Yes and
No tokens share ONE order book: 1 USDC mints 1 Yes + 1 No, so buying No *is* selling
Yes (see backfill_polymarket for the measurement). A trade is reported from the
aggressor's side in whichever token they traded, so the four cases fold into two:

    BUY  Yes @p  ->  lifts the Yes ask at p
    SELL Yes @p  ->  hits  the Yes bid at p
    BUY  No  @p  ->  hits  the Yes bid at 1-p     (buying No = selling Yes)
    SELL No  @p  ->  lifts the Yes ask at 1-p

Everything is stored folded into Yes space, so a resting Yes bid at q was filled by
the prints with ``yes_side == "SELL"`` and ``yes_price <= q``. Reading the raw
``side`` column without folding gets this backwards for roughly half the tape --
in a sampled Liga MX market 49 of 117 pre-kickoff prints were No-side.

Trades come back newest-first and the endpoint has no time filter, so reaching the
early pre-kickoff prints means paging through the (much larger) in-play tape. That is
the cost of the pull, not a bug.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import requests

from ligamx import config, paths
from ligamx.odds.backfill_polymarket import _kickoff_iso, _outcome_tokens
from ligamx.odds.capture_odds import _gamma, _match_polymarket_event, _search_query

TRADES_URL = "https://data-api.polymarket.com/trades"

COLUMNS = ["event_id", "outcome", "condition_id", "ts", "lead_h",
           "yes_price", "yes_side", "size", "notional"]

PAGE = 500
MAX_PAGES = 40  # 20k prints per market; far beyond any Liga MX book


def _true_kickoff(ev: dict, markets: dict, fallback: pd.Timestamp) -> pd.Timestamp:
    """Authoritative UTC kickoff from Polymarket, not from MEX_ligamx's Time column.

    MEX_ligamx mixes conventions: newer SofaScore rows store LOCAL Mexico time while
    older rows store UTC, and _kickoff_iso stamps a Z on whatever it finds. That puts
    the kickoff 6h early on roughly half the fixtures, which silently truncates the
    pre-kickoff window (it cannot leak in-play prints in, but it does drop real ones).
    Polymarket publishes gameStartTime per market, so take that and never guess.
    """
    for m in markets.values():
        raw = m.get("gameStartTime") or m.get("endDate")
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts
    ts = pd.to_datetime(ev.get("endDate"), utc=True, errors="coerce")
    return ts if pd.notna(ts) else fallback


def _outcome_markets(ev: dict, home: str, away: str) -> dict | None:
    """Map {home,draw,away} -> the market dict, reusing the Yes-token classifier.

    _outcome_tokens already solves the hard part (which sub-market is the draw, which
    team is which, tolerating Polymarket's naming); indexing the event's markets by
    their Yes token turns its answer into the market objects we need conditionIds from.
    """
    toks = _outcome_tokens(ev, home, away)
    if not toks:
        return None
    by_token = {}
    for m in ev.get("markets", []):
        try:
            by_token[json.loads(m.get("clobTokenIds") or "[]")[0]] = m
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
    out = {k: by_token.get(t) for k, t in toks.items()}
    return out if all(out.values()) else None


def _fetch_trades(condition_id: str, pause: float = 0.2) -> list[dict]:
    """Every print for one market, newest-first, paged to exhaustion."""
    rows, offset = [], 0
    for _ in range(MAX_PAGES):
        r = requests.get(
            TRADES_URL, params={"market": condition_id, "limit": PAGE, "offset": offset},
            headers={"User-Agent": config.HTTP_USER_AGENT, "Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        offset += len(batch)
        if len(batch) < PAGE:
            break
        time.sleep(pause)
    return rows


def _to_yes_space(trade: dict) -> tuple[float, str] | None:
    """Fold a print into (yes_price, yes_side) -- see the module docstring."""
    try:
        p = float(trade["price"])
    except (KeyError, TypeError, ValueError):
        return None
    side = str(trade.get("side", "")).upper()
    outcome = str(trade.get("outcome", "")).strip().lower()
    if outcome == "yes":
        return p, side
    if outcome == "no":
        # Buying No is selling Yes, at the complementary price.
        return 1.0 - p, ("SELL" if side == "BUY" else "BUY")
    return None


def _fixture_rows(ev: dict, markets: dict, kickoff: pd.Timestamp, pause: float) -> list[dict]:
    rows = []
    for name, m in markets.items():
        cond = m.get("conditionId")
        if not cond:
            continue
        for t in _fetch_trades(cond, pause):
            folded = _to_yes_space(t)
            if folded is None:
                continue
            yes_price, yes_side = folded
            try:
                ts = pd.Timestamp(int(t["timestamp"]), unit="s", tz="UTC")
                size = float(t["size"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "event_id": ev.get("slug"), "outcome": name, "condition_id": cond,
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_h": round((kickoff - ts).total_seconds() / 3600.0, 4),
                "yes_price": round(yes_price, 6), "yes_side": yes_side,
                "size": size, "notional": round(size * yes_price, 4),
            })
        time.sleep(pause)
    return rows


def load() -> pd.DataFrame:
    """Cached tape (empty frame with the right columns if absent)."""
    path = paths.polymarket_trades_csv()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in ("lead_h", "yes_price", "size", "notional"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["lead_h", "yes_price", "size"])


def build(since: str | None = None, limit: int | None = None,
          pause: float = 0.2, resume: bool = True) -> dict:
    """Walk played fixtures, pull each one's 3 markets, write the folded tape."""
    df = pd.read_csv(paths.ligamx_data_csv(), dtype=str)
    played = df[df["Res"].isin(["H", "D", "A"])]
    played = played[played["Home"].notna() & played["Away"].notna()]
    if since:
        played = played[played["Date"].str.replace("/", "-") >= since]
    played = played.sort_values("Date")
    if limit:
        played = played.tail(limit)

    have = set()
    existing = load() if resume else pd.DataFrame(columns=COLUMNS)
    if resume and not existing.empty:
        have = set(existing["event_id"].dropna())
        print(f"resuming: {len(have)} fixtures already in {paths.polymarket_trades_csv()}")

    stats = {"fixtures": len(played), "skipped": 0, "no_event": 0,
             "no_markets": 0, "no_trades": 0, "done": 0, "failed": 0, "rows": 0}
    out = [existing] if not existing.empty else []

    def checkpoint():
        """Persist what we have. The pull takes ~an hour, so never hold it all in RAM."""
        if out:
            pd.concat(out, ignore_index=True)[COLUMNS].to_csv(
                paths.polymarket_trades_csv(), index=False)

    for _, fx in played.iterrows():
        home, away = fx["Home"], fx["Away"]
        kickoff = pd.Timestamp(_kickoff_iso(fx["Date"], fx.get("Time", "")))
        try:
            res = _gamma("/public-search", {"q": _search_query(home, away), "limit_per_type": 10})
            ev = _match_polymarket_event(res.get("events", []), home, away,
                                         kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"), allow_closed=True)
            if not ev:
                stats["no_event"] += 1
                continue
            if ev.get("slug") in have:
                stats["skipped"] += 1
                continue
            markets = _outcome_markets(ev, home, away)
            if not markets:
                stats["no_markets"] += 1
                continue
            rows = _fixture_rows(ev, markets, _true_kickoff(ev, markets, kickoff), pause)
            if not rows:
                stats["no_trades"] += 1
                continue
            out.append(pd.DataFrame(rows))
            stats["done"] += 1
            stats["rows"] += len(rows)
            pre = sum(1 for r in rows if r["lead_h"] > 0)
            print(f"  {fx['Date']} {home:>14} v {away:<14} -> {len(rows):>5} prints "
                  f"({pre:>4} pre-KO)  {ev.get('slug')}")
            if stats["done"] % 20 == 0:
                checkpoint()
        except Exception as e:  # one bad fixture must not discard an hour of pulling
            stats["failed"] += 1
            print(f"  {fx['Date']} {home} v {away}: {type(e).__name__}: {e}")
        time.sleep(pause)

    checkpoint()
    print(f"\nWrote {stats['rows'] + len(existing)} prints -> {paths.polymarket_trades_csv()}")
    print("stats:", stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Pull the Polymarket trade tape for played fixtures.")
    ap.add_argument("--since", default="2025-10-01", help="only fixtures on/after this date")
    ap.add_argument("--limit", type=int, default=None, help="cap to the most recent N fixtures")
    ap.add_argument("--pause", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--no-resume", action="store_true", help="re-pull fixtures already cached")
    args = ap.parse_args()
    build(since=args.since, limit=args.limit, pause=args.pause, resume=not args.no_resume)


if __name__ == "__main__":
    main()
