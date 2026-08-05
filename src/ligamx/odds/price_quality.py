"""
Tell apart real traded Polymarket prices from pre-trading placeholders.

WHY THIS EXISTS. A Liga MX event is three independent binary markets (Will X win?
Will it be a draw? Will Y win?). Polymarket seeds each near 0.50 and it sits there
until somebody trades it. Nothing links the three to each other, so an untraded
event shows all three outcomes at ~0.47 and Sum(YES) ~= 1.4 -- a number that would
be a 39pp risk-free arbitrage if it were real (buy all three NO tokens for 1.61,
collect exactly 2.00, since two of three always win). It is not real. Beyond T-336h
the single most common stored triplet is literally 0.47 / 0.47 / 0.47, with the
draw priced 0.45-0.50 against a true Liga MX draw rate of ~0.25.

So the early half of our stored "price history" is not prices. That matters because
the early-line CLV conclusions were measured on it: measuring line movement from a
0.50 placeholder to a real price manufactures CLV out of nothing, and manufactures
most of it for whichever side ends up favourite.

THE TEST. Coherence, not freshness. A snapshot is a real book only if the three
outcomes price up to ~1. Time-since-last-price-change looks like the natural
staleness proxy but fails here: seeded prices jitter by a cent and score as fresh,
so filtering on it leaves Sum(YES) at 1.396 in the far bucket -- unchanged. The
coherence gate is what separates the two populations.

Coherence is necessary, not sufficient: it says a book existed, not that it had
depth. The store has no bid/ask, so executability at size remains unverifiable
retroactively.

    python -m ligamx.odds.price_quality [--write] [--delta-hours 48]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ligamx import config, paths
from ligamx.odds import snapshot_store

OUTCOME_COLS = {"home": "home_odds", "draw": "draw_odds", "away": "away_odds"}

# Lead-time buckets (hours before kickoff) used by every report here.
LEAD_BINS = [0, 12, 48, 72, 168, 336, 10_000]

# Sum(YES) may sit this far from 1.0 and still count as a real book. A traded
# Polymarket 1X2 runs ~0.3-1pp of overround, so 2pp is already generous; the
# placeholder population sits 19-40pp out, so the gate is not sensitive to it.
COHERENCE_TOL = 0.02


def _std(name: str) -> str:
    return config.ODDS_TO_STANDARD.get(name, name)


def load_polymarket(tol: float = COHERENCE_TOL) -> pd.DataFrame:
    """Pre-kickoff Polymarket snapshots with lead time, Sum(YES), and the gate flag.

    ``tol`` is the coherence tolerance; tightening it is the natural falsification
    test for anything measured through the gate (a finding that shrinks as the gate
    tightens was contamination, not signal).
    """
    df = snapshot_store.load()
    df = df[df["venue"] == "polymarket"].copy()
    for c in OUTCOME_COLS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["captured_at", "commence_time", *OUTCOME_COLS.values()])
    df["home_std"] = df["home_team"].map(_std)
    df["away_std"] = df["away_team"].map(_std)
    df["lead_h"] = (df["commence_time"] - df["captured_at"]).dt.total_seconds() / 3600.0
    df["sum_yes"] = sum(1.0 / df[c] for c in OUTCOME_COLS.values())
    df["traded"] = (df["sum_yes"] - 1.0).abs() < tol
    # Pre-kickoff only: in-play prints are a different market.
    df = df[df["lead_h"] > 0]
    return df.sort_values(["event_id", "captured_at"]).reset_index(drop=True)


def first_traded_lead(df: pd.DataFrame) -> pd.DataFrame:
    """Per fixture, the lead time at which its book first prices up to ~1."""
    rows = []
    for (event_id, ko), g in df.groupby(["event_id", "commence_time"], sort=False):
        ok = g[g["traded"]]
        rows.append({
            "event_id": event_id,
            "home_std": g["home_std"].iloc[0], "away_std": g["away_std"].iloc[0],
            "date": ko.strftime("%Y/%m/%d"),
            "earliest_lead_h": g["lead_h"].max(),
            "first_traded_lead_h": ok["lead_h"].max() if len(ok) else np.nan,
            "n_snapshots": len(g), "n_traded": len(ok),
        })
    return pd.DataFrame(rows)


def price_at(df: pd.DataFrame, delta_hours: float, traded_only: bool = False) -> pd.DataFrame:
    """The snapshot nearest T-delta per fixture (mirrors reduce_open_close's 'open').

    ``traded_only`` restricts the candidate set to coherent books first, so the
    returned row is the nearest *real* price rather than the nearest placeholder.
    Fixtures with no candidate are omitted.
    """
    rows = []
    for (event_id, ko), g in df.groupby(["event_id", "commence_time"], sort=False):
        cand = g[g["traded"]] if traded_only else g
        if cand.empty:
            continue
        pick = cand.loc[(cand["lead_h"] - delta_hours).abs().idxmin()]
        rows.append({
            "event_id": event_id, "home_std": pick["home_std"], "away_std": pick["away_std"],
            "date": ko.strftime("%Y/%m/%d"), "lead_h": pick["lead_h"],
            "sum_yes": pick["sum_yes"], "traded": bool(pick["traded"]),
            "home_odds": pick["home_odds"], "draw_odds": pick["draw_odds"],
            "away_odds": pick["away_odds"],
        })
    return pd.DataFrame(rows)


# --- joining store rows onto MEX_ligamx rows --------------------------------
# Snapshot dates come from the UTC kickoff, MEX_ligamx dates are local, so the two
# can differ by a day. Teams meet twice a season at most, far more than a day apart.
DATE_TOLERANCE_DAYS = 1


def row_index(df: pd.DataFrame) -> dict:
    """{(home, away): [(date, row_index), ...]} for nearest-date fixture lookup.

    reduce_open_close._find_row does the same job but assumes a string Date column
    (it is only ever called on a dtype=str frame); handed real datetimes it silently
    falls through to the first team-pair match, which is a fixture from another season.
    """
    idx = {}
    for i, r in df[["Home", "Away", "Date"]].iterrows():
        idx.setdefault((r["Home"], r["Away"]), []).append((r["Date"], i))
    return idx


def match_row(index: dict, home: str, away: str, date) -> int | None:
    """Row index for this fixture, or None if no date lands within tolerance."""
    date = pd.Timestamp(date)
    best, best_diff = None, None
    for d, i in index.get((home, away), ()):
        diff = abs((d - date).days)
        if diff <= DATE_TOLERANCE_DAYS and (best_diff is None or diff < best_diff):
            best, best_diff = i, diff
    return best


def traded_open_rows(df: pd.DataFrame, delta_hours: int = 48,
                     tol: float = COHERENCE_TOL) -> set[int]:
    """MEX_ligamx row indices whose T-delta Polymarket open was a real book.

    Lets the CLV harnesses drop fixtures whose stored "open" is a placeholder,
    without teaching them anything about the snapshot store's layout.
    """
    snaps = load_polymarket(tol)
    index = row_index(df)
    keep = set()
    for o in price_at(snaps, delta_hours).itertuples():
        if not o.traded:
            continue
        i = match_row(index, o.home_std, o.away_std, o.date)
        if i is not None:
            keep.add(i)
    return keep


def report_seeding(df: pd.DataFrame) -> None:
    """Show the placeholder population directly -- the diagnosis, not a summary of it."""
    d = df.copy()
    d["bucket"] = pd.cut(d["lead_h"], LEAD_BINS)
    for k, c in OUTCOME_COLS.items():
        d[f"p_{k}"] = 1.0 / d[c]
    print(f"\n{'='*72}\nWHAT THE STORED PRICES ACTUALLY ARE\n{'='*72}")
    print("mean implied probability per outcome (a real Liga MX book: ~.46/.25/.30)")
    print(d.groupby("bucket", observed=True)[["p_home", "p_draw", "p_away", "sum_yes"]]
          .mean().round(4).to_string())
    far = d[d["lead_h"] > 336]
    if not far.empty:
        trip = (far["p_home"].round(2).astype(str) + " / " + far["p_draw"].round(2).astype(str)
                + " / " + far["p_away"].round(2).astype(str))
        print("\nmost common (H/D/A) triplets beyond T-336h -- seeded, not traded:")
        print(trip.value_counts().head(5).to_string())


def report_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Share of snapshots that are real books, by lead time."""
    d = df.copy()
    d["bucket"] = pd.cut(d["lead_h"], LEAD_BINS)
    t = d.groupby("bucket", observed=True).agg(
        n=("traded", "size"), n_traded=("traded", "sum"),
        sum_yes_all=("sum_yes", "mean"))
    t["pct_traded"] = (100 * t["n_traded"] / t["n"]).round(1)
    t["sum_yes_traded"] = d[d["traded"]].groupby("bucket", observed=True)["sum_yes"].mean()
    print(f"\n{'='*72}\nCOHERENCE GATE -- how much of the history is a real book\n{'='*72}")
    print(t.round(4).to_string())
    return t


def report_coverage(ft: pd.DataFrame) -> None:
    """Share of fixtures that had a real price by each lead time -- the usable sample."""
    print(f"\n{'='*72}\nUSABLE SAMPLE FOR AN 'EARLY' STRATEGY\n{'='*72}")
    n = len(ft)
    print(f"fixtures: {n}   with any traded price: {ft['first_traded_lead_h'].notna().sum()}")
    print("\nshare of fixtures whose market was already trading at:")
    for lead in (336, 240, 168, 120, 72, 48, 24):
        pct = 100 * (ft["first_traded_lead_h"] >= lead).mean()
        bar = "#" * int(round(pct / 2.5))
        print(f"  T-{lead:<4}h  {pct:>5.1f}%  {bar}")


def main():
    ap = argparse.ArgumentParser(description="Separate traded Polymarket prices from placeholders.")
    ap.add_argument("--write", action="store_true",
                    help=f"write per-snapshot flags to {paths.polymarket_price_quality_csv()}")
    ap.add_argument("--delta-hours", type=int, default=48,
                    help="lead time whose 'open' snapshot gets summarised")
    ap.add_argument("--tolerance", type=float, default=COHERENCE_TOL,
                    help="max |Sum(YES) - 1| for a snapshot to count as a real book")
    args = ap.parse_args()

    df = load_polymarket(args.tolerance)
    print(f"Polymarket snapshots: {len(df)}  |  fixtures: {df['event_id'].nunique()}")

    report_seeding(df)
    report_gate(df)
    ft = first_traded_lead(df)
    report_coverage(ft)

    op = price_at(df, args.delta_hours)
    print(f"\n{'='*72}\nTHE T-{args.delta_hours}h 'OPEN' THE CLV BACKTESTS USE  (n={len(op)})\n{'='*72}")
    print(f"  real books: {100*op['traded'].mean():.1f}%   mean Sum(YES) {op['sum_yes'].mean():.4f}")

    if args.write:
        cols = ["event_id", "captured_at", "commence_time", "home_std", "away_std",
                "lead_h", "sum_yes", "traded"]
        df[cols].to_csv(paths.polymarket_price_quality_csv(), index=False)
        print(f"\nwrote {len(df)} rows -> {paths.polymarket_price_quality_csv()}")


if __name__ == "__main__":
    main()
