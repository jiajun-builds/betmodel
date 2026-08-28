"""
Re-test the "bet early for time value" thesis against the price-quality gate.

The thesis was that Polymarket's early prices are inefficient, and the supporting
measurement was that excess CLV rises with lead time (+4.32pp at T-240h, t=2.74)
while a naive back-the-favourite rule earns +2.3-2.7pp. Both were computed on the
raw snapshot store, which mixes real books with pre-trading placeholders sitting
near 0.50 per outcome (see odds/price_quality). Measuring movement from a 0.50
placeholder to a real price manufactures CLV, and hands most of it to whichever
side ends up favourite -- exactly the shape of the reported result.

This runs the same measurement twice: once on all snapshots (reproducing the
original claim) and once restricted to snapshots whose book prices up to ~1
(the honest sample). If the effect is real it survives the gate; if it was an
artifact of placeholder prices it collapses.

Prices come from the store rather than the reduced polymarket_open_* columns so
lead time is a free parameter. The model is refit weekly out-of-sample via
dc.fit_production_model, same as eval/clv_backtest.

    python -m ligamx.eval.early_price_clv [--test-start 2025-10-01] [--ev-threshold 0.0]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ligamx.eval.clv_backtest import _load, _model_probs
from ligamx.eval.rps_backtest import RESULT_IDX
from ligamx.odds import price_quality

warnings.filterwarnings("ignore")

from ligamx.odds.price_quality import match_row, row_index

LEAD_TIMES = [48, 72, 120, 168, 240, 336]


def _devig(odds):
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return np.array([x / s for x in inv])


def close_prices(snaps: pd.DataFrame, traded_only: bool = True) -> pd.DataFrame:
    """Last pre-kickoff snapshot per fixture -- the closing line.

    The close is always taken from a real book: a placeholder close would corrupt
    both arms of the comparison, not just the filtered one.
    """
    rows = []
    for (event_id, ko), g in snaps.groupby(["event_id", "commence_time"], sort=False):
        cand = g[g["traded"]] if traded_only else g
        if cand.empty:
            continue
        pick = cand.loc[cand["lead_h"].idxmin()]
        rows.append({"event_id": event_id, "home_std": pick["home_std"],
                     "away_std": pick["away_std"], "date": ko.strftime("%Y/%m/%d"),
                     "lead_h": pick["lead_h"], "close": [pick["home_odds"],
                     pick["draw_odds"], pick["away_odds"]]})
    return pd.DataFrame(rows)


def build_records(df, probs, snaps, delta, traded_only, index=None, closes=None):
    """Join model probs, the T-delta price, and the close onto one record per fixture."""
    opens = price_quality.price_at(snaps, delta, traded_only=traded_only)
    closes = close_prices(snaps) if closes is None else closes
    if opens.empty or closes.empty:
        return []
    index = row_index(df) if index is None else index
    ckey = {(r.event_id): (r.close, r.lead_h) for r in closes.itertuples()}

    recs = []
    for o in opens.itertuples():
        got = ckey.get(o.event_id)
        if got is None:
            continue
        close, close_lead = got
        # A price is only an "early" price if it genuinely precedes the close.
        if o.lead_h - close_lead < 1.0:
            continue
        idx = match_row(index, o.home_std, o.away_std, o.date)
        if idx is None or idx not in probs:
            continue
        recs.append({
            "m": probs[idx], "open": [o.home_odds, o.draw_odds, o.away_odds],
            "close": close, "res": RESULT_IDX[df.loc[idx, "Res"]],
            "lead_h": o.lead_h, "traded": o.traded,
        })
    return recs


def _drift(recs) -> np.ndarray:
    """Mean per-outcome (close - open) devigged movement: the free beta a pick inherits."""
    if not recs:
        return np.zeros(3)
    return np.mean([_devig(r["close"]) - _devig(r["open"]) for r in recs], axis=0)


def _excess_clv(recs, pick_fn, drift, ev_threshold):
    """Per-bet excess CLV in pp for a picking rule; returns the raw array."""
    out = []
    for r in recs:
        pick, ev = pick_fn(r)
        if pick is None or ev <= ev_threshold:
            continue
        po, pc = _devig(r["open"]), _devig(r["close"])
        out.append(100.0 * (pc[pick] - po[pick] - drift[pick]))
    return np.array(out)


def _ev_pick(r):
    ev = [r["m"][i] * r["open"][i] - 1 for i in range(3)]
    k = int(np.argmax(ev))
    return k, ev[k]


def _fav_pick(r):
    return int(np.argmin(r["open"])), np.inf


def _stat(a: np.ndarray) -> str:
    """n / mean / t for a per-bet excess-CLV array."""
    if len(a) < 2:
        return f"{len(a):>5}{'-':>10}{'-':>8}"
    se = a.std(ddof=1) / np.sqrt(len(a))
    t = a.mean() / se if se > 0 else 0.0
    star = "*" if abs(t) > 2 else " "
    return f"{len(a):>5}{a.mean():>+9.2f}{t:>+7.2f}{star}"


def run(df, probs, snaps, ev_threshold: float, index=None, closes=None):
    print(f"\n{'='*96}")
    print("EXCESS CLV BY LEAD TIME -- all snapshots vs real books only")
    print("excess = (close - open) devigged pp on the picked side, minus that outcome's drift")
    print(f"* marks |t| > 2.   EV threshold for model picks: {ev_threshold:g}")
    print(f"{'='*96}")
    hdr = (f"{'lead':>6} | {'ALL SNAPSHOTS':^24} | {'REAL BOOKS ONLY':^24} | "
           f"{'favourite (real books)':^24}")
    print(hdr)
    print(f"{'':>6} | {'n':>5}{'exc':>9}{'t':>8} | {'n':>5}{'exc':>9}{'t':>8} | {'n':>5}{'exc':>9}{'t':>8}")
    print("-" * 96)

    rows = []
    obtained = {}
    for delta in LEAD_TIMES:
        raw = build_records(df, probs, snaps, delta, False, index, closes)
        fil = build_records(df, probs, snaps, delta, True, index, closes)
        obtained[delta] = (raw, fil)
        a_raw = _excess_clv(raw, _ev_pick, _drift(raw), ev_threshold)
        a_fil = _excess_clv(fil, _ev_pick, _drift(fil), ev_threshold)
        a_fav = _excess_clv(fil, _fav_pick, _drift(fil), -np.inf)
        print(f"T-{delta:<4}h | {_stat(a_raw)} | {_stat(a_fil)} | {_stat(a_fav)}")
        rows.append({"delta": delta, "n_raw": len(a_raw), "exc_raw": a_raw.mean() if len(a_raw) else np.nan,
                     "n_fil": len(a_fil), "exc_fil": a_fil.mean() if len(a_fil) else np.nan})

    # What the gate removed: a real book may simply not exist at the requested lead.
    print("\nlead time actually obtained (no real book may exist that early):")
    for delta in LEAD_TIMES:
        raw, fil = obtained[delta]
        if fil:
            print(f"  T-{delta:<4}h requested -> {np.mean([r['lead_h'] for r in fil]):>6.1f}h obtained "
                  f"({len(fil)} fixtures; unfiltered {len(raw)} at "
                  f"{np.mean([r['lead_h'] for r in raw]):.1f}h, "
                  f"{100*np.mean([r['traded'] for r in raw]):.0f}% real books)")
    return pd.DataFrame(rows)


def run_decomposition(df, probs, snaps, ev_threshold: float, index=None, closes=None):
    """Does the model earn CLV when it disagrees with the market favourite?

    This is the bucket that matters: agreeing with the favourite inherits the
    market's own drift, so only the disagreement bucket can show model alpha.
    """
    print(f"\n{'='*96}")
    print("MODEL PICK vs MARKET FAVOURITE (real books only)")
    print(f"{'='*96}")
    print(f"{'lead':>6} | {'agrees with favourite':^24} | {'disagrees':^24}")
    print(f"{'':>6} | {'n':>5}{'exc':>9}{'t':>8} | {'n':>5}{'exc':>9}{'t':>8}")
    print("-" * 62)
    for delta in LEAD_TIMES:
        recs = build_records(df, probs, snaps, delta, True, index, closes)
        drift = _drift(recs)

        def agree(r):
            k, ev = _ev_pick(r)
            return (k, ev) if k == int(np.argmin(r["open"])) else (None, 0.0)

        def disagree(r):
            k, ev = _ev_pick(r)
            return (k, ev) if k != int(np.argmin(r["open"])) else (None, 0.0)

        print(f"T-{delta:<4}h | {_stat(_excess_clv(recs, agree, drift, ev_threshold))} | "
              f"{_stat(_excess_clv(recs, disagree, drift, ev_threshold))}")


def run_tolerance_sweep(df, probs, index, ev_threshold: float, leads=(168, 240, 336)):
    """Tighten the gate and watch what happens: contamination shrinks, signal doesn't."""
    print(f"\n{'='*96}")
    print("FALSIFICATION -- does the surviving effect just track a loose gate?")
    print("if the effect is leftover placeholder contamination it fades as the gate tightens")
    print(f"{'='*96}")
    print(f"{'gate':>9}{'snapshots':>11} | " + " | ".join(f"{'T-'+str(l)+'h':^22}" for l in leads))
    print(f"{'':>20} | " + " | ".join(f"{'n':>5}{'exc':>9}{'t':>8}" for _ in leads))
    print("-" * 96)
    for tol in (0.02, 0.01, 0.005, 0.002):
        snaps = price_quality.load_polymarket(tol)
        closes = close_prices(snaps)
        cells = []
        for lead in leads:
            recs = build_records(df, probs, snaps, lead, True, index, closes)
            cells.append(_stat(_excess_clv(recs, _ev_pick, _drift(recs), ev_threshold)))
        print(f"{100*tol:>7.1f}pp{int(snaps['traded'].sum()):>11} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--ev-threshold", type=float, default=0.0,
                    help="minimum model EV at the early price to place the bet")
    ap.add_argument("--tolerance", type=float, default=price_quality.COHERENCE_TOL,
                    help="max |Sum(YES) - 1| for a snapshot to count as a real book")
    args = ap.parse_args()

    df = _load()
    probs = _model_probs(df, args.test_start)
    snaps = price_quality.load_polymarket(args.tolerance)
    index, closes = row_index(df), close_prices(snaps)
    print(f"test-start {args.test_start} | model probs for {len(probs)} matches | "
          f"{snaps['event_id'].nunique()} fixtures in the snapshot store | "
          f"{len(closes)} with a real closing book")

    run(df, probs, snaps, args.ev_threshold, index, closes)
    run_decomposition(df, probs, snaps, args.ev_threshold, index, closes)
    run_tolerance_sweep(df, probs, index, args.ev_threshold)


if __name__ == "__main__":
    main()
