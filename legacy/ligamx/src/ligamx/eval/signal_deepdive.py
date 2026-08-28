"""Bootstrap-CI deep-dive on the candidate CLV/ROI signals from clv_signals.

Fits the production engine once (walk-forward, draw-recalibrated), then attaches a
90% bootstrap confidence interval to ROI@open and excess-CLV for the headline
slices, plus a reliability table (model prob vs realized frequency). The point is
to separate real edges from single-season variance: a slice whose ROI CI straddles
zero is not tradeable however large its point estimate.

Benchmark is polymarket_close (the largest, and the pure line-timing signal).

    python -m ligamx.eval.signal_deepdive [--draw-alpha 0.85] [--test-start 2025-10-01]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from ligamx.models.dc import DRAW_CALIBRATION_ALPHA
from ligamx.eval import clv_signals as cs

warnings.filterwarnings("ignore")

_RNG = np.random.default_rng(7)
CLOSE_KEY = "polymarket_close"


def boot_ci(vals, n=4000, q=(5, 95)):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    idx = _RNG.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return tuple(np.percentile(means, q))


def _slice(sample, drift, pick_filter, thr, odds_band=None):
    """Return per-bet (roi_open, clv_excess) arrays for a filtered slice."""
    roi, clvx = [], []
    for r in sample:
        po, _ = cs.devig(r["open"])
        pc, _ = cs.devig(r[CLOSE_KEY])
        k, ev = cs.ev_pick(r)
        if pick_filter is not None and k not in pick_filter:
            continue
        if ev <= thr:
            continue
        if odds_band is not None and not (odds_band[0] <= r["open"][k] < odds_band[1]):
            continue
        win = r["res"] == k
        roi.append((r["open"][k] - 1) if win else -1.0)
        clvx.append((pc[k] - po[k] - drift[k]) * 100)
    return roi, clvx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw-alpha", type=float, default=DRAW_CALIBRATION_ALPHA)
    ap.add_argument("--test-start", default="2025-10-01")
    args = ap.parse_args()

    df = cs.load()
    raw = cs.model_probs(df, test_start=args.test_start)
    probs = cs.apply_draw_alpha(raw, args.draw_alpha)
    recs = cs.build_records(df, probs)
    sample = [r for r in recs if r[CLOSE_KEY]]
    drift = cs.drift_vec(recs, CLOSE_KEY)
    print(f"benchmark={CLOSE_KEY}  n={len(sample)}  draw-alpha={args.draw_alpha}  "
          f"drift[H/D/A]={100*drift[0]:+.2f}/{100*drift[1]:+.2f}/{100*drift[2]:+.2f}pp")

    grid = []
    for thr in [0.0, 0.05, 0.10, 0.15]:
        grid.append((f"EV>{thr:g} all", None, thr, None))
        grid.append((f"EV>{thr:g} home-only", {0}, thr, None))
        grid.append((f"EV>{thr:g} no-draw", {0, 2}, thr, None))
    grid.append(("EV>0 away-only", {2}, 0.0, None))
    grid.append(("EV>0 draw-only", {1}, 0.0, None))
    grid.append(("EV>0 & odds 2.2-3.2", None, 0.0, (2.2, 3.2)))

    print(f"\n{'slice':<28}{'n':>5}{'ROI':>8}{'ROI 90% CI':>18}{'CLVexc':>8}{'CLVexc 90% CI':>18}")
    for label, pf, thr, band in grid:
        roi, clvx = _slice(sample, drift, pf, thr, band)
        if not roi:
            continue
        rlo, rhi = boot_ci(roi)
        clo, chi = boot_ci(clvx)
        print(f"{label:<28}{len(roi):>5}{100*np.mean(roi):>7.1f}%"
              f"  [{100*rlo:>+6.1f}%,{100*rhi:>+6.1f}%]{np.mean(clvx):>8.2f}"
              f"  [{clo:>+6.2f},{chi:>+6.2f}]")

    print("\ncalibration (pooled outcome slots): bucket   n  model_mean  realized")
    rows = sorted((r["m"][k], 1.0 if r["res"] == k else 0.0) for r in sample for k in range(3))
    for lo, hi in [(0, .15), (.15, .25), (.25, .35), (.35, .45), (.45, .6), (.6, 1.0)]:
        b = [(m, y) for m, y in rows if lo <= m < hi]
        if b:
            print(f"  {lo:.2f}-{hi:.2f}  {len(b):>4}  {np.mean([m for m, _ in b]):.3f}"
                  f"      {np.mean([y for _, y in b]):.3f}")

    mv = [abs(cs.devig(r[CLOSE_KEY])[0][k] - cs.devig(r["open"])[0][k])
          for r in sample for k in range(3)]
    print(f"\nmean |open->close| devigged move per slot: {100*np.mean(mv):.2f}pp "
          "(the prize if a movement predictor is found)")


if __name__ == "__main__":
    main()
