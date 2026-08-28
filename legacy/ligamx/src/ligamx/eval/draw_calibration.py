"""Select and validate the 1X2 draw-deflation factor (dc.DRAW_CALIBRATION_ALPHA).

Kept as a guard, not as a live correction: alpha is 1.0 since the continuous-target
fix removed the draw over-pricing it used to offset. This deflates the draw
probability by alpha (home/away renormalized) and picks alpha by RPS on a
calibration period, then reports RPS on a held-out out-of-sample period so the
choice can be seen to hold up rather than overfit. Also prints the raw model-vs-
realized draw rate on each period.

    python -m ligamx.eval.draw_calibration [--cal-start 2024-07-01]
        [--cal-end 2025-09-30] [--test-start 2025-10-01]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ligamx.models.dc import DRAW_CALIBRATION_ALPHA, calibrate_1x2
from ligamx.eval import clv_signals as cs
from ligamx.eval.rps_backtest import RESULT_IDX, rps

warnings.filterwarnings("ignore")

ALPHAS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-start", default="2024-07-01")
    ap.add_argument("--cal-end", default="2025-09-30")
    ap.add_argument("--test-start", default="2025-10-01")
    args = ap.parse_args()

    df = cs.load()
    # Fit from the calibration start so the calibration period is itself out-of-sample.
    probs = cs.model_probs(df, test_start=args.cal_start)

    cal_idx = [i for i in probs
               if pd.Timestamp(args.cal_start) <= df.loc[i, "Date"] <= pd.Timestamp(args.cal_end)]
    test_idx = [i for i in probs if df.loc[i, "Date"] >= pd.Timestamp(args.test_start)]
    print(f"production alpha={DRAW_CALIBRATION_ALPHA}   |   calibration n={len(cal_idx)}   "
          f"test(oos) n={len(test_idx)}")

    for label, idxs in [("CALIBRATION", cal_idx), ("TEST (out-of-sample)", test_idx)]:
        res = [RESULT_IDX[df.loc[i, "Res"]] for i in idxs]
        md = np.mean([probs[i][1] for i in idxs]) if idxs else float("nan")
        rd = np.mean([1.0 if r == 1 else 0.0 for r in res]) if res else float("nan")
        print(f"\n[{label}]  model draw {md:.3f}  vs realized {rd:.3f}")
        print(f"{'alpha':>7}{'RPS':>9}{'draw_mean':>11}")
        best = None
        for a in ALPHAS:
            cal = [calibrate_1x2(*probs[i], a) for i in idxs]
            r = np.mean([rps(cal[j], res[j]) for j in range(len(idxs))]) if idxs else float("nan")
            dm = np.mean([c[1] for c in cal]) if cal else float("nan")
            print(f"{a:>7.2f}{r:>9.4f}{dm:>11.3f}")
            if best is None or r < best[1]:
                best = (a, r)
        print(f"  -> RPS-best alpha on {label.split()[0].lower()}: {best[0]:.2f}  (RPS {best[1]:.4f})")


if __name__ == "__main__":
    main()
