"""Re-sweep time-decay xi and the training window on CALIBRATION, post-truncation-fix.

Every previous xi/lookback conclusion (model comparison/xi_lookback_grid_test.py)
was tuned against a fit whose lambda was 27% too low, so it is void. This re-runs
the grid against the corrected ContinuousPoissonGoalModel and ranks on calibration
(draw bias, AH-ladder bias, log-loss) rather than RPS, which is near-blind to the
uniform grid shift this bug caused.

Run (repo root):  PYTHONPATH=src python backtest/sweep_xi_lookback.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.models.continuous_poisson import ContinuousPoissonGoalModel  # noqa: E402
from csl.models.dc import MIN_TRAIN  # noqa: E402
from penaltyblog.models import dixon_coles_weights  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_truncation_fix import AH_LINES, ah_cover, load_matches, one_x_two  # noqa: E402

XI_GRID = (0.0005, 0.001, 0.002, 0.004)
LOOKBACK_GRID = (12, 18, 24, 30)


def evaluate(df: pd.DataFrame, xi: float, lookback: int) -> dict | None:
    graded = df[df.res.notna() & df["HExpG+"].notna()]
    rows = []
    for date in sorted(graded.Date.unique()):
        date = pd.Timestamp(date)
        test = graded[graded.Date == date]
        hist = df[
            (df.Date < date) & (df.Date >= date - pd.DateOffset(months=lookback))
        ].dropna(subset=["HExpG+", "AExpG+"])
        if len(hist) < MIN_TRAIN or test.empty:
            continue
        w = dixon_coles_weights(hist["Date"], xi=xi)
        try:
            clf = ContinuousPoissonGoalModel(
                hist["HExpG+"], hist["AExpG+"], hist["Home"], hist["Away"], w
            ).fit()
        except Exception:
            continue
        for r in test.itertuples(index=False):
            try:
                pred = clf.predict(r.Home, r.Away)
            except KeyError:
                continue
            grid = np.asarray(pred.grid)
            p = one_x_two(grid)
            rec = {"res": int(r.res), "pH": p[0], "pD": p[1], "pA": p[2],
                   "lam": pred.home_goal_expectation + pred.away_goal_expectation,
                   "tot": r.HG + r.AG}
            for line in AH_LINES:
                rec[f"ah{line}"] = ah_cover(grid, line)
                margin = r.HG - r.AG + line
                rec[f"ahres{line}"] = np.nan if margin == 0 else float(margin > 0)
            rows.append(rec)

    if len(rows) < 100:
        return None
    R = pd.DataFrame(rows)
    P = R[["pH", "pD", "pA"]].to_numpy()
    k = R.res.to_numpy(int)
    onehot = np.eye(3)[k]
    bias = (P - onehot).mean(0)
    ah = [abs((R[f"ah{l}"][R[f"ahres{l}"].notna()] - R[f"ahres{l}"][R[f"ahres{l}"].notna()]).mean())
          for l in AH_LINES]
    return {
        "n": len(R),
        "logloss": -np.log(np.clip(P[np.arange(len(k)), k], 1e-12, None)).mean(),
        "rps": (((P.cumsum(1) - onehot.cumsum(1))[:, :2] ** 2).sum(1) / 2).mean(),
        "draw_bias": bias[1] * 100,
        "draw_t": stats.ttest_1samp(P[:, 1] - onehot[:, 1], 0).statistic,
        "mean_abs_1x2": np.abs(bias).mean() * 100,
        "ah_bias": float(np.mean(ah)) * 100,
        "goals_gap": R.lam.mean() - R.tot.mean(),
    }


def main() -> int:
    df = load_matches()
    results = {}
    print(f"{'xi':>8} {'look':>5} {'n':>5} {'logloss':>9} {'RPS':>8} {'draw':>8} "
          f"{'draw t':>7} {'|1X2|':>7} {'AH':>7} {'goalgap':>8}")
    print("-" * 82)
    for lookback in LOOKBACK_GRID:
        for xi in XI_GRID:
            r = evaluate(df, xi, lookback)
            if r is None:
                print(f"{xi:>8.4f} {lookback:>5}   (too few rows)")
                continue
            results[(xi, lookback)] = r
            print(f"{xi:>8.4f} {lookback:>5} {r['n']:>5} {r['logloss']:>9.4f} {r['rps']:>8.4f} "
                  f"{r['draw_bias']:>+7.2f}pp {r['draw_t']:>+7.2f} {r['mean_abs_1x2']:>6.2f}pp "
                  f"{r['ah_bias']:>6.2f}pp {r['goals_gap']:>+8.3f}")
            sys.stdout.flush()

    print("\nBEST BY EACH CRITERION (calibration first; RPS shown only for contrast):")
    for key, lower_better in (("logloss", True), ("ah_bias", True), ("mean_abs_1x2", True),
                              ("rps", True)):
        best = min(results.items(), key=lambda kv: kv[1][key])
        print(f"  {key:>13}: xi={best[0][0]:.4f} lookback={best[0][1]}  -> {best[1][key]:.4f}")
    best = min(results.items(), key=lambda kv: abs(kv[1]["draw_t"]))
    print(f"  {'|draw t|':>13}: xi={best[0][0]:.4f} lookback={best[0][1]}  -> {best[1]['draw_t']:+.2f}")
    print("\nGuardrail: if a winner sits on the edge of XI_GRID or LOOKBACK_GRID, widen the")
    print("grid and re-run — a boundary optimum is unconverged (the dispersion=1000.0 lesson).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
