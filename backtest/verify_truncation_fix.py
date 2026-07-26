"""Walk-forward acceptance check for the continuous-target Poisson fix.

Checks the four pre-registered must-hold criteria from the fix plan against the
**shipped** production recipe (``csl.models.dc.fit_production_model``), not a
prototype:

  1. Poisson score-equation ratio in [0.999, 1.001] at every refit.
  2. Mean fitted ``lambda_h + lambda_a`` within 0.25 of realized mean total goals.
  3. Draw-bias |t| < 2.0                (truncated fit: +4.20pp, t=+2.70).
  4. Mean |Asian-handicap ladder bias| < 2.0pp  (truncated fit: 2.61pp).

Deliberately reports **calibration** (per-outcome bias with t, AH-ladder bias,
log-loss) rather than leaning on RPS. RPS is nearly blind to a uniform shift of
the whole scoreline grid: this bug cost 27% of lambda and moved RPS by only
0.0031 (0.1952 -> 0.1921), which is why every previous RPS-driven parameter sweep
walked straight past it.

Run (repo root, env with penaltyblog):
    PYTHONPATH=src python backtest/verify_truncation_fix.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.date_utils import parse_date_only_series  # noqa: E402
from csl.models.dc import (  # noqa: E402
    MIN_TRAIN,
    PROD_LOOKBACK_MONTHS,
    PROD_XI,
    fit_production_model,
)
from csl.paths import data_raw_dir  # noqa: E402

AH_LINES = (-2.5, -1.5, -0.5, 0.5, 1.5)

# Pre-registered thresholds — do not loosen these to make a run pass.
SCORE_EQ_TOL = 1e-3
MAX_TOTAL_GOALS_GAP = 0.25
MAX_DRAW_ABS_T = 2.0
MAX_AH_LADDER_BIAS = 0.020


def one_x_two(grid: np.ndarray) -> np.ndarray:
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n))
    v = np.array([grid[diff > 0].sum(), grid[diff == 0].sum(), grid[diff < 0].sum()])
    return v / v.sum()


def ah_cover(grid: np.ndarray, line: float) -> float:
    """P(home covers `line`), pushes excluded — matches how AH settles."""
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n)) + line
    win, lose = grid[diff > 0].sum(), grid[diff < 0].sum()
    return win / (win + lose)


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_raw_dir(), "CHN_Super League.csv"))
    df["Date"] = parse_date_only_series(df["Date"])
    df = df.dropna(subset=["Home", "Away"]).copy()
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    for col in ("HExpG+", "AExpG+", "HG", "AG"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["res"] = np.where(df.HG > df.AG, 0, np.where(df.HG == df.AG, 1, 2))
    df.loc[df.HG.isna() | df.AG.isna(), "res"] = np.nan
    return df


def walk_forward(df: pd.DataFrame):
    graded = df[df.res.notna() & df["HExpG+"].notna()].copy()
    rows, score_eq, deltas = [], [], []

    for date in sorted(graded.Date.unique()):
        date = pd.Timestamp(date)
        test = graded[graded.Date == date]
        hist = df[
            (df.Date < date) & (df.Date >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS))
        ].dropna(subset=["HExpG+", "AExpG+"])
        if len(hist) < MIN_TRAIN or test.empty:
            continue

        model = fit_production_model(hist, xi=PROD_XI)
        score_eq.append(model._clf.score_equation_ratio())
        deltas.append(model.draw_delta)

        for r in test.itertuples(index=False):
            try:
                raw = model.predict_raw(r.Home, r.Away)
                cal = model.predict(r.Home, r.Away)
            except KeyError:
                continue  # team promoted into this round; no strength estimate yet
            g_raw = np.asarray(raw.grid)
            p_raw, p_cal = one_x_two(g_raw), one_x_two(np.asarray(cal.grid))
            rec = {
                "res": int(r.res),
                "Season": int(r.Season),
                "delta_fit": model.draw_delta,
                "lam_h": raw.home_goal_expectation,
                "lam_a": raw.away_goal_expectation,
                "HG": r.HG,
                "AG": r.AG,
            }
            for tag, p in (("raw", p_raw), ("cal", p_cal)):
                rec[f"pH_{tag}"], rec[f"pD_{tag}"], rec[f"pA_{tag}"] = p
            for line in AH_LINES:
                rec[f"ah{line}"] = ah_cover(g_raw, line)
                margin = r.HG - r.AG + line
                rec[f"ahres{line}"] = np.nan if margin == 0 else float(margin > 0)
            rows.append(rec)

    return pd.DataFrame(rows), np.array(score_eq), np.array(deltas)


def report_outcomes(R: pd.DataFrame, tag: str, label: str) -> float:
    P = R[[f"pH_{tag}", f"pD_{tag}", f"pA_{tag}"]].to_numpy()
    k = R.res.to_numpy(int)
    onehot = np.eye(3)[k]
    logloss = -np.log(np.clip(P[np.arange(len(k)), k], 1e-12, None)).mean()
    rps = (((P.cumsum(1) - onehot.cumsum(1))[:, :2] ** 2).sum(1) / 2).mean()

    print(f"\n  {label}")
    draw_t = None
    for j, name in enumerate(("Home", "Draw", "Away")):
        bias = P[:, j] - onehot[:, j]
        t = stats.ttest_1samp(bias, 0)
        print(f"    {name:5s} bias {bias.mean() * 100:+.2f}pp   t={t.statistic:+.2f}")
        if name == "Draw":
            draw_t = t.statistic
    print(f"    log-loss {logloss:.4f}    RPS {rps:.4f}  (RPS is near-blind to this bug)")
    return abs(draw_t)


def main() -> int:
    R, score_eq, deltas = walk_forward(load_matches())
    if R.empty:
        print("no walk-forward rows produced — check the input CSV")
        return 1

    print("\n" + "=" * 72)
    print(f"WALK-FORWARD ACCEPTANCE  ({len(score_eq)} refits, n={len(R)} graded fixtures)")
    print("=" * 72)

    eq_min, eq_max = score_eq.min(), score_eq.max()
    print(f"\n  score equation sum(w*lambda)/sum(w*y): [{eq_min:.6f}, {eq_max:.6f}]")
    total_model = R.lam_h.mean() + R.lam_a.mean()
    total_actual = R.HG.mean() + R.AG.mean()
    print(f"  lambda_h {R.lam_h.mean():.4f} vs HG {R.HG.mean():.4f}")
    print(f"  lambda_a {R.lam_a.mean():.4f} vs AG {R.AG.mean():.4f}")
    print(f"  total goals: model {total_model:.3f} vs actual {total_actual:.3f} "
          f"(gap {total_model - total_actual:+.3f})")
    print(f"  draw delta APPLIED: mean {deltas.mean():.3f}  "
          f"range [{deltas.min():.3f}, {deltas.max():.3f}]")
    print("    (gated to 1.0 by DRAW_DELTA_SHRINK — the fitted value does not survive")
    print("     a per-season check; see the constant's comment in csl.models.dc)")

    draw_abs_t = report_outcomes(R, "raw", "RAW grid (pre-delta):")
    report_outcomes(R, "cal", "delta-CALIBRATED grid (what production ships anchorless):")

    print("\n  AH ladder, raw grid (model cover - realized):")
    biases = []
    for line in AH_LINES:
        ok = R[f"ahres{line}"].notna()
        bias = R[f"ah{line}"][ok] - R[f"ahres{line}"][ok]
        t = stats.ttest_1samp(bias, 0)
        biases.append(abs(bias.mean()))
        print(f"    {line:+.1f}: {bias.mean() * 100:+.2f}pp   t={t.statistic:+.2f}   n={ok.sum()}")
    ah_mean = float(np.mean(biases))
    print(f"    mean |bias| = {ah_mean * 100:.2f}pp")
    print("    NB: AH +-0.5 are algebraically 1X2 home/away with draws dropped, and the")
    print("    outer tiers have small variance — large t off a small bias is the same")
    print("    matches seen at low variance, not independent evidence.")

    checks = [
        ("score equation in [0.999, 1.001] every refit",
         abs(eq_min - 1.0) < SCORE_EQ_TOL and abs(eq_max - 1.0) < SCORE_EQ_TOL,
         f"[{eq_min:.6f}, {eq_max:.6f}]"),
        (f"|total goals gap| < {MAX_TOTAL_GOALS_GAP}",
         abs(total_model - total_actual) < MAX_TOTAL_GOALS_GAP,
         f"{total_model - total_actual:+.3f}"),
        (f"draw-bias |t| < {MAX_DRAW_ABS_T}", draw_abs_t < MAX_DRAW_ABS_T, f"{draw_abs_t:.2f}"),
        (f"mean |AH ladder bias| < {MAX_AH_LADDER_BIAS * 100:.1f}pp",
         ah_mean < MAX_AH_LADDER_BIAS, f"{ah_mean * 100:.2f}pp"),
    ]
    print("\n" + "-" * 72)
    failed = 0
    for name, ok, actual in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ->  {actual}")
        failed += not ok
    print("-" * 72)
    print(f"\n{len(checks) - failed}/{len(checks)} pre-registered criteria hold")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
