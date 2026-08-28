"""Calibration acceptance report for the goals model — the primary quality gate.

RPS is close to blind to the failure that actually matters for betting. The
integer-truncation bug (models/continuous_poisson) shifted the whole scoreline
grid toward low scores: it moved RPS by 0.0004 while leaving the draw 5.7pp too
high and Asian-handicap cover probabilities 6-8pp light on the home-giving lines.
An RPS-driven sweep will happily re-introduce that class of error. This report
measures the thing a bettor is exposed to — whether a stated probability is the
frequency that actually occurs.

Everything is walk-forward and out-of-sample: refit each ISO week on the trailing
window via dc.fit_production_model, predict that week, never look forward.

    python -m ligamx.eval.calibration_report [--test-start 2025-10-01] [--alpha 1.0]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ligamx.models.dc import (
    AH_LINES,
    DEFAULT_TARGET,
    DRAW_CALIBRATION_ALPHA,
    MODEL_XI,
    SHRINKAGE_K,
    TRAINING_WINDOW_MONTHS,
    calibrate_1x2,
    fit_production_model,
)
from ligamx.eval.clv_backtest import _load

warnings.filterwarnings("ignore")

RELIABILITY_EDGES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 1.01]


def settle_home(gd: float, line: float) -> float:
    """Fraction of a home-side stake returned at Asian handicap ``line``.

    Quarter lines split the stake across the two adjacent half-lines, so a
    half-win returns 0.75 and a half-loss 0.25.
    """
    if (line * 4) % 2 != 0:
        return 0.5 * settle_home(gd, line - 0.25) + 0.5 * settle_home(gd, line + 0.25)
    margin = gd + line
    if margin > 0:
        return 1.0
    return 0.5 if margin == 0 else 0.0


def collect(df: pd.DataFrame, test_start: str, alpha: float, xi: float = MODEL_XI,
            window: int = TRAINING_WINDOW_MONTHS, shrinkage_k: float = SHRINKAGE_K,
            target: tuple[str, str] = DEFAULT_TARGET, min_train: int = 80) -> pd.DataFrame:
    """Walk-forward predictions: one row per test match, model + realized.

    Defaults are the production settings; the overrides let a sweep candidate be
    judged on calibration rather than on RPS alone.
    """
    gh, ga = target
    fit_df = df.dropna(subset=[gh, ga])
    test = df[df["Date"] >= pd.Timestamp(test_start)].copy()
    test["_wk"] = test["Date"].dt.strftime("%G-W%V")

    rows = []
    for wk in sorted(test["_wk"].unique()):
        wk_rows = test[test["_wk"] == wk]
        cutoff = wk_rows["Date"].min()
        lo = cutoff - pd.DateOffset(months=window)
        train = fit_df[(fit_df["Date"] < cutoff) & (fit_df["Date"] >= lo)]
        if len(train) < min_train:
            continue
        try:
            model = fit_production_model(train, target=target, xi=xi,
                                         shrinkage_k=shrinkage_k, shrink=shrinkage_k > 0)
        except Exception:
            continue
        teams = set(model.teams)
        for _, mt in wk_rows.iterrows():
            if mt["Home"] not in teams or mt["Away"] not in teams:
                continue
            grid = model.predict(mt["Home"], mt["Away"])
            lam_h, lam_a = model.lambdas(mt["Home"], mt["Away"])
            ph, pd_, pa = calibrate_1x2(*grid.home_draw_away, alpha=alpha)
            rec = {
                "gd": mt["HG"] - mt["AG"], "hg": mt["HG"], "ag": mt["AG"],
                "lam_h": lam_h, "lam_a": lam_a,
                "p_home": ph, "p_draw": pd_, "p_away": pa,
                "score_p": float(grid.exact_score(int(mt["HG"]), int(mt["AG"]))),
            }
            for line in AH_LINES:
                probs = grid.asian_handicap_probs("home", line)
                rec[f"ah{line}"] = probs["win"] + 0.5 * probs["push"]
            rows.append(rec)
    return pd.DataFrame(rows)


def _paired(model_vals: np.ndarray, realized: np.ndarray) -> tuple[float, float, float]:
    """Mean bias, its standard error, and t — paired per match."""
    d = np.asarray(model_vals, dtype=float) - np.asarray(realized, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return float(d.mean()), float(se), float(d.mean() / se) if se else float("nan")


def report(R: pd.DataFrame, alpha: float) -> None:
    n = len(R)
    print(f"\n{'='*72}\nCALIBRATION REPORT   n = {n} walk-forward matches   "
          f"draw alpha = {alpha:g}\n{'='*72}")

    print("\n-- scoring rate (lambda scale self-check) --")
    print(f"  model lambda   home {R.lam_h.mean():.3f}   away {R.lam_a.mean():.3f}")
    print(f"  realized goals home {R.hg.mean():.3f}   away {R.ag.mean():.3f}")

    print("\n-- 1X2 --")
    print(f"{'outcome':>10}{'model':>9}{'realized':>10}{'bias':>9}{'SE':>7}{'t':>7}")
    for name, col, hit in (("home", "p_home", R.gd > 0),
                           ("draw", "p_draw", R.gd == 0),
                           ("away", "p_away", R.gd < 0)):
        b, se, t = _paired(R[col], hit.astype(float))
        print(f"{name:>10}{R[col].mean():>9.4f}{hit.mean():>10.4f}"
              f"{100*b:>+8.2f}pp{100*se:>6.2f}{t:>7.2f}")

    print("\n-- Asian handicap ladder (home side, expected stake return) --")
    print(f"{'line':>7}{'model':>9}{'realized':>10}{'bias':>9}{'SE':>7}{'t':>7}")
    biases = []
    for line in AH_LINES:
        realized = R.gd.apply(lambda g: settle_home(g, line)).to_numpy()
        b, se, t = _paired(R[f"ah{line}"], realized)
        print(f"{line:>+7g}{R[f'ah{line}'].mean():>9.4f}{realized.mean():>10.4f}"
              f"{100*b:>+8.2f}pp{100*se:>6.2f}{t:>7.2f}")
        biases.append(abs(b))
    print(f"  mean |bias| across the ladder: {100*np.mean(biases):.2f}pp")

    print("\n-- reliability (all three 1X2 outcomes pooled) --")
    preds = np.concatenate([R.p_home, R.p_draw, R.p_away])
    hits = np.concatenate([(R.gd > 0), (R.gd == 0), (R.gd < 0)]).astype(float)
    print(f"{'bucket':>14}{'n':>6}{'model':>9}{'realized':>10}{'bias':>9}")
    for lo, hi in zip(RELIABILITY_EDGES, RELIABILITY_EDGES[1:]):
        m = (preds >= lo) & (preds < hi)
        if m.sum() == 0:
            continue
        print(f"{f'[{lo:.2f},{hi:.2f})':>14}{int(m.sum()):>6}{preds[m].mean():>9.4f}"
              f"{hits[m].mean():>10.4f}{100*(preds[m].mean()-hits[m].mean()):>+8.2f}pp")

    ll = -np.log(np.clip(R.score_p, 1e-12, None)).mean()
    print(f"\n-- scoreline log-loss (direct target of the lambda scale): {ll:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--alpha", type=float, default=DRAW_CALIBRATION_ALPHA,
                    help="draw deflation; pass 1.0 to see the raw model")
    ap.add_argument("--xi", type=float, default=MODEL_XI)
    ap.add_argument("--window", type=int, default=TRAINING_WINDOW_MONTHS)
    ap.add_argument("--k", type=float, default=SHRINKAGE_K, help="shrinkage K (0 disables)")
    ap.add_argument("--target", default="xg", choices=["xg", "goals"],
                    help="training target: the xG blend (production) or raw goals")
    args = ap.parse_args()
    target = DEFAULT_TARGET if args.target == "xg" else ("HG", "AG")

    df = _load()
    df["HG"] = pd.to_numeric(df["HG"], errors="coerce")
    df["AG"] = pd.to_numeric(df["AG"], errors="coerce")
    df = df.dropna(subset=["HG", "AG"])
    R = collect(df, args.test_start, args.alpha, xi=args.xi, window=args.window,
                shrinkage_k=args.k, target=target)
    if R.empty:
        print("no walk-forward predictions produced")
        return
    print(f"config: target={'/'.join(target)}  xi={args.xi:g}  "
          f"window={args.window}mo  K={args.k:g}")
    report(R, args.alpha)


if __name__ == "__main__":
    main()
