"""Second idea: Polymarket CLOSE as a ~0-vig EXECUTION venue (not a CLV target).

`backtest_polymarket.py` showed betting Polymarket's OPEN is a flat-seed artifact.
This script tests the mirror idea: decouple the probability source from the execution
price and *bet at the Polymarket close* (overround ≈0.14% — the vig wall vanishes,
bar≈0). Because you execute at the close there is no "CLV vs close"; the honest judges
are realized ROI and calibration.

The paradox this must resolve: a 0-vig close is usually EFFICIENT, so EV≈0. The whole
question is whether Polymarket's niche-league CSL close is cheap-and-sharp (unbeatable)
or cheap-but-soft (beatable). Calibration answers it directly.

Three things are computed on the same fixture set (Pinnacle open+close + Polymarket
close + result; model needs the Pinnacle-open anchor, so all share that sample):

  CALIBRATION  — Brier + log-loss of three probability sources vs actual results:
       model (prod de-biased) | Pinnacle close no-vig | Polymarket close no-vig.
     If Polymarket-close loss ≈ Pinnacle-close loss, the close is sharp → both bets die.
     If it is meaningfully worse, the close is soft → there is something to hit.

  VARIANT (a)  Model vs Polymarket close — pick argmax(model_P × poly_close_odds − 1),
     bet above an EV threshold, grade by realized ROI. Pure model edge, vig-free venue.

  VARIANT (b)  Pinnacle close vs Polymarket close (NO model) — treat Pinnacle's no-vig
     close as sharp fair value; bet the outcome where it implies Polymarket's close is
     underpricing. "Use the sharp book to pick off the cheap-but-soft book." Cleanest
     test of the ~0-vig venue; independent of whether our model is any good.

Caveats printed at the end (single-season weight, close-time alignment, fill depth).

Reproduce (repo root): PYTHONPATH=src python backtest/backtest_polymarket_close.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_1x2 import (  # noqa: E402  reuse the validated harness
    MIN_TRAIN,
    PROD_LOOKBACK_MONTHS,
    PROD_XI,
    _t,
    fit_draw_delta,
    fit_model,
    one_x_two,
)
from backtest_1xbet import apply_debias  # noqa: E402  validated de-bias
from backtest_polymarket import novig  # noqa: E402  shared no-vig helper
from csl.date_utils import parse_date_only_series  # noqa: E402
from csl.models.continuous_poisson import ContinuousPoissonGoalModel  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

PIN_OPEN = ["pinnacle_open_h", "pinnacle_open_d", "pinnacle_open_a"]     # model anchor
PIN_CLOSE = ["pinnacle_close_h", "pinnacle_close_d", "pinnacle_close_a"]  # sharp ref (b)
POLY_CLOSE = ["polymarket_close_h", "polymarket_close_d", "polymarket_close_a"]  # exec price
res_map = {"H": 0, "D": 1, "A": 2}
OUTCOME = ["home", "draw", "away"]
EV_THRESHOLDS = [0.00, 0.05, 0.10, 0.20]
PROD_MODE = "pinnacle_anchor"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Date"] = parse_date_only_series(df["Date"])
    for c in [*PIN_OPEN, *PIN_CLOSE, *POLY_CLOSE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["res"] = df["Res"].map(res_map)
    return df.sort_values("Date").reset_index(drop=True)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    need = [*PIN_OPEN, *PIN_CLOSE, *POLY_CLOSE]
    graded = df[df["res"].notna() & df[need].notna().all(axis=1)].copy()
    recs = []
    for date in sorted(graded["Date"].unique()):
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < MIN_TRAIN or test.empty:
            continue
        weights = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        try:
            clf = fit_model(ContinuousPoissonGoalModel, hist, weights)
            delta = fit_draw_delta(clf, hist, weights)
        except Exception:
            continue
        for r in test.itertuples(index=False):
            try:
                p = one_x_two(np.asarray(clf.predict(r.Home, r.Away).grid))
            except Exception:
                continue
            rec = {"date": date, "Season": int(r.Season), "res": int(r.res),
                   "pH": p[0], "pD": p[1], "pA": p[2], "delta": delta}
            for cols in (PIN_OPEN, PIN_CLOSE, POLY_CLOSE):
                for c in cols:
                    rec[c] = float(getattr(r, c))
            recs.append(rec)
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)


def model_probs(b):
    P_raw = b[["pH", "pD", "pA"]].values
    return apply_debias(P_raw, novig(b, PIN_OPEN), PROD_MODE, b)


def brier_logloss(P, y):
    onehot = np.eye(3)[y]
    brier = ((P - onehot) ** 2).sum(1).mean()
    p_actual = np.clip(P[np.arange(len(y)), y], 1e-12, 1)
    logloss = -np.log(p_actual).mean()
    return brier, logloss


def calibration(b):
    y = b["res"].values
    srcs = {
        "model (prod de-biased)": model_probs(b),
        "Pinnacle close no-vig": novig(b, PIN_CLOSE),
        "Polymarket close no-vig": novig(b, POLY_CLOSE),
    }
    print(f"\n{'=' * 78}\nCALIBRATION  |  n={len(b)} fixtures, lower is better\n{'=' * 78}")
    print(f"  {'probability source':<26} {'Brier':>8} {'log-loss':>10}")
    for name, P in srcs.items():
        br, ll = brier_logloss(P, y)
        print(f"  {name:<26} {br:>8.4f} {ll:>10.4f}")
    qp = novig(b, PIN_CLOSE); qm = novig(b, POLY_CLOSE)
    print(f"\n  mean |Polymarket-close − Pinnacle-close| per-outcome prob: "
          f"{np.abs(qp - qm).mean():.4f}")
    print("  (near Pinnacle's loss ⇒ close is sharp/unbeatable; clearly worse ⇒ soft/beatable)")


def grade(prob_for_ev, O, y, thr, *, allow_draw):
    ev = prob_for_ev * O - 1.0
    ev_pick = ev.copy()
    if not allow_draw:
        ev_pick[:, 1] = -np.inf
    pick = ev_pick.argmax(1)
    m = ev_pick.max(1) > thr
    idx = np.arange(len(y))
    win = pick == y
    pl = np.where(win, O[idx, pick] - 1.0, -1.0)
    return pick, m, win, pl


def report_variant(title, prob_for_ev, b):
    O = b[POLY_CLOSE].values
    y = b["res"].values
    seasons = sorted(int(s) for s in b["Season"].unique())
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for allow_draw in (True, False):
        tag = "WITH draw" if allow_draw else "NO draw  "
        print(f"\n  [{tag}]  {'thr':>4} {'scope':>7} {'n':>4} {'draws':>5} "
              f"{'ROI':>8} {'t':>6} {'win%':>5} {'avgOdds':>7}")
        for thr in EV_THRESHOLDS:
            for scope, sub in [("pooled", b)] + [
                    (str(s), b[b["Season"] == s].reset_index(drop=True))
                    for s in seasons]:
                Osub = sub[POLY_CLOSE].values
                ysub = sub["res"].values
                pev = (prob_for_ev if scope == "pooled"
                       else prob_for_ev[b["Season"].values == int(scope)])
                pick, m, win, pl = grade(pev, Osub, ysub, thr, allow_draw=allow_draw)
                if m.sum() < 2:
                    continue
                ndraw = int((pick[m] == 1).sum())
                avg_odds = Osub[np.arange(len(ysub))[m], pick[m]].mean()
                print(f"  {'':>11} {thr:>4.2f} {scope:>7} {int(m.sum()):>4} {ndraw:>5} "
                      f"{pl[m].mean():>+8.3f} {_t(pl[m]):>+6.2f} {win[m].mean():>5.0%} "
                      f"{avg_odds:>7.2f}")


def main():
    df = load()
    b = walk_forward(df)
    if b.empty:
        raise SystemExit("No gradeable fixtures (need Pinnacle open+close + Polymarket close).")
    print(f"\nSAMPLE n={len(b)}, seasons={sorted(int(s) for s in b['Season'].unique())} "
          f"({dict(b.groupby('Season').size())})")
    calibration(b)
    report_variant("VARIANT (a)  Model  vs  Polymarket close   (execution price = Poly close)",
                   model_probs(b), b)
    report_variant("VARIANT (b)  Pinnacle close  vs  Polymarket close   (no model)",
                   novig(b, PIN_CLOSE), b)
    print(f"\n{'=' * 78}\nReading guide & caveats\n{'=' * 78}")
    print("  Execution price is the Polymarket CLOSE, so bar≈0 (R≈0.14%) — no vig wall to")
    print("  clear; the judge is realized ROI (and calibration above). A single Yes-binary")
    print("  bet pays the SPREAD, not the 3-way overround, so ~0-vig is the right cost frame.")
    print("  CAVEATS: sample is ~90% 2026 (2025 tail tiny) — not a cross-season result;")
    print("  assumes fills at the last close price (real cost = book depth/spread at kick);")
    print("  (b) assumes Pinnacle close is observable at the moment you hit Polymarket.")


if __name__ == "__main__":
    main()
