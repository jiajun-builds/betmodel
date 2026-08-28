"""Which opening line is the better market anchor for the draw de-bias — Pinnacle or 1xBet?

The market-anchored λ de-bias shrinks the model's draw prob toward a reference
book's no-vig OPENING draw prob m_D, returning freed mass to home/away pro-rata.
This asks: does anchoring on Pinnacle's open or 1xBet's open give a better
de-biased 1X2 forecast? "Better" = closer to reality → lower multiclass log-loss
and Brier score, and a draw prob nearer the actual rate.

Held fixed: the same NegBinom walk-forward model and the same fixtures carrying BOTH
books' opening lines (2024 + 2025 + 2026, the seasons the user backfilled). Only the
anchor book varies.

Reproduce (repo root): PYTHONPATH=src python backtest/anchor_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_1x2 import (  # noqa: E402
    MIN_TRAIN,
    PROD_LOOKBACK_MONTHS,
    PROD_XI,
    fit_draw_delta,
    fit_model,
    one_x_two,
)
from csl.date_utils import parse_date_only_series  # noqa: E402
from csl.models.continuous_poisson import ContinuousPoissonGoalModel  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")
PINN = ["pinnacle_open_h", "pinnacle_open_d", "pinnacle_open_a"]
ONEX = ["onexbet_open_h", "onexbet_open_d", "onexbet_open_a"]
res_map = {"H": 0, "D": 1, "A": 2}
LAM = 0.75


def load():
    df = pd.read_csv(CSV)
    df["Date"] = parse_date_only_series(df["Date"])
    for c in ["HExpG+", "AExpG+", "HG", "AG", *PINN, *ONEX]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str); df["Away"] = df["Away"].astype(str)
    df["res"] = df["Res"].map(res_map)
    return df.sort_values("Date").reset_index(drop=True)


def walk_forward(df):
    graded = df[df["res"].notna()
                & df[PINN].notna().all(axis=1)
                & df[ONEX].notna().all(axis=1)].copy()
    recs = []
    for date in sorted(graded["Date"].unique()):
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < MIN_TRAIN or test.empty:
            continue
        w = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        try:
            clf = fit_model(ContinuousPoissonGoalModel, hist, w)
            delta = fit_draw_delta(clf, hist, w)
        except Exception:
            continue
        for r in test.itertuples(index=False):
            try:
                p = one_x_two(np.asarray(clf.predict(r.Home, r.Away).grid))
            except Exception:
                continue
            recs.append({
                "res": int(r.res), "pH": p[0], "pD": p[1], "pA": p[2], "delta": delta,
                "P": [float(getattr(r, c)) for c in PINN],
                "X": [float(getattr(r, c)) for c in ONEX],
            })
    return pd.DataFrame(recs)


def novig(O):
    inv = 1.0 / O
    return inv / inv.sum(1, keepdims=True)


def anchor_lambda(P, mD, lam=LAM):
    pD = (1 - lam) * P[:, 1] + lam * mD
    scale = (1 - pD) / (1 - P[:, 1])
    return np.column_stack([P[:, 0] * scale, pD, P[:, 2] * scale])


def delta_debias(P, delta):
    d = P[:, 1]; z = 1 - d + delta * d
    return np.column_stack([P[:, 0] / z, delta * d / z, P[:, 2] / z])


def metrics(name, P, res):
    idx = np.arange(len(res))
    ll = -np.log(np.clip(P[idx, res], 1e-9, None)).mean()
    onehot = np.eye(3)[res]
    brier = ((P - onehot) ** 2).sum(1).mean()
    draw = P[:, 1].mean()
    # draw calibration: mean predicted draw vs actual, and draw-only log-loss
    return name, ll, brier, draw


def main():
    b = walk_forward(load())
    if b.empty:
        raise SystemExit("no fixtures")
    res = b["res"].values
    P = b[["pH", "pD", "pA"]].values
    delta = b["delta"].values
    Opinn = novig(np.array(b["P"].tolist()))
    Oonex = novig(np.array(b["X"].tolist()))
    actual_draw = (res == 1).mean()

    rows = [
        metrics("market-only: Pinnacle open", Opinn, res),
        metrics("market-only: 1xBet open", Oonex, res),
        metrics("NegBinom raw", P, res),
        metrics("NegBinom + delta-cal", delta_debias(P, delta), res),
        metrics(f"NegBinom + lam={LAM} @ Pinnacle open", anchor_lambda(P, Opinn[:, 1]), res),
        metrics(f"NegBinom + lam={LAM} @ 1xBet open", anchor_lambda(P, Oonex[:, 1]), res),
    ]

    print(f"\nAnchor test — n={len(b)} fixtures (seasons carrying both books' opens: "
          f"2024+2025+2026), actual draw rate {actual_draw:.3f}")
    print(f"per-fixture |Pinn draw − 1xBet draw| mean: {np.abs(Opinn[:,1]-Oonex[:,1]).mean():.4f} "
          f"(max {np.abs(Opinn[:,1]-Oonex[:,1]).max():.4f})")
    print(f"\n  {'forecast':<38} {'logloss':>8} {'Brier':>7} {'draw':>6} {'|draw−act|':>10}")
    print("  " + "-" * 74)
    for name, ll, brier, draw in rows:
        print(f"  {name:<38} {ll:>8.4f} {brier:>7.4f} {draw:>6.3f} {abs(draw-actual_draw):>10.4f}")
    print("\nLower log-loss / Brier = better forecast. The de-bias only moves the DRAW toward the"
          "\nanchor's draw prob and keeps the model's own home/away split, so the anchor choice"
          "\nmatters only as much as the two books' no-vig opening DRAW probs differ.")


if __name__ == "__main__":
    main()
