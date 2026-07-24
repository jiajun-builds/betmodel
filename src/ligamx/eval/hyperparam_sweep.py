"""
Hyperparameter sweep (RPS) for the goals model. Fixes the family to
NegativeBinomial (the RPS-best family from rps_backtest) and sweeps one axis at a
time via walk-forward, reporting RPS on the full test set and the Pinnacle-odds
subset. Coordinate descent: the best value of each axis is carried into the next.

Axes:
  A. Training target = xW * xG + (1 - xW) * goals   (xW: 0=pure goals .. 1=pure xG)
  B. Dixon-Coles time-decay xi
  C. Rolling training-window length (months)

    python -m ligamx.eval.hyperparam_sweep [--test-start 2024-07-01]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import penaltyblog as pb
from penaltyblog.models import dixon_coles_weights

from ligamx import paths
from ligamx.eval.rps_backtest import RESULT_IDX, _market_probs, log_loss, rps

warnings.filterwarnings("ignore")

DEFAULT_XW = 0.55
DEFAULT_XI = 0.0015
DEFAULT_WIN = 24


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(paths.ligamx_data_csv())
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["Res"] = df["Res"].astype(str).str.strip()
    for c in ["HxG", "AxG", "HG", "AG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (
        df[df["Res"].isin(["H", "D", "A"])]
        .dropna(subset=["HxG", "AxG", "HG", "AG"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def collect_ng(played: pd.DataFrame, x_weight: float, xi: float, window_months: int,
               test_start: str, min_train: int = 80) -> list:
    """Walk-forward with NegBinom on a blended target; return per-match records."""
    p = played.copy()
    p["th"] = x_weight * p["HxG"] + (1 - x_weight) * p["HG"]
    p["ta"] = x_weight * p["AxG"] + (1 - x_weight) * p["AG"]

    test = p[p["Date"] >= pd.Timestamp(test_start)].copy()
    test["week"] = test["Date"].dt.strftime("%G-W%V")

    recs = []
    for wk in sorted(test["week"].unique()):
        wm = test[test["week"] == wk]
        cutoff = wm["Date"].min()
        lo = cutoff - pd.DateOffset(months=window_months)
        train = p[(p["Date"] < cutoff) & (p["Date"] >= lo)]
        if len(train) < min_train:
            continue
        w = dixon_coles_weights(train["Date"], xi=xi)
        gh = train["th"].to_numpy(dtype=float, copy=True)
        ga = train["ta"].to_numpy(dtype=float, copy=True)
        try:
            m = pb.models.NegativeBinomialGoalModel(gh, ga, train["Home"], train["Away"], w)
            m.fit()
            teams = set(m.teams)
        except Exception:
            continue
        for _, mt in wm.iterrows():
            h, a = mt["Home"], mt["Away"]
            if h not in teams or a not in teams:
                continue
            try:
                pr = list(m.predict(h, a).home_draw_away)
            except Exception:
                continue
            recs.append({"oi": RESULT_IDX[mt["Res"]], "p": pr, "mkt": _market_probs(mt)})
    return recs


def score(recs: list):
    r = np.mean([rps(x["p"], x["oi"]) for x in recs]) if recs else float("nan")
    ll = np.mean([log_loss(x["p"], x["oi"]) for x in recs]) if recs else float("nan")
    mk = [x for x in recs if x["mkt"] is not None]
    rmk = np.mean([rps(x["p"], x["oi"]) for x in mk]) if mk else float("nan")
    mkt = np.mean([rps(x["mkt"], x["oi"]) for x in mk]) if mk else float("nan")
    return len(recs), r, ll, len(mk), rmk, mkt


def sweep(title: str, played: pd.DataFrame, configs: list, test_start: str):
    print(f"\n===== {title} =====")
    print(f"{'config':<26}{'n':>6}{'RPS':>9}{'LogLoss':>9}{'RPS_mkt':>9}{'Market':>8}")
    best = None
    for label, cfg in configs:
        n, r, ll, nm, rmk, mkt = score(collect_ng(played, *cfg, test_start))
        print(f"{label:<26}{n:>6}{r:>9.4f}{ll:>9.4f}{rmk:>9.4f}{mkt:>8.4f}")
        if best is None or r < best[1]:
            best = (label, r, cfg)
    print(f"  -> best: {best[0]}  (RPS {best[1]:.4f})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2024-07-01")
    ts = ap.parse_args().test_start
    played = _load_raw()

    # A) training target: xG blend weight (xi/window at defaults)
    cfgA = [(f"xW={w:.2f}", (w, DEFAULT_XI, DEFAULT_WIN)) for w in [0.0, 0.25, 0.45, 0.55, 0.75, 1.0]]
    bA = sweep("A) Training target (0=goals .. 1=xG)", played, cfgA, ts)
    xw = bA[2][0]

    # B) time-decay xi (best target, default window)
    cfgB = [(f"xi={xi:g}", (xw, xi, DEFAULT_WIN)) for xi in [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]]
    bB = sweep(f"B) Time-decay xi (xW={xw:.2f}, win={DEFAULT_WIN})", played, cfgB, ts)
    xi = bB[2][1]

    # C) rolling window months (best target + xi)
    cfgC = [(f"win={m}mo", (xw, xi, m)) for m in [12, 18, 24, 36, 120]]
    bC = sweep(f"C) Window months (xW={xw:.2f}, xi={xi:g})", played, cfgC, ts)

    print("\n===== best config =====")
    print(f"  xW={bC[2][0]:.2f}  xi={bC[2][1]:g}  window={bC[2][2]}mo  ->  RPS {bC[1]:.4f}")
    print(f"  (production baseline was xW=0.55, xi=0.0015, window=24)")


if __name__ == "__main__":
    main()
