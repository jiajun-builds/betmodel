"""Calibration diagnostic for the production model's probabilities.

ANALYSIS-LAYER ONLY — this never touches src/csl or the dashboard. It walk-forwards
the exact production recipe (ZIP-on-xG, 18-month lookback, xi=0.001, Dixon-Coles
weights), but instead of placing bets it records the model's probabilities and the
realized outcomes, then quantifies HOW overconfident the model is.

The opening-line backtest showed the model overstates its handicap-cover EV by
~20%/unit (t=6). That can only come from cover probabilities that are too extreme.
This script measures that directly:

  1. 1X2 reliability — model P(outcome) vs realized frequency, binned, with ECE.
  2. Handicap-cover reliability — for the side the backtest would back, model
     P(cover) vs realized cover rate, binned, with ECE. This is the number that
     drives the EV gap.
  3. Segment breakdown — cover calibration by handicap-line magnitude and by
     favourite vs underdog side.
  4. Implied temperature T* — the single exponent on P(goal difference) that best
     flattens the 1X2 distribution (in-sample readout: T*>1 == overconfident, and
     roughly how much a temperature-scaling fix would need to pull back).

Run (repo root, env with penaltyblog):
    PYTHONPATH=src python backtest/calibration_diagnostic.py
"""

import os

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize_scalar

from csl.date_utils import parse_date_only_series
from csl.models.continuous_poisson import ContinuousPoissonGoalModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

# Imported from the production module so this diagnostic cannot drift from what ships.
from csl.models.dc import PROD_LOOKBACK_MONTHS, PROD_XI  # noqa: E402,F401
res_map = {"H": 0, "D": 1, "A": 2}  # 0=home win, 1=draw, 2=away win


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Date"] = parse_date_only_series(df["Date"])
    for c in ("HExpG+", "AExpG+", "HG", "AG",
              "pinnacle_open_ah", "pinnacle_open_ah_h", "pinnacle_open_ah_a"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["res_numeric"] = df["Res"].map(res_map)
    return df.sort_values("Date").reset_index(drop=True)


def gd_distribution(grid: np.ndarray):
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n))
    gds = np.arange(-(n - 1), n)
    probs = np.array([grid[diff == d].sum() for d in gds])
    return gds, probs


def _one(adj: float) -> float:
    """Win-equivalent of a single settlement: 1 win, 0 lose, 0.5 push."""
    if adj > 1e-9:
        return 1.0
    if adj < -1e-9:
        return 0.0
    return 0.5


def cover_fraction(gd: float, line: float) -> float:
    """Fraction of stake won backing a side at `line` given goal diff `gd`
    (from that side's perspective). Quarter lines split half/half; whole-line
    pushes count as 0.5. Range [0, 1]."""
    if round(line * 4) % 2 != 0:
        return 0.5 * _one(gd + line - 0.25) + 0.5 * _one(gd + line + 0.25)
    return _one(gd + line)


def model_cover_prob(gds, probs, line: float) -> float:
    return float(sum(p * cover_fraction(g, line) for g, p in zip(gds, probs)))


def settle_profit(gd: float, line: float, odds: float) -> float:
    """Per-unit profit (to pick the backed side / sanity-tie to the backtest)."""
    def one(adj):
        if adj > 1e-9:
            return odds - 1.0
        if adj < -1e-9:
            return -1.0
        return 0.0
    if round(line * 4) % 2 != 0:
        return 0.5 * one(gd + line - 0.25) + 0.5 * one(gd + line + 0.25)
    return one(gd + line)


def one_x_two(gds, probs):
    """(P_home_win, P_draw, P_away_win) from a goal-difference distribution."""
    ph = float(probs[gds > 0].sum())
    pd_ = float(probs[gds == 0].sum())
    pa = float(probs[gds < 0].sum())
    return ph, pd_, pa


def reliability(pred, hit, n_bins=10):
    """Return a per-bin table (lo, hi, n, mean_pred, mean_realized) plus ECE.
    `pred` in [0,1], `hit` in [0,1] (fractional allowed for pushes)."""
    pred = np.asarray(pred, float)
    hit = np.asarray(hit, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi if hi < 1 else pred <= hi)
        if not m.any():
            continue
        mp, mr, k = pred[m].mean(), hit[m].mean(), int(m.sum())
        rows.append((lo, hi, k, mp, mr))
        ece += (k / len(pred)) * abs(mp - mr)
    return rows, ece


def temperature(probs, T):
    q = np.power(np.clip(probs, 1e-12, None), 1.0 / T)
    return q / q.sum()


def run():
    df = load()
    graded = df[df["res_numeric"].notna()].copy()
    run_dates = sorted(graded["Date"].unique())

    # Per-fixture records.
    onex = {"pred": [], "hit": []}          # pooled 1X2 (predicted prob, hit)
    llT_probs = []                          # goal-diff distros for temperature fit
    llT_out = []                            # realized 1X2 index
    cov_pred, cov_hit = [], []              # backed-side cover prob / realized
    seg = []                                # (line_abs_bucket, fav_or_dog, cov_pred, cov_hit)

    for date in run_dates:
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < 20 or test.empty:
            continue
        weights = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        try:
            # ContinuousPoisson, not ZIP: the xG targets are non-integer and every
            # penaltyblog family truncates them (see csl.models.continuous_poisson).
            clf = ContinuousPoissonGoalModel(
                hist["HExpG+"], hist["AExpG+"], hist["Home"], hist["Away"], weights)
            clf.fit()
        except Exception:
            continue

        for r in test.itertuples(index=False):
            try:
                grid = np.asarray(clf.predict(r.Home, r.Away).grid)
            except Exception:
                continue
            gds, probs = gd_distribution(grid)
            outcome = int(r.res_numeric)

            # 1X2 calibration (pool the 3 class probs vs their 0/1 hits).
            ph, pdw, pa = one_x_two(gds, probs)
            for cls, p in enumerate((ph, pdw, pa)):
                onex["pred"].append(p)
                onex["hit"].append(1.0 if outcome == cls else 0.0)
            llT_probs.append(probs)
            llT_out.append(outcome)

            # Handicap-cover calibration for the side the backtest would back.
            if pd.isna(r.pinnacle_open_ah) or pd.isna(r.pinnacle_open_ah_h) or pd.isna(r.pinnacle_open_ah_a):
                continue
            line = float(r.pinnacle_open_ah)
            oh, oa = float(r.pinnacle_open_ah_h), float(r.pinnacle_open_ah_a)
            gd = float(r.HG - r.AG)
            ev_home = float(sum(p * settle_profit(g, line, oh) for g, p in zip(gds, probs)))
            ev_away = float(sum(p * settle_profit(-g, -line, oa) for g, p in zip(gds, probs)))
            if ev_home >= ev_away:
                p_cov = model_cover_prob(gds, probs, line)
                realized = cover_fraction(gd, line)
                fav = "home"
            else:
                p_cov = model_cover_prob(-gds, probs, -line)
                realized = cover_fraction(-gd, -line)
                fav = "away"
            cov_pred.append(p_cov)
            cov_hit.append(realized)
            labs = abs(line)
            lb = "|line|<0.5" if labs < 0.5 else "0.5-1.0" if labs <= 1.0 else "1.25-2.0" if labs <= 2.0 else "|line|>2"
            seg.append((lb, fav, p_cov, realized))

    n1x2 = len(onex["pred"]) // 3
    print(f"Scored fixtures: {n1x2} (1X2)  |  backed-side cover bets: {len(cov_pred)}")

    # ---- 1X2 reliability -----------------------------------------------------
    rows, ece = reliability(onex["pred"], onex["hit"])
    print("\n=== 1X2 reliability (pooled H/D/A; predicted vs realized) ===")
    print(f"{'bin':>12}{'n':>7}{'pred':>9}{'realized':>10}{'gap':>8}")
    for lo, hi, k, mp, mr in rows:
        print(f"{f'{lo:.1f}-{hi:.1f}':>12}{k:>7}{mp:>9.3f}{mr:>10.3f}{mp-mr:>+8.3f}")
    print(f"  ECE (1X2) = {ece:.4f}   (0 = perfectly calibrated)")

    # ---- backed-side cover reliability --------------------------------------
    rows, ece_c = reliability(cov_pred, cov_hit)
    print("\n=== Handicap-cover reliability (backed side; the EV driver) ===")
    print(f"{'bin':>12}{'n':>7}{'pred':>9}{'realized':>10}{'gap':>8}")
    for lo, hi, k, mp, mr in rows:
        print(f"{f'{lo:.1f}-{hi:.1f}':>12}{k:>7}{mp:>9.3f}{mr:>10.3f}{mp-mr:>+8.3f}")
    print(f"  ECE (cover) = {ece_c:.4f}")
    print(f"  mean model P(cover) = {np.mean(cov_pred):.4f}  vs realized cover = {np.mean(cov_hit):.4f}"
          f"  -> overconfidence {np.mean(cov_pred)-np.mean(cov_hit):+.4f}")

    # ---- segment breakdown ---------------------------------------------------
    segdf = pd.DataFrame(seg, columns=["line_bucket", "side", "pred", "hit"])
    print("\n=== Cover calibration by handicap-line magnitude ===")
    print(f"{'bucket':>12}{'n':>6}{'pred':>9}{'realized':>10}{'gap':>8}")
    order = ["|line|<0.5", "0.5-1.0", "1.25-2.0", "|line|>2"]
    for b in order:
        s = segdf[segdf["line_bucket"] == b]
        if len(s):
            print(f"{b:>12}{len(s):>6}{s['pred'].mean():>9.3f}{s['hit'].mean():>10.3f}"
                  f"{s['pred'].mean()-s['hit'].mean():>+8.3f}")
    print("\n=== Cover calibration by backed side ===")
    for b in ("home", "away"):
        s = segdf[segdf["side"] == b]
        if len(s):
            print(f"{b:>12}{len(s):>6}{s['pred'].mean():>9.3f}{s['hit'].mean():>10.3f}"
                  f"{s['pred'].mean()-s['hit'].mean():>+8.3f}")

    # ---- implied temperature (in-sample readout) -----------------------------
    P = np.array(llT_probs)
    gds_full = np.arange(-(P.shape[1] // 2), P.shape[1] // 2 + 1)
    hw = gds_full > 0
    dw = gds_full == 0
    outs = np.array(llT_out)

    def nll(T):
        q = np.power(np.clip(P, 1e-12, None), 1.0 / T)
        q = q / q.sum(axis=1, keepdims=True)
        ph = q[:, hw].sum(1)
        pdw = q[:, dw].sum(1)
        pa = 1 - ph - pdw
        m = np.clip(np.where(outs == 0, ph, np.where(outs == 1, pdw, pa)), 1e-12, None)
        return -np.log(m).mean()

    res = minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded")
    T_star = res.x
    print(f"\n=== Implied temperature (in-sample 1X2 log-loss minimiser) ===")
    print(f"  T* = {T_star:.3f}   (T>1 => overconfident; a fix would raise P(goal diff)^(1/T*))")
    print(f"  1X2 log-loss:  T=1 (as-is) {nll(1.0):.4f}  ->  T=T* {nll(T_star):.4f}")
    print("  NOTE: in-sample readout only. A production fix must fit T walk-forward.")


if __name__ == "__main__":
    run()
