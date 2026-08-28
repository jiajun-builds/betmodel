# ============================================================================
# STALE RESULTS WARNING — this script's published conclusions predate the fix
# for penaltyblog's silent truncation of non-integer goal targets. It fits a
# penaltyblog family on xG-blend targets, and penaltyblog's BaseGoalsModel
# coerces those to int before the likelihood, so every lambda here is ~27% too
# low (score equation sum(w*lambda)/sum(w*y) = 0.733 instead of 1.0). Model
# families cannot be meaningfully ranked against each other on a truncated fit —
# they were all equally broken. See src/csl/models/continuous_poisson.py and
# backtest/verify_truncation_fix.py. Re-run against ContinuousPoissonGoalModel
# before citing any number from this file.
# ============================================================================
"""Phase 1: does walk-forward temperature scaling fix the model's overstated
handicap-cover EV? ANALYSIS-LAYER ONLY — nothing here touches src/csl or the
dashboard.

The calibration diagnostic showed the model's goal-difference distribution is too
narrow: cover probabilities are badly overconfident (ECE 0.086, worst on big
lines), even though 1X2 is only mildly off. Temperature scaling widens the whole
distribution — P_T(d) proportional to P(d)^(1/T), T>1 — which is the targeted fix.

Design (no leakage):
  Stage A  walk-forward the production recipe (ZIP-on-xG, 18mo, xi=0.001) once,
           caching each fixture's OUT-OF-SAMPLE goal-difference distribution and
           realized outcome/line.
  Stage B  walk-forward calibration: on each date, fit T on the cover outcomes of
           every fixture STRICTLY BEFORE it (minimising cover Brier), apply that T
           to the date's distributions, then settle bets at the opening line with
           the calibrated probabilities.

Prints the uncalibrated vs calibrated backtest side by side on the identical
(calibration-eligible) fixture set, so the effect of calibration is isolated.

Run (repo root, env with penaltyblog):
    PYTHONPATH=src python backtest/backtest_open_ah_calibrated.py
"""

import os

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize_scalar

from csl.date_utils import parse_date_only_series

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

PROD_XI = 0.001
PROD_LOOKBACK_MONTHS = 18
MIN_PRIOR = 100                 # fixtures needed before a T is fit (else skip)
EV_THRESHOLDS = [0.00, 0.02, 0.05, 0.10]
res_map = {"H": 0, "D": 1, "A": 2}


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


def _one_win(adj):
    return 1.0 if adj > 1e-9 else (0.0 if adj < -1e-9 else 0.5)


def cover_fraction(gd, line):
    if round(line * 4) % 2 != 0:
        return 0.5 * _one_win(gd + line - 0.25) + 0.5 * _one_win(gd + line + 0.25)
    return _one_win(gd + line)


def settle_profit(gd, line, odds):
    def one(adj):
        return (odds - 1.0) if adj > 1e-9 else (-1.0 if adj < -1e-9 else 0.0)
    if round(line * 4) % 2 != 0:
        return 0.5 * one(gd + line - 0.25) + 0.5 * one(gd + line + 0.25)
    return one(gd + line)


def temper(probs, T):
    q = np.power(np.clip(probs, 1e-12, None), 1.0 / T)
    return q / q.sum()


# ── Stage A: walk-forward, cache each fixture's OOS distribution ──────────────

def stage_a():
    df = load()
    graded = df[df["res_numeric"].notna()
                & df["pinnacle_open_ah"].notna()
                & df["pinnacle_open_ah_h"].notna()
                & df["pinnacle_open_ah_a"].notna()].copy()
    run_dates = sorted(graded["Date"].unique())
    recs = []
    for date in run_dates:
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < 20 or test.empty:
            continue
        w = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        try:
            clf = pb.models.ZeroInflatedPoissonGoalsModel(
                hist["HExpG+"], hist["AExpG+"], hist["Home"], hist["Away"], w)
            clf.fit()
        except Exception:
            continue
        for r in test.itertuples(index=False):
            try:
                grid = np.asarray(clf.predict(r.Home, r.Away).grid)
            except Exception:
                continue
            gds, probs = gd_distribution(grid)
            line = float(r.pinnacle_open_ah)
            gd = float(r.HG - r.AG)
            cf_home = np.array([cover_fraction(g, line) for g in gds])
            recs.append({
                "date": date, "Season": int(r.Season), "Round": int(r.Round),
                "Home": r.Home, "Away": r.Away, "line": line,
                "oh": float(r.pinnacle_open_ah_h), "oa": float(r.pinnacle_open_ah_a),
                "gd": gd, "outcome": int(r.res_numeric),
                "probs": probs, "cf_home": cf_home,
                "realized_home_cover": cover_fraction(gd, line),
            })
    return gds, recs


# ── Stage B: walk-forward temperature fit + settle ───────────────────────────

def fit_T_1x2(prior, gds):
    """Fit T by minimising 1X2 log-loss over prior fixtures — the standard
    temperature-scaling objective. (Fitting on cover Brier instead is degenerate:
    since the model has no cover-selection skill, Brier is minimised by predicting
    0.5 everywhere, i.e. T -> infinity, which just switches the signal off.)"""
    P = np.array([r["probs"] for r in prior])          # N x 29
    outs = np.array([r["outcome"] for r in prior])
    hw, dw = gds > 0, gds == 0

    def nll(T):
        q = np.power(np.clip(P, 1e-12, None), 1.0 / T)
        q = q / q.sum(axis=1, keepdims=True)
        ph = q[:, hw].sum(1)
        pdw = q[:, dw].sum(1)
        pa = 1 - ph - pdw
        m = np.clip(np.where(outs == 0, ph, np.where(outs == 1, pdw, pa)), 1e-12, None)
        return float(-np.log(m).mean())

    return minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded").x


def ev_pick(gds, probs, line, oh, oa):
    ev_h = float(sum(p * settle_profit(g, line, oh) for g, p in zip(gds, probs)))
    ev_a = float(sum(p * settle_profit(-g, -line, oa) for g, p in zip(gds, probs)))
    if ev_h >= ev_a:
        return "home", ev_h
    return "away", ev_a


def threshold_table(rows, ev_key, pl_key):
    out = []
    for thr in EV_THRESHOLDS:
        sel = [r for r in rows if r[ev_key] > thr]
        n = len(sel)
        if n == 0:
            out.append((thr, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        pls = np.array([r[pl_key] for r in sel])
        evs = np.array([r[ev_key] for r in sel])
        roi = pls.mean()
        se = pls.std(ddof=1) / np.sqrt(n)
        out.append((thr, n, roi, se, roi / se if se else np.nan, evs.mean()))
    return out


def honesty(rows, ev_key, pl_key):
    sel = [r for r in rows if r[ev_key] > 0]
    if not sel:
        return None
    ev = np.array([r[ev_key] for r in sel])
    pl = np.array([r[pl_key] for r in sel])
    d = ev - pl
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return len(sel), ev.mean(), pl.mean(), d.mean(), (d.mean() / se if se else np.nan)


def run():
    print("Stage A: walk-forward model fits (caching OOS distributions)…")
    gds, recs = stage_a()
    recs.sort(key=lambda r: r["date"])
    print(f"  cached {len(recs)} fixtures with an opening line + result")

    print("Stage B: walk-forward temperature calibration + settlement…")
    bets = []
    Ts = []
    dates = sorted({r["date"] for r in recs})
    for date in dates:
        prior = [r for r in recs if r["date"] < date]
        if len(prior) < MIN_PRIOR:
            continue
        T = fit_T_1x2(prior, gds)
        Ts.append((date, T))
        for r in (x for x in recs if x["date"] == date):
            cal = temper(r["probs"], T)
            side_c, ev_c = ev_pick(gds, cal, r["line"], r["oh"], r["oa"])
            side_u, ev_u = ev_pick(gds, r["probs"], r["line"], r["oh"], r["oa"])
            pl_c = (settle_profit(r["gd"], r["line"], r["oh"]) if side_c == "home"
                    else settle_profit(-r["gd"], -r["line"], r["oa"]))
            pl_u = (settle_profit(r["gd"], r["line"], r["oh"]) if side_u == "home"
                    else settle_profit(-r["gd"], -r["line"], r["oa"]))
            bets.append({**{k: r[k] for k in ("Season", "Round", "Home", "Away", "line")},
                         "T": T, "ev_cal": ev_c, "pl_cal": pl_c,
                         "ev_unc": ev_u, "pl_unc": pl_u})

    if not bets:
        raise SystemExit("No calibration-eligible bets; lower MIN_PRIOR.")

    tvals = np.array([t for _, t in Ts])
    print(f"\nCalibration-eligible bets: {len(bets)}  (from {Ts[0][0].date()} on)")
    print(f"Walk-forward T:  median {np.median(tvals):.3f} | "
          f"min {tvals.min():.3f} | max {tvals.max():.3f} | last {tvals[-1]:.3f}")

    print("\n=== Threshold table: UNCALIBRATED (T=1) vs CALIBRATED, same fixtures ===")
    print(f"{'thr':>5} | {'n':>4} {'ROI':>8} {'t':>6} {'avgEV':>7}  ||  "
          f"{'n':>4} {'ROI':>8} {'t':>6} {'avgEV':>7}")
    unc = threshold_table(bets, "ev_unc", "pl_unc")
    cal = threshold_table(bets, "ev_cal", "pl_cal")
    for (thr, nu, ru, su, tu, eu), (_, nc, rc, sc, tc, ec) in zip(unc, cal):
        du = f"{nu:>4} {ru:>+8.4f} {tu:>+6.2f} {eu:>+7.3f}" if nu else f"{'--':>4}"
        dc = f"{nc:>4} {rc:>+8.4f} {tc:>+6.2f} {ec:>+7.3f}" if nc else f"{'--':>4}"
        print(f"{thr:>5.2f} | {du}  ||  {dc}")

    print("\n=== Model-EV honesty (thr>0): predicted EV vs realized ROI ===")
    for label, key_ev, key_pl in (("UNCALIBRATED", "ev_unc", "pl_unc"),
                                   ("CALIBRATED  ", "ev_cal", "pl_cal")):
        h = honesty(bets, key_ev, key_pl)
        if h:
            n, ev, roi, over, t = h
            print(f"  {label}: n={n:>4}  predEV {ev:+.4f}  realized {roi:+.4f}  "
                  f"overstatement {over:+.4f}  (t={t:+.2f})")

    print("\n=== Calibrated EV bucketed (does higher cal-EV finally sort ROI?) ===")
    pos = [r for r in bets if r["ev_cal"] > 0]
    if len(pos) >= 8:
        dfp = pd.DataFrame(pos)
        dfp["bk"] = pd.qcut(dfp["ev_cal"], 4, duplicates="drop")
        g = dfp.groupby("bk", observed=True).agg(
            n=("pl_cal", "size"), pred_ev=("ev_cal", "mean"), roi=("pl_cal", "mean"))
        print(g.to_string(float_format=lambda v: f"{v:+.4f}"))
    else:
        print(f"  (only {len(pos)} +EV bets after calibration)")

    print("\n=== Per-season honesty (calibrated, thr>0) ===")
    dfb = pd.DataFrame(bets)
    for s in sorted(dfb["Season"].unique()):
        sub = dfb[(dfb["Season"] == s) & (dfb["ev_cal"] > 0)]
        if len(sub):
            ev, roi = sub["ev_cal"].mean(), sub["pl_cal"].mean()
            print(f"  {s}: n={len(sub):>4}  predEV {ev:+.3f}  realized {roi:+.3f}  "
                  f"overstate {ev - roi:+.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "backtest_open_ah_calibrated_bets.csv")
    pd.DataFrame(bets).to_csv(out, index=False)
    print(f"\nPer-bet detail written to: {out}")


if __name__ == "__main__":
    run()
