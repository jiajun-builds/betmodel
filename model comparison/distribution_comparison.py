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
"""Head-to-head of penaltyblog goal distributions on the corrected data.

The calibration diagnostic pinned the model's weakness on the SHAPE of the
goal-difference distribution: cover probabilities are overconfident, worst on big
handicap lines (big favourites), because Poisson/ZIP under-disperses margins.
Temperature scaling did not fix it. The proper fix is a distribution that
disperses margins better, so this compares the candidates on the SAME walk-forward
harness (18-month lookback, Dixon-Coles time-decay xi=0.001 where the model takes
weights), scored on the SAME fixtures.

For each model it reports:
  - 1X2 RPS and log-loss (raw predictive accuracy)
  - handicap-cover overconfidence overall AND by line-magnitude bucket
    (the metric that actually drives the EV overstatement)

Run (repo root, env with penaltyblog):
    PYTHONPATH=src python "model comparison/distribution_comparison.py"
"""

import os
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from csl.date_utils import parse_date_only_series  # noqa: E402

XI = 0.001
LOOKBACK_M = 18
START_SEASON = 2024               # score from here (needs prior history to train)
res_map = {"H": 0, "D": 1, "A": 2}

MODELS = {
    "Poisson": pb.models.PoissonGoalsModel,
    "ZIP (prod)": pb.models.ZeroInflatedPoissonGoalsModel,
    "DixonColes": pb.models.DixonColesGoalModel,
    "NegBinom": pb.models.NegativeBinomialGoalModel,
    "BivPoisson": pb.models.BivariatePoissonGoalModel,
    "WeibullCop": pb.models.WeibullCopulaGoalsModel,
}


def load():
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


def fit_model(Model, train, weights):
    """Fit with Dixon-Coles weights if the constructor accepts them, else without."""
    try:
        clf = Model(train["HExpG+"], train["AExpG+"], train["Home"], train["Away"], weights)
    except TypeError:
        clf = Model(train["HExpG+"], train["AExpG+"], train["Home"], train["Away"])
    clf.fit()
    return clf


def gd_dist(grid):
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n))
    gds = np.arange(-(n - 1), n)
    probs = np.array([grid[diff == d].sum() for d in gds])
    return gds, probs


def _one(adj):
    return 1.0 if adj > 1e-9 else (0.0 if adj < -1e-9 else 0.5)


def cover_fraction(gd, line):
    if round(line * 4) % 2 != 0:
        return 0.5 * _one(gd + line - 0.25) + 0.5 * _one(gd + line + 0.25)
    return _one(gd + line)


def settle_profit(gd, line, odds):
    def one(adj):
        return (odds - 1.0) if adj > 1e-9 else (-1.0 if adj < -1e-9 else 0.0)
    if round(line * 4) % 2 != 0:
        return 0.5 * one(gd + line - 0.25) + 0.5 * one(gd + line + 0.25)
    return one(gd + line)


def backed_bet(gds, probs, line, oh, oa, gd):
    """Pick the higher-EV side; return (predicted_EV, realized_pl)."""
    ev_h = float(sum(p * settle_profit(g, line, oh) for g, p in zip(gds, probs)))
    ev_a = float(sum(p * settle_profit(-g, -line, oa) for g, p in zip(gds, probs)))
    if ev_h >= ev_a:
        return ev_h, settle_profit(gd, line, oh)
    return ev_a, settle_profit(-gd, -line, oa)


def rps_one(p, o):
    cp = np.cumsum(p)
    co = np.cumsum([1.0 if i == o else 0.0 for i in range(3)])
    return float(np.sum((cp[:-1] - co[:-1]) ** 2) / 2)


def line_bucket(x):
    x = abs(x)
    return "0-0.5" if x <= 0.5 else "0.75-1.0" if x <= 1.0 else "1.25-2.0" if x <= 2.0 else ">2"


def run():
    df = load()
    graded = df[df["res_numeric"].notna()].copy()
    start = graded.query("Season == @START_SEASON")["Date"].min()
    run_dates = sorted(graded.loc[graded["Date"] >= start, "Date"].unique())
    print(f"Scoring from {pd.Timestamp(start).date()} | {len(run_dates)} dates | "
          f"models: {list(MODELS)}")

    # preds[model][fixture_key] = dict(rps, ll, p_cover, realized_cover, line, has_line)
    preds = {m: {} for m in MODELS}
    observed = set()

    for date in tqdm(run_dates, desc="dates"):
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=LOOKBACK_M)]
        if len(hist) < 20 or test.empty:
            continue
        weights = pb.models.dixon_coles_weights(hist["Date"], XI)
        fitted = {}
        for name, Model in MODELS.items():
            try:
                fitted[name] = fit_model(Model, hist, weights)
            except Exception:
                fitted[name] = None

        for r in test.itertuples(index=False):
            key = (date, r.Home, r.Away)
            outcome = int(r.res_numeric)
            has_line = (pd.notna(r.pinnacle_open_ah) and pd.notna(r.pinnacle_open_ah_h)
                        and pd.notna(r.pinnacle_open_ah_a))
            line = float(r.pinnacle_open_ah) if has_line else np.nan
            oh = float(r.pinnacle_open_ah_h) if has_line else np.nan
            oa = float(r.pinnacle_open_ah_a) if has_line else np.nan
            gd = float(r.HG - r.AG)
            ok = True
            row = {}
            for name in MODELS:
                clf = fitted[name]
                if clf is None:
                    ok = False
                    break
                try:
                    grid = np.asarray(clf.predict(r.Home, r.Away).grid)
                    gds, probs = gd_dist(grid)
                except Exception:
                    ok = False
                    break
                ph = float(probs[gds > 0].sum())
                pdw = float(probs[gds == 0].sum())
                pa = float(probs[gds < 0].sum())
                p3 = np.array([ph, pdw, pa])
                p3 = p3 / p3.sum()
                rec = {"rps": rps_one(p3, outcome),
                       "ll": -np.log(max(p3[outcome], 1e-12)),
                       "has_line": has_line}
                if has_line:
                    pcov = float(sum(p * cover_fraction(g, line) for g, p in zip(gds, probs)))
                    rec["p_cover"] = pcov
                    rec["realized"] = cover_fraction(gd, line)
                    rec["line"] = line
                    best_ev, best_pl = backed_bet(gds, probs, line, oh, oa, gd)
                    rec["best_ev"] = best_ev
                    rec["best_pl"] = best_pl
                row[name] = rec
            if ok:
                for name in MODELS:
                    preds[name][key] = row[name]
                observed.add(key)

    # Score on fixtures every model predicted.
    common = set(observed)
    for name in MODELS:
        common &= set(preds[name])
    common = sorted(common)
    print(f"\nCommon fixtures scored by all models: {len(common)}")

    print(f"\n{'model':<12}{'RPS':>9}{'logloss':>10}{'cover_over':>12}{'n_cov':>7}")
    summary = {}
    for name in MODELS:
        rps = np.mean([preds[name][k]["rps"] for k in common])
        ll = np.mean([preds[name][k]["ll"] for k in common])
        cov = [(preds[name][k]["p_cover"], preds[name][k]["realized"], preds[name][k]["line"])
               for k in common if preds[name][k]["has_line"]]
        over = np.mean([p - r for p, r, _ in cov]) if cov else np.nan
        summary[name] = cov
        print(f"{name:<12}{rps:>9.5f}{ll:>10.5f}{over:>+12.4f}{len(cov):>7}")

    print("\n=== Cover overconfidence by line-magnitude bucket (pred - realized) ===")
    buckets = ["0-0.5", "0.75-1.0", "1.25-2.0", ">2"]
    header = f"{'model':<12}" + "".join(f"{b:>11}" for b in buckets)
    print(header)
    for name in MODELS:
        cov = summary[name]
        cells = ""
        for b in buckets:
            vals = [(p - r) for p, r, ln in cov if line_bucket(ln) == b]
            cells += f"{(np.mean(vals) if vals else float('nan')):>+11.3f}"
        print(f"{name:<12}{cells}")
    n_by_bucket = {b: sum(1 for _, _, ln in summary["Poisson"] if line_bucket(ln) == b) for b in buckets}
    print(f"{'(n)':<12}" + "".join(f"{n_by_bucket[b]:>11}" for b in buckets))

    # ---- Betting metric: back the higher-EV side, EV honesty + ROI ----------
    print("\n=== Betting the model's +EV picks (thr>0): honesty + realized ROI ===")
    print(f"{'model':<12}{'n':>5}{'predEV':>9}{'realROI':>9}{'t(ROI)':>8}"
          f"{'overstate':>10}{'t(over)':>9}")
    roi_by_model = {}
    for name in MODELS:
        rows = [preds[name][k] for k in common
                if preds[name][k]["has_line"] and preds[name][k]["best_ev"] > 0]
        if len(rows) < 2:
            continue
        ev = np.array([r["best_ev"] for r in rows])
        pl = np.array([r["best_pl"] for r in rows])
        roi = pl.mean()
        se = pl.std(ddof=1) / np.sqrt(len(pl))
        d = ev - pl
        ose = d.std(ddof=1) / np.sqrt(len(d))
        roi_by_model[name] = [(preds[name][k]["best_pl"], preds[name][k]["line"])
                              for k in common
                              if preds[name][k]["has_line"] and preds[name][k]["best_ev"] > 0]
        print(f"{name:<12}{len(rows):>5}{ev.mean():>+9.3f}{roi:>+9.3f}{roi/se:>+8.2f}"
              f"{d.mean():>+10.3f}{d.mean()/ose:>+9.2f}")

    print("\n=== Realized ROI of +EV picks by line bucket (per model) ===")
    buckets2 = ["0-0.5", "0.75-1.0", "1.25-2.0", ">2"]
    print(f"{'model':<12}" + "".join(f"{b:>11}" for b in buckets2))
    for name, rows in roi_by_model.items():
        cells = ""
        for b in buckets2:
            vals = [pl for pl, ln in rows if line_bucket(ln) == b]
            cells += f"{(np.mean(vals) if vals else float('nan')):>+11.3f}"
        print(f"{name:<12}{cells}")

    print("\nLower RPS/logloss = more accurate 1X2. 'overstate' = predicted EV minus"
          " realized ROI on the backed side (the betting-relevant miscalibration);"
          " smaller is better. Production is 'ZIP (prod)'.")


if __name__ == "__main__":
    run()
