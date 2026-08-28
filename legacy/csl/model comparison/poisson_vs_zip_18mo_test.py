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
"""
Walk-forward RPS backtest: Poisson vs Zero-Inflated Poisson, aligned to the
production 18-month training window.

Same harness as the other model comparison scripts, but:
  - lookback is 18 MONTHS (production dc.py uses DateOffset(months=18)),
    not the 1 year used by poisson_model_test.py / zero_inf_ps_model_test.py
  - both models are fit on the SAME train slice each date and scored on an
    IDENTICAL set of fixtures, so the RPS difference is apples-to-apples

RPS (ranked probability score): lower is better.

Run (repo root, conda env csl-workflows or any env with penaltyblog):
    python "model comparison/poisson_vs_zip_18mo_test.py"
"""

import os
import sys

import pandas as pd
import penaltyblog as pb
from tqdm import tqdm

# Resolve the league CSV relative to the repo root (../data from this file),
# so the script is portable across machines.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

# The CSV dates are DD/MM/YYYY; a bare to_datetime() day/month-swaps them and
# drops every day>12 row. Use the production parser instead.
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from csl.date_utils import parse_date_only_series  # noqa: E402

XI = 0.001
LOOKBACK = pd.DateOffset(months=18)   # aligned to production dc.py
START_SEASON = 2025                   # first season to score, as in the other tests

df = pd.read_csv(CSV)
df["Date"] = parse_date_only_series(df["Date"])
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date", drop=False)

df["HExpG+"] = pd.to_numeric(df["HExpG+"], errors="coerce")
df["AExpG+"] = pd.to_numeric(df["AExpG+"], errors="coerce")

res_map = {"H": 0, "D": 1, "A": 2}
df["res_numeric"] = df["Res"].map(res_map)

start_date = df.query("Season == @START_SEASON")["Date"].min()
run_dates = df["Date"][df["Date"] >= start_date].unique()
print(
    f"Start date: {pd.Timestamp(start_date).date()} | run dates: {len(run_dates)} "
    f"| lookback: 18 months | xi={XI}"
)

MODELS = {
    "Poisson": pb.models.PoissonGoalsModel,
    "ZIP": pb.models.ZeroInflatedPoissonGoalsModel,
}

# Store predictions keyed by (date, home, away) so both models are scored on
# exactly the same fixtures (drop any fixture either model failed to predict).
preds = {name: {} for name in MODELS}
observed = {}

for date in tqdm(run_dates, desc="dates"):
    lookback = pd.Timestamp(date) - LOOKBACK
    train = df[(df["Date"] < date) & (df["Date"] >= lookback)]
    train = train.dropna(subset=["HExpG+", "AExpG+", "Home", "Away"])
    test = df[df["Date"] == date]
    if len(train) == 0 or len(test) == 0:
        continue

    weights = pb.models.dixon_coles_weights(train["Date"], XI)

    fitted = {}
    for name, Model in MODELS.items():
        try:
            clf = Model(train["HExpG+"], train["AExpG+"], train["Home"], train["Away"], weights)
            clf.fit()
            fitted[name] = clf
        except Exception:
            fitted[name] = None

    for row in test.itertuples(index=False):
        if pd.isna(row.res_numeric):
            continue
        key = (pd.Timestamp(date), row.Home, row.Away)
        row_preds = {}
        ok = True
        for name in MODELS:
            clf = fitted[name]
            if clf is None:
                ok = False
                break
            try:
                row_preds[name] = clf.predict(row.Home, row.Away).home_draw_away
            except Exception:
                ok = False
                break
        if not ok:
            continue
        for name in MODELS:
            preds[name][key] = row_preds[name]
        observed[key] = int(row.res_numeric)

# Score on the fixtures common to both models.
common = set(observed)
for name in MODELS:
    common &= set(preds[name])
common = sorted(common)
print(f"\nScored fixtures (common to both models): {len(common)}")

obs_list = [observed[k] for k in common]
print(f"{'Model':<10}{'RPS':>12}")
rps_vals = {}
for name in MODELS:
    p_list = [preds[name][k] for k in common]
    rps = pb.metrics.rps_average(p_list, obs_list)
    rps_vals[name] = rps
    print(f"{name:<10}{rps:>12.5f}")

diff = rps_vals["ZIP"] - rps_vals["Poisson"]
verdict = "ZIP better" if diff < 0 else "Poisson better" if diff > 0 else "tie"
print(f"\nZIP - Poisson RPS delta: {diff:+.6f}  ({verdict}; lower RPS is better)")
print(f"Relative: {diff / rps_vals['Poisson'] * 100:+.3f}% vs Poisson")
