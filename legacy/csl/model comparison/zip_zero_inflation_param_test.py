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
Step 1 of the ZIP validity checklist: is the zero-inflation actually doing
anything, or has production's ZeroInflatedPoissonGoalsModel effectively
collapsed to a plain Poisson?

Rationale: on the 362-fixture leaderboard ZIP beats Poisson by only ~0.0003
RPS. If the fitted zero-inflation parameter is ~0 at every refit, that tiny gap
is fully explained (ZIP == Poisson) and there is no evidence the extra
parameter earns its place in production.

This does NOT re-run the model comparison. It walk-forwards the SAME production
recipe (18-month lookback, xi=0.001, xG targets, Dixon-Coles time-decay
weights) but fits ONLY the ZIP model, and records the zero-inflation parameter
fitted at each run date, then summarises its distribution.

penaltyblog exposes model parameters via clf.get_params(); the exact key for
the zero-inflation term has changed across versions, so this script
auto-detects it and prints the full parameter dict on the first fit for you to
confirm.

Run (repo root, conda env csl-workflows or any env with penaltyblog):
    python "model comparison/zip_zero_inflation_param_test.py"
"""

import os
import statistics
import sys

import pandas as pd
import penaltyblog as pb
from tqdm import tqdm

# Resolve the league CSV relative to the repo root (../data from this file),
# so the script is portable across machines. Mirrors poisson_vs_zip_18mo_test.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

# The CSV dates are DD/MM/YYYY; a bare to_datetime() day/month-swaps them and
# drops every day>12 row. Use the production parser instead.
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from csl.date_utils import parse_date_only_series  # noqa: E402

XI = 0.001
LOOKBACK = pd.DateOffset(months=18)   # aligned to production dc.py
START_SEASON = 2025                   # first season to score, as in the other tests

# A fitted zero-inflation term at or below this is treated as "effectively
# zero" (ZIP has collapsed to Poisson for that refit).
ZERO_TOL = 1e-3

# Substrings we search for, in priority order, to locate the zero-inflation
# key inside clf.get_params(). Team ratings are attack_*/defence_*, plus a
# home advantage term, so the zero-inflation param is whatever is left.
ZI_KEY_CANDIDATES = ("zero_inflation", "zero", "inflat", "psi")


def find_zero_inflation_key(params):
    """Locate the zero-inflation key in a penaltyblog get_params() dict.

    Returns the key name, or None if nothing plausible is found. Ignores the
    per-team attack/defence entries and the home-advantage term.
    """
    keys = list(params.keys())
    for needle in ZI_KEY_CANDIDATES:
        for k in keys:
            if needle in k.lower():
                return k
    # Fallback: the single non-team, non-home scalar parameter, if unambiguous.
    leftovers = [
        k for k in keys
        if not k.lower().startswith(("attack", "defence", "defense"))
        and "home" not in k.lower()
    ]
    return leftovers[0] if len(leftovers) == 1 else None


df = pd.read_csv(CSV)
df["Date"] = parse_date_only_series(df["Date"])
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date", drop=False)

df["HExpG+"] = pd.to_numeric(df["HExpG+"], errors="coerce")
df["AExpG+"] = pd.to_numeric(df["AExpG+"], errors="coerce")

start_date = df.query("Season == @START_SEASON")["Date"].min()
run_dates = df["Date"][df["Date"] >= start_date].unique()
print(
    f"Start date: {pd.Timestamp(start_date).date()} | run dates: {len(run_dates)} "
    f"| lookback: 18 months | xi={XI}"
)

zi_key = None            # resolved once, on the first successful fit
zi_values = []           # (date, zero_inflation) per refit
n_fit = 0
n_failed = 0

for date in tqdm(run_dates, desc="dates"):
    lookback = pd.Timestamp(date) - LOOKBACK
    train = df[(df["Date"] < date) & (df["Date"] >= lookback)]
    train = train.dropna(subset=["HExpG+", "AExpG+", "Home", "Away"])
    if len(train) == 0:
        continue

    weights = pb.models.dixon_coles_weights(train["Date"], XI)
    try:
        clf = pb.models.ZeroInflatedPoissonGoalsModel(
            train["HExpG+"], train["AExpG+"], train["Home"], train["Away"], weights,
        )
        clf.fit()
        params = clf.get_params()
    except Exception as exc:  # noqa: BLE001 - a bad refit shouldn't abort the run
        n_failed += 1
        tqdm.write(f"  fit failed on {pd.Timestamp(date).date()}: {exc}")
        continue

    n_fit += 1
    if zi_key is None:
        # Show the full parameter dict once so the detected key can be verified.
        print("\nFirst successful fit — full get_params() dict:")
        for k, v in params.items():
            print(f"    {k!r}: {v}")
        zi_key = find_zero_inflation_key(params)
        print(f"\nDetected zero-inflation key: {zi_key!r}")
        if zi_key is None:
            raise SystemExit(
                "Could not auto-detect the zero-inflation parameter. Inspect the "
                "dict above and set zi_key manually."
            )

    zi_values.append((pd.Timestamp(date), float(params[zi_key])))

if not zi_values:
    raise SystemExit("No successful fits; nothing to summarise.")

vals = [v for _, v in zi_values]
n_zero = sum(1 for v in vals if abs(v) <= ZERO_TOL)

print("\n" + "=" * 56)
print(f"Zero-inflation parameter ({zi_key!r}) across {len(vals)} refits")
print(f"  (fits ok: {n_fit}, fits failed: {n_failed})")
print("=" * 56)
print(f"  min    : {min(vals):+.6f}")
print(f"  median : {statistics.median(vals):+.6f}")
print(f"  mean   : {statistics.fmean(vals):+.6f}")
print(f"  max    : {max(vals):+.6f}")
print(f"  stdev  : {statistics.pstdev(vals):.6f}")
print(f"  |value| <= {ZERO_TOL:g} (effectively Poisson): "
      f"{n_zero}/{len(vals)} refits ({n_zero / len(vals) * 100:.1f}%)")

median_abs = abs(statistics.median(vals))
print("\nVerdict:")
if median_abs <= ZERO_TOL:
    print("  Zero-inflation is ~0 at the typical refit -> ZIP has collapsed to")
    print("  Poisson. The ~0.0003 RPS edge over Poisson is noise; prefer the")
    print("  simpler/faster Poisson in production.")
else:
    print(f"  Median |zero-inflation| = {median_abs:.4f} is non-trivial -> ZIP is")
    print("  doing real work. Proceed to Step 2 (paired Wilcoxon) and Step 3")
    print("  (calibration) before deciding to keep or drop it.")

# Optional: dump the per-date series for plotting / inspection elsewhere.
out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "zip_zero_inflation_by_date.csv")
pd.DataFrame(zi_values, columns=["Date", "zero_inflation"]).to_csv(out_csv, index=False)
print(f"\nPer-date series written to: {out_csv}")
