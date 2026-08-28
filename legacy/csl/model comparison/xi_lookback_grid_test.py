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
Grid search over (xi, lookback window) for the production ZIP-on-xG recipe.

Production (dc.py) fits penaltyblog's ZeroInflatedPoissonGoalsModel on xG
targets (HExpG+/AExpG+) with Dixon-Coles time-decay weights, using a hard
18-month lookback and xi=0.001. Those two knobs are partly redundant (a long
window with a large xi down-weights old matches much like a short window), so
this searches them jointly rather than one at a time.

Method: walk-forward from the START_SEASON opener. On each match date, fit the
model on the trailing `lookback` window with decay `xi`, predict the day's
fixtures, and score the 1X2 outcome with the ranked probability score (RPS,
lower is better). Every config is scored on the SAME set of fixtures (the
intersection all configs could predict), so the comparison is apples-to-apples;
a shorter window that has never seen a promoted team would otherwise silently
drop the hard fixtures and flatter itself.

Because the RPS gaps between configs are tiny (~1e-3), the script also reports:
  - log-loss as a second, independent metric,
  - per-fixture RPS paired vs the current production config, with the
    bootstrap standard error of that mean difference, so we can tell a real
    edge from sampling noise.

Run (repo root, conda env csl-workflows or any env with penaltyblog):
    python "model comparison/xi_lookback_grid_test.py"
"""

import os
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

# The CSV dates are DD/MM/YYYY; a bare to_datetime() day/month-swaps them and
# drops every day>12 row. Use the production parser instead.
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from csl.date_utils import parse_date_only_series  # noqa: E402

START_SEASON = 2025  # first season to score, as in the other comparison tests

# The production knobs, used as the reference config for the paired comparison.
PROD_XI = 0.001
PROD_LOOKBACK = 18

# Search grid. `None` lookback means "use all history before the run date".
XI_GRID = [0.0, 0.0003, 0.0005, 0.0008, 0.001, 0.0013, 0.0016,
           0.002, 0.0025, 0.003, 0.004, 0.005, 0.007, 0.01]
LOOKBACK_GRID = [9, 12, 15, 18, 21, 24, 30, 36, None]

res_map = {"H": 0, "D": 1, "A": 2}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Date"] = parse_date_only_series(df["Date"])
    for c in ("HExpG+", "AExpG+"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["res_numeric"] = df["Res"].map(res_map)
    return df.sort_values("Date").reset_index(drop=True)


def rps_one(probs, outcome: int) -> float:
    """Ranked probability score for a single 3-way (H,D,A) prediction."""
    cum_p = np.cumsum(probs)
    cum_o = np.cumsum([1.0 if i == outcome else 0.0 for i in range(3)])
    return float(np.sum((cum_p[:-1] - cum_o[:-1]) ** 2) / (len(probs) - 1))


def logloss_one(probs, outcome: int) -> float:
    return float(-np.log(max(probs[outcome], 1e-12)))


def run() -> None:
    df = load()
    start_date = df.query("Season == @START_SEASON")["Date"].min()
    run_dates = sorted(df.loc[df["Date"] >= start_date, "Date"].unique())
    print(
        f"CSV: {os.path.relpath(CSV, REPO_ROOT)}\n"
        f"Start: {pd.Timestamp(start_date).date()} | run dates: {len(run_dates)} | "
        f"grid: {len(XI_GRID)} xi x {len(LOOKBACK_GRID)} lookback = "
        f"{len(XI_GRID) * len(LOOKBACK_GRID)} configs"
    )

    configs = [(xi, lb) for lb in LOOKBACK_GRID for xi in XI_GRID]
    # Per-config, per-fixture predictions: preds[cfg][fixture_key] = (rps, logloss)
    preds = {cfg: {} for cfg in configs}
    observed = {}

    for date in tqdm(run_dates, desc="dates"):
        date = pd.Timestamp(date)
        test = df[df["Date"] == date]
        test = test[test["res_numeric"].notna()]
        if test.empty:
            continue

        # Full trailing history once; each lookback is a suffix of it.
        hist = df[(df["Date"] < date)].dropna(subset=["HExpG+", "AExpG+"])
        if hist.empty:
            continue

        for lb in LOOKBACK_GRID:
            if lb is None:
                train = hist
            else:
                train = hist[hist["Date"] >= date - pd.DateOffset(months=lb)]
            if len(train) < 20:
                continue

            for xi in XI_GRID:
                weights = pb.models.dixon_coles_weights(train["Date"], xi)
                try:
                    clf = pb.models.ZeroInflatedPoissonGoalsModel(
                        train["HExpG+"], train["AExpG+"],
                        train["Home"], train["Away"], weights,
                    )
                    clf.fit()
                except Exception:
                    continue

                cfg = (xi, lb)
                for row in test.itertuples(index=False):
                    key = (date, row.Home, row.Away)
                    try:
                        p = clf.predict(row.Home, row.Away).home_draw_away
                    except Exception:
                        continue
                    outcome = int(row.res_numeric)
                    preds[cfg][key] = (rps_one(p, outcome), logloss_one(p, outcome))
                    observed[key] = outcome

    # Fair comparison: score every config on the fixtures ALL configs predicted.
    common = set(observed)
    for cfg in configs:
        common &= set(preds[cfg])
    common = sorted(common)
    total_fixtures = len(observed)
    print(f"\nScored fixtures common to all configs: {len(common)} / {total_fixtures} seen")

    if not common:
        raise SystemExit("No fixture is predictable by every config; shrink the grid.")

    rows = []
    prod_key = (PROD_XI, PROD_LOOKBACK)
    prod_rps = np.array([preds[prod_key][k][0] for k in common])
    for cfg in configs:
        r = np.array([preds[cfg][k][0] for k in common])
        ll = np.array([preds[cfg][k][1] for k in common])
        diff = r - prod_rps  # per-fixture RPS vs production; negative = better
        rows.append({
            "xi": cfg[0],
            "lookback": cfg[1] if cfg[1] is not None else "all",
            "coverage": len(preds[cfg]),
            "RPS": r.mean(),
            "logloss": ll.mean(),
            "d_vs_prod": diff.mean(),
            "se_boot": bootstrap_se(diff),
        })

    grid = pd.DataFrame(rows).sort_values("RPS").reset_index(drop=True)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.5f}")

    print("\n=== Top 15 configs by RPS (lower is better) ===")
    print(grid.head(15).to_string(index=False))

    prod_row = grid[(grid["xi"] == PROD_XI) & (grid["lookback"] == PROD_LOOKBACK)]
    print(f"\nProduction (xi={PROD_XI}, lookback={PROD_LOOKBACK}mo) rank: "
          f"{prod_row.index[0] + 1} / {len(grid)}")
    print(prod_row.to_string(index=False))

    best = grid.iloc[0]
    print("\n=== Best config ===")
    print(best.to_string())
    z = best["d_vs_prod"] / best["se_boot"] if best["se_boot"] else float("nan")
    print(f"\nBest vs production: mean per-fixture RPS diff = {best['d_vs_prod']:+.6f} "
          f"+/- {best['se_boot']:.6f} (boot SE)  ->  z = {z:+.2f}")
    print("|z| < ~2 means the edge is within sampling noise; the current "
          "settings are then statistically as good as the grid optimum.")

    # RPS heatmap (rows = lookback, cols = xi) for eyeballing the surface.
    heat = grid.pivot_table(index="lookback", columns="xi", values="RPS")
    order = [lb if lb is not None else "all" for lb in LOOKBACK_GRID]
    heat = heat.reindex(order)
    print("\n=== RPS surface (rows=lookback months, cols=xi) ===")
    print(heat.to_string())

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "xi_lookback_grid_results.csv")
    grid.to_csv(out, index=False)
    print(f"\nFull grid written to: {out}")


def bootstrap_se(diff: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    """Bootstrap standard error of the mean of a paired difference vector."""
    if len(diff) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    return float(diff[idx].mean(axis=1).std(ddof=1))


if __name__ == "__main__":
    run()
