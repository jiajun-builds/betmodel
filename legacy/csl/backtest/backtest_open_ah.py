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
Walk-forward backtest of the production model against Pinnacle OPENING
Asian-handicap lines.

This is the first unbiased, full-slate test of the CLV thesis. The main CSV now
carries, for every completed 2026 match, the Pinnacle opening handicap and both
prices (`pinnacle_open_ah` / `_h` / `_a`). We refit the production recipe
(ZeroInflatedPoissonGoalsModel on xG targets, 18-month lookback, xi=0.001)
strictly on data available BEFORE each round, convert the model's scoreline
grid into an exact expected return at that match's opening line, back any side
whose model EV clears a threshold, and settle the bet against the real result.

Why settle from the grid instead of `FootballProbabilityGrid.asian_handicap()`:
penaltyblog's helper does not expose quarter-line push mass cleanly (it returns
identical values for lines 0.0 / -0.25 / -0.50). We take the raw 15x15 scoreline
grid and settle each handicap ourselves, so quarter lines (half stake on each
neighbouring half-line) and integer-line pushes (stake refunded) are exact and
match how the lines settle at the book.

What it answers: if you had bet every model-vs-open divergence at these exact
opening prices, what was the realized ROI, hit rate, and does higher predicted
EV actually map to higher realized ROI (the only calibration that matters for
staking). NOTE: these are OPENING lines only, so this measures realized ROI at
open, not CLV (no closing lines captured yet) -- but ROI at open is the ground
truth that CLV is only a proxy for.

Requires the `pinnacle_open_ah` / `_h` / `_a` columns to be present in the main
CSV (`data/raw_data/CHN_Super League.csv`). Those opening lines are maintained
manually and are not part of the automation-refreshed production schema, so a
fresh checkout will have empty/missing columns until they are filled in.

Run (repo root, conda env csl-workflows or any env with penaltyblog):
    PYTHONPATH=src python backtest/backtest_open_ah.py
"""

import os

import numpy as np
import pandas as pd
import penaltyblog as pb
from tqdm import tqdm

from csl.date_utils import parse_date_only_series

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

# Production recipe (must mirror src/csl/models/dc.py).
PROD_XI = 0.001
PROD_LOOKBACK_MONTHS = 18

# Back a side when model EV per unit stake clears one of these thresholds.
EV_THRESHOLDS = [0.00, 0.02, 0.05, 0.10]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    # CSV dates are DD/MM/YYYY (legacy) or YYYY-MM-DD; parse_date_only_series
    # handles both. A plain to_datetime() would silently drop every day>12 row
    # and month/day-swap the rest, corrupting the walk-forward ordering.
    df["Date"] = parse_date_only_series(df["Date"])
    for c in ("HExpG+", "AExpG+", "HG", "AG",
              "pinnacle_open_ah", "pinnacle_open_ah_h", "pinnacle_open_ah_a"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    return df.sort_values("Date").reset_index(drop=True)


def settle(gd: float, line: float, odds: float) -> float:
    """Profit per unit stake for backing a side, given actual goal difference
    from that side's perspective (gd), the side's handicap line, and decimal
    odds. Handles quarter lines (half stake on each neighbouring half-line) and
    integer-line pushes (stake refunded)."""
    def one(adj: float) -> float:
        if adj > 1e-9:
            return odds - 1.0   # win
        if adj < -1e-9:
            return -1.0         # lose
        return 0.0              # push, stake refunded
    # Quarter line iff 4*line is odd (e.g. -0.25, -0.75): split the stake.
    if round(line * 4) % 2 != 0:
        return 0.5 * one(gd + line - 0.25) + 0.5 * one(gd + line + 0.25)
    return one(gd + line)


def gd_distribution(grid: np.ndarray):
    """Collapse a 15x15 scoreline grid (rows=home goals, cols=away goals) into
    P(goal difference) as parallel arrays (gd_values, probs), gd from home side."""
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n))  # home_i - away_j
    gds = np.arange(-(n - 1), n)
    probs = np.array([grid[diff == d].sum() for d in gds])
    return gds, probs


def model_ev(gds, probs, line: float, odds: float) -> float:
    """Model expected profit per unit stake for backing a side at (line, odds),
    where gds are goal differences from that side's perspective."""
    return float(sum(p * settle(g, line, odds) for g, p in zip(gds, probs)))


def run() -> None:
    df = load()
    graded = df[df["pinnacle_open_ah"].notna()
                & df["HG"].notna() & df["AG"].notna()].copy()
    if graded.empty:
        raise SystemExit("No rows with an opening AH line and a result to grade.")

    print(f"CSV: {os.path.relpath(CSV, REPO_ROOT)}")
    print(f"Gradeable matches (open AH line + result): {len(graded)} | "
          f"seasons {sorted(graded['Season'].unique())} | "
          f"rounds {graded['Round'].min()}-{graded['Round'].max()}")

    run_dates = sorted(graded["Date"].unique())
    records = []
    skipped = 0

    for date in tqdm(run_dates, desc="dates"):
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]

        # Trailing production window, strictly BEFORE this date -> no leakage.
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < 20:
            skipped += len(test)
            continue

        weights = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        try:
            clf = pb.models.ZeroInflatedPoissonGoalsModel(
                hist["HExpG+"], hist["AExpG+"], hist["Home"], hist["Away"], weights,
            )
            clf.fit()
        except Exception:
            skipped += len(test)
            continue

        for row in test.itertuples(index=False):
            try:
                grid = np.asarray(clf.predict(row.Home, row.Away).grid)
            except Exception:
                skipped += 1  # team unseen in the training window (e.g. promoted)
                continue

            gds, probs = gd_distribution(grid)
            line = float(row.pinnacle_open_ah)
            oh, oa = float(row.pinnacle_open_ah_h), float(row.pinnacle_open_ah_a)
            gd = float(row.HG - row.AG)

            ev_home = model_ev(gds, probs, line, oh)
            ev_away = model_ev(-gds, probs, -line, oa)  # away side: flip gd and line

            records.append({
                "Date": date.date(), "Season": row.Season, "Round": row.Round,
                "Home": row.Home, "Away": row.Away, "gd": gd, "line": line,
                "odds_home": oh, "odds_away": oa,
                "ev_home": ev_home, "ev_away": ev_away,
                "pl_home": settle(gd, line, oh),
                "pl_away": settle(-gd, -line, oa),
                # no-vig opening prob the market assigned to home covering
                "mkt_home_novig": (1 / oh) / (1 / oh + 1 / oa),
            })

    bets = pd.DataFrame(records)
    covered = len(bets)
    print(f"\nPredicted (bettable) matches: {covered} | "
          f"skipped (short history / unseen team): {skipped}")
    if bets.empty:
        raise SystemExit("Nothing predictable; check the training window.")

    # ---- Baselines: naive always-home / always-away at the opening line -------
    print("\n=== Baselines (bet every match, flat stake) ===")
    print(f"always-home ROI: {bets['pl_home'].mean():+.4f}   "
          f"always-away ROI: {bets['pl_away'].mean():+.4f}   (n={covered})")

    # ---- Strategy: back the higher-EV side when it clears the threshold -------
    # Choose the side with the greater model EV; bet it if that EV > threshold.
    best_side = np.where(bets["ev_home"] >= bets["ev_away"], "home", "away")
    best_ev = np.maximum(bets["ev_home"], bets["ev_away"])
    best_pl = np.where(best_side == "home", bets["pl_home"], bets["pl_away"])
    bets["best_side"], bets["best_ev"], bets["best_pl"] = best_side, best_ev, best_pl

    # ROI is a mean of per-bet P/L (sd ~1), so its standard error is
    # sd/sqrt(n) ~ 0.1 at n~100: any |ROI| within ~1 SE is indistinguishable
    # from zero. t = ROI/SE; |t| > ~2 is the bar for a real edge.
    print("\n=== Strategy: back the higher-EV side above an EV threshold ===")
    print(f"{'thr':>6} {'n_bets':>7} {'ROI':>8} {'SE':>7} {'t':>6} "
          f"{'profit':>8} {'win%':>7} {'avg_ev':>8}")
    summary = []
    for thr in EV_THRESHOLDS:
        sel = bets[bets["best_ev"] > thr]
        n = len(sel)
        if n == 0:
            print(f"{thr:>6.2f} {0:>7}      --      --     --       --      --       --")
            summary.append({"threshold": thr, "n_bets": 0, "roi": np.nan, "se": np.nan,
                            "t": np.nan, "profit": 0.0, "win_rate": np.nan})
            continue
        roi = sel["best_pl"].mean()
        se = sel["best_pl"].std(ddof=1) / np.sqrt(n)
        t = roi / se if se else np.nan
        winr = (sel["best_pl"] > 0).mean()
        print(f"{thr:>6.2f} {n:>7} {roi:>+8.4f} {se:>7.4f} {t:>+6.2f} "
              f"{sel['best_pl'].sum():>+8.2f} {winr:>7.1%} {sel['best_ev'].mean():>+8.4f}")
        summary.append({"threshold": thr, "n_bets": n, "roi": roi, "se": se, "t": t,
                        "profit": sel["best_pl"].sum(), "win_rate": winr})

    # Model EV vs realized: if predicted EVs sit far above realized ROI, the
    # model's handicap-cover probabilities are overconfident, not just unlucky.
    thr0 = bets[bets["best_ev"] > 0]
    over = thr0["best_ev"] - thr0["best_pl"]
    over_se = over.std(ddof=1) / np.sqrt(len(over)) if len(over) > 1 else float("nan")
    over_t = over.mean() / over_se if over_se else float("nan")
    print(f"\nModel-EV honesty check (thr>0): mean predicted EV "
          f"{thr0['best_ev'].mean():+.4f} vs realized ROI {thr0['best_pl'].mean():+.4f}"
          f"  -> model overstates its edge by {over.mean():+.4f}/unit")
    print(f"  paired (predicted EV - realized) mean {over.mean():+.4f} "
          f"+/- {over_se:.4f} (SE)  ->  t = {over_t:+.2f}  "
          f"(|t| > 2 => the overstatement is real, not variance)")

    # Cross-season replication: does the overstatement hold up season by season,
    # or is it one anomalous year? Each season is an independent sample.
    print("\n=== Cross-season replication (+EV bets, thr>0) ===")
    print(f"{'season':>7} {'n':>5} {'pred_ev':>9} {'realized':>9} {'t(ROI)':>7} {'overstate':>10}")
    for season in sorted(thr0["Season"].unique()):
        s = thr0[thr0["Season"] == season]
        roi = s["best_pl"].mean()
        se = s["best_pl"].std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else float("nan")
        t = roi / se if se else float("nan")
        print(f"{int(season):>7} {len(s):>5} {s['best_ev'].mean():>+9.3f} "
              f"{roi:>+9.3f} {t:>+7.2f} {s['best_ev'].mean() - roi:>+10.3f}")

    # ---- EV calibration: does higher predicted EV -> higher realized ROI? -----
    print("\n=== EV calibration (all +EV bets, thr>0, bucketed by model EV) ===")
    pos = bets[bets["best_ev"] > 0].copy()
    if len(pos) >= 10:
        pos["bucket"] = pd.qcut(pos["best_ev"], q=4, duplicates="drop")
        cal = pos.groupby("bucket", observed=True).agg(
            n=("best_pl", "size"), pred_ev=("best_ev", "mean"),
            realized_roi=("best_pl", "mean"))
        print(cal.to_string(float_format=lambda v: f"{v:+.4f}"))
    else:
        print(f"(only {len(pos)} +EV bets; too few to bucket)")

    # ---- Per-round P/L at thr>0, to eyeball variance --------------------------
    print("\n=== Per-round P/L (thr>0.00) ===")
    rnd = bets[bets["best_ev"] > 0].groupby("Round").agg(
        n=("best_pl", "size"), profit=("best_pl", "sum"), roi=("best_pl", "mean"))
    print(rnd.to_string(float_format=lambda v: f"{v:+.3f}"))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "backtest_open_ah_bets.csv")
    bets.to_csv(out, index=False)
    print(f"\nPer-bet detail written to: {out}")

    summ_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "backtest_open_ah_summary.csv")
    pd.DataFrame(summary).to_csv(summ_out, index=False)
    print(f"Threshold summary written to: {summ_out}")


if __name__ == "__main__":
    run()
