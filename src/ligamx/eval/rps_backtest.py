"""
Walk-forward RPS backtest comparing goal-model families on Liga MX.

For each ISO-week refit point, train on all played matches strictly before that
week (within the rolling window, Dixon-Coles weighted) and predict that week's
matches. Score the 1X2 predictions with the Ranked Probability Score (RPS,
lower = better), plus log-loss and accuracy for context.

Compared: several penaltyblog distributions (+ the production shrinkage variant)
against three baselines — uniform, training-window base rates, and a de-vigged
Pinnacle market benchmark (on the subset of matches that have pinnacle_close_h/d/a).
Runs both the xG-blend and actual-goals training targets.

    python -m ligamx.eval.rps_backtest [--test-start 2025-01-01] [--target xG|goals|both]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import penaltyblog as pb
from penaltyblog.models import dixon_coles_weights

from ligamx import paths
from ligamx.date_utils import parse_date_only_series
from ligamx.models.dc import (
    MODEL_XI,
    SHRINKAGE_K,
    TRAINING_WINDOW_MONTHS,
    _shrink_ratings,
    fit_production_model,
)

warnings.filterwarnings("ignore")

RESULT_IDX = {"H": 0, "D": 1, "A": 2}  # ordered [home, draw, away] for RPS

# name -> (penaltyblog class, apply_shrinkage). These are fed a ROUNDED target:
# penaltyblog truncates its goal arguments to integers, so handing them the raw
# xG blend would fit floor(xG) and deflate every rate by ~30% (see
# models/continuous_poisson). Rounding is mean-preserving, which is the fairest
# comparison the library allows.
MODELS = {
    "Poisson": (pb.models.PoissonGoalsModel, False),
    "DixonColes": (pb.models.DixonColesGoalModel, False),
    "NegBinom": (pb.models.NegativeBinomialGoalModel, False),
    "NegBinom+Shrink": (pb.models.NegativeBinomialGoalModel, True),
    "BivariatePoisson": (pb.models.BivariatePoissonGoalModel, False),
    "ZeroInfPoisson": (pb.models.ZeroInflatedPoissonGoalsModel, False),
    "WeibullCopula": (pb.models.WeibullCopulaGoalsModel, False),
}

# The production model, fitted on the exact continuous target via our own MLE.
PRODUCTION_MODELS = {
    "ContPoisson": False,
    "ContPoisson+Shrink": True,
}
BASELINES = ["Uniform", "BaseRate", "Market"]


def rps(p, oi: int) -> float:
    """Ranked Probability Score for a 3-outcome ordered forecast."""
    o = [0.0, 0.0, 0.0]
    o[oi] = 1.0
    cp = co = s = 0.0
    for i in range(2):  # r - 1 = 2 cumulative terms
        cp += p[i]
        co += o[i]
        s += (cp - co) ** 2
    return s / 2.0


def log_loss(p, oi: int) -> float:
    return float(-np.log(max(p[oi], 1e-12)))


def accuracy(p, oi: int) -> float:
    return 1.0 if int(np.argmax(p)) == oi else 0.0


def _shrink_inplace(clf, weights, home_series, away_series) -> None:
    """Apply the production shrinkage to a fitted model's team ratings in place."""
    teams = list(clf.teams)
    n = len(teams)
    eff = {t: 0.0 for t in teams}
    for w, h, a in zip(np.asarray(weights, dtype=float), home_series, away_series):
        if h in eff:
            eff[h] += w
        if a in eff:
            eff[a] += w
    n_arr = np.array([eff[t] for t in teams])
    clf._params[:n] = _shrink_ratings(np.asarray(clf._params[:n], dtype=float), n_arr, SHRINKAGE_K)
    clf._params[n : 2 * n] = _shrink_ratings(np.asarray(clf._params[n : 2 * n], dtype=float), n_arr, SHRINKAGE_K)


def _market_probs(row):
    """De-vig Pinnacle 1X2 closing odds (pinnacle_close_h/d/a decimals) to probabilities."""
    try:
        inv = [1.0 / float(row["pinnacle_close_h"]),
               1.0 / float(row["pinnacle_close_d"]),
               1.0 / float(row["pinnacle_close_a"])]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    s = sum(inv)
    if not np.isfinite(s) or s <= 0:
        return None
    return [x / s for x in inv]


def _load_played(target: str):
    gh, ga = ("HExpG+", "AExpG+") if target == "xG" else ("HG", "AG")
    df = pd.read_csv(paths.ligamx_data_csv())
    df["Date"] = parse_date_only_series(df["Date"])
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["Res"] = df["Res"].astype(str).str.strip()
    played = df[df["Res"].isin(["H", "D", "A"])].sort_values("Date").reset_index(drop=True)
    played[gh] = pd.to_numeric(played[gh], errors="coerce")
    played[ga] = pd.to_numeric(played[ga], errors="coerce")
    played = played.dropna(subset=[gh, ga]).reset_index(drop=True)
    return played, gh, ga


def collect(target: str, test_start: str, min_train: int = 80) -> list:
    """Walk-forward per ISO week; return per-test-match prediction records."""
    played, gh, ga = _load_played(target)
    test = played[played["Date"] >= pd.Timestamp(test_start)].copy()
    test["week"] = test["Date"].dt.strftime("%G-W%V")
    weeks = sorted(test["week"].unique())

    records = []
    for wk in weeks:
        wk_matches = test[test["week"] == wk]
        cutoff = wk_matches["Date"].min()
        lo = cutoff - pd.DateOffset(months=TRAINING_WINDOW_MONTHS)
        train = played[(played["Date"] < cutoff) & (played["Date"] >= lo)]
        if len(train) < min_train:
            continue

        w = dixon_coles_weights(train["Date"], xi=MODEL_XI)
        # penaltyblog's Cython loss needs writable float64 buffers; integer goals
        # (int64) otherwise raise "buffer source array is read-only". Round first —
        # the library truncates anyway, and rounding at least preserves the mean.
        gh_f = train[gh].round().to_numpy(dtype=float, copy=True)
        ga_f = train[ga].round().to_numpy(dtype=float, copy=True)
        fitted = {}
        for name, (cls, shrink) in MODELS.items():
            try:
                m = cls(gh_f, ga_f, train["Home"], train["Away"], w)
                m.fit()
                if shrink:
                    _shrink_inplace(m, w, train["Home"], train["Away"])
                fitted[name] = (m, set(m.teams))
            except Exception:
                fitted[name] = None
        for name, shrink in PRODUCTION_MODELS.items():
            try:
                m = fit_production_model(train, target=(gh, ga), shrink=shrink)
                fitted[name] = (m, set(m.teams))
            except Exception:
                fitted[name] = None

        br = train["Res"].value_counts(normalize=True)
        base = np.array([br.get("H", 1 / 3), br.get("D", 1 / 3), br.get("A", 1 / 3)], dtype=float)
        base = list(base / base.sum())

        for _, mt in wk_matches.iterrows():
            h, a = mt["Home"], mt["Away"]
            rec = {"oi": RESULT_IDX[mt["Res"]], "Uniform": [1 / 3, 1 / 3, 1 / 3],
                   "BaseRate": base, "Market": _market_probs(mt)}
            for name, f in fitted.items():
                if f is None:
                    rec[name] = None
                    continue
                m, teams = f
                if h not in teams or a not in teams:
                    rec[name] = None
                    continue
                try:
                    rec[name] = list(m.predict(h, a).home_draw_away)
                except Exception:
                    rec[name] = None
            records.append(rec)
    return records


def _mean(rows, name, fn):
    vals = [fn(r[name], r["oi"]) for r in rows if r.get(name) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(records: list, target: str) -> pd.DataFrame:
    model_names = list(MODELS.keys()) + list(PRODUCTION_MODELS.keys())
    # Comparable set: every model produced a prediction (both teams known in training).
    comp = [r for r in records if all(r.get(n) is not None for n in model_names)]
    print(f"\n{'='*66}\nTARGET = {target}   |   test matches = {len(records)}   |   comparable = {len(comp)}\n{'='*66}")

    rows = []
    for n in model_names + ["Uniform", "BaseRate"]:
        rows.append({"model": n, "RPS": _mean(comp, n, rps),
                     "LogLoss": _mean(comp, n, log_loss), "Acc": _mean(comp, n, accuracy)})
    tbl = pd.DataFrame(rows).sort_values("RPS").reset_index(drop=True)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Market head-to-head on the odds subset.
    mkt = [r for r in comp if r.get("Market") is not None]
    if mkt:
        print(f"\n-- vs Pinnacle market (subset with odds: {len(mkt)} matches) --")
        mrows = [{"model": "Market", "RPS": _mean(mkt, "Market", rps)}]
        for n in model_names + ["BaseRate"]:
            mrows.append({"model": n, "RPS": _mean(mkt, n, rps)})
        mt = pd.DataFrame(mrows).sort_values("RPS").reset_index(drop=True)
        print(mt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--target", default="both", choices=["xG", "goals", "both"])
    args = ap.parse_args()

    targets = ["xG", "goals"] if args.target == "both" else [args.target]
    for t in targets:
        recs = collect(t, args.test_start)
        summarize(recs, t)


if __name__ == "__main__":
    main()
