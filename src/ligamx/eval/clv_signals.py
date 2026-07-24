"""Multi-benchmark CLV / signal-slicing backtest for the Polymarket early play.

Walk-forward (weekly refit, out-of-sample) with the production engine, scoring the
"bet the best +EV outcome at the Polymarket open (~T-48h)" play against FOUR
closing benchmarks:

  polymarket_close  - the same venue's close (pure line-timing signal)
  pinnacle_close    - sharp benchmark (sparse: football-data dropped it after Oct-2025)
  avg_close (AvgC)  - market-average close (football-data, free)
  betfair_ex_close  - Betfair exchange close (football-data BFEC, free)

For each benchmark it reports, per EV threshold and per slice:
  hit%       realized win rate of the picked side
  ROI@open   realized unit P/L settling at the OPEN price (the money metric)
  CLV        devigged (close-open) pp on the picked side (raw line movement)
  CLVexc     CLV minus the same-outcome drift baseline (section-13 rule #1)
  gap        model_prob - close_prob on picks (does the model beat the close?)
  ROI@close  realized unit P/L at the close (edge over the closing line itself)

Model-free baselines (always-home/draw/away/favorite/longshot) are printed first
so any model slice can be read as excess over the naive market drift. All prices
are devigged before comparison. Draw probabilities are recalibrated with the
production DRAW_CALIBRATION_ALPHA unless overridden.

    python -m ligamx.eval.clv_signals [--family negbinom] [--xw 0.25] [--xi 0.0015]
        [--window 24] [--test-start 2025-10-01] [--draw-alpha 0.85] [--refresh]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import penaltyblog as pb
from penaltyblog.models import dixon_coles_weights

from ligamx.models.dc import (
    DRAW_CALIBRATION_ALPHA,
    MODEL_XI,
    TRAINING_WINDOW_MONTHS,
    calibrate_1x2,
)
from ligamx.eval.football_data import load_enriched
from ligamx.eval.rps_backtest import RESULT_IDX, _shrink_inplace

warnings.filterwarnings("ignore")

# Production xG-blend weight (HExpG+ = 0.25*HxG + 0.75*HG); mirrored here so the
# backtest recomputes the same training target from raw HxG/HG columns.
DEFAULT_XW = 0.25

FAMILIES = {
    "poisson": pb.models.PoissonGoalsModel,
    "dixoncoles": pb.models.DixonColesGoalModel,
    "negbinom": pb.models.NegativeBinomialGoalModel,
    "bivariate": pb.models.BivariatePoissonGoalModel,
    "zip": pb.models.ZeroInflatedPoissonGoalsModel,
    "weibull": pb.models.WeibullCopulaGoalsModel,
}

CLOSES = {
    "polymarket_close": ("polymarket_close_h", "polymarket_close_d", "polymarket_close_a"),
    "pinnacle_close": ("pinnacle_close_h", "pinnacle_close_d", "pinnacle_close_a"),
    "avg_close": ("AvgCH", "AvgCD", "AvgCA"),
    "betfair_ex_close": ("BFECH", "BFECD", "BFECA"),
}


def devig(odds):
    """Decimal 3-way odds -> (devigged probs [h,d,a], overround = sum of inverses)."""
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv], s


def triplet(row, cols):
    """[h, d, a] decimals for the given columns, or None if incomplete/invalid."""
    try:
        v = [float(row[c]) for c in cols]
    except (TypeError, ValueError, KeyError):
        return None
    return v if all(np.isfinite(x) and x > 1.0 for x in v) else None


def load():
    """Enriched match history, restricted to played matches with usable xG."""
    df = load_enriched()
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Res"] = df["Res"].astype(str).str.strip()
    df = df[df["Res"].isin(["H", "D", "A"])].sort_values("Date").reset_index(drop=True)
    for c in ("HxG", "AxG", "HG", "AG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def model_probs(df, family=None, xw=DEFAULT_XW, xi=MODEL_XI,
                window=TRAINING_WINDOW_MONTHS, test_start="2025-10-01",
                use_shrink=True, min_train=80):
    """Walk-forward 1X2 probabilities per test-match row index -> [ph, pd, pa].

    Refits weekly on the trailing rolling window (Dixon-Coles weighted) on the
    xG-blend target; optional production shrinkage. Returns raw (uncalibrated) probs.
    """
    cls = FAMILIES[family or "negbinom"]
    p = df.copy()
    p["th"] = xw * p["HxG"] + (1 - xw) * p["HG"]
    p["ta"] = xw * p["AxG"] + (1 - xw) * p["AG"]
    fit_df = p.dropna(subset=["th", "ta"])
    test = p[p["Date"] >= pd.Timestamp(test_start)].copy()
    test["_wk"] = test["Date"].dt.strftime("%G-W%V")

    out = {}
    for wk in sorted(test["_wk"].unique()):
        wk_rows = test[test["_wk"] == wk]
        cutoff = wk_rows["Date"].min()
        lo = cutoff - pd.DateOffset(months=window)
        train = fit_df[(fit_df["Date"] < cutoff) & (fit_df["Date"] >= lo)]
        if len(train) < min_train:
            continue
        w = dixon_coles_weights(train["Date"], xi=xi)
        gh = train["th"].to_numpy(dtype=float, copy=True)
        ga = train["ta"].to_numpy(dtype=float, copy=True)
        try:
            m = cls(gh, ga, train["Home"], train["Away"], w)
            m.fit()
            if use_shrink:
                _shrink_inplace(m, w, train["Home"], train["Away"])
        except Exception:
            continue
        teams = set(m.teams)
        for idx, mt in wk_rows.iterrows():
            if mt["Home"] in teams and mt["Away"] in teams:
                try:
                    out[idx] = list(m.predict(mt["Home"], mt["Away"]).home_draw_away)
                except Exception:
                    pass
    return out


def apply_draw_alpha(probs, alpha):
    """Return a new {idx: [h,d,a]} with the production draw recalibration applied."""
    if alpha == 1.0:
        return dict(probs)
    return {i: list(calibrate_1x2(p[0], p[1], p[2], alpha)) for i, p in probs.items()}


def build_records(df, probs):
    """Per-match records carrying model probs, the Polymarket open, and every close."""
    recs = []
    for idx, mp in probs.items():
        row = df.loc[idx]
        po = triplet(row, ("polymarket_open_h", "polymarket_open_d", "polymarket_open_a"))
        if po is None:
            continue
        r = {"m": mp, "open": po, "res": RESULT_IDX[row["Res"]],
             "date": row["Date"], "home": row["Home"], "away": row["Away"]}
        for key, cols in CLOSES.items():
            r[key] = triplet(row, cols)
        recs.append(r)
    return recs


def drift_vec(recs, close_key):
    """Same-outcome mean (close - open) devigged movement across the close's sample."""
    sample = [r for r in recs if r[close_key]]
    mov = np.zeros(3)
    for r in sample:
        po, _ = devig(r["open"])
        pc, _ = devig(r[close_key])
        mov += np.array(pc) - np.array(po)
    return (mov / len(sample)) if sample else mov


def ev_pick(r, drop_draw=False):
    """Best +EV outcome at the open and its EV; optionally never pick the draw."""
    ev = [r["m"][i] * r["open"][i] - 1 for i in range(3)]
    if drop_draw:
        ev[1] = -np.inf
    k = int(np.argmax(ev))
    return k, ev[k]


def accumulate(sample, close_key, drift, pick_fn, thr=-np.inf):
    """Aggregate a slice: pick_fn(r) -> (pick_idx | None, ev); keep picks with ev>thr."""
    st = dict(n=0, hit=0, clv=0.0, clv_exc=0.0, gap=0.0, roi_open=0.0, roi_close=0.0)
    for r in sample:
        po, _ = devig(r["open"])
        pc, _ = devig(r[close_key])
        pick, ev = pick_fn(r)
        if pick is None or ev <= thr:
            continue
        st["n"] += 1
        win = r["res"] == pick
        st["hit"] += win
        st["clv"] += (pc[pick] - po[pick]) * 100
        st["clv_exc"] += (pc[pick] - po[pick] - drift[pick]) * 100
        st["gap"] += (r["m"][pick] - pc[pick]) * 100
        st["roi_open"] += (r["open"][pick] - 1) if win else -1
        st["roi_close"] += (r[close_key][pick] - 1) if win else -1
    return st


def _line(st, label):
    n = st["n"]
    if n == 0:
        print(f"  {label:<26}{'-':>6}")
        return
    print(f"  {label:<26}{n:>6}{100*st['hit']/n:>7.1f}%"
          f"{100*st['roi_open']/n:>9.2f}%{st['clv']/n:>8.2f}{st['clv_exc']/n:>8.2f}"
          f"{st['gap']/n:>8.2f}{100*st['roi_close']/n:>9.2f}%")


def analyze(recs, close_key):
    sample = [r for r in recs if r[close_key]]
    if not sample:
        print(f"\nBENCHMARK {close_key}: no matches with open+close, skipped")
        return
    drift = drift_vec(recs, close_key)
    print(f"\n{'='*92}\nBENCHMARK {close_key}   n={len(sample)}   drift[H/D/A] = "
          f"{100*drift[0]:+.2f}/{100*drift[1]:+.2f}/{100*drift[2]:+.2f} pp")
    print(f"  {'slice':<26}{'n':>6}{'hit':>8}{'ROI@open':>9}{'CLV':>8}{'CLVexc':>8}{'gap':>8}{'ROI@cl':>9}")

    for name, k in [("always-home", 0), ("always-draw", 1), ("always-away", 2)]:
        _line(accumulate(sample, close_key, drift, lambda r, k=k: (k, 0.0)), f"[base] {name}")
    _line(accumulate(sample, close_key, drift, lambda r: (int(np.argmin(r["open"])), 0.0)),
          "[base] always-favorite")
    _line(accumulate(sample, close_key, drift, lambda r: (int(np.argmax(r["open"])), 0.0)),
          "[base] always-longshot")

    for thr in [-1.0, 0.0, 0.05, 0.10]:
        _line(accumulate(sample, close_key, drift, ev_pick, thr), f"EV>{thr:g}")
    for thr in [0.0, 0.05]:
        _line(accumulate(sample, close_key, drift, lambda r: ev_pick(r, True), thr),
              f"no-draw EV>{thr:g}")
    for name, k in [("home", 0), ("draw", 1), ("away", 2)]:
        def pf(r, k=k):
            pick, ev = ev_pick(r)
            return (pick, ev) if pick == k else (None, 0.0)
        _line(accumulate(sample, close_key, drift, pf, 0.0), f"EV>0 & pick={name}")
    for name, lohi in [("odds<2.2", (1.0, 2.2)), ("2.2-3.2", (2.2, 3.2)), (">3.2", (3.2, 99.0))]:
        def pf(r, lohi=lohi):
            pick, ev = ev_pick(r)
            return (pick, ev) if lohi[0] <= r["open"][pick] < lohi[1] else (None, 0.0)
        _line(accumulate(sample, close_key, drift, pf, 0.0), f"EV>0 & {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="negbinom", choices=list(FAMILIES))
    ap.add_argument("--xw", type=float, default=DEFAULT_XW)
    ap.add_argument("--xi", type=float, default=MODEL_XI)
    ap.add_argument("--window", type=int, default=TRAINING_WINDOW_MONTHS)
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--draw-alpha", type=float, default=DRAW_CALIBRATION_ALPHA,
                    help="draw deflation applied to model probs (1.0 disables)")
    ap.add_argument("--refresh", action="store_true", help="re-download football-data first")
    args = ap.parse_args()

    if args.refresh:
        from ligamx.eval import football_data
        football_data.refresh()

    df = load()
    raw = model_probs(df, args.family, args.xw, args.xi, args.window, args.test_start)
    probs = apply_draw_alpha(raw, args.draw_alpha)
    recs = build_records(df, probs)
    print(f"engine={args.family} xW={args.xw} xi={args.xi} win={args.window}mo "
          f"draw-alpha={args.draw_alpha}")
    print(f"matches with model prob + polymarket_open: {len(recs)}")

    ors_open = [devig(r["open"])[1] for r in recs]
    print(f"\npolymarket_open overround: mean {100*(np.mean(ors_open)-1):.2f}%")
    for key in CLOSES:
        s = [devig(r[key])[1] for r in recs if r[key]]
        if s:
            print(f"{key} overround: mean {100*(np.mean(s)-1):.2f}%  (n={len(s)})")

    md = np.mean([r["m"][1] for r in recs])
    mo = np.mean([devig(r["open"])[0][1] for r in recs])
    rd = np.mean([1.0 if r["res"] == 1 else 0.0 for r in recs])
    print(f"\ndraw calibration: model {md:.3f} | market open {mo:.3f} | realized {rd:.3f}")

    for close_key in CLOSES:
        analyze(recs, close_key)


if __name__ == "__main__":
    main()
