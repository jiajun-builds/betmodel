"""
CLV / line-timing backtest for the Polymarket early-price strategy.

The play (per LIGAMX_PORTING_GUIDE section 13): bet the model's best +EV outcome at the
earliest catchable price (polymarket_open, ~T-48h) and win on closing-line value,
not on prediction accuracy. This scores that, in parallel, against TWO closing
benchmarks — polymarket_close and pinnacle_close — reporting:

  CLV  (line-timing): devigged (close - open) prob movement on the picked side.
       raw, plus EXCESS over the same-outcome drift baseline (section 13 rule #1: the
       market drifts every season and a naive pick inherits that for free).
  GAP  (model vs close): mean(model_prob - close_prob) on picks, and the realized
       ROI of those picks priced AT the close. Positive => the model has an edge
       over the closing line itself (not just good timing).
  ROI@open (realized): the actual money metric — picks settled at the open odds.

Walk-forward, out-of-sample: the production model is refit each ISO week on the
prior rolling window via dc.fit_production_model, same as eval/rps_backtest. All prices are devigged before comparison (Pinnacle ~5% vs
Polymarket ~0% overround would otherwise bias the gap).

    python -m ligamx.eval.clv_backtest [--test-start 2025-10-01]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ligamx import paths
from ligamx.models.dc import DEFAULT_TARGET, TRAINING_WINDOW_MONTHS, fit_production_model
from ligamx.eval.rps_backtest import RESULT_IDX

warnings.filterwarnings("ignore")

OUTCOMES = ("home", "draw", "away")
THRESHOLDS = [-1.0, 0.0, 0.02, 0.05, 0.10]  # min EV-at-open to place the bet


def _devig(odds):
    """Decimal 3-way odds -> de-vigged probabilities [home, draw, away]."""
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def _triplet(row, prefix):
    """Return [h, d, a] decimal odds for a venue prefix, or None if incomplete."""
    try:
        vals = [float(row[f"{prefix}_{k}"]) for k in ("h", "d", "a")]
    except (TypeError, ValueError, KeyError):
        return None
    return vals if all(np.isfinite(v) and v > 1.0 for v in vals) else None


def _load():
    df = pd.read_csv(paths.ligamx_data_csv())
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["Res"] = df["Res"].astype(str).str.strip()
    df = df[df["Res"].isin(["H", "D", "A"])].sort_values("Date").reset_index(drop=True)
    for c in ("HExpG+", "AExpG+"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _model_probs(df, test_start, min_train=80):
    """Walk-forward production-model probs per test match -> {row_index: [ph,pd,pa]}."""
    gh, ga = DEFAULT_TARGET
    fit_df = df.dropna(subset=[gh, ga])
    test = df[df["Date"] >= pd.Timestamp(test_start)].copy()
    test["_wk"] = test["Date"].dt.strftime("%G-W%V")
    out = {}
    for wk in sorted(test["_wk"].unique()):
        wk_rows = test[test["_wk"] == wk]
        cutoff = wk_rows["Date"].min()
        lo = cutoff - pd.DateOffset(months=TRAINING_WINDOW_MONTHS)
        train = fit_df[(fit_df["Date"] < cutoff) & (fit_df["Date"] >= lo)]
        if len(train) < min_train:
            continue
        try:
            m = fit_production_model(train)
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


def _records(df, probs):
    """Assemble per-match records carrying model probs, open, and both closes."""
    recs = []
    for idx, mp in probs.items():
        row = df.loc[idx]
        pm_open = _triplet(row, "polymarket_open")
        if pm_open is None:
            continue
        recs.append({
            "m": mp,
            "open": pm_open,
            "polymarket_close": _triplet(row, "polymarket_close"),
            "pinnacle_close": _triplet(row, "pinnacle_close"),
            "res": RESULT_IDX[row["Res"]],
        })
    return recs


def _drift(recs, close_key):
    """Same-outcome mean (close - open) devigged prob movement across the sample."""
    sample = [r for r in recs if r[close_key]]
    mov = np.zeros(3)
    for r in sample:
        po, pc = _devig(r["open"]), _devig(r[close_key])
        mov += np.array(pc) - np.array(po)
    return mov / len(sample) if sample else mov


def analyze(recs, close_key: str, drop_draw: bool = False):
    sample = [r for r in recs if r[close_key]]
    drift = _drift(recs, close_key)  # per-outcome baseline line movement
    tag = "  [DRAW DROPPED: pick home/away only]" if drop_draw else ""
    print(f"\n{'='*88}\nBENCHMARK: {close_key}   |   matches with open+close = {len(sample)}"
          f"   |   drift[H/D/A] = {100*drift[0]:+.2f}/{100*drift[1]:+.2f}/{100*drift[2]:+.2f} pp{tag}\n{'='*88}")
    header = f"{'EV_thr':>7}{'nbets':>7}{'hit%':>7}{'ROI@open':>10}{'CLV_raw':>9}{'CLV_exc':>9}{'gap_pp':>8}{'ROI@close':>10}"
    print(header)
    for thr in THRESHOLDS:
        n = hit = 0
        clv = clv_exc = gap = roi_open = roi_close = 0.0
        for r in sample:
            po, pc = _devig(r["open"]), _devig(r[close_key])
            ev = [r["m"][i] * r["open"][i] - 1 for i in range(3)]
            if drop_draw:
                ev[1] = -np.inf  # never bet the draw
            pick = int(np.argmax(ev))
            if ev[pick] <= thr:
                continue
            n += 1
            win = (r["res"] == pick)
            hit += win
            clv += (pc[pick] - po[pick]) * 100
            clv_exc += (pc[pick] - po[pick] - drift[pick]) * 100
            gap += (r["m"][pick] - pc[pick]) * 100
            roi_open += (r["open"][pick] - 1) if win else -1
            roi_close += (r[close_key][pick] - 1) if win else -1
        if n == 0:
            print(f"{thr:>7.2f}{0:>7}{'-':>7}{'-':>10}{'-':>9}{'-':>9}{'-':>8}{'-':>10}")
            continue
        print(f"{thr:>7.2f}{n:>7}{100*hit/n:>6.1f}%{roi_open/n*100:>9.2f}%"
              f"{clv/n:>9.2f}{clv_exc/n:>9.2f}{gap/n:>8.2f}{roi_close/n*100:>9.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--drop-draw", action="store_true", help="pick home/away only (section 13 fix)")
    args = ap.parse_args()

    df = _load()
    probs = _model_probs(df, args.test_start)
    recs = _records(df, probs)
    print(f"Test matches with model prob + polymarket_open: {len(recs)}  (test-start {args.test_start})")
    print("CLV_raw/exc = devigged (close-open) pp on picked side, raw & excess-over-drift.")
    print("gap_pp = model_prob - close_prob on picks. ROI = mean unit P/L at that price.")
    for close_key in ("polymarket_close", "pinnacle_close"):
        analyze(recs, close_key, drop_draw=args.drop_draw)


if __name__ == "__main__":
    main()
