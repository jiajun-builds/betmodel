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
from ligamx.date_utils import parse_date_only_series

warnings.filterwarnings("ignore")

OUTCOMES = ("home", "draw", "away")
THRESHOLDS = [-1.0, 0.0, 0.02, 0.05, 0.10]  # min EV-at-open to place the bet


def _devig(odds):
    """Decimal 3-way odds -> de-vigged probabilities [home, draw, away]."""
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


# A pre-kickoff 3-way close never resolves an outcome. When one does, the stored
# "close" is a post-kickoff or settled price and carries the result -- pure
# look-ahead that inflates both CLV and ROI@close. Two Polymarket rows trip this
# (Tijuana-Mazatlan 2026-02-22 priced 0.999 on the draw it finished as; America-
# Tigres 2026-03-01 priced 0.000 on the home side while 1-4 down).
DEGENERATE_LO, DEGENERATE_HI = 0.01, 0.95


def _triplet(row, prefix):
    """Return [h, d, a] decimal odds for a venue prefix, or None if incomplete."""
    try:
        vals = [float(row[f"{prefix}_{k}"]) for k in ("h", "d", "a")]
    except (TypeError, ValueError, KeyError):
        return None
    return vals if all(np.isfinite(v) and v > 1.0 for v in vals) else None


def _is_prekickoff(odds):
    """False when a 3-way price has already resolved (in-play/settled leakage)."""
    if odds is None:
        return False
    p = _devig(odds)
    return min(p) >= DEGENERATE_LO and max(p) <= DEGENERATE_HI


def _load():
    df = pd.read_csv(paths.ligamx_data_csv())
    df["Date"] = parse_date_only_series(df["Date"])
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


def _records(df, probs, keep: set | None = None, open_key: str = "polymarket_open",
             close_keys=("polymarket_close", "pinnacle_close")):
    """Assemble per-match records carrying model probs, the open, and each close.

    ``keep`` restricts to fixtures whose Polymarket open was a real book (see
    odds/price_quality): the store seeds untraded events near 0.50 per outcome, and
    CLV measured from a placeholder is manufactured, not observed.

    ``open_key`` is the price you actually bet into, so the same harness scores an
    attack on any venue's opener (polymarket_open, betano_open, ...) as long as the
    schema carries its h/d/a triplet.
    """
    recs = []
    for idx, mp in probs.items():
        if keep is not None and idx not in keep:
            continue
        row = df.loc[idx]
        open_odds = _triplet(row, open_key)
        if open_odds is None:
            continue
        rec = {"m": mp, "open": open_odds, "res": RESULT_IDX[row["Res"]],
               "date": row["Date"]}
        for ck in close_keys:
            close = _triplet(row, ck)
            rec[ck] = close if _is_prekickoff(close) else None
        recs.append(rec)
    return recs


MIN_DRIFT_HISTORY = 20  # prior matches needed before a walk-forward baseline is usable


def _drift(recs, close_key):
    """Same-outcome mean (close - open) devigged prob movement across the sample.

    FULL-SAMPLE, so it peeks: a bet placed in October is scored against a baseline
    that includes the following May. Fine as a descriptive statistic, not fine as a
    backtest input -- use _walk_forward_drift for that.
    """
    sample = [r for r in recs if r[close_key]]
    mov = np.zeros(3)
    for r in sample:
        po, pc = _devig(r["open"]), _devig(r[close_key])
        mov += np.array(pc) - np.array(po)
    return mov / len(sample) if sample else mov


def _walk_forward_drift(recs, close_key):
    """Per-record drift baseline computed only from STRICTLY EARLIER matches.

    The excess-CLV metric subtracts the market's average movement on that outcome,
    because the market drifts every season and a naive pick inherits that for free.
    Estimating it on the whole sample means a bet is scored against movement that
    had not happened yet. This returns {id(record): drift vector} built from an
    expanding window of prior matches only; records without ``MIN_DRIFT_HISTORY``
    predecessors get None and are dropped by the caller rather than scored against
    a baseline that does not exist yet.
    """
    sample = sorted((r for r in recs if r[close_key]), key=lambda r: r["date"])
    out, running, n = {}, np.zeros(3), 0
    for r in sample:
        out[id(r)] = (running / n).copy() if n >= MIN_DRIFT_HISTORY else None
        po, pc = _devig(r["open"]), _devig(r[close_key])
        running += np.array(pc) - np.array(po)
        n += 1
    return out


def analyze(recs, close_key: str, drop_draw: bool = False, wf_drift: bool = True):
    sample = [r for r in recs if r[close_key]]
    drift = _drift(recs, close_key)  # per-outcome baseline line movement
    wf = _walk_forward_drift(recs, close_key) if wf_drift else None
    if wf is not None:
        sample = [r for r in sample if wf[id(r)] is not None]
    tag = "  [DRAW DROPPED: pick home/away only]" if drop_draw else ""
    mode = "walk-forward (prior matches only)" if wf_drift else "FULL SAMPLE (peeks)"
    print(f"\n{'='*88}\nBENCHMARK: {close_key}   |   matches with open+close = {len(sample)}"
          f"   |   drift baseline: {mode}\n"
          f"   full-sample drift[H/D/A] = {100*drift[0]:+.2f}/{100*drift[1]:+.2f}/{100*drift[2]:+.2f} pp{tag}\n{'='*88}")
    header = (f"{'EV_thr':>7}{'nbets':>7}{'hit%':>7}{'ROI@open':>10}{'CLV_raw':>9}"
              f"{'CLV_exc':>9}{'t_exc':>7}{'95% CI on CLV_exc':>21}{'gap_pp':>8}{'ROI@close':>10}")
    print(header)
    for thr in THRESHOLDS:
        clv_l, exc_l, gap_l, ropen_l, rclose_l, wins = [], [], [], [], [], 0
        for r in sample:
            po, pc = _devig(r["open"]), _devig(r[close_key])
            ev = [r["m"][i] * r["open"][i] - 1 for i in range(3)]
            if drop_draw:
                ev[1] = -np.inf  # never bet the draw
            pick = int(np.argmax(ev))
            if ev[pick] <= thr:
                continue
            win = (r["res"] == pick)
            wins += win
            base = drift if wf is None else wf[id(r)]
            clv_l.append((pc[pick] - po[pick]) * 100)
            exc_l.append((pc[pick] - po[pick] - base[pick]) * 100)
            gap_l.append((r["m"][pick] - pc[pick]) * 100)
            ropen_l.append((r["open"][pick] - 1) if win else -1)
            rclose_l.append((r[close_key][pick] - 1) if win else -1)
        n = len(clv_l)
        if n == 0:
            print(f"{thr:>7.2f}{0:>7}{'-':>7}{'-':>10}{'-':>9}{'-':>9}{'-':>7}{'-':>21}{'-':>8}{'-':>10}")
            continue
        exc = np.array(exc_l)
        # Standard error over bets. The drift baseline is estimated on the same
        # sample, so this slightly understates uncertainty -- it is an upper bound
        # on how much the number can be trusted, not a lower one.
        se = exc.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        t = exc.mean() / se if n > 1 and se > 0 else np.nan
        lo, hi = (exc.mean() - 1.96 * se, exc.mean() + 1.96 * se) if n > 1 else (np.nan, np.nan)
        print(f"{thr:>7.2f}{n:>7}{100*wins/n:>6.1f}%{np.mean(ropen_l)*100:>9.2f}%"
              f"{np.mean(clv_l):>9.2f}{exc.mean():>9.2f}{t:>7.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>21}"
              f"{np.mean(gap_l):>8.2f}{np.mean(rclose_l)*100:>9.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--drop-draw", action="store_true", help="pick home/away only (section 13 fix)")
    ap.add_argument("--traded-only", action="store_true",
                    help="drop fixtures whose Polymarket open was a pre-trading placeholder")
    ap.add_argument("--open-source", default="polymarket_open",
                    help="schema prefix of the price bet into (e.g. betano_open)")
    ap.add_argument("--benchmarks", default="polymarket_close,pinnacle_close",
                    help="comma-separated closing benchmarks to score against")
    ap.add_argument("--full-sample-drift", action="store_true",
                    help="score against a drift baseline fitted on the WHOLE sample "
                         "(peeks at the future; walk-forward is the default)")
    args = ap.parse_args()

    close_keys = tuple(k.strip() for k in args.benchmarks.split(",") if k.strip())
    df = _load()
    probs = _model_probs(df, args.test_start)
    keep = None
    if args.traded_only:
        from ligamx.odds import price_quality
        keep = price_quality.traded_open_rows(df)
    recs = _records(df, probs, keep, args.open_source, close_keys)
    print(f"Test matches with model prob + {args.open_source}: {len(recs)}  (test-start {args.test_start})"
          + ("  [real books only]" if args.traded_only else ""))
    print("CLV_raw/exc = devigged (close-open) pp on picked side, raw & excess-over-drift.")
    print("gap_pp = model_prob - close_prob on picks. ROI = mean unit P/L at that price.")
    for close_key in close_keys:
        analyze(recs, close_key, drop_draw=args.drop_draw,
                wf_drift=not args.full_sample_drift)


if __name__ == "__main__":
    main()
