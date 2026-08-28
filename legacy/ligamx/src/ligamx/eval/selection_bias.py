"""
Is the model's problem the model, or the way bets are picked from it?

THE PUZZLE. Overall calibration is fine ([[continuous-target-fix]]), yet on the
argmax-EV picks the model says 34.76%, the close says 30.09% and realized is 28.26%.
Both can be true at once: argmax selects the outcome whose model error is most
positive, so a perfectly calibrated model still produces overconfident *picks*. That
is the winner's curse, and it is a defect of the selector, not of the forecaster.

THE TEST. Blend the model with the price the market showed at bet time, as a log
opinion pool over the three outcomes:

    p_blend  ~  p_market^(1-L) * p_model^L        (renormalised)

L=0 is the market, L=1 is the model, and L is fitted by maximum likelihood on real
results. So L answers the only question that matters here: **how much of the model's
disagreement with the market is information rather than noise?**

  L ~ 1   the model's deviations are real; the selector is what is broken, and
          shrinking picks toward the market should fix it.
  L ~ 0   the model adds nothing over the price. No selection rule can rescue that,
          because there is nothing to select on.
  L < 0   the deviations are actively wrong and should be faded.

Fitting one parameter over whole matches (rather than pooling 3 correlated legs into
a binary regression) keeps the mutually-exclusive outcomes honest; the CI comes from
a match-level bootstrap for the same reason.

Prices are the T-48h Polymarket open -- what was actually available when the bet is
placed. Using the close as the prior would measure something untradeable.

    python -m ligamx.eval.selection_bias [--test-start 2025-10-01] [--boot 2000]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ligamx.eval.clv_backtest import _devig, _load, _model_probs, _triplet
from ligamx.eval.rps_backtest import RESULT_IDX

warnings.filterwarnings("ignore")

EPS = 1e-6
OUTCOMES = ("home", "draw", "away")


def blend(mkt: np.ndarray, mod: np.ndarray, lam: float) -> np.ndarray:
    """Log opinion pool: p ~ mkt^(1-lam) * mod^lam, renormalised per match."""
    z = (1.0 - lam) * np.log(np.clip(mkt, EPS, 1)) + lam * np.log(np.clip(mod, EPS, 1))
    z -= z.max(axis=1, keepdims=True)          # stabilise before exponentiating
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _nll(lam: float, mkt: np.ndarray, mod: np.ndarray, res: np.ndarray) -> float:
    p = blend(mkt, mod, lam)
    return -np.log(np.clip(p[np.arange(len(res)), res], EPS, 1)).sum()


def fit_lambda(mkt: np.ndarray, mod: np.ndarray, res: np.ndarray) -> float:
    """MLE of the blend weight. Unbounded on purpose: L>1 and L<0 are real answers."""
    r = minimize_scalar(_nll, args=(mkt, mod, res), bounds=(-2.0, 3.0), method="bounded")
    return float(r.x)


def rps(p: np.ndarray, res: np.ndarray) -> float:
    """Ranked probability score (lower is better), normalised by r-1 so it is
    comparable with eval/rps_backtest -- Liga MX sits around 0.20, not 0.40."""
    cum_p = np.cumsum(p, axis=1)
    obs = np.zeros_like(p)
    obs[np.arange(len(res)), res] = 1.0
    cum_o = np.cumsum(obs, axis=1)
    return float(((cum_p - cum_o) ** 2)[:, :2].sum(axis=1).mean() / (p.shape[1] - 1))


def load_records(test_start: str, traded_only: bool = True):
    """Per-match (market open, model, close, result), restricted to real books."""
    df = _load()
    probs = _model_probs(df, test_start)
    keep = None
    if traded_only:
        from ligamx.odds import price_quality
        keep = price_quality.traded_open_rows(df)

    mkt, mod, close, res = [], [], [], []
    for idx, mp in probs.items():
        if keep is not None and idx not in keep:
            continue
        row = df.loc[idx]
        op = _triplet(row, "polymarket_open")
        cl = _triplet(row, "polymarket_close")
        if op is None or cl is None:
            continue
        mkt.append(_devig(op))
        mod.append(list(mp))
        close.append(_devig(cl))
        res.append(RESULT_IDX[row["Res"]])
    return (np.array(mkt), np.array(mod), np.array(close), np.array(res))


# football-data closing triplets, best first: the market average is the most complete
# column, Betfair's exchange close is the sharpest, Bet365 fills a few remaining gaps.
FD_CLOSES = (("AvgC", "market average"), ("BFEC", "Betfair exchange"), ("B365C", "Bet365"))


def load_bookmaker_records(test_start: str, prefix: str = "AvgC"):
    """Same records but priced off a football-data CLOSE, which covers far more matches.

    The Polymarket window is only ~184 coherence-gated fixtures, which leaves the
    blend weight's CI spanning the whole decision boundary. Beating a *closing* line
    is the harder question anyway, and it is the one that decides whether any edge
    exists at all -- so pay the extra difficulty to buy the sample size.
    """
    from ligamx.eval.football_data import load_enriched

    fd = load_enriched()
    fd["Res"] = fd["Res"].astype(str).str.strip()
    fd = fd[fd["Res"].isin(["H", "D", "A"])].sort_values("Date").reset_index(drop=True)
    for c in ("HExpG+", "AExpG+"):
        fd[c] = pd.to_numeric(fd[c], errors="coerce")
    probs = _model_probs(fd, test_start)

    mkt, mod, res = [], [], []
    for idx, mp in probs.items():
        row = fd.loc[idx]
        try:
            o = [float(row[f"{prefix}{k}"]) for k in ("H", "D", "A")]
        except (TypeError, ValueError, KeyError):
            continue
        if not all(np.isfinite(v) and v > 1.0 for v in o):
            continue
        mkt.append(_devig(o))
        mod.append(list(mp))
        res.append(RESULT_IDX[row["Res"]])
    return np.array(mkt), np.array(mod), np.array(res)


def report_lambda(mkt, mod, res, n_boot: int):
    print(f"\n{'='*94}\nHOW MUCH OF THE MODEL'S DISAGREEMENT IS INFORMATION?\n{'='*94}")
    lam = fit_lambda(mkt, mod, res)
    boot = np.array([
        fit_lambda(mkt[s], mod[s], res[s])
        for s in (np.random.randint(0, len(res), len(res)) for _ in range(n_boot))
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  matches: {len(res)}")
    print(f"  fitted L = {lam:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"P(L>0) = {(boot > 0).mean():.3f}")
    print()
    print(f"  {'blend':<22}{'log-lik':>12}{'RPS':>10}")
    for name, l in (("market only (L=0)", 0.0), ("model only (L=1)", 1.0),
                    (f"fitted (L={lam:+.2f})", lam)):
        print(f"  {name:<22}{-_nll(l, mkt, mod, res):>12.2f}{rps(blend(mkt, mod, l), res):>10.5f}")
    print()
    if hi < 0.15:
        print("  READ: the model adds ~nothing over the price it is quoted against.")
        print("  No selection rule can fix that -- there is nothing to select on.")
    elif lo > 0.15:
        print("  READ: the model's deviations carry real information; shrinking picks")
        print("  toward the market should convert calibration into usable selection.")
    else:
        print("  READ: inconclusive at this sample size -- CI spans the decision boundary.")
    return lam, (lo, hi)


def walk_forward_lambda(mkt, mod, res, min_train: int = 60) -> np.ndarray:
    """L refitted using only earlier matches, so the betting sim has no lookahead."""
    out = np.full(len(res), np.nan)
    for i in range(len(res)):
        if i >= min_train:
            out[i] = fit_lambda(mkt[:i], mod[:i], res[:i])
    return out


def report_selection(mkt, mod, close, res, lams, thresholds=(0.0, 0.02, 0.05, 0.10)):
    """What each selection rule actually picks, and how those picks behave.

    ``gap`` is the winner's curse made visible: model probability on the picked side
    minus what actually happened. A calibrated forecaster with a broken selector
    shows a large positive gap here while scoring fine overall.
    """
    print(f"\n{'='*94}\nSELECTION RULES  (bet at the T-48h open, CLV vs the Polymarket close)\n{'='*94}")
    print(f"{'rule':<26}{'EV thr':>7}{'bets':>6}{'model%':>8}{'close%':>8}{'real%':>7}"
          f"{'gap':>7}{'CLV pp':>8}{'t':>6}{'ROI%':>8}")
    ok = ~np.isnan(lams)
    for label, use in (("raw model (argmax EV)", "raw"), ("shrunk to market (L_wf)", "shrunk")):
        for thr in thresholds:
            n = 0
            mp = cp = rl = clv = roi = 0.0
            clvs = []
            for i in range(len(res)):
                if use == "shrunk":
                    if not ok[i]:
                        continue
                    p = blend(mkt[i:i+1], mod[i:i+1], lams[i])[0]
                else:
                    p = mod[i]
                # EV against the devigged open we could actually have taken.
                ev = p / np.clip(mkt[i], EPS, 1) - 1.0
                pick = int(np.argmax(ev))
                if ev[pick] <= thr:
                    continue
                n += 1
                win = 1.0 if res[i] == pick else 0.0
                mp += mod[i][pick]; cp += close[i][pick]; rl += win
                c = (close[i][pick] - mkt[i][pick]) * 100
                clvs.append(c); clv += c
                roi += (1.0 / mkt[i][pick] - 1.0) if win else -1.0
            if n < 2:
                print(f"{label:<26}{thr:>7.0%}{n:>6}")
                continue
            t = np.mean(clvs) / (np.std(clvs, ddof=1) / np.sqrt(n))
            print(f"{label:<26}{thr:>7.0%}{n:>6}{100*mp/n:>8.2f}{100*cp/n:>8.2f}"
                  f"{100*rl/n:>7.2f}{100*(mp-rl)/n:>+7.2f}{clv/n:>+8.2f}{t:>+6.2f}{100*roi/n:>+8.2f}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose winner's curse in bet selection.")
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--boot", type=int, default=2000, help="match-level bootstrap draws")
    ap.add_argument("--all-books", action="store_true",
                    help="keep fixtures whose open was a pre-trading placeholder")
    args = ap.parse_args()

    np.random.seed(0)
    mkt, mod, close, res = load_records(args.test_start, traded_only=not args.all_books)
    print(f"matches with model + real open + close: {len(res)}"
          + ("" if args.all_books else "   [coherence-gated opens only]"))

    print("\n\n### A. PRICED OFF THE TRADEABLE T-48h POLYMARKET OPEN")
    report_lambda(mkt, mod, res, args.boot)
    lams = walk_forward_lambda(mkt, mod, res)
    report_selection(mkt, mod, close, res, lams)

    # The Polymarket window is too small to resolve L. football-data closes cover the
    # whole history, so they answer the same question with usable power.
    print("\n\n### B. PRICED OFF FOOTBALL-DATA CLOSES (bigger sample, harder benchmark)")
    for prefix, label in FD_CLOSES:
        try:
            b_mkt, b_mod, b_res = load_bookmaker_records(args.test_start, prefix)
        except (FileNotFoundError, KeyError):
            continue
        if len(b_res) < 50:
            continue
        print(f"\n--- {prefix} ({label}) ---")
        report_lambda(b_mkt, b_mod, b_res, args.boot)


if __name__ == "__main__":
    main()
