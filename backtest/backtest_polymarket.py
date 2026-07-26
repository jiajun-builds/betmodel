"""Head-to-head: bet 1xBet OPEN vs Polymarket OPEN, same strategy, same fixtures.

The user's ask: keep the shipped strategy (NegBinom walk-forward, market-anchored
de-bias, EV-threshold picks, CLV graded vs Pinnacle CLOSE) but swap the *bet price*
from 1xBet's opening 1X2 to Polymarket's opening 1X2 — and report what changes vs
1xBet. To make it a clean comparison, both books are graded on the IDENTICAL fixture
set: the intersection where a fixture has 1xBet open AND Polymarket open AND Pinnacle
open+close AND a result (n≈155, seasons 2025 tail + 2026).

Everything but the bet-price column is held fixed and reused from ``backtest_1xbet`` /
``backtest_1x2`` (the validated harness):
  * CLV(pp) = novig(pinnacle_close)[pick] − novig(bet_open)[pick]
  * bar(pp) = novig(bet_open)[pick] × R      (R = the bet book's OWN overround)
  * gap     = CLV − bar                      (>0 ⇒ the bet clears the book's vig wall)
  * exCLV   = CLV − the (season,outcome) model-free home-drift baseline

The one structural difference the numbers will expose: Polymarket lists each outcome as
an independent binary market seeded near 0.5, so at OPEN the three "Yes" prices sum to
~1.44 — a ~44% synthetic overround — versus 1xBet's ~4.9%. That inflates ``R`` (hence
``bar``) by ~9× and makes the open a near-flat ~2.0/2.0/2.0 line. Read the gap column
with that in mind; see the printed reading guide.

Reproduce (repo root): PYTHONPATH=src python backtest/backtest_polymarket.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_1x2 import (  # noqa: E402  reuse the validated harness
    MIN_TRAIN,
    PROD_LOOKBACK_MONTHS,
    PROD_XI,
    _t,
    fit_draw_delta,
    fit_model,
    one_x_two,
)
from backtest_1xbet import apply_debias, season_drift  # noqa: E402  validated de-bias
from csl.date_utils import parse_date_only_series  # noqa: E402
from csl.models.continuous_poisson import ContinuousPoissonGoalModel  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

ONEXBET_OPEN = ["onexbet_open_h", "onexbet_open_d", "onexbet_open_a"]
POLY_OPEN = ["polymarket_open_h", "polymarket_open_d", "polymarket_open_a"]
PIN_CLOSE = ["pinnacle_close_h", "pinnacle_close_d", "pinnacle_close_a"]  # CLV ref
PIN_OPEN = ["pinnacle_open_h", "pinnacle_open_d", "pinnacle_open_a"]      # de-bias anchor
res_map = {"H": 0, "D": 1, "A": 2}
EV_THRESHOLDS = [0.00, 0.10, 0.20]

# Variants whose de-bias anchor does NOT depend on the bet price (so both books use the
# identical de-biased model probs — only the price they are cashed against differs).
VARIANTS = [
    ("NegBinom raw", 0.00),
    ("NegBinom + delta-cal (market-free)", "delta"),
    ("NegBinom + lam=0.75 (Pinnacle-open anchor, prod)", "pinnacle_anchor"),
]

# The two candidate bet prices, graded on the same sample.
BOOKS = [
    ("1xBet open", ONEXBET_OPEN),
    ("Polymarket open", POLY_OPEN),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Date"] = parse_date_only_series(df["Date"])
    for c in [*ONEXBET_OPEN, *POLY_OPEN, *PIN_CLOSE, *PIN_OPEN]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["res"] = df["Res"].map(res_map)
    return df.sort_values("Date").reset_index(drop=True)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """NegBinom walk-forward over the intersection carrying BOTH books' opens + refs."""
    need = [*ONEXBET_OPEN, *POLY_OPEN, *PIN_CLOSE, *PIN_OPEN]
    graded = df[df["res"].notna() & df[need].notna().all(axis=1)].copy()
    recs = []
    for date in sorted(graded["Date"].unique()):
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < MIN_TRAIN or test.empty:
            continue
        weights = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        try:
            clf = fit_model(ContinuousPoissonGoalModel, hist, weights)
            delta = fit_draw_delta(clf, hist, weights)
        except Exception:
            continue
        for r in test.itertuples(index=False):
            try:
                p = one_x_two(np.asarray(clf.predict(r.Home, r.Away).grid))
            except Exception:
                continue
            rec = {
                "date": date, "Season": int(r.Season), "Round": int(r.Round),
                "Home": r.Home, "Away": r.Away, "res": int(r.res),
                "pH": p[0], "pD": p[1], "pA": p[2], "delta": delta,
            }
            for cols in (ONEXBET_OPEN, POLY_OPEN, PIN_CLOSE, PIN_OPEN):
                for c in cols:
                    rec[c] = float(getattr(r, c))
            recs.append(rec)
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)


def novig(df, cols):
    o = df[cols].values
    inv = 1.0 / o
    return inv / inv.sum(1, keepdims=True)


def grade(b, mode, bet_cols, *, allow_draw: bool):
    P_raw = b[["pH", "pD", "pA"]].values
    O = b[bet_cols].values                 # bet price (this book's open)
    novig_o = novig(b, bet_cols)
    novig_c = novig(b, PIN_CLOSE)          # Pinnacle close = CLV reference
    novig_anchor = novig(b, PIN_OPEN)      # de-bias anchor (bet-price independent)
    P = apply_debias(P_raw, novig_anchor, mode, b)
    ev = P * O - 1.0
    ev_pick = ev.copy()
    if not allow_draw:
        ev_pick[:, 1] = -np.inf
    pick = ev_pick.argmax(1)
    best_ev = ev_pick.max(1)
    idx = np.arange(len(b))
    win = pick == b["res"].values
    pl = np.where(win, O[idx, pick] - 1.0, -1.0)
    clv = novig_c[idx, pick] - novig_o[idx, pick]
    drift = season_drift(b, novig_o, novig_c)
    seasons = b["Season"].values
    exclv = clv - np.array([drift[(seasons[i], pick[i])] for i in idx])
    bar = novig_o[idx, pick] * ((1 / O).sum(1) - 1.0)
    return dict(novig_o=novig_o, pick=pick, best_ev=best_ev,
                win=win, pl=pl, clv=clv, exclv=exclv, bar=bar)


def print_overround(b):
    print(f"\n{'=' * 78}")
    print(f"SAMPLE  |  n={len(b)} fixtures, seasons={sorted(b['Season'].unique())} "
          f"(same fixtures for both books)")
    print(f"{'=' * 78}")
    print(f"  {'price':<20} {'mean overround R':>16} {'implied p×R bar (@p=.40)':>26}")
    for label, cols in BOOKS:
        R = (1 / b[cols].values).sum(1) - 1.0
        print(f"  {label:<20} {R.mean() * 100:>15.2f}% {0.40 * R.mean() * 100:>24.2f}pp")


def report_book(book_label, bet_cols, b):
    print(f"\n{'#' * 78}\n# BET PRICE: {book_label}\n{'#' * 78}")
    for label, mode in VARIANTS:
        print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
        for allow_draw in (True, False):
            g = grade(b, mode, bet_cols, allow_draw=allow_draw)
            tag = "WITH draw" if allow_draw else "NO draw  "
            print(f"  [{tag}]  {'thr':>4} {'n':>4} {'draws':>5} {'ROI':>8} {'t':>6} "
                  f"{'CLV':>7} {'exCLV':>7} {'t':>6} {'bar':>7} {'gap':>8} {'win%':>5}")
            for thr in EV_THRESHOLDS:
                m = g["best_ev"] > thr
                if m.sum() < 2:
                    print(f"  {'':>11} {thr:>4.2f} {int(m.sum()):>4}  (n<2)")
                    continue
                ndraw = int((g["pick"][m] == 1).sum())
                gap = (g["clv"][m].mean() - g["bar"][m].mean()) * 100
                print(f"  {'':>11} {thr:>4.2f} {int(m.sum()):>4} {ndraw:>5} "
                      f"{g['pl'][m].mean():>+8.3f} {_t(g['pl'][m]):>+6.2f} "
                      f"{g['clv'][m].mean() * 100:>+7.2f} "
                      f"{g['exclv'][m].mean() * 100:>+7.2f} {_t(g['exclv'][m]):>+6.2f} "
                      f"{g['bar'][m].mean() * 100:>7.2f} {gap:>+8.2f} {g['win'][m].mean():>5.0%}")


def head_to_head(b, thr=0.20):
    """Compact side-by-side at the headline thr>0.20 cell, prod variant, WITH draw."""
    mode = "pinnacle_anchor"
    print(f"\n{'=' * 78}")
    print(f"HEAD-TO-HEAD  |  prod variant (λ=0.75 Pinnacle anchor), WITH draw, thr>{thr:.2f}")
    print(f"{'=' * 78}")
    print(f"  {'bet price':<18} {'n':>4} {'ROI':>8} {'t':>6} {'CLV':>7} {'exCLV':>7} "
          f"{'bar':>7} {'gap':>8}")
    for label, cols in BOOKS:
        g = grade(b, mode, cols, allow_draw=True)
        m = g["best_ev"] > thr
        if m.sum() < 2:
            print(f"  {label:<18} (n<2)"); continue
        gap = (g["clv"][m].mean() - g["bar"][m].mean()) * 100
        print(f"  {label:<18} {int(m.sum()):>4} {g['pl'][m].mean():>+8.3f} "
              f"{_t(g['pl'][m]):>+6.2f} {g['clv'][m].mean() * 100:>+7.2f} "
              f"{g['exclv'][m].mean() * 100:>+7.2f} {g['bar'][m].mean() * 100:>7.2f} "
              f"{gap:>+8.2f}")


def diagnostics(b, thr=0.20):
    """Separate the flat-seed ARTIFACT from genuine selection, and split by season."""
    mode = "pinnacle_anchor"
    print(f"\n{'=' * 78}\nDIAGNOSTICS  |  prod variant, WITH draw, thr>{thr:.2f}\n{'=' * 78}")
    for label, cols in BOOKS:
        g = grade(b, mode, cols, allow_draw=True)
        m = g["best_ev"] > thr
        picks = g["pick"][m]
        novig_o_pick = g["novig_o"][np.arange(len(b))[m], picks]
        print(f"\n  {label}: on the picked bets (n={int(m.sum())})")
        print(f"    no-vig OPEN prob of the pick: mean {novig_o_pick.mean():.3f} "
              f"(min {novig_o_pick.min():.3f}, max {novig_o_pick.max():.3f})")
        print(f"    -> a flat ~0.333 here means the open carries no matchup info (pure seed)")
        print(f"    {'season':>7} {'n':>4} {'ROI':>8} {'t':>6} {'CLV':>7} {'exCLV':>7} {'gap':>7} {'win%':>5}")
        for scope, sub in [("pooled", b)] + [
                (str(s), b[b["Season"] == s].reset_index(drop=True))
                for s in sorted(b["Season"].unique())]:
            gs = grade(sub, mode, cols, allow_draw=True)
            ms = gs["best_ev"] > thr
            if ms.sum() < 2:
                print(f"    {scope:>7} {int(ms.sum()):>4}  (n<2)"); continue
            gap = (gs["clv"][ms].mean() - gs["bar"][ms].mean()) * 100
            print(f"    {scope:>7} {int(ms.sum()):>4} {gs['pl'][ms].mean():>+8.3f} "
                  f"{_t(gs['pl'][ms]):>+6.2f} {gs['clv'][ms].mean() * 100:>+7.2f} "
                  f"{gs['exclv'][ms].mean() * 100:>+7.2f} {gap:>+7.2f} {gs['win'][ms].mean():>5.0%}")


def main():
    df = load()
    b = walk_forward(df)
    if b.empty:
        raise SystemExit("No gradeable fixtures on the 1xBet∩Polymarket intersection.")
    print_overround(b)
    for label, cols in BOOKS:
        report_book(label, cols, b)
    head_to_head(b, thr=0.20)
    diagnostics(b, thr=0.20)
    print(f"\n{'=' * 78}\nReading guide\n{'=' * 78}")
    print("  gap = CLV − bar, bar = novig(bet)×R = the book's own vig wall.")
    print("  1xBet open R≈4.9% → bar≈1.7pp. Polymarket OPEN R≈44% (3 binaries seeded")
    print("  near 0.5) → bar≈13–16pp, a wall no realistic CLV clears. A high Polymarket")
    print("  ROI/CLV is the seed mispricing favourites at ~2.0, NOT a fill you could get")
    print("  for size — the informative Polymarket price is the CLOSE (R≈0.1%), not the open.")


if __name__ == "__main__":
    main()
