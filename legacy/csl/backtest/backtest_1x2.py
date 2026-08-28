"""Walk-forward 1X2 (home/draw/away) opening-line backtest + open->close CLV.

NEGATIVE RESULT as specified — but the direction is NOT dead. Full write-up:
`backtest/backtest.md` §11 (read §11.3 and §11.7 before trusting any CLV number,
including the ones this script prints).

What this script reports (611 matches, 2024 R1 - 2026 R18; 2023 excluded, no
training history) is: bet the highest-EV 1X2 outcome at Pinnacle's opening price.
That loses — -4.8% ROI at EV>0.10, and full Kelly goes to zero. Two reasons, and
only the first is fatal:

1. THE DRAW BUG (the actual defect). The model pins draw prob near 0.28 vs the
   market's ~0.234 and an actual ~0.242 — high by ~4pp in every season and every
   match type (a structural artifact of independent-Poisson scoring). With draw
   prob 0.28, EV>0.10 fires whenever the draw is priced above 1.10/0.28 = 3.93,
   and the CSL median opening draw price is 3.79 — so EVERY above-median draw
   becomes a bet. Result: 61% of all stake sits on the bug, carrying ZERO CLV
   (+0.03pp, t=0.22). Worst bucket: draws priced 4.5-6, where the model claims 28%
   and reality delivers 8% (n=40, ROI -62%).
   Drop the draw and CLV triples (+0.66 -> +2.15pp), surviving the baseline
   adjustment below (+1.73pp, t=2.51) and positive in all three seasons.

2. THE VIG WALL (why even +CLV loses). Betting at opening price with no-vig prob p
   and opening overround R: EV > 0 <=> CLV > p * R. Here p~0.344 and Pinnacle's
   opening R = 7.55%, so breakeven needs CLV > 2.61pp. The no-draw strategy gets
   +2.15pp — still short. This is why "always bet home" earns +0.91pp CLV and
   still returns -4.8%.

DRAW DE-BIAS VARIANTS (roadmap #9, results in backtest.md §12). Each model is also
run with a market-anchored draw shrink applied after prediction, before EV:
    p'_D = (1-lam) * p_D + lam * m_D        (m_D = no-vig OPENING draw prob)
    p'_H = p_H * (1-p'_D)/(1-p_D)           (freed mass returned pro-rata)
    p'_A = p_A * (1-p'_D)/(1-p_D)
No leakage: the opening price is known at bet time. lam=1 pins the draw to the
market (a draw is then never +EV), but unlike the naive "never bet the draw" rule
of §11.6 it also scales home/away up by ~(1-m_D)/(1-p_D) ≈ 1.06, changing picks
and EVs. The lam grid is reported in full — a single good lam cell is noise; only
a *region* of lam clearing the bar in all three seasons counts.
OUTCOME (§12): as a model fix it works (draw prob 0.245 vs actual 0.242 at
lam=0.75; excess CLV doubles to +1.4pp at thr>0.10), but the success bar is NOT
met — the per-season gap is negative in 2024 and 2025 at every lam; only 2026
clears. Betting Pinnacle's open stays closed; the surviving signal would clear
the bar at a <=5%-overround book (roadmap #8).

A MARKET-FREE variant ("delta-cal") is also run: per round, fit a scale delta
for the scoreline-grid diagonal on the TRAINING window (Dixon-Coles-weighted 1X2
log-likelihood, `fit_draw_delta`), then apply it to that round's predictions.
Needs no market anchor, so it is the only mechanism deployable in production
(dc.py / dashboard, which have no 1X2 odds). Run here first to validate the
draw repair matches the market-anchored version before deploying (§12.4).

Every variant's table now reports, per §11.3 and §11.7:
  exCLV — excess CLV = per-bet CLV minus the same-season/same-outcome market
          drift (the model-free baseline; "always home" alone earns +0.91pp).
  bar   — the per-bet breakeven p*R (no-vig opening prob of the pick x opening
          overround), averaged over the selected bets. EV > 0 <=> CLV > bar.
  gap   — CLV - bar. Positive gap in all three seasons is the success criterion.

Method: the production recipe (xG targets, 18-month lookback, xi=0.001,
Dixon-Coles time-decay weights), walk-forward with no leakage, scoring 1X2 instead
of AH. Compares ZIP (production) vs NegBinom (better calibrated; fixes nothing
here). CLV = no-vig closing prob - no-vig opening prob of the backed outcome.

Run (repo root, env with penaltyblog):
    PYTHONPATH=src python backtest/backtest_1x2.py
"""

import os

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.optimize import minimize_scalar

from csl.date_utils import parse_date_only_series
from csl.models.continuous_poisson import ContinuousPoissonGoalModel

# Recipe constants come from the production module so a backtest cannot silently
# drift from what actually ships. They used to be re-declared here and in seven
# other backtest files. See the MIN_TRAIN=100 rationale in csl.models.dc.
from csl.models.dc import MIN_TRAIN, PROD_LOOKBACK_MONTHS, PROD_XI  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

EV_THRESHOLDS = [0.00, 0.10, 0.20]
KELLY_FRACTIONS = [1.0, 0.5, 0.25]
res_map = {"H": 0, "D": 1, "A": 2}  # 0=home win, 1=draw, 2=away win
OUTCOME = ["home", "draw", "away"]

# ContinuousPoisson is production. The two penaltyblog families are retained only
# as a before/after reference: both silently truncate the non-integer xG targets to
# integers, so every conclusion previously drawn from comparing them was drawn on a
# lambda ~27% too low, and the choice between them was never meaningful. See
# csl.models.continuous_poisson.
MODELS = {
    "ContinuousPoisson (prod)": ContinuousPoissonGoalModel,
    "ZIP (truncating)": pb.models.ZeroInflatedPoissonGoalsModel,
    "NegBinom (truncating)": pb.models.NegativeBinomialGoalModel,
}

# (frame name, de-bias mode, report label, per-bet CSV slug).
# mode: 0.0 = raw model; float lam in (0,1] = market-anchored shrink; "delta" =
# the market-free calibration (grid diagonal x delta, delta fit on the training
# window — the production-deployable variant, no market anchor needed).
# NegBinom carries the lam grid (the roadmap-#9 candidate); ZIP+lam=1 isolates
# "how much is the de-bias vs how much is the distribution swap".
# The lam grid on ContinuousPoisson is the DEBIAS_LAMBDA re-sweep: lam was tuned to
# 0.75 against the truncated fit, where the model sat ABOVE the market draw so the
# anchor pulled down. On the corrected fit the model sits BELOW it, so the same lam
# now pulls the draw the other way — the old optimum carries no information.
VARIANTS = [
    ("ContinuousPoisson (prod)", 0.00, "ContinuousPoisson (raw)", "cpois"),
    ("ContinuousPoisson (prod)", 0.25, "ContinuousPoisson + anchor lam=0.25", "cpois_lam25"),
    ("ContinuousPoisson (prod)", 0.50, "ContinuousPoisson + anchor lam=0.50", "cpois_lam50"),
    ("ContinuousPoisson (prod)", 0.75, "ContinuousPoisson + anchor lam=0.75", "cpois_lam75"),
    ("ContinuousPoisson (prod)", 1.00, "ContinuousPoisson + anchor lam=1.00", "cpois_lam100"),
    ("ContinuousPoisson (prod)", "delta", "ContinuousPoisson + delta-cal", "cpois_delta"),
    # Truncated-fit reference points — kept to quantify the bug, not to choose from.
    ("NegBinom (truncating)", 0.00, "NegBinom (truncating, raw)", "negbinom"),
    ("NegBinom (truncating)", 0.75, "NegBinom (truncating) + anchor lam=0.75", "negbinom_lam75"),
]

OPEN_COLS = ["pinnacle_open_h", "pinnacle_open_d", "pinnacle_open_a"]
CLOSE_COLS = ["pinnacle_close_h", "pinnacle_close_d", "pinnacle_close_a"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Date"] = parse_date_only_series(df["Date"])
    for c in ["HExpG+", "AExpG+", "HG", "AG", *OPEN_COLS, *CLOSE_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Home", "Away"])
    df["Home"] = df["Home"].astype(str)
    df["Away"] = df["Away"].astype(str)
    df["res"] = df["Res"].map(res_map)
    return df.sort_values("Date").reset_index(drop=True)


def one_x_two(grid: np.ndarray) -> np.ndarray:
    """(P_home_win, P_draw, P_away_win) from a 15x15 scoreline grid."""
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n))
    v = np.array([grid[diff > 0].sum(), grid[diff == 0].sum(), grid[diff < 0].sum()])
    return v / v.sum()


def fit_model(Model, train, weights):
    """Fit with Dixon-Coles weights if the constructor accepts them, else without."""
    try:
        clf = Model(train["HExpG+"], train["AExpG+"], train["Home"], train["Away"], weights)
    except TypeError:
        clf = Model(train["HExpG+"], train["AExpG+"], train["Home"], train["Away"])
    clf.fit()
    return clf


def fit_draw_delta(clf, hist: pd.DataFrame, weights) -> float:
    """Market-free draw calibration (the production-deployable de-bias).

    Finds the scale delta for the scoreline-grid diagonal that maximizes the
    Dixon-Coles-weighted 1X2 log-likelihood of the *training* fixtures — no
    market data involved, so it is computable in production for every team pair.
    With raw (pH, pD, pA) and z = 1 - pD + delta*pD (the renormalizer), the
    adjusted probs are (pH/z, delta*pD/z, pA/z) — identical to scaling the grid
    diagonal by delta and renormalizing.
    """
    cache: dict = {}
    P, ks, ws = [], [], []
    w_arr = np.asarray(weights)
    for i, r in enumerate(hist.itertuples(index=False)):
        if pd.isna(r.res):
            continue
        key = (r.Home, r.Away)
        if key not in cache:
            try:
                cache[key] = one_x_two(np.asarray(clf.predict(r.Home, r.Away).grid))
            except Exception:
                cache[key] = None
        if cache[key] is None:
            continue
        P.append(cache[key])
        ks.append(int(r.res))
        ws.append(w_arr[i])
    if len(P) < 20:
        return 1.0
    P = np.asarray(P)
    k = np.asarray(ks)
    w = np.asarray(ws)
    d = P[:, 1]
    raw = P[np.arange(len(k)), k]

    def nll(delta):
        z = 1.0 - d + delta * d
        pk = np.where(k == 1, delta * d, raw) / z
        return -(w * np.log(np.clip(pk, 1e-12, None))).sum()

    return float(minimize_scalar(nll, bounds=(0.3, 2.0), method="bounded").x)


def walk_forward(df: pd.DataFrame):
    """One pass fitting every model per date; returns {model_name: DataFrame}."""
    graded = df[df["res"].notna()
                & df[OPEN_COLS].notna().all(axis=1)
                & df[CLOSE_COLS].notna().all(axis=1)].copy()
    recs = {name: [] for name in MODELS}

    for date in sorted(graded["Date"].unique()):
        date = pd.Timestamp(date)
        test = graded[graded["Date"] == date]
        hist = df[df["Date"] < date].dropna(subset=["HExpG+", "AExpG+"])
        hist = hist[hist["Date"] >= date - pd.DateOffset(months=PROD_LOOKBACK_MONTHS)]
        if len(hist) < MIN_TRAIN or test.empty:
            continue
        weights = pb.models.dixon_coles_weights(hist["Date"], PROD_XI)
        fitted, deltas = {}, {}
        for name, Model in MODELS.items():
            try:
                fitted[name] = fit_model(Model, hist, weights)
                deltas[name] = fit_draw_delta(fitted[name], hist, weights)
            except Exception:
                fitted[name] = None
        for r in test.itertuples(index=False):
            for name in MODELS:
                if fitted[name] is None:
                    continue
                try:
                    p = one_x_two(np.asarray(fitted[name].predict(r.Home, r.Away).grid))
                except Exception:
                    continue
                recs[name].append({
                    "date": date, "Season": int(r.Season), "Round": int(r.Round),
                    "Home": r.Home, "Away": r.Away, "res": int(r.res),
                    "pH": p[0], "pD": p[1], "pA": p[2],
                    "delta": deltas[name],
                    "oH": float(r.pinnacle_open_h), "oD": float(r.pinnacle_open_d),
                    "oA": float(r.pinnacle_open_a),
                    "cH": float(r.pinnacle_close_h), "cD": float(r.pinnacle_close_d),
                    "cA": float(r.pinnacle_close_a),
                })
    return {name: pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            for name, rows in recs.items()}


def apply_debias(P: np.ndarray, novig_o: np.ndarray, mode, b: pd.DataFrame) -> np.ndarray:
    """Apply one de-bias mode to raw (pH, pD, pA).

    mode 0.0        -> unchanged.
    mode lam (float)-> market-anchored shrink: p_D toward the no-vig opening draw
                       prob, freed mass returned to home/away pro-rata.
    mode "delta"    -> market-free calibration: the per-round training-window
                       delta from ``fit_draw_delta`` (grid-diagonal scale).
    """
    if isinstance(mode, str) and mode == "delta":
        delta = b["delta"].values
        d = P[:, 1]
        z = 1.0 - d + delta * d
        return np.column_stack([P[:, 0] / z, delta * d / z, P[:, 2] / z])
    if mode == 0.0:
        return P
    pD = (1.0 - mode) * P[:, 1] + mode * novig_o[:, 1]
    scale = (1.0 - pD) / (1.0 - P[:, 1])
    return np.column_stack([P[:, 0] * scale, pD, P[:, 2] * scale])


def season_drift(b: pd.DataFrame) -> dict:
    """Mean open->close no-vig drift per (season, outcome) over ALL graded
    matches — the §11.3 model-free baseline a CLV claim is measured against."""
    O = b[["oH", "oD", "oA"]].values
    C = b[["cH", "cD", "cA"]].values
    novig_o = (1 / O) / (1 / O).sum(1, keepdims=True)
    novig_c = (1 / C) / (1 / C).sum(1, keepdims=True)
    drift = novig_c - novig_o
    seasons = b["Season"].values
    return {(s, k): drift[seasons == s, k].mean()
            for s in np.unique(seasons) for k in range(3)}


def grade(b: pd.DataFrame, mode=0.0) -> dict:
    """Add EV / pick / P&L / CLV / excess-CLV / vig-bar columns for one variant."""
    P_raw = b[["pH", "pD", "pA"]].values
    O = b[["oH", "oD", "oA"]].values
    C = b[["cH", "cD", "cA"]].values
    novig_o = (1 / O) / (1 / O).sum(1, keepdims=True)
    novig_c = (1 / C) / (1 / C).sum(1, keepdims=True)
    P = apply_debias(P_raw, novig_o, mode, b)
    ev = P * O - 1.0
    pick = ev.argmax(1)
    best_ev = ev.max(1)
    win = pick == b["res"].values
    idx = np.arange(len(b))
    pl = np.where(win, O[idx, pick] - 1.0, -1.0)
    clv = novig_c[idx, pick] - novig_o[idx, pick]
    # §11.3: excess CLV = CLV minus the same-season/same-outcome market drift.
    drift = season_drift(b)
    seasons = b["Season"].values
    exclv = clv - np.array([drift[(seasons[i], pick[i])] for i in idx])
    # §11.7: per-bet breakeven bar. EV > 0 at the opening price <=> CLV > p * R.
    R = (1 / O).sum(1) - 1.0
    bar = novig_o[idx, pick] * R
    return {"P_raw": P_raw, "P": P, "O": O, "novig_o": novig_o, "pick": pick,
            "best_ev": best_ev, "win": win, "pl": pl, "clv": clv,
            "exclv": exclv, "bar": bar}


def _t(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def print_market_baselines(b: pd.DataFrame):
    """The §11.3 control, printed once: what model-free rules earn in this market."""
    O = b[["oH", "oD", "oA"]].values
    C = b[["cH", "cD", "cA"]].values
    novig_o = (1 / O) / (1 / O).sum(1, keepdims=True)
    novig_c = (1 / C) / (1 / C).sum(1, keepdims=True)
    drift = novig_c - novig_o
    seasons = b["Season"].values

    print(f"\n{'=' * 66}\nMARKET BASELINES (§11.3) — n={len(b)} graded matches\n{'=' * 66}")
    print("  open->close no-vig drift (pp) by season x outcome:")
    print(f"    {'season':>8} {'home':>7} {'draw':>7} {'away':>7}")
    for s in np.unique(seasons):
        m = seasons == s
        print(f"    {int(s):>8} {drift[m, 0].mean() * 100:>+7.2f} "
              f"{drift[m, 1].mean() * 100:>+7.2f} {drift[m, 2].mean() * 100:>+7.2f}")
    print(f"    {'all':>8} {drift[:, 0].mean() * 100:>+7.2f} "
          f"{drift[:, 1].mean() * 100:>+7.2f} {drift[:, 2].mean() * 100:>+7.2f}")
    print("  model-free CLV baselines:")
    for k, lbl in enumerate(["always home", "always draw", "always away"]):
        print(f"    {lbl:<12} {drift[:, k].mean() * 100:>+6.2f}pp (t={_t(drift[:, k]):>+5.2f})")
    rnd = drift.mean(1)
    print(f"    {'random pick':<12} {rnd.mean() * 100:>+6.2f}pp (t={_t(rnd):>+5.2f})")


def report(name: str, b: pd.DataFrame, g: dict, mode):
    pick, best_ev, pl, clv, win = g["pick"], g["best_ev"], g["pl"], g["clv"], g["win"]
    P, P_raw, novig_o = g["P"], g["P_raw"], g["novig_o"]
    exclv, bar = g["exclv"], g["bar"]
    act = np.array([(b["res"].values == k).mean() for k in range(3)])
    debiased = mode != 0.0

    print(f"\n{'=' * 66}\n{name}   (n={len(b)}, "
          f"{int(b['Season'].min())} R{int(b['Round'].min())}-{int(b['Round'].max())})\n{'=' * 66}")
    shrunk = f" | shrunk {P[:, 1].mean():.3f}" if debiased else ""
    print(f"  draw prob: model {P_raw[:, 1].mean():.3f}{shrunk} | "
          f"market {novig_o[:, 1].mean():.3f} | actual {act[1]:.3f}")
    if isinstance(mode, str) and mode == "delta":
        print(f"  training-window delta: mean {b['delta'].mean():.3f} "
              f"(min {b['delta'].min():.3f}, max {b['delta'].max():.3f})")
    dist = {OUTCOME[k]: int((pick == k).sum()) for k in range(3)}
    print(f"  bets by pick: {dist}")

    print(f"\n  {'thr':>5} {'n':>4} {'ROI':>9} {'t':>6} {'CLV(pp)':>9} {'t':>6} "
          f"{'exCLV':>7} {'t':>6} {'bar':>6} {'gap':>6} {'win%':>5}")
    for thr in EV_THRESHOLDS:
        m = best_ev > thr
        if m.sum() < 2:
            continue
        gap = (clv[m].mean() - bar[m].mean()) * 100
        print(f"  {thr:>5.2f} {int(m.sum()):>4} {pl[m].mean():>+9.3f} {_t(pl[m]):>+6.2f} "
              f"{clv[m].mean() * 100:>+9.2f} {_t(clv[m]):>+6.2f} "
              f"{exclv[m].mean() * 100:>+7.2f} {_t(exclv[m]):>+6.2f} "
              f"{bar[m].mean() * 100:>6.2f} {gap:>+6.2f} {win[m].mean():>5.0%}")

    print("  by picked outcome (thr>0):")
    for k in range(3):
        m = (pick == k) & (best_ev > 0)
        if m.sum() >= 3:
            print(f"    {OUTCOME[k]:<5} n={int(m.sum()):>3}  ROI {pl[m].mean():>+.3f} "
                  f"(t={_t(pl[m]):>+.2f})  CLV {clv[m].mean() * 100:>+.2f}pp (t={_t(clv[m]):>+.2f})  "
                  f"exCLV {exclv[m].mean() * 100:>+.2f}pp (t={_t(exclv[m]):>+.2f})")

    # Cross-season replication: each season is an independent sample. The success
    # criterion (roadmap #9) is gap > 0 in ALL seasons, not one lucky year.
    print("  cross-season replication (thr>0.10):")
    for s in sorted(b["Season"].unique()):
        m = (b["Season"].values == s) & (best_ev > 0.10)
        if m.sum() >= 5:
            gap = (clv[m].mean() - bar[m].mean()) * 100
            print(f"    {int(s)}: n={int(m.sum()):>3}  ROI {pl[m].mean():>+.3f} (t={_t(pl[m]):>+.2f})  "
                  f"CLV {clv[m].mean() * 100:>+.2f}pp (t={_t(clv[m]):>+.2f})  "
                  f"exCLV {exclv[m].mean() * 100:>+.2f}pp (t={_t(exclv[m]):>+.2f})  "
                  f"bar {bar[m].mean() * 100:.2f}  gap {gap:>+.2f}")

    # Kelly reality check (chronological, bet the +EV pick, bankroll 100).
    print("  Kelly bankroll (start 100, chronological, +EV picks only):")
    for kf in KELLY_FRACTIONS:
        B, peak, mdd = 100.0, 100.0, 0.0
        for i in range(len(b)):
            if best_ev[i] <= 0:
                continue
            p, o = P[i, pick[i]], g["O"][i, pick[i]]
            f = max(((o - 1) * p - (1 - p)) / (o - 1), 0.0) * kf
            B += B * f * (o - 1) if win[i] else -B * f
            peak = max(peak, B)
            mdd = max(mdd, (peak - B) / peak)
            if B <= 0:
                B = 0.0
                break
        print(f"    {kf:.2f}-Kelly: end {B:6.1f}  max drawdown {mdd * 100:.0f}%")


def run():
    df = load()
    frames = walk_forward(df)
    any_frame = next(iter(frames.values()))
    if any_frame.empty:
        raise SystemExit("No gradeable 1X2 fixtures (need pinnacle_open_*/close_* + result).")

    print_market_baselines(any_frame)

    for frame_name, mode, label, slug in VARIANTS:
        b = frames[frame_name]
        if b.empty:
            print(f"\n{label}: no fixtures graded.")
            continue
        g = grade(b, mode)
        report(label, b, g, mode)
        out = b.copy()
        if mode != 0.0:
            out[["pH_adj", "pD_adj", "pA_adj"]] = g["P"]
        out["pick"] = [OUTCOME[k] for k in g["pick"]]
        out["best_ev"] = g["best_ev"]
        out["pl"] = g["pl"]
        out["clv"] = g["clv"]
        out["clv_excess"] = g["exclv"]
        out["vig_bar"] = g["bar"]
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"backtest_1x2_{slug}_bets.csv")
        out.to_csv(path, index=False)
        print(f"  per-bet detail -> {os.path.relpath(path, REPO_ROOT)}")

    print("\nRead the exCLV and gap columns, not raw CLV or ROI: the market itself drifts"
          "\ntoward the home team (+0.91pp/season — a coin beats raw CLV), and EV > 0 at the"
          "\nopen requires CLV > p*R (the bar/gap columns). The success criterion for the"
          "\ndraw de-bias (roadmap #9) is gap > 0 in ALL three seasons, across a lam region"
          "\nrather than a single cell. Full write-up: backtest/backtest.md §11-§12.")


if __name__ == "__main__":
    run()
