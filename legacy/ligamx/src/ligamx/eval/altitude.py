"""
Does altitude add anything the team ratings do not already know?

WHY ASK. The model fits ONE global home advantage (0.2820) for a league running from
Toluca at 2,660m to Mazatlan at 10m. Raw, the differential looks strong: on all 1,042
matches, xG difference regressed on altitude difference gives +0.141 xG/km (t=+3.69).

WHY THAT NUMBER IS NOT AN EDGE. Altitude is confounded with club quality -- America,
Cruz Azul, Pumas, Toluca and Pachuca are both high and strong, while Mazatlan,
Tijuana, Juarez and Atlante are low and weak. A raw regression cannot tell "altitude
hurts visitors" from "good teams happen to live up high", and the model's attack and
defence ratings already absorb the second one.

THE IDENTIFICATION. A home team's own altitude is a constant, but its opponents'
altitudes vary, so the DIFFERENTIAL varies within one team's home fixtures. Demeaning
the residual within home team therefore throws away everything about how good or how
high the host is, and asks only: does this host do relatively better against visitors
who came up further? That question team ratings cannot answer, so a surviving
coefficient is real information.

Run as a two-stage boost so production code stays untouched: fit the model, take the
residual it could not explain, and regress that on the altitude gap. Stage 2 checks
whether feeding the fitted coefficient back improves out-of-sample RPS -- the only
metric that matters, since the target is the 0.0057 RPS gap to the market close.

    python -m ligamx.eval.altitude [--test-start 2025-10-01]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ligamx.eval.clv_backtest import _load
from ligamx.eval.rps_backtest import RESULT_IDX
from ligamx.models.dc import DEFAULT_TARGET, TRAINING_WINDOW_MONTHS, fit_production_model

warnings.filterwarnings("ignore")

# Stadium elevation in metres. Mexico City clubs share the valley floor (~2,240m);
# Pumas sit slightly higher at Ciudad Universitaria.
ALTITUDE_M = {
    "Toluca": 2660, "Pachuca": 2400, "UNAM Pumas": 2270,
    "Club America": 2240, "Cruz Azul": 2240, "Puebla": 2135,
    "Necaxa": 1880, "Atletico San Luis": 1860, "Queretaro": 1820, "Leon": 1815,
    "Guadalajara": 1560, "Atlas": 1560,
    "FC Juarez": 1140, "Santos Laguna": 1120,
    "Monterrey": 540, "Tigres UANL": 500,
    "Tijuana": 20, "Mazatlan": 10, "Atlante": 10,
}

MAX_GOALS = 10


def with_altitude(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["alt_h"] = d["Home"].map(ALTITUDE_M)
    d["alt_a"] = d["Away"].map(ALTITUDE_M)
    d = d.dropna(subset=["alt_h", "alt_a"])
    d["d_alt"] = (d["alt_h"] - d["alt_a"]) / 1000.0
    return d


def walk_forward_fits(df: pd.DataFrame, test_start: str, min_train: int = 80):
    """Per test match: the model's lambdas from a fit that never saw that match."""
    gh, ga = DEFAULT_TARGET
    fit_df = df.dropna(subset=[gh, ga])
    test = df[df["Date"] >= pd.Timestamp(test_start)].copy()
    test["_wk"] = test["Date"].dt.strftime("%G-W%V")
    out = {}
    for wk in sorted(test["_wk"].unique()):
        rows = test[test["_wk"] == wk]
        cutoff = rows["Date"].min()
        lo = cutoff - pd.DateOffset(months=TRAINING_WINDOW_MONTHS)
        train = fit_df[(fit_df["Date"] < cutoff) & (fit_df["Date"] >= lo)]
        if len(train) < min_train:
            continue
        try:
            m = fit_production_model(train)
        except Exception:
            continue
        teams = set(m.teams)
        for idx, r in rows.iterrows():
            if r["Home"] in teams and r["Away"] in teams:
                out[idx] = m.lambdas(r["Home"], r["Away"])
    return out


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, t, p for y ~ x (with intercept)."""
    r = stats.linregress(x, y)
    return r.slope, (r.slope / r.stderr if r.stderr else np.nan), r.pvalue


def stage1_residuals(df: pd.DataFrame, lam: dict) -> pd.DataFrame:
    """Residual xG the fitted ratings could not explain, per match."""
    gh, ga = DEFAULT_TARGET
    recs = []
    for idx, (lh, la) in lam.items():
        r = df.loc[idx]
        if not np.isfinite(r[gh]) or not np.isfinite(r[ga]):
            continue
        recs.append({
            "idx": idx, "home": r["Home"], "away": r["Away"], "d_alt": r["d_alt"],
            # differential residual: how much more the host outscored expectation
            "resid_diff": (r[gh] - lh) - (r[ga] - la),
            "lam_diff": lh - la, "res": RESULT_IDX[r["Res"]],
        })
    return pd.DataFrame(recs)


def report_stage1(rd: pd.DataFrame):
    print(f"\n{'='*92}\nSTAGE 1 -- does altitude explain what the ratings could not?\n{'='*92}")
    print(f"  test matches with an out-of-sample fit: {len(rd)}")

    b, t, p = _ols(rd["d_alt"].to_numpy(), rd["resid_diff"].to_numpy())
    print(f"\n  POOLED       residual ~ altitude gap : {b:+.4f} xG/km  (t={t:+.2f}, p={p:.4f})")
    print("               still confounded: a high host is usually also a strong host.")

    # The real test. Demean within home team, so the host's own quality and altitude
    # are differenced away and only the visitor's climb varies.
    d = rd.copy()
    d["x"] = d["d_alt"] - d.groupby("home")["d_alt"].transform("mean")
    d["y"] = d["resid_diff"] - d.groupby("home")["resid_diff"].transform("mean")
    keep = d.groupby("home")["home"].transform("size") >= 5
    d = d[keep]
    b2, t2, p2 = _ols(d["x"].to_numpy(), d["y"].to_numpy())
    print(f"\n  HOME-FIXED   residual ~ altitude gap : {b2:+.4f} xG/km  (t={t2:+.2f}, p={p2:.4f})")
    print(f"               n={len(d)} over {d['home'].nunique()} hosts -- this is the identified one.")
    print("               Host quality and host altitude are differenced out; only how far")
    print("               the VISITOR climbed still varies.")
    return b2, p2


def rps_and_ll(lams, res, beta: float, d_alt) -> tuple[float, float]:
    """RPS and log-likelihood when lambdas are tilted by exp(+/- beta*d_alt)."""
    from scipy.stats import poisson
    k = np.arange(MAX_GOALS + 1)
    rps_tot = ll_tot = 0.0
    for (lh, la), r, da in zip(lams, res, d_alt):
        adj = np.exp(beta * da)
        grid = np.outer(poisson.pmf(k, lh * adj), poisson.pmf(k, la / adj))
        ph = np.tril(grid, -1).sum(); pa = np.triu(grid, 1).sum(); pdw = np.trace(grid)
        p = np.array([ph, pdw, pa]); p = p / p.sum()
        cum_p, cum_o = np.cumsum(p), np.cumsum(np.eye(3)[r])
        rps_tot += ((cum_p - cum_o) ** 2)[:2].sum() / 2.0
        ll_tot += np.log(max(p[r], 1e-9))
    n = len(res)
    return rps_tot / n, ll_tot


def report_stage2(rd: pd.DataFrame, lam: dict, beta_grid=np.arange(-0.05, 0.31, 0.025)):
    """Does feeding the coefficient back actually improve the forecast?"""
    print(f"\n{'='*92}\nSTAGE 2 -- does it improve the forecast? (RPS is the target metric)\n{'='*92}")
    lams = [lam[i] for i in rd["idx"]]
    res = rd["res"].to_numpy()
    da = rd["d_alt"].to_numpy()
    base_rps, base_ll = rps_and_ll(lams, res, 0.0, da)
    print(f"  baseline (no altitude term): RPS {base_rps:.5f}   log-lik {base_ll:.2f}")
    print(f"\n  {'beta':>8}{'RPS':>11}{'dRPS':>10}{'log-lik':>11}{'dLL':>8}")
    best = (base_rps, 0.0)
    for b in beta_grid:
        r, ll = rps_and_ll(lams, res, float(b), da)
        mark = ""
        if r < best[0]:
            best = (r, float(b)); mark = "  <- best"
        print(f"  {b:>8.3f}{r:>11.5f}{r-base_rps:>+10.5f}{ll:>11.2f}{ll-base_ll:>+8.2f}{mark}")
    print(f"\n  best in-sample beta {best[1]:+.3f} -> RPS {best[0]:.5f} "
          f"({best[0]-base_rps:+.5f} vs baseline)")
    print("  NOTE: beta chosen on the same matches it is scored on, so this is the")
    print("  OPTIMISTIC bound. It only matters if it is large enough to chase.")
    print(f"  For scale, the gap to the market close is about 0.0057 RPS.")
    return best


def report_stage3(rd: pd.DataFrame, lam: dict, k: float = 8.0):
    """The general form of the same hypothesis: per-team home advantage.

    Altitude is only one parameterisation of "the single global home advantage is
    mis-specified". Estimating a free home-advantage deviation per host nests it --
    if even that cannot buy forecast quality, the whole line is closed rather than
    just this one proxy. Estimated on the first half of the test period and scored on
    the second, so nothing is fitted and scored on the same match.
    """
    print(f"\n{'='*92}\nSTAGE 3 -- the general form: a free home advantage per team\n{'='*92}")
    rd = rd.sort_values("idx").reset_index(drop=True)
    cut = len(rd) // 2
    tr, te = rd.iloc[:cut], rd.iloc[cut:]
    # Shrunk per-host deviation, on the same logic as the model's rating shrinkage.
    g = tr.groupby("home")["resid_diff"]
    dev = (g.sum() / (g.size() + k)).to_dict()
    print(f"  estimated on {len(tr)} matches, scored on {len(te)}")
    top = sorted(dev.items(), key=lambda kv: -abs(kv[1]))[:5]
    print("  largest home-advantage deviations (xG differential):")
    for t, v in top:
        print(f"    {t:<20}{v:+.3f}   (altitude {ALTITUDE_M.get(t, 0)}m)")

    lams = [lam[i] for i in te["idx"]]
    res = te["res"].to_numpy()
    base_rps, base_ll = rps_and_ll(lams, res, 0.0, np.zeros(len(te)))
    # Feed the deviation in as a per-match tilt, reusing the same machinery.
    tilt = te["home"].map(dev).fillna(0.0).to_numpy()
    best = (base_rps, 0.0)
    print(f"\n  baseline on the held-out half: RPS {base_rps:.5f}   log-lik {base_ll:.2f}")
    print(f"  {'scale':>8}{'RPS':>11}{'dRPS':>10}{'log-lik':>11}{'dLL':>8}")
    for s in (0.25, 0.5, 0.75, 1.0):
        r, ll = rps_and_ll(lams, res, 1.0, tilt * s)
        mark = ""
        if r < best[0]:
            best = (r, s); mark = "  <- best"
        print(f"  {s:>8.2f}{r:>11.5f}{r-base_rps:>+10.5f}{ll:>11.2f}{ll-base_ll:>+8.2f}{mark}")
    if best[1] == 0.0:
        print("\n  No scale beats the global home advantage out of sample.")
    else:
        print(f"\n  best scale {best[1]:.2f} -> RPS {best[0]:.5f} ({best[0]-base_rps:+.5f})")
    return best


def main():
    ap = argparse.ArgumentParser(description="Test altitude as a model feature.")
    ap.add_argument("--test-start", default="2025-10-01")
    args = ap.parse_args()

    df = with_altitude(_load())
    print(f"matches with altitude for both clubs: {len(df)}  "
          f"(gap {df['d_alt'].min():+.2f}..{df['d_alt'].max():+.2f} km)")
    lam = walk_forward_fits(df, args.test_start)
    rd = stage1_residuals(df, lam)
    if rd.empty:
        print("no out-of-sample fits -- widen --test-start")
        return
    report_stage1(rd)
    report_stage2(rd, lam)
    report_stage3(rd, lam)


if __name__ == "__main__":
    main()
