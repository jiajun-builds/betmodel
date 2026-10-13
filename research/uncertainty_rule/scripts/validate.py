"""Pre-registered validation of the uncertainty rule (research/uncertainty_rule/PLAN.md)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ev_correction", "scripts"))
from lib import sides, boot, lg, pick_per_fixture  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402

BANDS = [1.0, 2.5, 4.0, 7.01]; LAB = ["<2.5", "2.5-4", ">4"]
d = sides().copy()
d["ob"] = pd.cut(d.odds, BANDS, labels=LAB)
d["err"] = lg(d.p_close) - lg(d.p_new)
SEASONS = (2024, 2025, 2026)

def s_from(mask):
    return d[mask].groupby("ob", observed=True).err.std().to_dict()

# leave-one-season-out s
d["s"] = np.nan
for held in SEASONS:
    s = s_from(d.season != held)
    m = d.season == held
    d.loc[m, "s"] = d.loc[m, "ob"].map(s).astype(float)
    print(f"s estimated without {held}:", {k: round(v, 3) for k, v in s.items()})
d["u"] = d.err / d.s
d["sd_ev"] = d.odds * d.p_new * (1 - d.p_new) * d.s
d["z"] = d.ev_new / d.sd_ev

ok = {}
print("\n[V1] uncertainty model on held-out seasons: sd(u) target 1, coverage P(|u|<=1) target 0.68")
v1 = True
for held in SEASONS:
    g = d[d.season == held]; sd = g.u.std(); cov = (g.u.abs() <= 1).mean()
    print(f"  {held}: sd(u)={sd:.3f} mean(u)={g.u.mean():+.3f} coverage={cov:.3f} n={len(g)}"); v1 &= 0.85 <= sd <= 1.15
for b, g in d.groupby("ob", observed=True):
    sd = g.u.std(); print(f"  band {b}: sd(u)={sd:.3f} coverage={(g.u.abs()<=1).mean():.3f} n={len(g)}"); v1 &= 0.85 <= sd <= 1.15
cov = (d.u.abs() <= 1).mean(); print(f"  pooled coverage {cov:.3f}"); v1 &= 0.60 <= cov <= 0.76
ok["V1"] = v1

ev_p = pick_per_fixture(d, "ev_new"); z_p = pick_per_fixture(d, "z")
rng = np.random.default_rng(11)
fx = ev_p.fixture.unique()
cA = ev_p.set_index("fixture").ev_close; cB = z_p.set_index("fixture").ev_close
def diff_ci(setA, setB, n=4000):
    out = []
    for _ in range(n):
        smp = rng.choice(fx, len(fx))
        a = [cA[f] for f in smp if f in setA]; b = [cB[f] for f in smp if f in setB]
        if a and b: out.append(np.mean(b) - np.mean(a))
    return np.percentile(out, [2.5, 97.5])
print("\n[V2a] equal selectivity, pooled held-out seasons: z-rank minus EV-rank, mean CLV")
v2a = True
for t in (0.05, 0.10, 0.20):
    n = int((ev_p.ev_new > t).sum()); A = ev_p.nlargest(n, "ev_new"); B = z_p.nlargest(n, "z")
    lo, hi = diff_ci(set(A.fixture), set(B.fixture))
    print(f"  n={n:3d} (EV>{t:.2f}): EV-rank CLV {A.ev_close.mean():+.3f} underdog {A.underdog.mean():.2f} | z-rank CLV {B.ev_close.mean():+.3f} underdog {B.underdog.mean():.2f} | diff {B.ev_close.mean()-A.ev_close.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
    v2a &= lo > -0.05
ok["V2a"] = v2a
print("\n[V2b] z>=1 (EV floor 0.05) vs EV>0.20, pooled held-out seasons")
Z = z_p[(z_p.z >= 1) & (z_p.ev_new > 0.05)]; E = ev_p[ev_p.ev_new > 0.20]
zc = boot(Z.ev_close); lo, hi = diff_ci(set(E.fixture), set(Z.fixture))
for lab, s in (("EV>0.20", E), ("z>=1", Z)):
    ci = boot(s.ev_close); r = boot(s.ret)
    print(f"  {lab:7s} n={len(s):3d} CLV {s.ev_close.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] realised {s.ret.mean():+.3f} [{r[0]:+.3f},{r[1]:+.3f}] underdog {s.underdog.mean():.2f} odds<1.8: {(s.odds<1.8).sum()} favourites: {(~s.underdog).sum()}")
print(f"  z>=1 minus EV>0.20 mean CLV: {Z.ev_close.mean()-E.ev_close.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
ok["V2b"] = zc[0] > 0 and lo > -0.05
base = d.underdog.mean()
ok["V3"] = abs(Z.underdog.mean() - base) < abs(E.underdog.mean() - base)
print(f"\n[V3] candidate base underdog share {base:.2f}: EV>0.20 {E.underdog.mean():.2f}, z>=1 {Z.underdog.mean():.2f}")
print("\n[V4] report only")
print(f"  z>=1 bets with EV<=0.05 (floor binding): {int(((z_p.z >= 1) & (z_p.ev_new <= 0.05)).sum())}; smallest EV among z>=1: {z_p[z_p.z>=1].ev_new.min():+.3f}")
for zs in (0.75, 1.0, 1.25):
    s = z_p[(z_p.z >= zs) & (z_p.ev_new > 0.05)]
    print(f"  z*={zs}: n={len(s)} CLV {s.ev_close.mean():+.3f} underdog {s.underdog.mean():.2f}")
for held in SEASONS:
    s = Z[Z.season == held]; e = E[E.season == held]
    print(f"  {held}: z>=1 n={len(s)} CLV {s.ev_close.mean():+.3f} underdog {s.underdog.mean():.2f} | EV>0.20 n={len(e)} CLV {e.ev_close.mean():+.3f} underdog {e.underdog.mean():.2f}")
print("\nVERDICT:", ok, "->", "PASS" if all(ok.values()) else "FAIL")
s_all = s_from(d.season.isin(SEASONS)); print("production s (all seasons):", {k: round(v, 3) for k, v in s_all.items()})
pd.to_pickle(d, os.path.join(HERE, "validated.pkl"))
