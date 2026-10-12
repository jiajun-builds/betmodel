import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403
d = sides(); pd.set_option("display.width", 220)
d["ob"] = pd.cut(d.odds, [1, 2.5, 4, 7.01], labels=["<2.5", "2.5-4", ">4"])
tr = d[d.split == "train"]
# B2: s from TRAIN only, D38 probabilities vs close, per pre-registered rule
e = lg(tr.p_new) - lg(tr.p_close); s_b = e.groupby(tr.ob, observed=True).std()
spread = s_b.max() / s_b.min() - 1
rule = "per-bucket" if spread > 0.15 else "constant"
s_map = s_b.to_dict() if rule == "per-bucket" else {k: float(e.std()) for k in s_b.index}
print(f"[B2] train s by odds {s_b.round(3).to_dict()}  max/min-1={spread:.2f} -> {rule}: {dict((k, round(v,3)) for k,v in s_map.items())}")
json.dump({str(k): v for k, v in s_map.items()}, open(os.path.join(HERE, "s.json"), "w"))
d["sd_ev"] = d.odds * d.p_new * (1 - d.p_new) * d.ob.map(s_map).astype(float)
d["z"] = d.ev_new / d.sd_ev
# what z>=1 demands, vs EV>0.20
print("\n[B3] EV needed to reach z=1, by odds (at p ~ 1.1/odds):")
for o in (1.5, 2.0, 3.0, 4.5, 6.5):
    ob = "<2.5" if o < 2.5 else ("2.5-4" if o <= 4 else ">4"); s = s_map[ob]
    # solve p*o-1 = o*p*(1-p)*s  -> p = 1/(o*(1 - (1-p)s)) iterate
    p = 1 / o
    for _ in range(50): p = 1 / (o * (1 - (1 - p) * s))
    print(f"  odds {o:.1f}: EV >= {p*o-1:+.3f}")
ev_p = {sp: pick_per_fixture(d[d.split == sp], "ev_new") for sp in ("train", "test")}
z_p = {sp: pick_per_fixture(d[d.split == sp], "z") for sp in ("train", "test")}
print("\n[C1] equal selectivity: top-n by EV vs top-n by z (D38 probabilities). n = bets EV>t would take.")
rng = np.random.default_rng(5)
for sp in ("train", "test"):
    for t in (0.05, 0.10, 0.20):
        n = int((ev_p[sp].ev_new > t).sum())
        if n < 3: continue
        A = ev_p[sp].nlargest(n, "ev_new"); B = z_p[sp].nlargest(n, "z")
        # paired bootstrap over fixtures for the difference in mean CLV
        fx = ev_p[sp].fixture.unique(); ia = set(A.fixture); ib = set(B.fixture)
        ca = ev_p[sp].set_index("fixture").ev_close; cb = z_p[sp].set_index("fixture").ev_close
        diffs = []
        for _ in range(3000):
            smp = rng.choice(fx, len(fx))
            a = [ca[f] for f in smp if f in ia]; b = [cb[f] for f in smp if f in ib]
            if a and b: diffs.append(np.mean(b) - np.mean(a))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {sp:5s} n={n:3d} | EV-rank: CLV {A.ev_close.mean():+.3f} ret {A.ret.mean():+.3f} underdog {A.underdog.mean():.2f} odds {A.odds.mean():.2f}"
              f" | z-rank: CLV {B.ev_close.mean():+.3f} ret {B.ret.mean():+.3f} underdog {B.underdog.mean():.2f} odds {B.odds.mean():.2f}"
              f" | CLV z-EV {B.ev_close.mean()-A.ev_close.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
print("\n[C2] pre-registered rule z>=1 vs current EV>0.20")
for sp in ("train", "test"):
    for lab, s in (("EV>0.20", ev_p[sp][ev_p[sp].ev_new > 0.20]), ("z>=1", z_p[sp][z_p[sp].z >= 1])):
        ci = boot(s.ev_close); rc = boot(s.ret)
        print(f"  {sp:5s} {lab:7s} n={len(s):3d} CLV {s.ev_close.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  realised {s.ret.mean():+.3f} [{rc[0]:+.3f},{rc[1]:+.3f}]  EV {s.ev_new.mean():+.3f}  underdog {s.underdog.mean():.2f}  odds {s.odds.mean():.2f}")
    s = z_p[sp][z_p[sp].z >= 1]
    print(f"        z>=1 by odds bucket:", s.groupby("ob", observed=True).agg(n=("ev_close", "size"), clv=("ev_close", "mean"), ev=("ev_new", "mean")).round(3).to_dict("index"))
print("\n[C3] calibration against results (D38): non-draw matches, result ~ logit(home share); and selected bets predicted vs realised win rate")
from scipy.optimize import minimize
for sp in ("train", "test"):
    g = d[(d.split == sp) & (d.side == "H")].copy()
    a = d[(d.split == sp) & (d.side == "A")].set_index("fixture")
    g = g.join(a[["p_new", "won"]], on="fixture", rsuffix="_a").dropna(subset=["p_new_a"])
    g = g[g.won | g.won_a]                 # non-draws
    x = lg(g.p_new / (g.p_new + g.p_new_a)).to_numpy(); y = g.won.to_numpy(float)
    X = np.c_[np.ones(len(x)), x]
    nll = lambda b: -(y * (X @ b) - np.logaddexp(0, X @ b)).sum()
    b = minimize(nll, np.zeros(2)).x; p = sg(X @ b); H = X.T @ (X * (p * (1 - p))[:, None]); se = np.sqrt(np.diag(np.linalg.inv(H)))
    sel = z_p[sp][z_p[sp].z >= 1]
    print(f"  {sp:5s} slope {b[1]:.2f}±{se[1]:.2f} intercept {b[0]:+.2f}±{se[0]:.2f} (n={len(g)}) | z>=1 bets: predicted win {sel.p_new.mean():.3f} realised {sel.won.mean():.3f} (n={len(sel)})")
