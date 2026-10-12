"""ONE evaluation on 2026: monotonicity, the full curve, and t*, 0.075, 0.20 head to head."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403
c = candidates(); te = c[c["split"] == "test"]
t_star = json.load(open(os.path.join(HERE, "choice.json")))["t_star"]
b, ci, n = slope(te)
print(f"[1] CLV ~ EV slope, TEST, EV>0 (n={n}): {b:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
pd.set_option("display.width", 200)
print(curve(te).round(3).to_string(index=False))
def paired_total_diff(a, b, n=4000, seed=3):
    """total CLV at threshold a minus at b, resampling fixtures."""
    rng = np.random.default_rng(seed); x = te.reset_index(drop=True); out = []
    for _ in range(n):
        s = x.iloc[rng.integers(0, len(x), len(x))]
        out.append(s[s.ev > a].clv.sum() - s[s.ev > b].clv.sum())
    full = x[x.ev > a].clv.sum() - x[x.ev > b].clv.sum()
    return full, tuple(np.percentile(out, [2.5, 97.5]))
for t in (t_star, 0.075):
    d, dci = paired_total_diff(t, 0.20)
    m = te[te.ev > t].clv
    print(f"[2] t={t}: n={len(m)} mean CLV {m.mean():+.3f} {np.round(boot(m),3)}  total {m.sum():+.2f} | total minus 0.20: {d:+.2f} [{dci[0]:+.2f},{dci[1]:+.2f}]")
m = te[te.ev > 0.20].clv; print(f"    t=0.20: n={len(m)} mean CLV {m.mean():+.3f} {np.round(boot(m),3)}  total {m.sum():+.2f}")
print("\nby half (n, mean CLV):")
for half, g in (("2026H1", te[te.day < pd.Timestamp('2026-07-01')]), ("2026H2", te[te.day >= pd.Timestamp('2026-07-01')])):
    print(" ", half, {t: (int((g.ev > t).sum()), round(g[g.ev > t].clv.mean(), 3) if (g.ev > t).any() else None) for t in (t_star, 0.075, 0.20)})
