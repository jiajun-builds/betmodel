"""TRAIN ONLY (2024-2025): choose xi, then the offset (xi_o, lam). Writes choice.json."""
import json
from lib import *

W = load("wf.pkl")
def train(F): return F[(F.era == "train") & F.clean]

print("[xi] home-share xent vs clean close, train; difference vs 0.0015 with 95% CI")
base = train(W[0.0015]); q = close_share(base); b0 = bin_xent(share_logit(base), q)
rows = {}
for xi, F in W.items():
    T = train(F); assert (T.index == base.index).all()
    x = bin_xent(share_logit(T), q); rows[xi] = x
    seas = {s: (x - b0)[(T.Season == s).values].mean() for s in ("Clausura 2024", "Apertura 2024", "Clausura 2025", "Apertura 2025")}
    print(f"  xi={xi:<7} xent={x.mean():.5f}  diff {fmt(x - b0)}  by season " + " ".join(f"{v:+.4f}" for v in seas.values()))
best = min(rows, key=lambda k: rows[k].mean())
lo, hi = boot(rows[best] - b0)
for yr in ("2024", "2025"):
    m = base._day.dt.year.astype(str).eq(yr).values
    print(f"  chosen on {yr} alone: xi={min(rows, key=lambda k: rows[k][m].mean())}")
xi = best if hi < 0 else 0.0015
print(f"  arg-min {best}, CI upper {hi:+.5f} -> xi = {xi}")

print(f"\n[offset] on xi={xi}: home-share xent, train, vs no offset")
F = W[xi]; cfg = base_config(xi)
res = residuals(F, cfg)
T = train(F); q = close_share(T); b = bin_xent(share_logit(T), q)
grid = {}
for xo in OFF_XI:
    line = []
    for lam in OFF_LAM:
        s = offsets_for(T, res, xo, lam); d = bin_xent(share_logit(T) + s, q) - b; grid[(xo, lam)] = d.mean()
        line.append(f"{d.mean():+.5f}")
    print(f"  xi_o={xo}: " + "  ".join(f"lam{l}={v}" for l, v in zip(OFF_LAM, line)))
xo, lam = min(grid, key=grid.get)
d = bin_xent(share_logit(T) + offsets_for(T, res, xo, lam), q) - b
print(f"  chosen xi_o={xo} lam={lam}: {fmt(d)}")
json.dump({"xi": xi, "off_xi": xo, "off_lam": lam}, open(os.path.join(HERE, "choice.json"), "w"))
