"""Contingency, added after step 2 (xi 0.004 failed its test criterion, so production keeps 0.0015).
The offset is re-selected on TRAIN ONLY at xi 0.0015, then tested once with step 2's criteria."""
import json
from lib import *

xi = 0.0015
F = W = load("wf.pkl")[xi]; res = residuals(F, base_config(xi))
tr = (F.clean & (F.era == "train")).values; te = (F.clean & (F.era == "test")).values
q = np.array([c[0] / (c[0] + c[2]) if c else np.nan for c in F.close])
m = share_logit(F)
def gain(s, mask): return bin_xent(m[mask] + s[mask], q[mask]) - bin_xent(m[mask], q[mask])

grid = {(x, l): gain(offsets_for(F, res, x, l), tr).mean() for x in OFF_XI for l in OFF_LAM}
for x in OFF_XI: print(f"  train xi_o={x}: " + "  ".join(f"lam{l}={grid[(x, l)]:+.5f}" for l in OFF_LAM))
xo, lam = min(grid, key=grid.get); print(f"selected on train: xi_o={xo} lam={lam}")
json.dump({"xi": xi, "off_xi": xo, "off_lam": lam}, open(os.path.join(HERE, "choice_current_xi.json"), "w"))

s = offsets_for(F, res, xo, lam)
pd.to_pickle(dict(shift=s, index=F.index), os.path.join(HERE, "shift_current_xi.pkl"))
print(f"\nTRAIN {fmt(gain(s, tr))}\nTEST  {fmt(gain(s, te))}")
for h in ("Clausura 2026", "Apertura 2026"): print(f"  {h}: {gain(s, te & (F.half == h).values).mean():+.5f}")
P = lambda sh: p0(model_1x2(F, sh), F.open_odds)
rt = (F.era == "test").values; l = logloss3(P(s)[rt], F.Res.values[rt]) - logloss3(P(0)[rt], F.Res.values[rt])
print(f"  logloss TEST (n={rt.sum()}) {fmt(l)}")
real = gain(s, te).mean()
pl = np.array([(gain(t := offsets_for(F, res, xo, lam, perm_seed=k), tr).mean(), gain(t, te).mean()) for k in range(20)])
print(f"[placebo] train mean {pl[:,0].mean():+.5f} test mean {pl[:,1].mean():+.5f} best {pl[:,1].min():+.5f}; as good as real: {(pl[:,1] <= real).sum()}")
print("[neighbours] TEST [train]: " + "  ".join(f"({x},{l}) {gain(t := offsets_for(F, res, x, l), te).mean():+.5f} [{gain(t, tr).mean():+.5f}]"
                                           for x in OFF_XI for l in (2, 4, 8)))
