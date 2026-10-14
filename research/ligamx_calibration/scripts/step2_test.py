"""ONE evaluation of the frozen xi and offset choices on 2026, plus the anti-overfitting checks."""
import json
from lib import *

ch = json.load(open(os.path.join(HERE, "choice.json"))); xi, xo, lam = ch["xi"], ch["off_xi"], ch["off_lam"]
W = load("wf.pkl"); cur, new = W[0.0015], W[xi]
assert (cur.index == new.index).all()
res = residuals(new, base_config(xi))
shift = offsets_for(new, res, xo, lam)
pd.to_pickle(dict(shift=shift, index=new.index), os.path.join(HERE, "shift.pkl"))

def prod_probs(F, s):   # what production publishes: offset model, draw anchored where an opener exists
    return p0(model_1x2(F, s), F.open_odds)

q_all = np.array([c[0] / (c[0] + c[2]) if c else np.nan for c in new.close])
arms = {"current (xi 0.0015)": (share_logit(cur), prod_probs(cur, 0)),
        f"xi {xi}": (share_logit(new), prod_probs(new, 0)),
        f"xi {xi} + offset": (share_logit(new) + shift, prod_probs(new, shift))}

def compare(a, b, mask_close, mask_res, label):
    x = bin_xent(arms[a][0][mask_close], q_all[mask_close]) - bin_xent(arms[b][0][mask_close], q_all[mask_close])
    res_ = new.Res.values[mask_res]
    l = logloss3(arms[a][1][mask_res], res_) - logloss3(arms[b][1][mask_res], res_)
    lc = boot(l)
    print(f"  {label:14s} n_close={mask_close.sum():3d}  xent {fmt(x)}   logloss(n={mask_res.sum()}) {l.mean():+.4f} [{lc[0]:+.4f}, {lc[1]:+.4f}]")

clean = new.clean.values
for a, b in ((f"xi {xi}", "current (xi 0.0015)"), (f"xi {xi} + offset", f"xi {xi}"), (f"xi {xi} + offset", "current (xi 0.0015)")):
    print(f"\n[{a}]  minus  [{b}]")
    compare(a, b, clean & (new.era == "train").values, (new.era == "train").values, "TRAIN 2024-25")
    compare(a, b, clean & (new.era == "test").values, (new.era == "test").values, "TEST 2026")
    for h in ("Clausura 2026", "Apertura 2026"):
        compare(a, b, clean & (new.half == h).values, (new.half == h).values, h)

te = clean & (new.era == "test").values; tr = clean & (new.era == "train").values
base_te = bin_xent(share_logit(new)[te], q_all[te]); base_tr = bin_xent(share_logit(new)[tr], q_all[tr])
real = (bin_xent(share_logit(new)[te] + shift[te], q_all[te]) - base_te).mean()
print(f"\n[placebo] offset gain on TEST, real labels {real:+.5f}")
pl = []
for seed in range(20):
    s = offsets_for(new, res, xo, lam, perm_seed=seed)
    pl.append(((bin_xent(share_logit(new)[tr] + s[tr], q_all[tr]) - base_tr).mean(),
               (bin_xent(share_logit(new)[te] + s[te], q_all[te]) - base_te).mean()))
pl = np.array(pl)
print(f"  shuffled clubs x20: train mean {pl[:,0].mean():+.5f}  test mean {pl[:,1].mean():+.5f} "
      f"(best {pl[:,1].min():+.5f}); runs at least as good as real: {(pl[:,1] <= real).sum()}")

print("\n[neighbours] offset gain, TEST [train]")
for x2 in OFF_XI:
    print("  " + "  ".join(f"xi_o={x2} lam={l}: {(bin_xent(share_logit(new)[te] + (s := offsets_for(new, res, x2, l))[te], q_all[te]) - base_te).mean():+.5f} "
                           f"[{(bin_xent(share_logit(new)[tr] + s[tr], q_all[tr]) - base_tr).mean():+.5f}]" for l in OFF_LAM[:3]))

print("\n[sensitivity] offset residuals without the >5% closes")
res_clean = residuals(new.assign(**{f"pinnacle_close_{x}": new[f"pinnacle_close_{x}"].where(new.clean) for x in "hda"}), base_config(xi))
s2 = offsets_for(new, res_clean, xo, lam)
print(f"  TEST offset gain {fmt(bin_xent(share_logit(new)[te] + s2[te], q_all[te]) - base_te)}")
