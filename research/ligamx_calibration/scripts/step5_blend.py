"""xg_blend on the full production recipe (D41). Chosen on 2024-25, reported on 2026.

Every other setting is the committed config's: xi, shrinkage, team offsets, pool.
Scored on what production publishes: log loss and RPS on results, home-share
cross-entropy against the clean close.
"""
from lib import *

GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)
BASE = 0.25


def run(b):
    cfg = wf.configure(load_league("ligamx"), blend=b)
    F = annotate(wf.predict("ligamx", cfg, "2024-01-01")); S = wf.score("ligamx", cfg, F)
    P = S[["p_h", "p_d", "p_a"]].to_numpy(); y = np.array(["HDA".index(r) for r in F.Res])
    rps = 0.5 * ((P.cumsum(1)[:, :2] - np.eye(3)[y].cumsum(1)[:, :2]) ** 2).sum(1)
    xent = np.full(len(F), np.nan); c = F.clean.to_numpy()
    xent[c] = bin_xent(lg(P[c, 0] / (P[c, 0] + P[c, 2])), close_share(F[F.clean]))
    return pd.DataFrame({"era": F.era.to_numpy(), "half": F.half.to_numpy(), "season": F.Season.to_numpy(),
                         "clean": c, "ll": -np.log(P[np.arange(len(P)), y]), "rps": rps, "xent": xent})


R = {b: run(b) for b in GRID}
base = R[BASE]
tr = base.era == "train"; trc = tr & base.clean

print("[train 2024-25] per season arg-min: close xent | log loss | rps")
for s in base.season[tr].unique():
    m = base.season == s; mc = m & base.clean
    am = lambda col, mask: min(GRID, key=lambda b: R[b][col][mask].mean())
    print(f"  {s:14s} {am('xent', mc):.2f} | {am('ll', m):.2f} | {am('rps', m):.2f}")
opt = min(GRID, key=lambda b: R[b].xent[trc].mean())
print(f"\n[train] close-xent optimum {opt}; each blend against it (95% CI)")
chosen = None
for b in GRID:
    d = (R[b].xent - R[opt].xent)[trc]; lo, hi = boot(d)
    if chosen is None and b > BASE and lo <= 0:
        chosen = b   # the smallest move the close cannot tell from its optimum
    print(f"  {b:.2f}  {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
print(f"  -> chosen {chosen}")

for era in ("train", "test"):
    print(f"\n[{era}] against {BASE}: close xent | log loss | rps")
    m = base.era == era; mc = m & base.clean
    for b in GRID:
        if b == BASE: continue
        d = R[b]
        print(f"  {b:.2f}  {fmt((d.xent - base.xent)[mc])} | {fmt((d.ll - base.ll)[m])} | {fmt((d.rps - base.rps)[m])}")

print(f"\n[test halves] {chosen} against {BASE}: close xent | log loss")
for h in ("Clausura 2026", "Apertura 2026"):
    m = base.half == h; mc = m & base.clean; d = R[chosen]
    print(f"  {h}: {(d.xent - base.xent)[mc].mean():+.5f} | {(d.ll - base.ll)[m].mean():+.5f}")
for era, m in (("train", tr), ("test", base.era == "test"), ("all", base.era != "pre")):
    print(f"[log loss {era}] {BASE}: {base.ll[m].mean():.5f}  {chosen}: {R[chosen].ll[m].mean():.5f}")
