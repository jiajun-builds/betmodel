import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403
d = sides(); pd.set_option("display.width", 220)
print("side candidates:", d.groupby("split").size().to_dict(), " fixtures:", d.groupby("split").fixture.nunique().to_dict())
print("\n[A1] composition of selected bets (one per fixture, engine rule)")
for k in ("old", "new"):
    p = pick_per_fixture(d, f"ev_{k}")
    for t in (0.05, 0.10, 0.20):
        for sp in ("train", "test"):
            s = p[(p.split == sp) & (p[f"ev_{k}"] > t)]
            print(f"  {k:3s} EV>{t:.2f} {sp:5s} n={len(s):3d} underdog share={s.underdog.mean() if len(s) else float('nan'):.2f} mean odds={s.odds.mean() if len(s) else float('nan'):.2f}")
allc = d.groupby("split").underdog.mean().round(2).to_dict(); print("  (all side candidates: underdog share", allc, ")")
print("\n[A2] logit disagreement with the opener needed to reach EV 0.20 (soft odds = Pinnacle no-vig * (1 - 5% margin))")
for po in (0.15, 0.25, 0.35, 0.50, 0.65):
    odds = 1 / po * 0.95; need = 1.2 / odds
    print(f"  opener p={po:.2f} odds={odds:.2f}: model must say p>={need:.3f}  (+{(need-po)*100:.1f}pp, logit +{lg(need)-lg(po):.2f})")
for k in ("old", "new"):
    p = pick_per_fixture(d, f"ev_{k}"); s = p[p[f"ev_{k}"] > 0.20]
    s = s.assign(dl=lg(s[f"p_{k}"]) - lg(s.p_open))
    print(f"  observed {k}: selected EV>0.20 logit(model)-logit(open): underdog {s[s.underdog].dl.mean():+.2f} (n={s.underdog.sum()}), favourite {s[~s.underdog].dl.mean():+.2f} (n={(~s.underdog).sum()})")
print("\n[A3] EV inflation = EV - EV_close (= (p_model - p_close) * odds), and EV - realised return")
d["ob"] = pd.cut(d.odds, [1, 2.5, 4, 7.01], labels=["<2.5", "2.5-4", ">4"])
for k in ("old", "new"):
    d[f"inf_{k}"] = d[f"ev_{k}"] - d.ev_close
    print(f"  -- {k} calibration")
    for sp in ("train", "test"):
        g = d[d.split == sp]
        un = g.groupby("ob", observed=True)[f"inf_{k}"].mean().round(3).to_dict()
        p = pick_per_fixture(g, f"ev_{k}"); s = p[p[f"ev_{k}"] > 0.10]
        se = s.groupby("ob", observed=True)[f"inf_{k}"].agg(["mean", "size"]).round(3)
        s20 = p[p[f"ev_{k}"] > 0.20]
        print(f"   {sp}: all candidates by odds {un} | selected EV>0.10 by odds {se['mean'].to_dict()} n={se['size'].to_dict()} | "
              f"EV>0.20: EV {s20[f'ev_{k}'].mean():+.3f} vs CLV {s20.ev_close.mean():+.3f} vs realised {s20.ret.mean():+.3f} (n={len(s20)})")
print("\n[A4] logit error vs close, sd by odds bucket (all side candidates)")
for k in ("old", "new", "open"):
    for sp in ("train", "test"):
        g = d[d.split == sp]; e = lg(g[f"p_{k}"]) - lg(g.p_close)
        print(f"  {k:4s} {sp:5s} sd by odds:", e.groupby(g.ob, observed=True).std().round(3).to_dict(), " mean by odds:", e.groupby(g.ob, observed=True).mean().round(3).to_dict())
