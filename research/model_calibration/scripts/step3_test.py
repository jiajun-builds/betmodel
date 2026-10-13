"""ONE evaluation on the 2026 test set with the frozen choices from choice.json."""
import json, sys
sys.path.insert(0,__file__.rsplit("/",1)[0])
from lib import *
ch=json.load(open(__file__.replace("step3_test.py","choice.json"))); o=ch["offset"]
P["mO"]=adjusted(P,o["variant"],o["xi"],o["lam"])
def build(g):
    m0=g.m.values; mO=g.mO.values
    return {"current (P0, O0)":pool(g,m0,"P0",None),
            "team offset only (P0, O*)":pool(g,mO,"P0",None),
            "HA pool (O0)":pool(g,m0,"HA",ch["w_HA_O0"]),
            "HDA pool (O0)":pool(g,m0,"HDA",ch["w_HDA_O0"]),
            "HA pool + team offset":pool(g,mO,"HA",ch["w_HA_O*"]),
            "HDA pool + team offset":pool(g,mO,"HDA",ch["w_HDA_O*"]),
            "opener only":pool(g,m0,"OPEN",None)}
for lab,g in (("TRAIN 2024-25",P[P.era.isin(["2024","2025"])]),("TEST 2026 all",P[P.season_is_2026] if False else P[P.era.isin(["2026H1","2026H2"])]),
              ("TEST 2026H1",P[P.era=="2026H1"]),("TEST 2026H2",P[P.era=="2026H2"])):
    C=closev(g); Q=build(g); base=xent3(Q["current (P0, O0)"],C); lb=logloss(Q["current (P0, O0)"],g.Res.values)
    print(f"\n[{lab}] n={len(g)}   current xent3={base.mean():.4f}")
    for k,q in Q.items():
        x=xent3(q,C)-base; l=logloss(q,g.Res.values)-lb; r2=rule_clv(g,q,0.2); r1=rule_clv(g,q,0.1)
        ci=boot(x) if k!="current (P0, O0)" else (0,0); cl=boot(l) if k!="current (P0, O0)" else (0,0)
        print(f"  {k:26s} xent3 {x.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  logloss {l.mean():+.4f} [{cl[0]:+.4f},{cl[1]:+.4f}]  @0.20 n={len(r2):3d} clv={r2.mean() if len(r2) else np.nan:+.3f}  @0.10 n={len(r1):3d} clv={r1.mean() if len(r1) else np.nan:+.3f}")
    for a,b in (("HA pool + team offset","HDA pool + team offset"),("HA pool (O0)","HDA pool (O0)")):
        x=xent3(Q[a],C)-xent3(Q[b],C); ci=boot(x); print(f"  >> {a} minus {b}: {x.mean():+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]")
P.to_pickle(__file__.replace("step3_test.py","P_test.pkl"))
