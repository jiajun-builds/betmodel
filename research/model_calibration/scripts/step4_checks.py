import json, sys
sys.path.insert(0,__file__.rsplit("/",1)[0])
from lib import *
ch=json.load(open(__file__.replace("step4_checks.py","choice.json"))); o=ch["offset"]
P=pd.read_pickle(__file__.replace("step4_checks.py","P_test.pkl"))
tr=P.era.isin(["2024","2025"]).values; te=P.era.isin(["2026H1","2026H2"]).values
C=closev(P)
def gain_offset(mO, mask):   # O + P0 vs current, xent3
    return (xent3(pool(P[mask],mO[mask],"P0",None),C[mask])-xent3(pool(P[mask],P.m.values[mask],"P0",None),C[mask])).mean()
real_tr=gain_offset(P.mO.values,tr); real_te=gain_offset(P.mO.values,te)
print(f"[placebo] real offsets: train {real_tr:+.5f}  test {real_te:+.5f}")
pl=[]
for s in range(20):
    mp=adjusted(P,o["variant"],o["xi"],o["lam"],perm_seed=s); pl.append((gain_offset(mp,tr),gain_offset(mp,te)))
pl=np.array(pl); print(f"  permuted team labels (20x): train mean {pl[:,0].mean():+.5f} (min {pl[:,0].min():+.5f})  test mean {pl[:,1].mean():+.5f} (min {pl[:,1].min():+.5f}, max {pl[:,1].max():+.5f})")
print(f"  share of placebo runs at least as good as real: train {np.mean(pl[:,0]<=real_tr):.2f} test {np.mean(pl[:,1]<=real_te):.2f}")
print("\n[sensitivity] offset neighbours on TEST (O + P0 vs current, xent3); train value in brackets")
for xi in (0.004,0.008):
    for lam in (2,4,8,16):
        mO=adjusted(P,"a",xi,lam); print(f"  xi={xi} lam={lam:2d}: test {gain_offset(mO,te):+.5f}  [train {gain_offset(mO,tr):+.5f}]")
print("\n[w sensitivity] HDA + team offset on TEST at weights fitted on 2024 only / 2025 only / both")
g=P[te]; base=xent3(pool(g,g.m.values,"P0",None),C[te])
for lab,mask in (("2024 only",(P.era=="2024").values),("2025 only",(P.era=="2025").values)):
    for kind in ("HA","HDA"):
        w=fit_w(P[mask],P.mO.values[mask],kind); x=xent3(pool(g,g.mO.values,kind,w),C[te])-base; ci=boot(x)
        print(f"  {kind:3s} w({lab})={w:.3f}: test {x.mean():+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]")
for kind in ("HA","HDA"):
    w=ch[f"w_{kind}_O*"]; x=xent3(pool(g,g.mO.values,kind,w),C[te])-base; print(f"  {kind:3s} w(both)={w:.3f}: test {x.mean():+.5f}")
print("\n[incremental] does the team offset still help once pooled? HDA+O minus HDA(O0)")
for lab,mask in (("train",tr),("test",te)):
    g=P[mask]; x=xent3(pool(g,g.mO.values,"HDA",ch["w_HDA_O*"]),C[mask])-xent3(pool(g,g.m.values,"HDA",ch["w_HDA_O0"]),C[mask]); ci=boot(x)
    print(f"  {lab}: {x.mean():+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]")
print("\n[symptoms] model minus close on bet-relevant sides, by opener-implied strength (TEST 2026, home share logit)")
g=P[te].copy(); 
def sh(Q): return lg(Q[:,0]/(Q[:,0]+Q[:,2]))
g["cur"]=sh(pool(g,g.m.values,"P0",None))-g.c; g["new"]=sh(pool(g,g.mO.values,"HDA",ch["w_HDA_O*"]))-g.c
g["b"]=pd.cut(sg(g.o),[0,.3,.45,.55,.7,1])
print(g.groupby("b",observed=True)[["cur","new"]].agg(["mean","size"]).round(3).to_string())
