"""TRAIN ONLY (2024-2025): select offset hyperparameters, then pooling weights. Writes frozen choices."""
import json, itertools, sys
sys.path.insert(0,__file__.rsplit("/",1)[0])
from lib import *
T=P[P.era.isin(["2024","2025"])]
print("train fixtures",len(T))
base=bin_xent(T.m,T.c).mean(); print(f"O0 home-share xent {base:.5f}")
rows=[]
for v,xi,lam in itertools.product(("a","b"),(0.002,0.004,0.008),(2,4,8,16,32)):
    ma=adjusted(T,v,xi,lam); rows.append(dict(variant=v,xi=xi,lam=lam,xent=bin_xent(ma,T.c).mean()))
G=pd.DataFrame(rows); G["delta"]=G.xent-base
print(G.pivot_table(index=["variant","xi"],columns="lam",values="delta").round(5).to_string())
best=G.sort_values("xent").iloc[0].to_dict(); print("selected offsets:",best)
choice={"offset":{"variant":best["variant"],"xi":best["xi"],"lam":best["lam"]}}
mO0=T.m.values; mO=adjusted(T,best["variant"],best["xi"],best["lam"])
C=closev(T)
for lab,ma in (("O0",mO0),("O*",mO)):
    for kind in ("HA","HDA"):
        w=fit_w(T,ma,kind); choice[f"w_{kind}_{lab}"]=w
        print(f"  {lab} {kind}: w={w:.3f} train xent3={xent3(pool(T,ma,kind,w),C).mean():.5f}  (P0 {xent3(pool(T,ma,'P0',None),C).mean():.5f})")
    # stability: fit on each season alone
    for s in ("2024","2025"):
        Ts=T[T.era==s]; ms=ma[(T.era==s).values]
        print(f"     {lab} fitted on {s} only: w_HA={fit_w(Ts,ms,'HA'):.3f} w_HDA={fit_w(Ts,ms,'HDA'):.3f}")
print("train xent3: current P0+O0 =",xent3(pool(T,mO0,"P0",None),C).mean().round(5)," OPEN =",xent3(pool(T,mO0,"OPEN",None),C).mean().round(5))
json.dump(choice,open(__file__.replace("step2_train.py","choice.json"),"w"),indent=1); print(choice)
