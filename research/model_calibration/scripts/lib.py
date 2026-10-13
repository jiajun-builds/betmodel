import pickle, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
lg=lambda p: np.log(p/(1-p)); sg=lambda x: 1/(1+np.exp(-x))
S=pickle.load(open(__file__.replace("lib.py","step1.pkl"),"rb"))
frame,F,INS=S["frame"],S["fixtures"],S["insample"]
def share(c): return lg(c[0]/(c[0]+c[2]))
CL={i:share(c) for i,c in frame.close.items() if c is not None}
DAY=frame._day
# evaluation panel: priced fixtures with opener and close, from 2024-03
P=F.join(frame[["Season","open","close","Res","onexbet_open_h","onexbet_open_d","onexbet_open_a","duel_open_h","duel_open_d","duel_open_a"]])
P=P[(P.day>=pd.Timestamp("2024-03-01"))&P.open.notna()&P.close.notna()].copy()
P["era"]=np.where(P.day>=pd.Timestamp("2026-07-01"),"2026H2",np.where(P.Season==2026,"2026H1",P.Season.astype(str)))
P["m"]=lg(P.raw_h/(P.raw_h+P.raw_a))
P["o"]=[share(x) for x in P.open]; P["c"]=[share(x) for x in P.close]
def history(day, variant, perm=None):
    """(age_days, home, away, resid) for matches before `day` within 18 months."""
    lo=day-pd.DateOffset(months=18)
    if variant=="a":
        h=F[(F.day<day)&(F.day>=lo)]; h=h[h.index.isin(CL.keys())]
        res=np.array([CL[i] for i in h.index])-lg(h.raw_h/(h.raw_h+h.raw_a)).values
        out=pd.DataFrame(dict(age=(day-h.day).dt.days.values,home=h.home.values,away=h.away.values,res=res))
    else:
        rows=INS.get(day,[])
        if not rows: return None
        idx=[r[0] for r in rows]; ph=np.array([r[1] for r in rows]); pa=np.array([r[3] for r in rows])
        res=np.array([CL[i] for i in idx])-lg(ph/(ph+pa))
        out=pd.DataFrame(dict(age=(day-DAY.loc[idx]).dt.days.values,home=frame.loc[idx,"Home"].values,away=frame.loc[idx,"Away"].values,res=res))
    if perm is not None:   # placebo: relabel teams with a random permutation (consistent within this history)
        teams=sorted(set(out.home)|set(out.away)); sh=perm.permutation(teams); mp=dict(zip(teams,sh))
        out["home"]=out.home.map(mp); out["away"]=out.away.map(mp)
    return out
def offsets(h, xi, lam):
    if h is None or len(h)<30: return 0.0,{}
    teams=sorted(set(h.home)|set(h.away)); ti={t:i for i,t in enumerate(teams)}
    X=np.zeros((len(h),len(teams)+1)); X[:,-1]=1
    X[np.arange(len(h)),[ti[t] for t in h.home]]+=1; X[np.arange(len(h)),[ti[t] for t in h.away]]-=1
    w=np.exp(-xi*h.age.values); A=X*np.sqrt(w)[:,None]; y=h.res.values*np.sqrt(w)
    Pn=np.eye(X.shape[1])*lam; Pn[-1,-1]=0
    b=np.linalg.solve(A.T@A+Pn,A.T@y); return b[-1],{t:b[ti[t]] for t in teams}
def adjusted(P, variant, xi, lam, perm_seed=None):
    if variant is None: return P.m.values.copy()
    rng=np.random.default_rng(perm_seed) if perm_seed is not None else None
    out=np.empty(len(P)); cache={}
    for k,(i,r) in enumerate(P.iterrows()):
        if r.day not in cache: cache[r.day]=offsets(history(r.day,variant,rng),xi,lam)
        kap,d=cache[r.day]; out[k]=r.m+kap+d.get(r.home,0.0)-d.get(r.away,0.0)
    return out
def model_1x2(P, m_adj):
    """model 1X2 with its own draw, H/A split from the (adjusted) share."""
    d=P.raw_d.values; s=sg(m_adj); return np.c_[(1-d)*s,d,(1-d)*(1-s)]
def opener(P): return np.array(list(P.open.values))
def closev(P): return np.array(list(P.close.values))
def pool(P, m_adj, kind, w):
    O=opener(P)
    if kind=="P0":  s=sg(m_adj); return np.c_[(1-O[:,1])*s,O[:,1],(1-O[:,1])*(1-s)]
    if kind=="HA":  s=sg(w*m_adj+(1-w)*lg(O[:,0]/(O[:,0]+O[:,2]))); return np.c_[(1-O[:,1])*s,O[:,1],(1-O[:,1])*(1-s)]
    if kind=="HDA":
        M=model_1x2(P,m_adj); L=w*np.log(M)+(1-w)*np.log(O); E=np.exp(L-L.max(1,keepdims=True)); return E/E.sum(1,keepdims=True)
    if kind=="OPEN": return O
def xent3(Q,C): return -(C*np.log(Q)).sum(1)
def bin_xent(s,c): p=sg(s); q=sg(c); return -(q*np.log(p)+(1-q)*np.log(1-p))
def fit_w(P, m_adj, kind):
    C=closev(P); r=minimize_scalar(lambda w: xent3(pool(P,m_adj,kind,w),C).mean(),bounds=(0,1),method="bounded"); return r.x
def logloss(Q,res): return -np.log(Q[np.arange(len(Q)),["HDA".index(x) for x in res]])
def rule_clv(P,Q,thr=0.2,cap=7.0):
    C=closev(P); out=[]
    for k,(i,r) in enumerate(P.iterrows()):
        best=None
        for j,leg in ((0,"h"),(2,"a")):
            q=[r[f"{b}_open_{leg}"] for b in ("onexbet","duel")]; q=[x for x in q if pd.notna(x) and x>1]
            if q:
                od=max(q); e=Q[k,j]*od-1
                if best is None or e>best[0]: best=(e,od,j)
        if best and best[0]>thr and best[1]<=cap: out.append(C[k,best[2]]*best[1]-1)
    return np.array(out)
def boot(x,n=4000,seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x); m=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(n)]; return np.percentile(m,[2.5,97.5])
