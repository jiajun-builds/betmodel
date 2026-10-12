import sys, os, logging
sys.path.insert(0,__file__.rsplit("/",1)[0])
from lib import *
R=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0,R+"/scripts"); sys.path.insert(0,R+"/src"); os.chdir(R)
import importlib, walkforward as wf
from betmodel.config import load_league
from betmodel.signals import debias
logging.disable(logging.WARNING)
cfg=load_league("csl"); assert cfg.model.team_offset.enabled and cfg.signals.debias.pool_weight==0.38
f=wf.predict("csl",cfg,"2024-03-01")
Pt=pd.read_pickle(__file__.replace("consistency.py","P_test.pkl"))
j=Pt.join(f[["raw_h","raw_d","raw_a","offset"]],rsuffix="_prod",how="inner")
print("fixtures research",len(Pt),"production",len(f),"joined",len(j))
research=model_1x2(j,j.mO.values)
prod=j[["raw_h_prod","raw_d_prod","raw_a_prod"]].to_numpy()
print("max |model 1X2 production - research|:",np.abs(prod-research).max())
# final probabilities through the production debias
fin=np.array([debias.apply(tuple(p),cfg.signals.debias,anchor_odds=tuple(r[f"pinnacle_open_{l}"] for l in "hda"))[0]
              for p,(_,r) in zip(prod,j.join(f[["pinnacle_open_h","pinnacle_open_d","pinnacle_open_a"]]).iterrows())])
ref=pool(j,j.mO.values,"HDA",0.38)
print("max |final probability production - research (w=0.38)|:",np.abs(fin-ref).max())
C=closev(j); cur=pool(j,j.m.values,"P0",None)
for lab,mask in (("train",j.era.isin(["2024","2025"]).values),("test",j.era.isin(["2026H1","2026H2"]).values)):
    x=xent3(fin[mask],C[mask])-xent3(cur[mask],C[mask]); ci=boot(x); print(f"  production path, {lab}: xent3 vs current {x.mean():+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]")
