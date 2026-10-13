"""Walk-forward (production protocol: train on matches dated before the match day).
Stores (a) each fixture's prediction and (b) for each match day, the current fit's prediction of every
training-window match, so team offsets can be estimated without refitting."""
import sys, os, logging, pickle, numpy as np, pandas as pd
R=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0,R+"/scripts"); sys.path.insert(0,R+"/src"); os.chdir(R)
import walkforward as wf
from betmodel.config import load_league
from betmodel.models.dc import FitError, fit_production_model, prepare_training_frame
from betmodel.xg.blend import recompute
from betmodel.dates import parse_date_only_series
from betmodel.signals.ev import devig
logging.disable(logging.WARNING)
cfg=load_league("csl")
frame=pd.read_csv(R+"/data/csl/matches.csv")
frame,_=recompute(frame,cfg)
frame["_day"]=parse_date_only_series(frame["Date"])
frame=frame[frame["Res"].isin(wf.RESULT)].sort_values("_day",kind="stable")
def trip(r,p):
    o=tuple(r[f"{p}_{l}"] for l in "hda"); return devig(o) if all(pd.notna(x) and x>1 for x in o) else None
frame["close"]=[trip(r,"pinnacle_close") for _,r in frame.iterrows()]
frame["open"]=[trip(r,"pinnacle_open") for _,r in frame.iterrows()]
fixtures=[]; insample={}
for day in sorted(frame._day.unique()):
    try:
        tr=prepare_training_frame(frame[frame._day<day].drop(columns=["_day","close","open"]),cfg)
        fit=fit_production_model(tr,cfg)
    except FitError:
        continue
    teams=set(fit.teams)
    for idx,row in frame[frame._day==day].iterrows():
        if row.Home in teams and row.Away in teams:
            o=fit.model.outcome_probabilities(row.Home,row.Away)
            fixtures.append(dict(idx=idx,day=day,home=row.Home,away=row.Away,raw_h=o.home,raw_d=o.draw,raw_a=o.away))
    # in-sample predictions of training-window matches that have a close
    win=frame.loc[tr.index] if set(tr.index)<=set(frame.index) else None
    rows=[]
    for idx,row in frame[(frame._day<day)&(frame._day>=day-pd.DateOffset(months=18))].iterrows():
        if row.close is None or row.Home not in teams or row.Away not in teams: continue
        o=fit.model.outcome_probabilities(row.Home,row.Away)
        rows.append((idx,o.home,o.draw,o.away))
    insample[day]=rows
F=pd.DataFrame(fixtures).set_index("idx")
pickle.dump(dict(frame=frame,fixtures=F,insample=insample),open(sys.argv[1],"wb"))
print(len(F), F.day.min(), F.day.max(), len(insample))
