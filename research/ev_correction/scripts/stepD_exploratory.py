"""EXPLORATORY (not in the pre-registered plan): among selected bets, do underdogs do worse than favourites?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403
d = sides()
for k in ("old", "new"):
    for sp in ("train", "test"):
        p = pick_per_fixture(d[d.split == sp], f"ev_{k}")
        for t in (0.05, 0.10, 0.20):
            s = p[p[f"ev_{k}"] > t]
            u, f = s[s.underdog], s[~s.underdog]
            cu, cf = boot(u.ev_close), boot(f.ev_close)
            print(f"  {k:3s} {sp:5s} EV>{t:.2f}: underdog n={len(u):3d} EV {u[f'ev_{k}'].mean():+.3f} CLV {u.ev_close.mean():+.3f} [{cu[0]:+.3f},{cu[1]:+.3f}] | "
                  f"favourite n={len(f):3d} EV {f[f'ev_{k}'].mean():+.3f} CLV {f.ev_close.mean():+.3f} [{cf[0]:+.3f},{cf[1]:+.3f}]")
