"""Build the candidates and check them against walkforward.score at 0.20."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403
from lib import _triple
c = candidates()
print(c.groupby("split").size().to_dict())
config = load_league("csl")
frame = wf.predict("csl", config, "2024-03-01")
scored = wf.score("csl", config, frame)
anchored = [(_triple(r, "pinnacle", "open") is not None) for _, r in frame.iterrows()]
ours = c[c["ev"] > config.signals.ev_min]
theirs = scored.reset_index(drop=True)[pd.Series(anchored).values].dropna(subset=["clv"])
print(f"at ev_min {config.signals.ev_min}: candidates {len(ours)} bets, walkforward.score {len(theirs)} bets")
assert len(ours) == len(theirs) and np.allclose(sorted(ours["clv"]), sorted(theirs["clv"]))
print("match: identical bets and CLV")
