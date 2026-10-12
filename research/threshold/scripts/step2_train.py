"""TRAIN ONLY (2024-25): monotonicity, t*, and its stability. Freezes t* to choice.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403
c = candidates(); tr = c[c["split"] == "train"]
b, ci, n = slope(tr)
print(f"[1] CLV ~ EV slope, train, EV>0 (n={n}): {b:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
t_star, k = select(tr)
pd.set_option("display.width", 200)
print(k.round(3).to_string(index=False))
print("t* (train) =", t_star)
for s in (2024, 2025):
    print(f"  t* on {s} alone:", select(tr[tr["season"] == s])[0])
rng = np.random.default_rng(7); picks = []
for _ in range(500):
    picks.append(select(tr.iloc[rng.integers(0, len(tr), len(tr))])[0])
print("  bootstrap distribution of t* (500 resamples):",
      pd.Series(picks).value_counts(normalize=True, dropna=False).sort_index().round(3).to_dict())
json.dump({"t_star": t_star}, open(os.path.join(HERE, "choice.json"), "w"))
