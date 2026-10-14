"""Walk-forward predictions for every xi in the grid (production loop, offsets off). Writes wf.pkl."""
import pickle
from lib import *

out = {}
for xi in XI_GRID:
    F = wf.predict("ligamx", base_config(xi), "2023-01-01")
    out[xi] = annotate(F)
    print(f"xi={xi}: {len(F)} priced, {F._day.min().date()}..{F._day.max().date()}", flush=True)
pickle.dump(out, open(os.path.join(HERE, "wf.pkl"), "wb"))
