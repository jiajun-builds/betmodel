"""Side-level candidates under the old and the D38 calibration (production code)."""
import os, sys, logging, math
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, os.path.join(R, "src"))
import walkforward as wf  # noqa: E402
from betmodel.config import load_league  # noqa: E402
from betmodel.signals import debias  # noqa: E402
from betmodel.signals.ev import devig  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def _triple(row, prefix, phase):
    o = tuple(row.get(f"{prefix}_{phase}_{leg}") for leg in "hda")
    return o if all(pd.notna(x) and x > 1.0 for x in o) else None


def sides():
    path = os.path.join(HERE, "sides.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    logging.disable(logging.WARNING); os.chdir(R)
    cfg = load_league("csl")
    f = wf.predict("csl", cfg, "2024-03-01")
    books = [b.schema_prefix for b in cfg.odds.bet_books if b.schema_prefix]
    rows = []
    for idx, r in f.iterrows():
        op, cl = _triple(r, "pinnacle", "open"), _triple(r, "pinnacle", "close")
        if not (op and cl):
            continue
        o, c = devig(op), devig(cl)
        new, _ = debias.apply((r.raw_h, r.raw_d, r.raw_a), cfg.signals.debias, anchor_odds=op)
        share_old = sg(math.log(r.raw_h / r.raw_a) - r.offset)       # model split before the team offset
        old = ((1 - o[1]) * share_old, o[1], (1 - o[1]) * (1 - share_old))  # draw replaced, split = model
        for i, leg in ((0, "h"), (2, "a")):
            q = [r.get(f"{p}_open_{leg}") for p in books]; q = [x for x in q if pd.notna(x) and x > 1.0]
            if not q:
                continue
            odds = max(q)
            if odds > cfg.signals.odds_cap:
                continue
            won = r.Res == "HDA"[i]
            rows.append(dict(fixture=idx, day=r._day, season=r.Season, side="HDA"[i], odds=odds,
                             p_open=o[i], p_close=c[i], p_old=old[i], p_new=new[i],
                             underdog=o[i] < o[2 - i], ret=(odds - 1.0) if won else -1.0, won=won))
    d = pd.DataFrame(rows)
    d["split"] = np.where(d.season == 2026, "test", "train")
    for k in ("old", "new"):
        d[f"ev_{k}"] = d[f"p_{k}"] * d.odds - 1
    d["ev_close"] = d.p_close * d.odds - 1          # = CLV of the side
    d.to_pickle(path)
    return d


def boot(x, stat=np.mean, n=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    return tuple(np.percentile([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)], [2.5, 97.5]))


def pick_per_fixture(d, score):
    """The engine's rule: one side per fixture, the one with the highest score."""
    return d.loc[d.groupby("fixture")[score].idxmax()]
