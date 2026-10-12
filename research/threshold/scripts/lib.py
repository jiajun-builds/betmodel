"""Candidate bets under the production calibration, and the statistics on them."""
import os, sys, logging
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, os.path.join(R, "src"))
import walkforward as wf  # noqa: E402
from betmodel.config import load_league  # noqa: E402
from betmodel.signals import debias  # noqa: E402
from betmodel.signals.ev import devig  # noqa: E402

GRID = (0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30)
HERE = os.path.dirname(os.path.abspath(__file__))


def _triple(row, prefix, phase):
    o = tuple(row.get(f"{prefix}_{phase}_{leg}") for leg in "hda")
    return o if all(pd.notna(x) and x > 1.0 for x in o) else None


def candidates():
    """One row per anchored fixture with a close: the highest-EV side under the cap."""
    path = os.path.join(HERE, "candidates.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    logging.disable(logging.WARNING)
    os.chdir(R)
    config = load_league("csl")
    frame = wf.predict("csl", config, "2024-03-01")
    books = [b.schema_prefix for b in config.odds.bet_books if b.schema_prefix]
    rows = []
    for index, row in frame.iterrows():
        opener, close = _triple(row, "pinnacle", "open"), _triple(row, "pinnacle", "close")
        if not (opener and close):
            continue
        probs, method = debias.apply((row["raw_h"], row["raw_d"], row["raw_a"]),
                                     config.signals.debias, anchor_odds=opener)
        assert method == debias.MARKET_ANCHOR
        fair = devig(close)
        best = None
        for i, leg in ((0, "h"), (2, "a")):
            quoted = [row.get(f"{p}_open_{leg}") for p in books]
            quoted = [q for q in quoted if pd.notna(q) and q > 1.0]
            if quoted:
                odds = max(quoted)
                ev = probs[i] * odds - 1.0
                if best is None or ev > best[0]:
                    best = (ev, odds, i)
        if best is None or best[1] > config.signals.odds_cap:
            continue
        ev, odds, i = best
        rows.append({"index": index, "day": row["_day"], "season": row["Season"],
                     "home": row["Home"], "away": row["Away"], "side": "HDA"[i],
                     "odds": odds, "ev": ev, "clv": fair[i] * odds - 1.0})
    out = pd.DataFrame(rows).set_index("index")
    out["split"] = np.where(out["season"] == 2026, "test", "train")
    out.to_pickle(path)
    return out


def boot(values, stat=np.mean, n=4000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return tuple(np.percentile(draws, [2.5, 97.5]))


def curve(c, grid=GRID):
    rows = []
    for t in grid:
        b = c[c["ev"] > t]["clv"].to_numpy()
        lo, hi = boot(b)
        tlo, thi = boot(b, np.sum)
        rows.append({"t": t, "n": len(b), "mean_clv": b.mean() if len(b) else np.nan,
                     "mean_lo": lo, "mean_hi": hi, "total_clv": b.sum(),
                     "total_lo": tlo, "total_hi": thi})
    return pd.DataFrame(rows)


def select(c, grid=GRID):
    """t* = the highest total CLV among thresholds whose mean-CLV CI clears zero."""
    k = curve(c, grid)
    ok = k[k["mean_lo"] > 0]
    return (None if ok.empty else float(ok.loc[ok["total_clv"].idxmax(), "t"])), k


def slope(c):
    """OLS slope of CLV on EV over candidates with EV > 0, with a bootstrap CI."""
    d = c[c["ev"] > 0]
    x, y = d["ev"].to_numpy(), d["clv"].to_numpy()
    b = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(4000):
        i = rng.integers(0, len(x), len(x))
        draws.append(np.polyfit(x[i], y[i], 1)[0])
    return b, tuple(np.percentile(draws, [2.5, 97.5])), len(d)
