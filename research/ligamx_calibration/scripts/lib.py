"""Shared helpers. Everything that production also does is done by production's own code."""
import dataclasses, logging, os, pickle, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, R + "/scripts"); sys.path.insert(0, R + "/src"); os.chdir(R)
import walkforward as wf  # noqa: E402
from betmodel.config import load_league  # noqa: E402
from betmodel.config.schema import TeamOffset  # noqa: E402
from betmodel.models import team_offset  # noqa: E402
from betmodel.models.poisson import Outcome  # noqa: E402
from betmodel.signals import debias  # noqa: E402
from betmodel.signals.ev import devig  # noqa: E402
logging.disable(logging.WARNING)

XI_GRID = (0.0015, 0.002, 0.003, 0.004, 0.006, 0.008)
OFF_XI = (0.002, 0.004, 0.008); OFF_LAM = (2, 4, 8, 16, 32)
MAX_MARGIN = 0.05
lg = lambda p: np.log(p / (1 - p)); sg = lambda x: 1 / (1 + np.exp(-x))


def base_config(xi=None):
    cfg = load_league("ligamx")
    cfg = wf.configure(cfg, xi=xi)
    model = dataclasses.replace(cfg.model, team_offset=dataclasses.replace(cfg.model.team_offset, enabled=False))
    return dataclasses.replace(cfg, model=model)


def triple(row, prefix, phase):
    o = tuple(row.get(f"{prefix}_{phase}_{x}") for x in "hda")
    return o if all(pd.notna(v) and v > 1 for v in o) else None


def annotate(F):
    """Clean close, opener, era. A close above MAX_MARGIN overround is not a yardstick."""
    F = F.copy()
    cl, op, clean = [], [], []
    for _, r in F.iterrows():
        c = triple(r, "pinnacle", "close"); o = triple(r, "pinnacle", "open")
        ok = c is not None and sum(1 / v for v in c) - 1 <= MAX_MARGIN
        clean.append(ok); cl.append(devig(c) if ok else None); op.append(o)
    F["close"] = cl; F["clean"] = clean; F["open_odds"] = op
    d = F["_day"]
    F["era"] = np.where(d < "2024-01-01", "pre", np.where(d < "2026-01-01", "train", "test"))
    F["half"] = np.where(d < "2026-01-01", F["Season"], np.where(d < "2026-07-01", "Clausura 2026", "Apertura 2026"))
    return F


def load(name):
    return pickle.load(open(os.path.join(HERE, name), "rb"))


def share_logit(F, h="raw_h", a="raw_a"):
    return lg(F[h] / (F[h] + F[a])).to_numpy()


def close_share(F):
    return np.array([c[0] / (c[0] + c[2]) for c in F["close"]])


def bin_xent(s_logit, q):
    p = sg(s_logit); return -(q * np.log(p) + (1 - q) * np.log(1 - p))


def logloss3(Q, res):
    return -np.log(Q[np.arange(len(Q)), ["HDA".index(x) for x in res]])


def xent3(Q, C):
    return -(C * np.log(Q)).sum(1)


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def fmt(x):
    ci = boot(x); return f"{np.mean(x):+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]"


# ---------- team offsets, through production's estimate() ----------

def settings(xi, lam):
    return TeamOffset(enabled=True, xi=xi, lam=lam)


def offsets_for(F, res, xi, lam, perm_seed=None):
    """Logit shift for every row of F, fitted on residuals strictly before its day."""
    rng = np.random.default_rng(perm_seed) if perm_seed is not None else None
    s = settings(xi, lam); cache = {}; out = np.zeros(len(F))
    for k, (day, h, a) in enumerate(zip(F["_day"], F["Home"], F["Away"])):
        if day not in cache:
            r = res
            if rng is not None:   # placebo: relabel clubs within this history
                lo = day - pd.DateOffset(months=s.window_months)
                r = res[(res.day < day) & (res.day >= lo)].copy()
                teams = sorted(set(r.home) | set(r.away)); mp = dict(zip(teams, rng.permutation(teams)))
                r["home"] = r.home.map(mp); r["away"] = r.away.map(mp)
            cache[day] = team_offset.estimate(r, day, s)
        out[k] = cache[day].shift(h, a)
    return out


def model_1x2(F, shift):
    """Model 1X2 with the home/away split moved by `shift`, draw untouched (= Offsets.adjust)."""
    d = F.raw_d.to_numpy(); s = sg(share_logit(F) + shift)
    return np.c_[(1 - d) * s, d, (1 - d) * (1 - s)]


# ---------- pooling, through production's debias ----------

def p0(M, opens):
    """The previous calibration: the draw replaced by the opener's, whatever the live config now says."""
    cfg = dataclasses.replace(load_league("ligamx").signals.debias, pool_weight=None)
    return np.array([debias.apply(tuple(m), cfg, anchor_odds=o)[0] for m, o in zip(M, opens)])


def hda(M, opens, w):
    return np.array([debias.pool(tuple(m), o, w) for m, o in zip(M, opens)])


def fit_w(M, opens, C):
    return minimize_scalar(lambda w: xent3(hda(M, opens, w), C).mean(), bounds=(0, 1), method="bounded").x


def residuals(F, cfg):
    """Production's residuals: every close in the table (cleaned or not), model at the time."""
    P = ["raw_h", "raw_d", "raw_a"]
    return team_offset.residuals(F.drop(columns=P + ["evidence", "close", "clean", "open_odds"]), F[P], cfg)
