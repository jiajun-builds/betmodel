"""Pool on Apertura 2026 fixtures with a Pinnacle opener and a clean close.
Model = xi 0.0015 + team offset. P0 (draw anchored) vs HDA w=0.38 (CSL's, no Liga MX fit) vs HDA rolling w."""
from lib import *

F = load("wf.pkl")[0.0015]; S = pd.read_pickle(os.path.join(HERE, "shift_current_xi.pkl"))
assert (S["index"] == F.index).all()
F = F.assign(shift=S["shift"])
H = F[F.clean & F.open_odds.notna()].sort_values("_day")         # every opener + clean close (2024's 4 included)
M = model_1x2(H, H["shift"].to_numpy()); O = list(H.open_odds); C = np.array(list(H.close))

# rolling w: fitted on strictly earlier days, once at least 20 exist
w_roll = np.full(len(H), np.nan)
for k, day in enumerate(H._day):
    past = (H._day < day).to_numpy()
    if past.sum() >= 20:
        w_roll[k] = fit_w(M[past], [o for o, p in zip(O, past) if p], C[past])
ev = (H.half == "Apertura 2026").to_numpy()
print(f"opener + clean close: {len(H)} (Apertura 2026: {ev.sum()}); rolling w active on {(ev & ~np.isnan(w_roll)).sum()}")
print("rolling w trajectory:", " ".join(f"{d.date()}:{w:.2f}" for d, w in zip(H._day[ev], w_roll[ev]) if not np.isnan(w))[:900])

base = p0(M, O); arms = {"HDA w=0.38": hda(M, O, 0.38), "HDA w=0.27": hda(M, O, 0.27), "HDA w=0.54": hda(M, O, 0.54),
                          "opener only": hda(M, O, 0.0)}
roll = base.copy(); act = ~np.isnan(w_roll)
roll[act] = np.array([debias.pool(tuple(m), o, w) for m, o, w, a in zip(M, O, w_roll, act) if a])
arms["HDA rolling w"] = roll
res = H.Res.values

def report(mask, label):
    n = mask.sum(); idx = np.where(mask)[0]; first, second = idx[: n // 2], idx[n // 2:]
    print(f"\n[{label}] n={n}   P0 xent3 {xent3(base[mask], C[mask]).mean():.4f}")
    for k, Q in arms.items():
        x = xent3(Q, C) - xent3(base, C); l = logloss3(Q, res) - logloss3(base, res); lc = boot(l[mask])
        print(f"  {k:14s} xent3 {fmt(x[mask])}  halves {x[first].mean():+.4f} / {x[second].mean():+.4f}"
              f"   logloss {l[mask].mean():+.4f} [{lc[0]:+.4f}, {lc[1]:+.4f}]")
    d = xent3(arms["HDA rolling w"], C) - xent3(arms["HDA w=0.38"], C)
    print(f"  rolling minus fixed 0.38: {fmt(d[mask])}")

report(ev, "Apertura 2026, all with opener")
report(ev & act, "Apertura 2026, where rolling w is active")
