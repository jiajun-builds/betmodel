"""Production path (walkforward.predict + score, leagues/ligamx.yml as committed) against the research
arrays, then the whole change against the previous calibration."""
from lib import *

new_cfg = load_league("ligamx")
assert new_cfg.model.team_offset.enabled and new_cfg.signals.debias.pool_weight == 0.38 and new_cfg.model.xi == 0.0015
old_cfg = dataclasses.replace(base_config(), signals=dataclasses.replace(
    new_cfg.signals, debias=dataclasses.replace(new_cfg.signals.debias, pool_weight=None)))

fn = wf.predict("ligamx", new_cfg, "2024-01-01"); fo = wf.predict("ligamx", old_cfg, "2024-01-01")
sn = wf.score("ligamx", new_cfg, fn).set_index(fn.index); so = wf.score("ligamx", old_cfg, fo).set_index(fo.index)

# --- consistency with the research arrays
F = annotate(load("wf.pkl")[0.0015]); S = pd.read_pickle(os.path.join(HERE, "shift_current_xi.pkl"))
F["shift"] = S["shift"]; F = F.loc[fn.index]
M = model_1x2(F, F["shift"].to_numpy())
print("max |model 1X2 production - research|:", np.abs(fn[["raw_h", "raw_d", "raw_a"]].to_numpy() - M).max())
print("max |offset production - research|:  ", np.abs(fn["offset"].to_numpy() - F["shift"].to_numpy()).max())
R_ = np.array([debias.pool(tuple(m), o, 0.38) if o else tuple(m) for m, o in zip(M, F.open_odds)])
print("max |final 1X2 production - research|:", np.abs(sn[["p_h", "p_d", "p_a"]].to_numpy() - R_).max())

# --- how much better, against the previous calibration
F = annotate(fn)
Pn = sn[["p_h", "p_d", "p_a"]].to_numpy(); Po = so.loc[fn.index, ["p_h", "p_d", "p_a"]].to_numpy()
q = np.array([c[0] / (c[0] + c[2]) if c else np.nan for c in F.close]); Cl = np.array([c if c else (np.nan,) * 3 for c in F.close])
sl = lambda P: lg(P[:, 0] / (P[:, 0] + P[:, 2]))
for era in ("train", "test"):
    e = (F.era == era).values; c = e & F.clean.values; o = c & F.open_odds.notna().values
    print(f"\n[{era}]")
    print(f"  home-share xent vs close  (n={c.sum():3d}) {fmt(bin_xent(sl(Pn)[c], q[c]) - bin_xent(sl(Po)[c], q[c]))}"
          f"   (previous {bin_xent(sl(Po)[c], q[c]).mean():.4f})")
    if o.sum() > 5:
        print(f"  3-way xent vs close, opener (n={o.sum():3d}) {fmt(xent3(Pn[o], Cl[o]) - xent3(Po[o], Cl[o]))}"
              f"   (previous {xent3(Po[o], Cl[o]).mean():.4f})")
    print(f"  3-way log loss on results (n={e.sum():3d}) {fmt(logloss3(Pn[e], F.Res.values[e]) - logloss3(Po[e], F.Res.values[e]))}"
          f"   (previous {logloss3(Po[e], F.Res.values[e]).mean():.4f})")

print("\n[descriptive only] CLV of the rule at ev_min 0.10 on Duel's openers")
for lab, cfg, f in (("previous", old_cfg, fo), ("new", new_cfg, fn)):
    t = wf.report(wf.score("ligamx", cfg, f, book="duel"))
    print(f"  {lab:8s} bets {int(t.loc['all','bets'])}  clv {t.loc['all','clv']:+.4f} [{t.loc['all','clv_low']:+.4f}, {t.loc['all','clv_high']:+.4f}]")
