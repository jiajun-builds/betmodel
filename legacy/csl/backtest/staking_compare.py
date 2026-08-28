"""Compare staking schemes on the recommended bet set (lam75 · EV>0.20 · odds<=7).

Q1: 0.5-Kelly final + maxDD.
Q2: three schemes -
  M1 fractional Kelly (0.25, per-bet, as in the chart)
  M2 flat unit      = 5% of INITIAL bankroll ($5 flat), never recompounded
  M3 compound weekly = 5% of bankroll, rebalanced each week (Season,Round); the
                       week's new bankroll = last week's + last week's P/L
All start at $100. maxDD = peak-to-trough on the after-bet equity curve.

Reproduce (repo root):  PYTHONPATH=src python backtest/staking_compare.py
Writes backtest/staking_compare.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_1xbet import load, walk_forward  # noqa: E402  reuse validated harness

HERE = os.path.dirname(os.path.abspath(__file__))
b = walk_forward(load())
b["date"] = pd.to_datetime(b["date"])
b = b.sort_values("date").reset_index(drop=True)  # match plot_bankroll.py ordering

O = b[["oH", "oD", "oA"]].values
C = b[["cH", "cD", "cA"]].values
P_raw = b[["pH", "pD", "pA"]].values
novig_o = (1 / O) / (1 / O).sum(1, keepdims=True)
res = b["res"].values
pD = 0.25 * P_raw[:, 1] + 0.75 * novig_o[:, 1]
sc = (1 - pD) / (1 - P_raw[:, 1])
P = np.column_stack([P_raw[:, 0] * sc, pD, P_raw[:, 2] * sc])
ev = P * O - 1.0
pick = ev.argmax(1)
best = ev.max(1)
i = np.arange(len(b))
pl = np.where(pick == res, O[i, pick] - 1.0, -1.0)   # per-UNIT P/L (win: odds-1, lose: -1)
odds = O[i, pick]
p = P[i, pick]

mask = (best > 0.20) & (odds <= 7.0)
idx = np.where(mask)[0]
sub = b.iloc[idx].copy()
pl_s = pl[idx]
odds_s = odds[idx]
p_s = p[idx]
BANK0 = 100.0


def maxdd(banks):
    banks = np.asarray(banks)
    return (1 - banks / np.maximum.accumulate(banks)).max() * 100


def kelly(frac, cap=0.9):
    bank = BANK0
    curve = [BANK0]
    binds = 0
    for k in range(len(idx)):
        o = odds_s[k]
        f_raw = max((p_s[k] * o - 1) / (o - 1), 0.0) * frac
        f = min(f_raw, cap)
        binds += f_raw > cap
        bank += bank * f * pl_s[k]
        curve.append(bank)
    return np.array(curve), binds


def flat_unit(pct=0.05):
    """M2: unit = pct of INITIAL bankroll, fixed dollar stake."""
    unit = BANK0 * pct
    bank = BANK0
    curve = [BANK0]
    for k in range(len(idx)):
        bank += unit * pl_s[k]
        curve.append(bank)
    return np.array(curve)


def compound_weekly(pct=0.05):
    """M3: unit = pct of bankroll, rebalanced each (Season,Round) week."""
    weeks = list(sub.groupby(["Season", "Round"], sort=False).indices.items())
    # groupby on already-chronological sub preserves order via sort=False
    bank = BANK0
    curve = [BANK0]
    order = sorted(range(len(sub)), key=lambda j: (sub["date"].iloc[j], sub["Round"].iloc[j]))
    # iterate week by week in chronological order
    seen = []
    wk_keys = []
    for j in order:
        key = (int(sub["Season"].iloc[j]), int(sub["Round"].iloc[j]))
        if key not in wk_keys:
            wk_keys.append(key)
    posmap = {j: (int(sub["Season"].iloc[j]), int(sub["Round"].iloc[j])) for j in range(len(sub))}
    for key in wk_keys:
        unit = bank * pct                      # rebalance at week start
        members = [j for j in order if posmap[j] == key]
        for j in members:
            bank += unit * pl_s[j]             # all bets this week use same unit
            curve.append(bank)
    return np.array(curve)


# ---- Q1: 0.5 Kelly ----
c25, _ = kelly(0.25)
c50, binds50 = kelly(0.50)
print("Q1  0.50-Kelly : final ${:.0f}   maxDD {:.0f}%   (0.9-cap bound on {} bets)".format(
    c50[-1], maxdd(c50), binds50))
print("    0.25-Kelly : final ${:.0f}   maxDD {:.0f}%  (reference / chart)".format(c25[-1], maxdd(c25)))

# ---- Q2: three schemes ----
m1 = c25
m2 = flat_unit(0.05)
m3 = compound_weekly(0.05)
print("\nQ2  scheme comparison (n={} bets, {} weeks):".format(
    len(idx), sub.groupby(['Season', 'Round']).ngroups))
for name, cv in [("M1 0.25-Kelly (chart)", m1), ("M2 flat 5% ($5)", m2), ("M3 compound 5%/week", m3)]:
    print(f"    {name:<24} final ${cv[-1]:>6.0f}   x{cv[-1]/100:>4.2f}   maxDD {maxdd(cv):>4.0f}%")

# per-season endpoints for M2/M3
seas = sub["Season"].values
print("\n   per-season flat-ROI reminder:", {int(s): round(pl_s[seas == s].mean() * 100, 1) for s in (2024, 2025, 2026)})

# ---- comparison chart ----
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN, ORANGE = "#2a78d6", "#008300", "#eb6834"

# x = bet index 0..n (curves have n+1 points incl. start)
x = np.arange(len(idx) + 1)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
fig.subplots_adjust(left=0.088, right=0.80, top=0.86, bottom=0.16)

ax.axhline(100, color=BASE, lw=1, ls=(0, (4, 3)), zorder=1)
series = [("M3 compound 5%/week", m3, GREEN, 2.4, "-"),
          ("M1 0.25-Kelly", m1, BLUE, 2.4, "-"),
          ("M2 flat 5% ($5)", m2, ORANGE, 2.2, "-")]
for name, cv, col, lw, ls in series:
    ax.plot(x, cv, color=col, lw=lw, ls=ls, zorder=5, solid_capstyle="round")
    ax.scatter([x[-1]], [cv[-1]], s=40, color=col, zorder=6, edgecolors=SURF, linewidths=1.5)

# end labels, de-collide by value
ends = sorted([("M3", m3[-1], GREEN), ("M1", m1[-1], BLUE), ("M2", m2[-1], ORANGE)],
              key=lambda t: t[1])
for lbl, val, col in ends:
    ax.annotate(f"${val:.0f}", (x[-1], val), textcoords="offset points",
                xytext=(9, 0), va="center", ha="left", fontsize=11.5, fontweight="bold", color=col)

ax.set_ylabel("Bankroll (USD, log scale)", fontsize=10.5, color=INK2)
ax.set_yscale("log")
ax.set_yticks([80, 100, 150, 200, 300, 400])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}"))
ax.set_xlabel("Bet number (chronological)", fontsize=10, color=INK2)
ax.set_xlim(-2, len(idx) + 9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(BASE)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

leg = ax.legend(handles=[
    Line2D([0], [0], color=BLUE, lw=2.4, label=f"M1  0.25-Kelly (fractional) — maxDD {maxdd(m1):.0f}%"),
    Line2D([0], [0], color=GREEN, lw=2.4, label=f"M3  compound 5%/week — maxDD {maxdd(m3):.0f}%"),
    Line2D([0], [0], color=ORANGE, lw=2.2, label=f"M2  flat 5% = $5 fixed — maxDD {maxdd(m2):.0f}%"),
], loc="upper left", frameon=False, fontsize=9, handlelength=2.0, bbox_to_anchor=(0.01, 0.99))
for t in leg.get_texts():
    t.set_color(INK2)

fig.suptitle("Staking schemes compared — same bets, same edge",
             x=0.088, y=0.955, ha="left", fontsize=14, fontweight="bold", color=INK)
ax.set_title("1xBet open · λ=0.75 · EV>0.20 · odds≤7 · n=100 bets, start $100 · CLV graded vs Pinnacle close",
             loc="left", fontsize=9.5, color=MUTED, pad=8)
fig.text(0.088, 0.028,
         "M2 (flat $5) risks least, ends lowest; M1/M3 compound gains but ride deeper drawdowns.\n"
         "One lucky ordering of a high-variance process — a variance illustration, not a forecast; "
         "judge the edge by CLV/gap.",
         ha="left", va="bottom", fontsize=7.4, color=MUTED)

out = os.path.join(HERE, "staking_compare.png")
fig.savefig(out, facecolor=SURF)
print("\nsaved ->", out)
