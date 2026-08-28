"""0.25-Kelly bankroll path for the recommended live config (backtest.md §13.4).

Config: 1xBet OPEN price · NegBinom + lam=0.75 market-anchored draw de-bias ·
max-EV (draws eligible) · EV threshold > 0.20 · pick-odds cap <= 7.0 · 0.25-Kelly,
bankroll starting at $100. A reference line drops the odds<=7 cap to show the cap
is a free variance reduction (it strips the long-shot tail that carries the least
edge; see §13.1). CLV is graded against Pinnacle's close, so this is a value proxy,
not realized 1xBet settlement.

Reproduce (repo root):  PYTHONPATH=src python backtest/plot_bankroll.py
Writes backtest/bankroll_recommended.png.
"""
from __future__ import annotations

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
LAM = 0.75          # market-anchored de-bias strength
EV_THR = 0.20       # selectivity — load-bearing (§13.1)
ODDS_CAP = 7.0      # drop the long-shot tail (§13.4)
KELLY = 0.25
BANK0 = 100.0

# design-system palette (dataviz skill reference instance)
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"


def picks(b: pd.DataFrame):
    O = b[["oH", "oD", "oA"]].values
    C = b[["cH", "cD", "cA"]].values
    P_raw = b[["pH", "pD", "pA"]].values
    novig_o = (1 / O) / (1 / O).sum(1, keepdims=True)
    novig_c = (1 / C) / (1 / C).sum(1, keepdims=True)
    pD = (1 - LAM) * P_raw[:, 1] + LAM * novig_o[:, 1]
    scale = (1 - pD) / (1 - P_raw[:, 1])
    P = np.column_stack([P_raw[:, 0] * scale, pD, P_raw[:, 2] * scale])
    ev = P * O - 1.0
    pick = ev.argmax(1)
    i = np.arange(len(b))
    pl = np.where(pick == b["res"].values, O[i, pick] - 1.0, -1.0)
    return dict(pick=pick, best_ev=ev.max(1), pl=pl,
                odds=O[i, pick], p=P[i, pick], novig_c=novig_c, novig_o=novig_o)


def kelly_path(b, g, mask):
    idx = np.where(mask)[0]
    bank = BANK0
    dates = [b["date"].iloc[idx[0]] - pd.Timedelta(days=3)]
    banks = [BANK0]
    for k in idx:
        o = g["odds"][k]
        f = min(max((g["p"][k] * o - 1) / (o - 1), 0.0) * KELLY, 0.5)
        bank += bank * f * g["pl"][k]
        dates.append(b["date"].iloc[k])
        banks.append(bank)
    return np.array(dates), np.array(banks)


def main():
    b = walk_forward(load())
    b["date"] = pd.to_datetime(b["date"])
    b = b.sort_values("date").reset_index(drop=True)
    g = picks(b)
    seasons = b["Season"].values

    m_cap = (g["best_ev"] > EV_THR) & (g["odds"] <= ODDS_CAP)
    m_nocap = g["best_ev"] > EV_THR
    d_cap, bank_cap = kelly_path(b, g, m_cap)
    d_nocap, bank_nocap = kelly_path(b, g, m_nocap)

    dd = 1 - bank_cap / np.maximum.accumulate(bank_cap)
    print(f"RECOMMENDED (odds<=7): n={int(m_cap.sum())}  final=${bank_cap[-1]:.0f}  "
          f"maxDD={dd.max()*100:.0f}%  flatROI={g['pl'][m_cap].mean()*100:+.1f}%")
    print(f"reference (no cap):    n={int(m_nocap.sum())}  final=${bank_nocap[-1]:.0f}")

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    fig.subplots_adjust(left=0.085, right=0.82, top=0.87, bottom=0.175)

    season_dates = {s: b["date"][seasons == s].min() for s in (2024, 2025, 2026)}
    xmax = d_cap[-1] + pd.Timedelta(days=20)
    bounds = [d_cap[0]] + [season_dates[s] for s in (2025, 2026)] + [xmax]
    for j, s in enumerate((2024, 2025, 2026)):
        ax.axvspan(bounds[j], bounds[j + 1], color=["#f4f4f1", "#ecebe6", "#f4f4f1"][j], zorder=0)
        mid = bounds[j] + (bounds[j + 1] - bounds[j]) / 2
        ax.text(mid, 0.035, str(s), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=MUTED)

    ax.axhline(BANK0, color=BASE, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(d_nocap, bank_nocap, color=MUTED, lw=1.6, ls=(0, (5, 2)), zorder=3, solid_capstyle="round")
    ax.plot(d_cap, bank_cap, color=BLUE, lw=2.4, zorder=5, solid_capstyle="round")
    ax.scatter([d_cap[-1]], [bank_cap[-1]], s=42, color=BLUE, zorder=6, edgecolors=SURF, linewidths=1.5)
    ax.annotate(f"${bank_cap[-1]:.0f}", (d_cap[-1], bank_cap[-1]), textcoords="offset points",
                xytext=(9, 0), va="center", ha="left", fontsize=12, fontweight="bold", color=BLUE)
    ax.annotate(f"${bank_nocap[-1]:.0f} (no cap)", (d_nocap[-1], bank_nocap[-1]),
                textcoords="offset points", xytext=(9, -15), va="center", ha="left", fontsize=9, color=MUTED)

    tri = int(np.argmax(dd))
    ax.annotate(f"max drawdown −{dd[tri]*100:.0f}%", (d_cap[tri], bank_cap[tri]),
                textcoords="offset points", xytext=(6, -30), ha="left", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.set_ylabel("Bankroll (USD, log scale)", fontsize=10.5, color=INK2)
    ax.set_yscale("log")
    ax.set_yticks([100, 150, 200, 300, 400, 600])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.set_xlim(d_cap[0] - pd.Timedelta(days=10), xmax + pd.Timedelta(days=95))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=2.4, label="Recommended: EV>0.20 + odds≤7"),
        Line2D([0], [0], color=MUTED, lw=1.6, ls=(0, (5, 2)), label="Reference: no odds cap"),
    ], loc="upper left", frameon=False, fontsize=9, handlelength=2.2, bbox_to_anchor=(0.01, 0.99))
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("0.25-Kelly bankroll — 1xBet open · λ=0.75 de-bias · 1X2",
                 x=0.085, y=0.965, ha="left", fontsize=14, fontweight="bold", color=INK)
    ax.set_title("Walk-forward, 3 backfilled seasons (2024–2026), n=100 bets, CLV graded vs Pinnacle close",
                 loc="left", fontsize=9.5, color=MUTED, pad=8)
    fig.text(0.085, 0.028,
             "Caveat: 2025 posted the strongest +CLV of the three seasons yet −9.8% flat per-bet ROI "
             "(compounded +18% here on ordering) — judge by CLV/gap, not one season's ROI. Past "
             "results ≠ live: graded vs Pinnacle's close, not real 1xBet settlement.",
             ha="left", va="bottom", fontsize=7.4, color=MUTED)

    out = os.path.join(HERE, "bankroll_recommended.png")
    fig.savefig(out, facecolor=SURF)
    print("saved ->", out)


if __name__ == "__main__":
    main()
