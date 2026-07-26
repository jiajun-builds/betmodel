"""
Export upcoming CSL fixtures as a **1xBet-open bet-signal board** (dashboard
v2.7). Each fixture carries de-biased model 1X2 probabilities, the 1xBet OPENING
1X2 price (the line we actually bet), model EV against that price, and a betting
signal flag per the backtest.md §13.4 recommended config.

Two books, two distinct roles (they are NOT the same book):

  * **Pinnacle open — the de-bias anchor** (never displayed). Pinnacle is the
    sharp reference; its opening no-vig draw is the target the draw shrinks
    toward. This keeps the model probability identical to the all-pairs
    prediction surface (one coherent prob per fixture).
  * **1xBet open — the bet price.** Cheaper book (~4.9% open overround vs
    Pinnacle's ~7.5%, backtest.md §13), so the same edge can clear the vig wall.
    EV, the displayed Open odds, and the signal are all on this line.

Draw de-bias is hybrid (AGENTS.md roadmap #10, validated in backtest.md §12):

  * Fixture WITH a captured PINNACLE opening 1X2 -> market-anchored shrink at
    ``DEBIAS_LAMBDA``: starting from the RAW (un-δ'd) model grid,
    ``p'_D = (1-λ)·p_D + λ·m_D`` where ``m_D`` is Pinnacle's no-vig opening draw
    probability; the freed mass is returned to H/A pro-rata. Anchoring on the
    raw grid (``predict_raw``) avoids stacking λ on top of the δ calibration.
  * Fixture WITHOUT a captured Pinnacle open -> the ``predict`` path, which since
    backtest.md §15.3 applies **no** draw correction (δ is retired,
    ``DRAW_DELTA_SHRINK = 0.0``) and therefore returns the raw grid.

At the shipped ``DEBIAS_LAMBDA = 1.0`` the anchored draw is exactly ``m_D``, so the
model contributes only the home/away split; see §15.4 for why 1.0 rather than the
measured argmax of 1.25.

The ``debias_method`` column records which path produced each row's
probabilities ("market_anchor" or "delta"). EV is computed against the 1xBet
opening price: ``onexbet_open_EV_k = p'_k * onexbet_open_odds_k - 1``.

Signal (backtest.md §13.4, §15.4): pick = argmax EV over ``SIGNAL_SIDES`` —
**{home, away} only**, draws cannot fire (``SIGNAL_ALLOW_DRAW = False``) — priced on
the 1xBet open; ``signal_state`` is "bet" when that pick's EV > ``SIGNAL_EV_MIN``
and its 1xBet odds <= ``SIGNAL_ODDS_CAP``, "odds_cap" when the EV clears but the
long-shot cap does not, "" otherwise.

Usage (仓库根目录，PYTHONPATH=src):
    python -m csl.odds.export_upcoming_market_comparison

Outputs:
    data/output_data/CHN_upcoming_market_comparison.csv
    data/dashboard/csv/upcoming_market_comparison.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from csl.models.dc import fit_dixon_coles_model_from_csv
from csl.odds.fetch_pinnacle_spreads import BOOKMAKER as ANCHOR_BOOKMAKER
from csl.odds.snapshot_store import HISTORY_CSV, load_history
from csl.paths import data_dashboard_csv_dir, data_output_dir, data_raw_dir

# The cheap book we bet into; its opening 1X2 is the displayed line, the EV
# basis, and the signal price (backtest.md §13). Distinct from ANCHOR_BOOKMAKER
# (Pinnacle), whose open only feeds the λ draw anchor and is never shown.
BET_BOOKMAKER = "onexbet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_XI = 0.001

# Market-anchored draw shrink weight. λ=1.0 means the draw probability IS the
# anchor book's no-vig draw — the model contributes nothing to it, and only splits
# home/away. backtest.md §15.4.
#
# Why exactly 1.0 and not the measured argmax (1.25, gap +3.58 vs +3.32): λ>1
# extrapolates PAST the market draw, which gives p_D a *negative* weight on the
# model's own draw estimate — and §15.3 showed that estimate's bias changes sign
# season to season (2023 +0.81pp, 2025 -4.37pp). Depending negatively on an
# unstable quantity is the fragility that retiring δ removed; at λ=1.0 the draw is
# independent of the model, so that instability cannot propagate. λ=1.0 captures
# ~93% of the available gap, and 1.25 was chosen by argmax on the evaluation set —
# structurally the same mistake that produced δ=1.258. Earning 1.25 requires a
# pre-registered forward test, not a re-read of the same table.
#
# NB this stays load-bearing even though draws are never bet (SIGNAL_ALLOW_DRAW):
# p_home/p_away are renormalized by (1-p_D_new)/(1-p_D), so λ moves their EV.
DEBIAS_LAMBDA = 1.0

# Betting signal thresholds (backtest.md §13.4 recommended config). A pick fires
# ("bet") only when its 1xBet-open EV clears SIGNAL_EV_MIN AND its price is within
# the long-shot cap; picks over the cap are flagged "odds_cap" (visible, not bet)
# because the odds>7 tail is the least-edge slice in the book (§13.4b).
SIGNAL_EV_MIN = 0.20
SIGNAL_ODDS_CAP = 7.0

# Draws are never bet. At DEBIAS_LAMBDA >= 1.0 the draw probability carries no model
# information — it is the anchor book's no-vig draw — so a draw signal would mean
# only "1xBet's draw price beats Pinnacle's implied fair draw", a cross-book price
# disagreement rather than a model view. Measured cost of excluding them: none.
# backtest.md §15.4, at λ=1.25/thr>0.20/cap<=7: with draws n=110 gap +3.58, without
# n=104 gap +3.60. The draw is still modelled and displayed; it just cannot fire.
SIGNAL_ALLOW_DRAW = False
SIGNAL_SIDES = ("home", "draw", "away") if SIGNAL_ALLOW_DRAW else ("home", "away")

# 1xBet opening-price columns joined from the capture history (snapshot_type=open,
# bookmaker=onexbet). These are the displayed line, the EV basis, and the signal
# price. Blank for fixtures whose 1xBet open has not been captured yet.
ONEXBET_OPEN_COLUMNS = [
    "onexbet_open_home_odds",
    "onexbet_open_draw_odds",
    "onexbet_open_away_odds",
    "onexbet_open_home_ev",
    "onexbet_open_draw_ev",
    "onexbet_open_away_ev",
    "onexbet_open_last_update",
]

# Pinnacle opening odds — the λ draw anchor only, never surfaced. Retained in the
# full archive CSV for reproducibility of debias_method.
PINNACLE_OPEN_COLUMNS = [
    "open_home_odds",
    "open_draw_odds",
    "open_away_odds",
    "open_last_update",
]

SIGNAL_COLUMNS = ["signal_pick", "signal_state"]

FULL_COLUMNS = [
    "fixture_id",
    "round",
    "match_date",
    "match_time",
    "kickoff_at",
    "home_team",
    "away_team",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "debias_method",
    # Pinnacle "Now" line — still captured (roadmap #3 close/CLV data) but not a
    # betting basis here; kept in the archive for reference.
    "home_odds",
    "draw_odds",
    "away_odds",
    "bookmaker",
    "market",
    "regions",
    "last_update",
    "fetched_at",
    *ONEXBET_OPEN_COLUMNS,
    *PINNACLE_OPEN_COLUMNS,
    *SIGNAL_COLUMNS,
]

# Dashboard contract: probabilities + 1xBet open line/EV + signal. No Pinnacle
# Now line, no Move — the board is the 1xBet-open signal surface only.
DASHBOARD_COLUMNS = [
    "fixture_id",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "debias_method",
    *ONEXBET_OPEN_COLUMNS,
    *SIGNAL_COLUMNS,
    "fetched_at",
]


@dataclass(frozen=True)
class ExportPaths:
    upcoming_csv: str = os.path.join(data_dashboard_csv_dir(), "upcoming_fixtures.csv")
    pinnacle_csv: str = os.path.join(data_raw_dir(), "CHN_pinnacle_spreads.csv")
    matches_csv: str = os.path.join(data_raw_dir(), "CHN_Super League.csv")
    history_csv: str = HISTORY_CSV
    full_out_csv: str = os.path.join(data_output_dir(), "CHN_upcoming_market_comparison.csv")
    dashboard_out_csv: str = os.path.join(data_dashboard_csv_dir(), "upcoming_market_comparison.csv")


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _read_csv_required(path: str, required: Iterable[str], label: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    _require_columns(df, required, label)
    return df


def load_upcoming(path: str) -> pd.DataFrame:
    df = _read_csv_required(
        path,
        ["fixture_id", "round", "match_date", "match_time", "kickoff_at", "home_team", "away_team"],
        "upcoming_fixtures.csv",
    ).copy()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    return df


def load_pinnacle(path: str) -> pd.DataFrame:
    df = _read_csv_required(
        path,
        [
            "event_id",
            "commence_time",
            "home_team",
            "away_team",
            "home_odds",
            "draw_odds",
            "away_odds",
            "bookmaker",
            "market",
            "regions",
            "last_update",
            "fetched_at",
        ],
        "CHN_pinnacle_spreads.csv",
    ).copy()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    numeric_cols = ["home_odds", "draw_odds", "away_odds"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    dupes = df.duplicated(subset=["home_team", "away_team"], keep=False)
    if dupes.any():
        duplicated_pairs = df.loc[dupes, ["home_team", "away_team"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Pinnacle odds table has duplicate home/away pairs: {duplicated_pairs}")

    return df


def _open_snapshot_columns(prefix: str) -> list[str]:
    return [
        "home_team",
        "away_team",
        f"{prefix}_home_odds",
        f"{prefix}_draw_odds",
        f"{prefix}_away_odds",
        f"{prefix}_last_update",
    ]


def load_open_snapshots(
    path: str, bookmaker: str = ANCHOR_BOOKMAKER, *, prefix: str = "open"
) -> pd.DataFrame:
    """One opening-price row per fixture from the capture history (may be empty).

    Reads the append-only capture history, keeps ``snapshot_type == "open"`` rows
    **for ``bookmaker`` only** — since roadmap #8 the history carries several books'
    prices at the same window, and the two roles here need different books: the λ
    anchor is Pinnacle's open (``bookmaker=ANCHOR_BOOKMAKER``, ``prefix="open"``),
    the bet price is 1xBet's open (``bookmaker=BET_BOOKMAKER``,
    ``prefix="onexbet_open"``). Since a line can in principle be captured more than
    once, takes the earliest ``fetched_at`` per fixture as the true opening prices.
    Returns a frame keyed by (home_team, away_team) with ``{prefix}_*`` 1X2 odds
    columns, or an empty (correctly-columned) frame when no opens exist yet.
    """
    columns = _open_snapshot_columns(prefix)
    hist = load_history(path)
    if not hist.empty:
        opens = hist[(hist["snapshot_type"] == "open") & (hist["bookmaker"] == bookmaker)]
    else:
        opens = hist
    if opens.empty:
        return pd.DataFrame(columns=columns)

    opens = opens.copy()
    opens["home_team"] = opens["home_team"].astype(str).str.strip()
    opens["away_team"] = opens["away_team"].astype(str).str.strip()
    for col in ("home_odds", "draw_odds", "away_odds"):
        opens[col] = pd.to_numeric(opens[col], errors="coerce")

    opens = opens.sort_values("fetched_at").drop_duplicates(
        subset=["home_team", "away_team"], keep="first"
    )
    opens = opens.rename(
        columns={
            "home_odds": f"{prefix}_home_odds",
            "draw_odds": f"{prefix}_draw_odds",
            "away_odds": f"{prefix}_away_odds",
            "last_update": f"{prefix}_last_update",
        }
    )
    return opens[columns]


def build_base_frame(
    upcoming: pd.DataFrame,
    pinnacle: pd.DataFrame,
    pinnacle_opens: pd.DataFrame,
    onexbet_opens: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    merged = upcoming.merge(
        pinnacle,
        on=["home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        pinnacle_opens, on=["home_team", "away_team"], how="left", validate="one_to_one"
    )
    merged = merged.merge(
        onexbet_opens, on=["home_team", "away_team"], how="left", validate="one_to_one"
    )
    # Keep a fixture if it has a Pinnacle Now line (event_id) OR a captured opening
    # line from either book. Open-only fixtures — captured before they appeared in a
    # Now-line fetch — are shown rather than dropped, so a freshly captured 1xBet open
    # surfaces immediately. Now-line fixtures come from the live feed and are inherently
    # upcoming; open-only fixtures are gated to a future kickoff so already-kicked-off
    # matches don't linger on the board until the daily upcoming CSV trims them.
    now = now or pd.Timestamp.now(tz="UTC")
    kickoff = pd.to_datetime(merged["kickoff_at"], utc=True, errors="coerce")
    is_upcoming = kickoff.isna() | (kickoff >= now)
    has_now = merged["event_id"].notna()
    has_pin_open = merged["open_home_odds"].notna()
    has_bet_open = merged["onexbet_open_home_odds"].notna()
    keep = has_now | ((has_pin_open | has_bet_open) & is_upcoming)
    return merged[keep].copy()


def _grid_1x2(pred) -> tuple[float, float, float]:
    """Aggregate a scoreline grid to normalized (home, draw, away) probabilities."""
    grid = np.asarray(pred.grid, dtype=float)
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n))
    v = np.array([grid[diff > 0].sum(), grid[diff == 0].sum(), grid[diff < 0].sum()])
    v = v / v.sum()
    return float(v[0]), float(v[1]), float(v[2])


def anchored_probs(
    raw: tuple[float, float, float],
    open_odds: tuple[float, float, float],
    lam: float,
) -> tuple[float, float, float]:
    """Market-anchored draw shrink (backtest.md §12) on raw model probabilities.

    ``m_D`` is the no-vig draw probability implied by the opening 1X2 prices;
    the draw moves λ of the way to it and the freed mass goes back to H/A
    pro-rata, preserving their relative strength.
    """
    p_h, p_d, p_a = raw
    inv = np.array([1.0 / o for o in open_odds])
    m_d = float(inv[1] / inv.sum())
    p_d_new = (1.0 - lam) * p_d + lam * m_d
    scale = (1.0 - p_d_new) / (1.0 - p_d)
    return p_h * scale, p_d_new, p_a * scale


def attach_model_probabilities(
    frame: pd.DataFrame, matches_csv: str, xi: float, lam: float = DEBIAS_LAMBDA
) -> pd.DataFrame:
    clf = fit_dixon_coles_model_from_csv(matches_csv, xi=xi)
    out = frame.copy()

    nan = float("nan")
    probs_h: list[float] = []
    probs_d: list[float] = []
    probs_a: list[float] = []
    methods: list[str] = []

    for row in out.itertuples(index=False):
        # Anchor is PINNACLE's open (prefix "open"), never 1xBet's — the sharp
        # reference draw is the de-bias target even though we bet the 1xBet line.
        open_odds = (row.open_home_odds, row.open_draw_odds, row.open_away_odds)
        try:
            if all(pd.notna(o) and float(o) > 1.0 for o in open_odds):
                raw = _grid_1x2(clf.predict_raw(row.home_team, row.away_team))
                p = anchored_probs(raw, tuple(float(o) for o in open_odds), lam)
                method = "market_anchor"
            else:
                p = _grid_1x2(clf.predict(row.home_team, row.away_team))
                method = "delta"
        except Exception as exc:
            raise ValueError(
                f"Model prediction failed for {row.home_team} vs {row.away_team}: {exc}"
            ) from exc
        probs_h.append(p[0])
        probs_d.append(p[1])
        probs_a.append(p[2])
        methods.append(method)

    out["home_win_prob"] = probs_h
    out["draw_prob"] = probs_d
    out["away_win_prob"] = probs_a
    out["debias_method"] = methods

    # EV per outcome against the 1xBet OPENING price (the line we bet). The
    # de-biased probabilities are the same ones anchored on Pinnacle's open above,
    # so EV isolates the model's disagreement with 1xBet's cheaper line.
    for side, prob_col in (("home", "home_win_prob"), ("draw", "draw_prob"), ("away", "away_win_prob")):
        bet_odds = pd.to_numeric(out[f"onexbet_open_{side}_odds"], errors="coerce")
        out[f"onexbet_open_{side}_ev"] = np.where(bet_odds.notna(), out[prob_col] * bet_odds - 1.0, nan)

    return out


def attach_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag the max-EV 1xBet-open pick per fixture (backtest.md §13.4).

    ``signal_pick`` is the outcome with the highest EV among the ``SIGNAL_SIDES``
    that have a 1xBet opening price — home/away only by default, see
    ``SIGNAL_ALLOW_DRAW``; ``signal_state`` is:

      * "bet"      — pick EV > SIGNAL_EV_MIN and pick odds <= SIGNAL_ODDS_CAP.
      * "odds_cap" — pick EV > SIGNAL_EV_MIN but odds over the long-shot cap
                     (surfaced, greyed, not a bet — §13.4b).
      * ""         — no pick clears the EV floor (or no 1xBet open captured).
    """
    out = frame.copy()
    picks: list[str] = []
    states: list[str] = []
    for row in out.itertuples(index=False):
        best_key = ""
        best_ev = float("-inf")
        for side in SIGNAL_SIDES:
            odds = getattr(row, f"onexbet_open_{side}_odds")
            ev = getattr(row, f"onexbet_open_{side}_ev")
            if pd.isna(odds) or pd.isna(ev):
                continue
            if float(ev) > best_ev:
                best_ev = float(ev)
                best_key = side
        if best_key and best_ev > SIGNAL_EV_MIN:
            pick_odds = float(getattr(row, f"onexbet_open_{best_key}_odds"))
            state = "bet" if pick_odds <= SIGNAL_ODDS_CAP else "odds_cap"
            picks.append(best_key)
            states.append(state)
        else:
            picks.append("")
            states.append("")
    out["signal_pick"] = picks
    out["signal_state"] = states
    return out


def validate_model_probabilities(frame: pd.DataFrame) -> None:
    prob_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
    if frame[prob_cols].isna().any(axis=1).any():
        bad = frame.loc[frame[prob_cols].isna().any(axis=1), ["fixture_id", "home_team", "away_team"]].to_dict("records")
        raise ValueError(f"Missing model 1X2 probabilities: {bad}")
    total = frame[prob_cols].sum(axis=1)
    if not ((total - 1.0).abs() <= 1e-6).all():
        bad = frame.loc[(total - 1.0).abs() > 1e-6, ["fixture_id", "home_team", "away_team"]].to_dict("records")
        raise ValueError(f"Model 1X2 probabilities do not sum to 1: {bad}")
    if ((frame[prob_cols] <= 0) | (frame[prob_cols] >= 1)).any(axis=1).any():
        bad = frame.loc[
            ((frame[prob_cols] <= 0) | (frame[prob_cols] >= 1)).any(axis=1),
            ["fixture_id", "home_team", "away_team"],
        ].to_dict("records")
        raise ValueError(f"Model 1X2 probabilities out of (0, 1): {bad}")

    # 1xBet-open EVs must exist exactly where a 1xBet opening price does.
    for side in ("home", "draw", "away"):
        odds_col = f"onexbet_open_{side}_odds"
        ev_col = f"onexbet_open_{side}_ev"
        has_odds = frame[odds_col].notna()
        missing = has_odds & frame[ev_col].isna()
        if missing.any():
            bad = frame.loc[missing, ["fixture_id", "home_team", "away_team"]].to_dict("records")
            raise ValueError(f"Missing EV in {ev_col} where {odds_col} present: {bad}")

    # A fired signal ("bet"/"odds_cap") must name an outcome that actually has a
    # 1xBet open price; an empty state must have an empty pick.
    for row in frame.itertuples(index=False):
        state = getattr(row, "signal_state")
        pick = getattr(row, "signal_pick")
        if state in ("bet", "odds_cap"):
            if pick not in ("home", "draw", "away") or pd.isna(getattr(row, f"onexbet_open_{pick}_odds")):
                raise ValueError(f"Signal {state} without a priced pick: {row.home_team} vs {row.away_team}")
        elif pick:
            raise ValueError(f"Empty signal_state with non-empty pick '{pick}': {row.home_team} vs {row.away_team}")


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(df))


def run(
    *,
    upcoming_csv: str,
    pinnacle_csv: str,
    matches_csv: str,
    history_csv: str,
    full_out_csv: str,
    dashboard_out_csv: str,
    xi: float,
    lam: float = DEBIAS_LAMBDA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    upcoming = load_upcoming(upcoming_csv)
    pinnacle = load_pinnacle(pinnacle_csv)

    pinnacle_opens = load_open_snapshots(history_csv, ANCHOR_BOOKMAKER, prefix="open")
    onexbet_opens = load_open_snapshots(history_csv, BET_BOOKMAKER, prefix="onexbet_open")
    base = build_base_frame(upcoming, pinnacle, pinnacle_opens, onexbet_opens)
    n_now = int(base["event_id"].notna().sum())
    n_pin_open = int(base["open_home_odds"].notna().sum())
    n_bet_open = int(base["onexbet_open_home_odds"].notna().sum())
    log.info(
        "Comparison fixtures: %d of %d upcoming (%d Now line, %d Pinnacle open anchor, %d 1xBet open bet-price)",
        len(base), len(upcoming), n_now, n_pin_open, n_bet_open,
    )

    if base.empty:
        log.info("No fixtures matched with Pinnacle odds; writing empty outputs and skipping model fit")
        full_df = pd.DataFrame(columns=FULL_COLUMNS)
        dashboard_df = pd.DataFrame(columns=DASHBOARD_COLUMNS)
        write_csv(full_df, full_out_csv)
        write_csv(dashboard_df, dashboard_out_csv)
        return full_df, dashboard_df

    enriched = attach_model_probabilities(base, matches_csv, xi, lam)
    enriched = attach_signals(enriched)
    validate_model_probabilities(enriched)
    log.info(
        "De-bias split: %d market_anchor (λ=%.2f), %d delta fallback",
        int((enriched["debias_method"] == "market_anchor").sum()),
        lam,
        int((enriched["debias_method"] == "delta").sum()),
    )
    log.info(
        "Signals: %d bet, %d odds_cap (EV>%.2f, cap odds<=%.0f, sides=%s)",
        int((enriched["signal_state"] == "bet").sum()),
        int((enriched["signal_state"] == "odds_cap").sum()),
        SIGNAL_EV_MIN, SIGNAL_ODDS_CAP, "/".join(SIGNAL_SIDES),
    )

    full_df = enriched[FULL_COLUMNS].copy()
    dashboard_df = enriched[DASHBOARD_COLUMNS].copy()

    write_csv(full_df, full_out_csv)
    write_csv(dashboard_df, dashboard_out_csv)
    return full_df, dashboard_df


def main() -> None:
    paths = ExportPaths()
    parser = argparse.ArgumentParser(
        description="Export upcoming CSL fixtures with de-biased model 1X2 probabilities and Pinnacle h2h comparison"
    )
    parser.add_argument("--upcoming", default=paths.upcoming_csv, help="Path to upcoming_fixtures.csv")
    parser.add_argument("--pinnacle", default=paths.pinnacle_csv, help="Path to CHN_pinnacle_spreads.csv")
    parser.add_argument("--matches", default=paths.matches_csv, help="Path to CHN_Super League.csv")
    parser.add_argument("--history", default=paths.history_csv, help="Path to CHN_pinnacle_spreads_history.csv")
    parser.add_argument("--out", default=paths.full_out_csv, help="Path to full comparison CSV output")
    parser.add_argument(
        "--dashboard-out",
        default=paths.dashboard_out_csv,
        help="Path to dashboard comparison CSV output",
    )
    parser.add_argument("--xi", type=float, default=MODEL_XI, help="Dixon-Coles time-decay factor")
    parser.add_argument("--lam", type=float, default=DEBIAS_LAMBDA,
                        help="Market-anchored draw shrink weight λ (default: 1.0 = take the "
                             "anchor book's no-vig draw outright, backtest.md §15.4)")
    args = parser.parse_args()

    try:
        run(
            upcoming_csv=args.upcoming,
            pinnacle_csv=args.pinnacle,
            matches_csv=args.matches,
            history_csv=args.history,
            full_out_csv=args.out,
            dashboard_out_csv=args.dashboard_out,
            xi=args.xi,
            lam=args.lam,
        )
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
