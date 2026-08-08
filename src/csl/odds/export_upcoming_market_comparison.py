"""
Export upcoming CSL fixtures as an **opening-line bet-signal board** (dashboard
v2.8). Each fixture carries de-biased model 1X2 probabilities, every bet book's
OPENING 1X2 price, model EV against each, and a betting signal flag priced on the
**best** of them, per the backtest.md §13.4 recommended config.

Three books, two distinct roles (they are NOT interchangeable):

  * **Pinnacle open — the de-bias anchor** (never displayed, never bet). Pinnacle
    is the sharp reference; its opening no-vig draw is the target the draw shrinks
    toward. This keeps the model probability identical to the all-pairs
    prediction surface (one coherent prob per fixture).
  * **1xBet + Duel opens — the bet prices** (``books.BET_BOOKS``). Cheap books
    (~4.9% and ~3.0% open overround vs Pinnacle's ~7.5%, backtest.md §13), so the
    same edge can clear the vig wall. EV, the displayed Open odds and the signal
    all live here.

Best price, not "the cheaper book" (2026-08-08)
----------------------------------------------
EV is scored against ``max(odds)`` across ``BET_BOOKS`` per outcome, because that is
the price we would actually take. Choosing one book by its headline overround would
be wrong: measured the day Duel was wired in, its overround was ~2.4pp lower than
1xBet's on *every* fixture, yet 1xBet still held the better price on a third of
outcomes. Duel's advantage sits almost entirely in the **draw**, which
``SIGNAL_ALLOW_DRAW = False`` means we never bet; on home/away the two books split
evenly. Per-outcome best price was worth +2.06% mean on those sides.

⚠️ ``SIGNAL_EV_MIN`` was calibrated on 1xBet **alone** (§13.4), and a max over books
is upward-biased, so best-of-two fires strictly more signals at the same threshold.
The per-book EV columns are retained precisely so 1xBet-only can be reconstructed
and the threshold re-derived; see ``BET_OPEN_COLUMNS``.

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
probabilities ("market_anchor" or "delta"). EV is computed per book against that
book's opening price, ``{prefix}_EV_k = p'_k * {prefix}_odds_k - 1``, and again
against the best price as ``best_open_{k}_ev``.

Signal (backtest.md §13.4, §15.4): pick = argmax EV over ``SIGNAL_SIDES`` —
**{home, away} only**, draws cannot fire (``SIGNAL_ALLOW_DRAW = False``) — priced on
the **best** open; ``signal_state`` is "bet" when that pick's EV > ``SIGNAL_EV_MIN``
and its odds <= ``SIGNAL_ODDS_CAP``, "odds_cap" when the EV clears but the
long-shot cap does not, "" otherwise. ``signal_book`` names the book to bet;
``signal_books`` lists every book independently clearing both bars (the board's
logos) and is populated only on a "bet" row — see ``attach_signals``.

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
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from csl.models.dc import fit_dixon_coles_model_from_csv
from csl.odds.books import (  # re-exported: the registry's canonical home is books.py
    BET_BOOKS,
    BOOK_BY_KEY,
    DUEL_BOOK,
    ONEXBET_BOOK,
    SIDES,
    BetBook,
)
from csl.odds.fetch_pinnacle_spreads import BOOKMAKER as ANCHOR_BOOKMAKER
from csl.odds.snapshot_store import HISTORY_CSV, load_history
from csl.paths import data_dashboard_csv_dir, data_output_dir, data_raw_dir

# The books we bet into. Each one's OPENING 1X2 is a displayed line and an EV basis;
# the *best* price across them is the signal price (backtest.md §13, extended to two
# books 2026-08-08). All are distinct from ANCHOR_BOOKMAKER (Pinnacle), whose open
# only feeds the λ draw anchor and is never shown or bet.
#
# Back-compat alias: `BET_BOOKMAKER` was the single-book world's name for this. Kept
# because it is referenced in prose across AGENTS.md and oddsapi_io's docstring.
BET_BOOKMAKER = ONEXBET_BOOK.key

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

# Each outcome paired with the probability column it is scored against. One definition
# so the EV loop and the best-price layer can never disagree about that pairing.
_SIDE_PROB_COLS = (
    ("home", "home_win_prob"),
    ("draw", "draw_prob"),
    ("away", "away_win_prob"),
)

# Per-book opening-price columns joined from the capture history (snapshot_type=open,
# bookmaker=<book.key>): 3 odds, 3 EV and a last_update per book. Blank for fixtures
# whose open that book has not posted yet — which is normal and permanent for some,
# since the two books do not open together and Duel has no `backfill_open` safety net.
#
# These stay PER BOOK rather than collapsing into the best-price layer so 1xBet-only
# performance remains reconstructible by column selection alone. That is the
# instrument that makes deploying best-of-two auditable: SIGNAL_EV_MIN was calibrated
# on 1xBet in isolation (backtest.md §13.4) and taking a max over books is upward
# biased, so the threshold has to be re-derivable from what actually shipped.
BET_OPEN_COLUMNS = [col for book in BET_BOOKS for col in book.columns]

# Back-compat alias for the single-book world.
ONEXBET_OPEN_COLUMNS = ONEXBET_BOOK.columns

# The best price available across BET_BOOKS, per outcome — the EV basis and the
# signal price. `_book` names which book quotes it, which is what the board's logo
# and the Telegram alert's "where to bet" both key off. "" when nobody has priced
# that side.
BEST_OPEN_COLUMNS = (
    [f"best_open_{side}_odds" for side in SIDES]
    + [f"best_open_{side}_ev" for side in SIDES]
    + [f"best_open_{side}_book" for side in SIDES]
)

# Pinnacle opening odds — the λ draw anchor only, never surfaced. Retained in the
# full archive CSV for reproducibility of debias_method.
PINNACLE_OPEN_COLUMNS = [
    "open_home_odds",
    "open_draw_odds",
    "open_away_odds",
    "open_last_update",
]

# signal_book  — the single book whose price to actually take (== best_open_{pick}_book).
# signal_books — "|"-joined keys of EVERY book that independently clears both bars for
#                the picked side; this is what drives the logos on the board. Emitted
#                only when signal_state == "bet" (see attach_signals).
# Both are always STRINGS, never lists: a list reaching a JSON row value raises inside
# export_dashboard_json._clean_scalar's pd.isna.
SIGNAL_COLUMNS = ["signal_pick", "signal_state", "signal_book", "signal_books"]

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
    *BET_OPEN_COLUMNS,
    *BEST_OPEN_COLUMNS,
    *PINNACLE_OPEN_COLUMNS,
    *SIGNAL_COLUMNS,
]

# Dashboard contract: probabilities + every book's open line/EV + the best-price layer
# + signal. No Pinnacle Now line, no Move — the board is the opening-line signal
# surface only.
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
    *BET_OPEN_COLUMNS,
    *BEST_OPEN_COLUMNS,
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
    bet_opens: Mapping[str, pd.DataFrame],
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join fixtures to the Now line, the Pinnacle open anchor and every bet book's open.

    ``bet_opens`` maps ``book.key`` -> that book's opens frame (as produced by
    ``load_open_snapshots`` with the book's own ``prefix``).
    """
    merged = upcoming.merge(
        pinnacle,
        on=["home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        pinnacle_opens, on=["home_team", "away_team"], how="left", validate="one_to_one"
    )
    # Iterate BET_BOOKS, not bet_opens: column order stays deterministic regardless of
    # mapping order, and a book missing from the mapping raises here instead of quietly
    # vanishing from the output schema.
    for book in BET_BOOKS:
        merged = merged.merge(
            bet_opens[book.key], on=["home_team", "away_team"], how="left",
            validate="one_to_one",
        )

    # Force numeric dtypes on every book's odds. load_open_snapshots coerces only on its
    # non-empty path; the empty-frame path returns object dtype, which is exactly the
    # state of a book that has not captured an open yet (all of Duel, until its first
    # capture lands). Normalizing once here means the best-price scan, the validators and
    # the written CSV all see float64 instead of object-dtype comparisons.
    for book in BET_BOOKS:
        for side in SIDES:
            merged[book.odds_col(side)] = pd.to_numeric(
                merged[book.odds_col(side)], errors="coerce"
            )

    # Keep a fixture if it has a Pinnacle Now line (event_id) OR a captured opening line
    # from any bet book — open-only fixtures, captured before they appeared in a Now-line
    # fetch, are shown rather than dropped so a freshly captured open surfaces
    # immediately — AND its kickoff is still ahead.
    #
    # The `is_upcoming` gate applies to Now-line fixtures too (2026-08-02). It used to be
    # skipped for them, on the premise that "Now-line fixtures come from the live feed and
    # are inherently upcoming". That premise holds only at *fetch* time: the request sends
    # commenceTimeFrom=now, so the API can't return a kicked-off fixture — but
    # CHN_pinnacle_spreads.csv is a snapshot on disk, and the property decays with exactly
    # the refresh interval. `upcoming_fixtures.csv` is not kickoff-filtered either (it is
    # rebuilt daily), so this gate was the only thing trimming finished matches, and
    # Now-line fixtures bypassed it. That capped the staleness bug at ~3h under the old 3h
    # odds cron; when that cron dropped to 12h it would have become a half-day of
    # kicked-off matches lingering on the board. Gating on kickoff_at — the repo's own
    # schedule data — makes the board independent of the odds-fetch cadence.
    now = now or pd.Timestamp.now(tz="UTC")
    kickoff = pd.to_datetime(merged["kickoff_at"], utc=True, errors="coerce")
    # A fixture with no parseable kickoff is kept: absent data must not silently hide it.
    is_upcoming = kickoff.isna() | (kickoff >= now)
    has_now = merged["event_id"].notna()
    has_pin_open = merged["open_home_odds"].notna()
    # Any book having posted an open is enough. Checking the `home` column alone is
    # safe: a book's open is all-three-or-nothing (oddsapi_io.extract_ml returns None
    # unless home, draw AND away parse), so a priced side implies a priced fixture.
    has_bet_open = merged[[b.odds_col("home") for b in BET_BOOKS]].notna().any(axis=1)
    keep = (has_now | has_pin_open | has_bet_open) & is_upcoming
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

    # EV per outcome against each book's OPENING price. The de-biased probabilities are
    # the same ones anchored on Pinnacle's open above, so EV isolates the model's
    # disagreement with each cheap book's line. The signal uses the BEST of these
    # (attach_best_prices); the per-book values are retained for reconstruction.
    for book in BET_BOOKS:
        for side, prob_col in _SIDE_PROB_COLS:
            bet_odds = pd.to_numeric(out[book.odds_col(side)], errors="coerce")
            out[book.ev_col(side)] = np.where(
                bet_odds.notna(), out[prob_col] * bet_odds - 1.0, nan
            )

    return out


def attach_best_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Best available price per outcome across ``BET_BOOKS``, and who quotes it.

    This is the execution layer: for a given fixture and side we would take whichever
    book pays more, so that is the price EV must be computed against. It is *not* a
    "pick the cheaper book" rule — book-level overround and per-outcome price are
    different things. Measured 2026-08-08: Duel's overround was 2.4pp lower than
    1xBet's on every fixture, yet 1xBet held the better price on a third of outcomes
    and on the away side specifically 2 times in 3. Duel's cheapness lives almost
    entirely in the draw, which SIGNAL_ALLOW_DRAW excludes from betting.

    Tie-break: books are scanned in ``BET_BOOKS`` order with a strict ``>``, so an
    exact tie goes to the earlier book (1xBet). That determinism is load-bearing —
    ``signal_book`` derives from this and forms part of the Telegram dedup key, so a
    tie-break that could flip would re-alert every run.

    Sides nobody has priced get ``NaN`` odds/EV and ``""`` for the book: never
    ``-inf``, never ``0``, so downstream ``notna()`` gates behave exactly as they did
    for an uncaptured fixture in the single-book world.
    """
    out = frame.copy()
    for side, prob_col in _SIDE_PROB_COLS:
        odds_by_book = [
            (book, pd.to_numeric(out[book.odds_col(side)], errors="coerce"))
            for book in BET_BOOKS
        ]
        best_odds = pd.Series(np.nan, index=out.index, dtype=float)
        best_book = pd.Series("", index=out.index, dtype=object)
        for book, odds in odds_by_book:
            # Strict `>` keeps the incumbent on an exact tie; `fillna(-inf)` on the
            # running max makes the first priced book win against "nothing yet".
            wins = odds.notna() & (odds > best_odds.fillna(-np.inf))
            best_odds = best_odds.mask(wins, odds)
            best_book = best_book.mask(wins, book.key)
        out[f"best_open_{side}_odds"] = best_odds
        out[f"best_open_{side}_book"] = best_book
        out[f"best_open_{side}_ev"] = np.where(
            best_odds.notna(), out[prob_col] * best_odds - 1.0, np.nan
        )
    return out


def attach_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag the max-EV best-price pick per fixture (backtest.md §13.4).

    ``signal_pick`` is the outcome with the highest **best-price** EV among the
    ``SIGNAL_SIDES`` that any book has priced — home/away only by default, see
    ``SIGNAL_ALLOW_DRAW``; ``signal_state`` is:

      * "bet"      — pick EV > SIGNAL_EV_MIN and pick odds <= SIGNAL_ODDS_CAP.
      * "odds_cap" — pick EV > SIGNAL_EV_MIN but odds over the long-shot cap
                     (surfaced, greyed, not a bet — §13.4b).
      * ""         — no pick clears the EV floor (or no book has opened).

    ``signal_book`` is the book quoting that best price — the one to actually bet.
    ``signal_books`` is every book that *independently* clears both bars on the picked
    side, "|"-joined in ``BET_BOOKS`` order; it drives the logos on the board.

    Why ``signal_books`` is emitted ONLY when state == "bet"
    -------------------------------------------------------
    State is decided on the best price, ``signal_books`` on each book separately, so
    they can disagree: if the best price is over the odds cap while a second book's
    price is under it and still clears EV, the row is greyed "do not bet" — and
    showing a bettable book's logo on it would be incoherent. Emitting nothing there
    keeps one invariant true everywhere: state == "bet" iff signal_books is non-empty,
    and every displayed logo is a bet you should actually place.

    The cost, recorded deliberately: adding a second book can therefore *remove* a bet
    that 1xBet alone would have fired. This is the same "argmax EV first, then apply
    the cap" ordering the single-book version already used (§13.4b treats the odds>7
    tail as the least-edge slice), now applied to a cross-book price. Measured rarity:
    1 of 68 captured home/away opens ever exceeded odds 7, and the divergence needs a
    stricter conjunction than that.
    """
    out = frame.copy()
    picks: list[str] = []
    states: list[str] = []
    books: list[str] = []
    book_lists: list[str] = []
    for row in out.itertuples(index=False):
        best_key = ""
        best_ev = float("-inf")
        for side in SIGNAL_SIDES:
            odds = getattr(row, f"best_open_{side}_odds")
            ev = getattr(row, f"best_open_{side}_ev")
            if pd.isna(odds) or pd.isna(ev):
                continue
            if float(ev) > best_ev:
                best_ev = float(ev)
                best_key = side
        if best_key and best_ev > SIGNAL_EV_MIN:
            pick_odds = float(getattr(row, f"best_open_{best_key}_odds"))
            state = "bet" if pick_odds <= SIGNAL_ODDS_CAP else "odds_cap"
            picks.append(best_key)
            states.append(state)
            books.append(str(getattr(row, f"best_open_{best_key}_book") or ""))
            if state == "bet":
                clearing = [
                    book.key for book in BET_BOOKS
                    if not pd.isna(getattr(row, book.odds_col(best_key)))
                    and not pd.isna(getattr(row, book.ev_col(best_key)))
                    and float(getattr(row, book.ev_col(best_key))) > SIGNAL_EV_MIN
                    and float(getattr(row, book.odds_col(best_key))) <= SIGNAL_ODDS_CAP
                ]
                book_lists.append("|".join(clearing))
            else:
                book_lists.append("")
        else:
            picks.append("")
            states.append("")
            books.append("")
            book_lists.append("")
    out["signal_pick"] = picks
    out["signal_state"] = states
    out["signal_book"] = books
    out["signal_books"] = book_lists
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

    ident = ["fixture_id", "home_team", "away_team"]

    # Every book's EV must exist exactly where that book's opening price does.
    for book in BET_BOOKS:
        for side in SIDES:
            has_odds = frame[book.odds_col(side)].notna()
            missing = has_odds & frame[book.ev_col(side)].isna()
            if missing.any():
                bad = frame.loc[missing, ident].to_dict("records")
                raise ValueError(
                    f"Missing EV in {book.ev_col(side)} where "
                    f"{book.odds_col(side)} present: {bad}"
                )

    # Best-price layer coherence. Each of these catches a distinct silent corruption:
    # a book key that no longer matches the capture vocabulary, a max that is not a
    # max, or an EV computed against a different price than the one displayed.
    for side, prob_col in _SIDE_PROB_COLS:
        best_odds = frame[f"best_open_{side}_odds"]
        best_book = frame[f"best_open_{side}_book"].fillna("").astype(str)
        best_ev = frame[f"best_open_{side}_ev"]
        any_priced = frame[[b.odds_col(side) for b in BET_BOOKS]].notna().any(axis=1)

        if not best_odds.notna().equals(any_priced):
            bad = frame.loc[best_odds.notna() != any_priced, ident].to_dict("records")
            raise ValueError(f"best_open_{side}_odds disagrees with book coverage: {bad}")
        if not (best_book != "").equals(best_odds.notna()):
            bad = frame.loc[(best_book != "") != best_odds.notna(), ident].to_dict("records")
            raise ValueError(f"best_open_{side}_book set without a price (or vice versa): {bad}")

        named = best_book != ""
        unknown = named & ~best_book.isin([b.key for b in BET_BOOKS])
        if unknown.any():
            raise ValueError(
                f"best_open_{side}_book names an unknown book: "
                f"{frame.loc[unknown, ident].to_dict('records')}"
            )
        # The named book's own price must BE the best price, and no book may beat it.
        for book in BET_BOOKS:
            own = pd.to_numeric(frame[book.odds_col(side)], errors="coerce")
            is_named = named & (best_book == book.key)
            if is_named.any() and not np.allclose(
                own[is_named], best_odds[is_named], rtol=0, atol=1e-9
            ):
                bad = frame.loc[is_named, ident].to_dict("records")
                raise ValueError(f"best_open_{side}_book does not quote the best price: {bad}")
            beats = own.notna() & best_odds.notna() & (own > best_odds + 1e-9)
            if beats.any():
                bad = frame.loc[beats, ident].to_dict("records")
                raise ValueError(f"{book.odds_col(side)} beats best_open_{side}_odds: {bad}")

        priced = best_odds.notna()
        if (priced & best_ev.isna()).any():
            bad = frame.loc[priced & best_ev.isna(), ident].to_dict("records")
            raise ValueError(f"Missing best_open_{side}_ev where a best price exists: {bad}")
        expected = frame.loc[priced, prob_col] * best_odds[priced] - 1.0
        if priced.any() and not np.allclose(best_ev[priced], expected, rtol=0, atol=1e-9):
            bad = frame.loc[priced, ident].to_dict("records")
            raise ValueError(f"best_open_{side}_ev != p * best odds - 1: {bad}")
        # EV is monotone in odds for fixed p (and p is strictly in (0,1), asserted
        # above), so the best price must also carry the best EV.
        per_book_ev = frame[[b.ev_col(side) for b in BET_BOOKS]].max(axis=1)
        if priced.any() and not np.allclose(
            best_ev[priced], per_book_ev[priced], rtol=0, atol=1e-9
        ):
            bad = frame.loc[priced, ident].to_dict("records")
            raise ValueError(f"best_open_{side}_ev is not the max of per-book EV: {bad}")

    valid_keys = {b.key for b in BET_BOOKS}
    order = [b.key for b in BET_BOOKS]
    # A fired signal ("bet"/"odds_cap") must name an outcome that actually has a best
    # price; an empty state must have an empty pick, book and book list.
    for row in frame.itertuples(index=False):
        state = getattr(row, "signal_state")
        pick = getattr(row, "signal_pick")
        book = str(getattr(row, "signal_book") or "")
        listed = str(getattr(row, "signal_books") or "")
        where = f"{row.home_team} vs {row.away_team}"

        if state not in ("", "bet", "odds_cap"):
            raise ValueError(f"Unknown signal_state '{state}': {where}")

        if state in ("bet", "odds_cap"):
            if pick not in SIDES or pd.isna(getattr(row, f"best_open_{pick}_odds")):
                raise ValueError(f"Signal {state} without a priced pick: {where}")
            if book != str(getattr(row, f"best_open_{pick}_book") or ""):
                raise ValueError(f"signal_book is not the best-price book: {where}")
            if book not in valid_keys:
                raise ValueError(f"signal_book '{book}' is not a known book: {where}")
        elif pick or book or listed:
            raise ValueError(
                f"Empty signal_state with non-empty pick/book/books: {where}"
            )

        # signal_books is non-empty exactly when state == "bet" (see attach_signals).
        if state == "odds_cap" and listed:
            raise ValueError(f"signal_books set on an odds_cap row: {where}")
        if state == "bet":
            if not listed:
                raise ValueError(f"signal_state 'bet' with no clearing book: {where}")
            keys = listed.split("|")
            if len(set(keys)) != len(keys) or not set(keys) <= valid_keys:
                raise ValueError(f"signal_books malformed: '{listed}' — {where}")
            if keys != [k for k in order if k in set(keys)]:
                raise ValueError(f"signal_books not in BET_BOOKS order: '{listed}' — {where}")
            if book not in keys:
                raise ValueError(f"signal_book '{book}' missing from signal_books: {where}")
            for key in keys:
                b = BOOK_BY_KEY[key]
                if not (float(getattr(row, b.ev_col(pick))) > SIGNAL_EV_MIN
                        and float(getattr(row, b.odds_col(pick))) <= SIGNAL_ODDS_CAP):
                    raise ValueError(f"signal_books lists '{key}' which clears neither bar: {where}")


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
    bet_opens = {
        book.key: load_open_snapshots(history_csv, book.key, prefix=book.prefix)
        for book in BET_BOOKS
    }
    base = build_base_frame(upcoming, pinnacle, pinnacle_opens, bet_opens)
    n_now = int(base["event_id"].notna().sum())
    n_pin_open = int(base["open_home_odds"].notna().sum())
    log.info(
        "Comparison fixtures: %d of %d upcoming (%d Now line, %d Pinnacle open anchor)",
        len(base), len(upcoming), n_now, n_pin_open,
    )
    # Per-book counts, warning on zero. A book contributing nothing is either genuinely
    # un-opened (normal, and expected for Duel until its first capture lands) or a
    # `key`/`stored_key` mismatch against the capture vocabulary — which produces an
    # all-NaN column and raises nothing anywhere else. Making it a warning is what turns
    # that silent failure into something visible in the workflow log.
    for book in BET_BOOKS:
        n = int(base[book.odds_col("home")].notna().sum()) if not base.empty else 0
        (log.info if n else log.warning)(
            "%s opening lines joined: %d", book.label, n
        )

    if base.empty:
        log.info("No fixtures matched with Pinnacle odds; writing empty outputs and skipping model fit")
        full_df = pd.DataFrame(columns=FULL_COLUMNS)
        dashboard_df = pd.DataFrame(columns=DASHBOARD_COLUMNS)
        write_csv(full_df, full_out_csv)
        write_csv(dashboard_df, dashboard_out_csv)
        return full_df, dashboard_df

    enriched = attach_model_probabilities(base, matches_csv, xi, lam)
    enriched = attach_best_prices(enriched)
    enriched = attach_signals(enriched)
    validate_model_probabilities(enriched)
    log.info(
        "De-bias split: %d market_anchor (λ=%.2f), %d delta fallback",
        int((enriched["debias_method"] == "market_anchor").sum()),
        lam,
        int((enriched["debias_method"] == "delta").sum()),
    )
    log.info(
        "Signals: %d bet, %d odds_cap (EV>%.2f, cap odds<=%.0f, sides=%s, best of %s)",
        int((enriched["signal_state"] == "bet").sum()),
        int((enriched["signal_state"] == "odds_cap").sum()),
        SIGNAL_EV_MIN, SIGNAL_ODDS_CAP, "/".join(SIGNAL_SIDES),
        "/".join(b.label for b in BET_BOOKS),
    )
    fired = enriched[enriched["signal_state"] == "bet"]
    if not fired.empty:
        log.info(
            "Bet price by book: %s",
            ", ".join(f"{BOOK_BY_KEY[k].label} {v}"
                      for k, v in fired["signal_book"].value_counts().items()),
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
