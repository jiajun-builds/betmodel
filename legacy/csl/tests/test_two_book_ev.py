"""Tests for the two-book best-price layer and the signal it drives.

Live data cannot exercise this yet — Duel had captured zero opening lines when it was
wired in, so every real row goes down the single-book path. These synthetic frames
cover the states that only appear once both books are pricing, which is exactly when a
bug would start moving money:

* **who wins a side** — including the exact tie, whose resolution must be deterministic
  because ``signal_book`` feeds the Telegram dedup key (a flapping tie-break would
  re-alert every run);
* **partial coverage** — one book priced, the other not. This is the permanent normal
  state, not an edge case: the books do not open together and Duel has no
  ``backfill_open`` safety net, so its coverage stays sparser indefinitely;
* **``signal_books`` vs ``signal_state``** — the two are computed from different prices
  (best vs per-book) and the deliberate rule is that logos appear only on a real bet.

``test_odds_cap_suppresses_an_executable_second_book`` pins the one case where adding a
book *removes* a bet. That is a real, accepted cost of the design, recorded so a future
reader meets it as an intended behaviour rather than a regression.

Runnable either way::

    pytest tests/test_two_book_ev.py
    python tests/test_two_book_ev.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.odds.books import DUEL_BOOK, ONEXBET_BOOK  # noqa: E402
from csl.odds.export_upcoming_market_comparison import (  # noqa: E402
    SIGNAL_EV_MIN,
    SIGNAL_ODDS_CAP,
    attach_best_prices,
    attach_signals,
    validate_model_probabilities,
)

NAN = float("nan")


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a post-``attach_model_probabilities`` frame from compact specs.

    Each row spec gives probabilities and, per book, a (home, draw, away) odds triple —
    ``None`` for "this book has not opened". EV is filled in the same way the export
    does it, so the fixtures cannot drift from the production formula.
    """
    out = []
    for i, r in enumerate(rows):
        rec = {
            "fixture_id": f"f{i}",
            "home_team": r.get("home_team", f"Home{i}"),
            "away_team": r.get("away_team", f"Away{i}"),
            "home_win_prob": r["p"][0],
            "draw_prob": r["p"][1],
            "away_win_prob": r["p"][2],
        }
        for book, key in ((ONEXBET_BOOK, "onexbet"), (DUEL_BOOK, "duel")):
            triple = r.get(key)
            for j, side in enumerate(("home", "draw", "away")):
                odds = NAN if triple is None else float(triple[j])
                rec[book.odds_col(side)] = odds
                rec[book.ev_col(side)] = NAN if triple is None else rec[
                    {"home": "home_win_prob", "draw": "draw_prob", "away": "away_win_prob"}[side]
                ] * odds - 1.0
        out.append(rec)
    return pd.DataFrame(out)


def _run(rows: list[dict]) -> pd.DataFrame:
    return attach_signals(attach_best_prices(_frame(rows)))


# ------------------------------------------------------------------ best price


def test_best_price_is_chosen_per_outcome_not_per_book() -> None:
    """The whole point: a book can win one side and lose another on the same fixture.

    These are the real 2026-08-08 Liaoning prices — Duel wins home and draw, 1xBet wins
    away. A "use the lower-overround book" rule would have taken 3.05 instead of 3.12.
    """
    df = _run([{"p": (0.45, 0.27, 0.28),
                "onexbet": (2.152, 3.72, 3.12),
                "duel": (2.22, 3.95, 3.05)}])
    r = df.iloc[0]
    assert r["best_open_home_book"] == "duel" and r["best_open_home_odds"] == 2.22
    assert r["best_open_draw_book"] == "duel" and r["best_open_draw_odds"] == 3.95
    assert r["best_open_away_book"] == "onexbet" and r["best_open_away_odds"] == 3.12


def test_exact_tie_goes_to_the_incumbent() -> None:
    """Deterministic tie-break. signal_book feeds the alert dedup key — it must not flap."""
    df = _run([{"p": (0.45, 0.27, 0.28),
                "onexbet": (2.50, 3.40, 3.00),
                "duel": (2.50, 3.40, 3.00)}])
    r = df.iloc[0]
    for side in ("home", "draw", "away"):
        assert r[f"best_open_{side}_book"] == "onexbet", side


def test_single_book_coverage_each_way() -> None:
    """One book priced is the normal state, not an error — and must reproduce it exactly."""
    only_x = _run([{"p": (0.45, 0.27, 0.28), "onexbet": (2.10, 3.50, 4.00), "duel": None}]).iloc[0]
    assert only_x["best_open_home_book"] == "onexbet" and only_x["best_open_home_odds"] == 2.10

    only_d = _run([{"p": (0.45, 0.27, 0.28), "onexbet": None, "duel": (2.10, 3.50, 4.00)}]).iloc[0]
    assert only_d["best_open_home_book"] == "duel" and only_d["best_open_home_odds"] == 2.10


def test_no_book_priced_yields_nan_and_empty_book() -> None:
    """Never -inf, never 0 — downstream notna() gates must behave as for an uncaptured row."""
    r = _run([{"p": (0.45, 0.27, 0.28), "onexbet": None, "duel": None}]).iloc[0]
    for side in ("home", "draw", "away"):
        assert pd.isna(r[f"best_open_{side}_odds"])
        assert pd.isna(r[f"best_open_{side}_ev"])
        assert r[f"best_open_{side}_book"] == ""
    assert r["signal_state"] == "" and r["signal_pick"] == ""
    assert r["signal_book"] == "" and r["signal_books"] == ""


def test_best_ev_matches_best_odds() -> None:
    df = _run([{"p": (0.50, 0.25, 0.25), "onexbet": (2.60, 3.72, 3.12), "duel": (2.80, 3.95, 3.05)}])
    r = df.iloc[0]
    assert abs(r["best_open_home_ev"] - (0.50 * 2.80 - 1.0)) < 1e-12


# --------------------------------------------------------------------- signals


def test_signal_prices_on_the_best_book() -> None:
    """Duel's better home price both raises EV and becomes the book to bet."""
    r = _run([{"p": (0.50, 0.25, 0.25),
               "onexbet": (2.30, 3.72, 3.12),   # EV 0.15 — below the floor alone
               "duel": (2.60, 3.50, 3.00)}]).iloc[0]  # EV 0.30 — fires
    assert r["signal_pick"] == "home"
    assert r["signal_state"] == "bet"
    assert r["signal_book"] == "duel"
    # Only Duel clears the EV floor, so only Duel's logo shows.
    assert r["signal_books"] == "duel"


def test_signal_books_lists_every_clearing_book_in_registry_order() -> None:
    r = _run([{"p": (0.50, 0.25, 0.25),
               "onexbet": (2.45, 3.72, 3.12),   # EV 0.225 — clears
               "duel": (2.60, 3.50, 3.00)}]).iloc[0]  # EV 0.30  — clears, and is best
    assert r["signal_state"] == "bet"
    assert r["signal_book"] == "duel"
    assert r["signal_books"] == "onexbet|duel", "must be BET_BOOKS order, not best-first"


def test_signal_books_excludes_a_book_below_the_ev_floor() -> None:
    r = _run([{"p": (0.50, 0.25, 0.25),
               "onexbet": (2.20, 3.72, 3.12),   # EV 0.10 — below floor
               "duel": (2.60, 3.50, 3.00)}]).iloc[0]
    assert r["signal_books"] == "duel"
    assert ONEXBET_BOOK.key not in r["signal_books"].split("|")


def test_odds_cap_suppresses_an_executable_second_book() -> None:
    """ACCEPTED COST, pinned deliberately: adding a book can remove a bet.

    Best price (Duel 7.5) clears EV but breaks the long-shot cap, so the row is
    "odds_cap" — greyed, not a bet. 1xBet's 6.5 is under the cap and clears EV, i.e. an
    executable bet exists, but no logo is shown: state is decided on the best price and
    a "do not bet" row must not display a bettable book. See attach_signals' docstring.
    """
    r = _run([{"p": (0.20, 0.25, 0.25),
               "onexbet": (6.5, 3.72, 3.12),    # EV 0.30, under cap
               "duel": (7.5, 3.50, 3.00)}]).iloc[0]  # EV 0.50, OVER cap
    assert r["best_open_home_book"] == "duel"
    assert r["signal_state"] == "odds_cap"
    assert r["signal_pick"] == "home"
    assert r["signal_book"] == "duel", "the capped price still names who quotes it"
    assert r["signal_books"] == "", "no logos on a row we are not betting"
    # Guard the premise, so this test fails loudly if the thresholds ever move.
    assert 6.5 <= SIGNAL_ODDS_CAP < 7.5 and 0.20 * 6.5 - 1.0 > SIGNAL_EV_MIN


def test_pick_can_flip_side_versus_one_book_alone() -> None:
    """Best-of-two argmaxes over a different price vector, so the picked side can change."""
    r = _run([{"p": (0.40, 0.25, 0.35),
               "onexbet": (3.20, 3.60, 2.90),   # home EV 0.28 > away EV 0.015
               "duel": (3.00, 3.60, 3.60)}]).iloc[0]  # away best 3.60 -> EV 0.26 > home 0.28? no
    # home best = 3.20 (1xBet) -> EV 0.28 ; away best = 3.60 (Duel) -> EV 0.26
    assert r["signal_pick"] == "home" and r["signal_book"] == "onexbet"
    # Nudging Duel's away price past the crossover flips both the side and the book.
    r2 = _run([{"p": (0.40, 0.25, 0.35),
                "onexbet": (3.20, 3.60, 2.90),
                "duel": (3.00, 3.60, 3.80)}]).iloc[0]  # away EV 0.33 > home 0.28
    assert r2["signal_pick"] == "away" and r2["signal_book"] == "duel"


# ------------------------------------------------------------------- validator


def test_validator_accepts_a_healthy_two_book_frame() -> None:
    df = _run([
        {"p": (0.50, 0.25, 0.25), "onexbet": (2.45, 3.72, 3.12), "duel": (2.60, 3.50, 3.00)},
        {"p": (0.45, 0.27, 0.28), "onexbet": (2.10, 3.50, 4.00), "duel": None},
        {"p": (0.40, 0.30, 0.30), "onexbet": None, "duel": None},
    ])
    validate_model_probabilities(df)  # must not raise


def test_validator_catches_a_corrupted_best_book() -> None:
    """The silent-corruption case: best_open_*_book naming a book that isn't the best."""
    df = _run([{"p": (0.50, 0.25, 0.25), "onexbet": (2.45, 3.72, 3.12), "duel": (2.60, 3.50, 3.00)}])
    df.loc[0, "best_open_home_book"] = "onexbet"   # Duel actually holds 2.60
    try:
        validate_model_probabilities(df)
    except ValueError as exc:
        assert "best price" in str(exc)
        return
    raise AssertionError("validator accepted a best_open_home_book that is not the best")


def test_validator_catches_stale_best_odds() -> None:
    df = _run([{"p": (0.50, 0.25, 0.25), "onexbet": (2.45, 3.72, 3.12), "duel": (2.60, 3.50, 3.00)}])
    df.loc[0, "best_open_home_odds"] = 2.45       # a book now beats the "best"
    try:
        validate_model_probabilities(df)
    except ValueError:
        return
    raise AssertionError("validator accepted a best price a book beats")


def _run_all() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
