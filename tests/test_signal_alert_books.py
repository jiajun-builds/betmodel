"""Tests for the two-book Telegram alert: dedup migration and message content.

The dedup baseline is not a state file — it is ``git show HEAD:<comparison csv>``, one
snapshot deep. Adding ``signal_book`` to the dedup key therefore has a specific hazard:
the committed baseline was written by the single-book exporter and has no such column,
so a naive comparison reads "" != "onexbet" for every currently-firing signal and
re-sends all of them. ``test_pre_migration_baseline_does_not_realert`` is the guard.

The mirror-image requirement is that a *genuine* book change still alerts: opening lines
are immutable once banked, so ``signal_book`` moves exactly once — when the second book
opens at a better price — and that is real new information the user should act on.

Runnable either way::

    pytest tests/test_signal_alert_books.py
    python tests/test_signal_alert_books.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.notify.signal_alert import format_message, new_signals  # noqa: E402


def _row(**kw):
    """A firing bet row with the post-migration columns; override anything via kwargs."""
    row = {
        "fixture_id": "f1",
        "home_team": "Henan Songshan Longmen",
        "away_team": "Qingdao West Coast",
        "kickoff_at": "2026-08-09T12:00:00Z",
        "match_time": "12:00",
        "home_win_prob": "0.565",
        "draw_prob": "0.218",
        "away_win_prob": "0.218",
        "signal_state": "bet",
        "signal_pick": "away",
        "signal_book": "onexbet",
        "signal_books": "onexbet",
        "onexbet_open_away_odds": "5.58",
        "onexbet_open_away_ev": "0.2138",
        "duel_open_away_odds": "",
        "duel_open_away_ev": "",
        "best_open_away_odds": "5.58",
        "best_open_away_ev": "0.2138",
        "best_open_away_book": "onexbet",
    }
    row.update(kw)
    return row


def _legacy_row(**kw):
    """A baseline row as the SINGLE-BOOK exporter wrote it — no signal_book column."""
    row = _row(**kw)
    for k in ("signal_book", "signal_books", "duel_open_away_odds", "duel_open_away_ev",
              "best_open_away_odds", "best_open_away_ev", "best_open_away_book"):
        row.pop(k, None)
    return row


# ------------------------------------------------------------------ dedup


def test_pre_migration_baseline_does_not_realert() -> None:
    """THE migration guard: a baseline without signal_book must swallow the book compare.

    Without the wildcard this re-sends every firing signal the first time the two-book
    exporter runs — silently, and to the user's phone.
    """
    assert new_signals([_row()], [_legacy_row()]) == []


def test_same_book_is_not_realerted() -> None:
    assert new_signals([_row()], [_row()]) == []


def test_a_better_price_at_the_other_book_alerts_again() -> None:
    """Duel opening later at a better price is new information, not a duplicate."""
    now = _row(signal_book="duel", signal_books="onexbet|duel",
               duel_open_away_odds="6.10", duel_open_away_ev="0.3298",
               best_open_away_odds="6.10", best_open_away_ev="0.3298",
               best_open_away_book="duel")
    fresh = new_signals([now], [_row()])
    assert len(fresh) == 1 and fresh[0]["signal_book"] == "duel"


def test_a_brand_new_fixture_alerts() -> None:
    fresh = new_signals([_row(fixture_id="f2")], [_row()])
    assert len(fresh) == 1 and fresh[0]["fixture_id"] == "f2"


def test_a_non_bet_row_never_alerts() -> None:
    assert new_signals([_row(signal_state="odds_cap")], []) == []
    assert new_signals([_row(signal_state="")], []) == []


# ---------------------------------------------------------------- message


def test_message_names_the_book_and_its_price() -> None:
    """The price and the book must travel together — the right side at the wrong book
    is a losing bet."""
    msg = format_message(_row(signal_book="duel", signal_books="duel",
                              duel_open_away_odds="6.10", duel_open_away_ev="0.3298",
                              best_open_away_odds="6.10", best_open_away_ev="0.3298",
                              best_open_away_book="duel"))
    assert "Duel 开盘价" in msg and "6.10" in msg
    assert "duel.com" in msg, "the link must point at the book being recommended"
    assert "1xBet" not in msg


def test_message_lists_the_alternate_book_with_its_own_price() -> None:
    msg = format_message(_row(signal_book="duel", signal_books="onexbet|duel",
                              duel_open_away_odds="6.10", duel_open_away_ev="0.3298",
                              best_open_away_odds="6.10", best_open_away_ev="0.3298",
                              best_open_away_book="duel"))
    assert "备选" in msg
    assert "1xBet 5.58" in msg, "the fallback needs ITS price, not just a name"


def test_message_omits_alternates_when_only_one_book_clears() -> None:
    assert "备选" not in format_message(_row())


def test_message_labels_1_over_p_as_fair_odds_not_a_betting_floor() -> None:
    """1/p is the zero-EV fair price, and the label must not imply it is a bet-down-to line.

    The signal fired at EV > 0.20, so the tradeable price is ~1.20/p — well above 1/p.
    The old "底线赔率 (≥ 才下注)" wording invited betting the band between the two, which
    the backtest never validated and which §11.7's vig wall says loses.
    """
    msg = format_message(_row())          # p(away) = 0.218 -> fair odds 4.59
    assert "Fair odds" in msg and "4.59" in msg
    assert "底线" not in msg and "才下注" not in msg


def test_message_uses_the_best_price_not_a_single_book() -> None:
    msg = format_message(_row(signal_book="duel", signal_books="onexbet|duel",
                              onexbet_open_away_odds="5.58",
                              duel_open_away_odds="6.10",
                              best_open_away_odds="6.10", best_open_away_ev="0.3298",
                              best_open_away_book="duel"))
    assert "开盘价: <b>6.10</b>" in msg


def test_message_falls_back_to_a_pre_migration_row() -> None:
    """An old CSV has no best_* columns; the notifier must still produce a usable alert."""
    msg = format_message(_legacy_row())
    assert "5.58" in msg and "BET 信号" in msg


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
