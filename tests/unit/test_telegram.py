"""Signal alerts.

Two failure modes matter and both are silent. Alerting twice trains the reader to
ignore the channel; failing to alert defeats the point of having one. The baseline
is a git object rather than a state file, so a path change breaks it without an
error, which is why the mechanism gets its own tests.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from betmodel import display
from betmodel.config import load_league
from betmodel.notify import telegram


def _signal(fixture="F1", side="home", book="duel", odds=3.8, ev=0.21, fires=True):
    return {
        "fixture_id": fixture,
        "kickoff_utc": "2026-08-29T01:00:00Z",
        "home_team": "Necaxa", "away_team": "Cruz Azul", "round": "6",
        "model": {"home": 0.32, "draw": 0.24, "away": 0.44, "method": "raw"},
        "quotes": [{"book": book, "side": side, "odds": odds, "ev": ev,
                    "captured_at": None, "last_update": None, "proof": "observed"}],
        "best": {side: {"book": book, "odds": odds, "ev": ev}},
        "judged": {"side": side, "book": book, "odds": odds, "ev": ev,
                   "proof": "observed", "captured_at": None},
        "bet": ({"side": side, "book": book, "odds": odds, "ev": ev,
                 "books": [book]} if fires else None),
        "state": "bet" if fires else "",
    }


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #

def test_an_already_alerted_signal_is_not_repeated():
    current = [_signal()]
    assert telegram.new_signals(current, [_signal()]) == []


def test_a_new_fixture_is_alerted():
    assert len(telegram.new_signals([_signal("F2")], [_signal("F1")])) == 1


def test_the_same_side_at_a_better_book_is_alerted_again():
    """The price is the best across books and the answer is not always the same
    one. A second book opening better is genuinely new information."""
    fresh = telegram.new_signals([_signal(book="betano")], [_signal(book="duel")])
    assert len(fresh) == 1


def test_a_baseline_without_a_book_swallows_the_comparison():
    """Otherwise the first run after a key change re-alerts every live signal.
    It self-heals after one committed run."""
    old = _signal()
    old["bet"]["book"] = ""
    assert telegram.new_signals([_signal(book="betano")], [old]) == []


def test_a_row_that_does_not_fire_is_never_alerted():
    assert telegram.new_signals([_signal(fires=False)], []) == []


# --------------------------------------------------------------------------- #
# the git baseline
# --------------------------------------------------------------------------- #

def test_the_baseline_is_read_from_the_last_commit():
    """The mechanism itself: it reads a git object, so a moved path breaks it
    silently. This asserts it still resolves against the real repository."""
    path = "public/csl/signals.json"
    committed = telegram.previous_signals(path)
    assert committed is not None, (
        "no committed baseline for the published signals file; either it is not "
        "tracked or the path moved"
    )
    assert isinstance(committed, list)


def test_no_baseline_means_send_nothing(tmp_path):
    """A first run must not blast every signal that happens to be live."""
    outside = tmp_path / "signals.json"
    outside.write_text(json.dumps({"schema": 1, "signals": [_signal()]}))
    assert telegram.previous_signals(str(outside)) is None


def test_an_unreadable_baseline_is_treated_as_no_baseline(monkeypatch):
    def broken(*args, **kwargs):
        class Result:
            stdout = "not json at all"
        return Result()

    monkeypatch.setattr(subprocess, "run", broken)
    assert telegram.previous_signals("public/csl/signals.json") is None


# --------------------------------------------------------------------------- #
# fail-open
# --------------------------------------------------------------------------- #

def test_a_missing_token_does_not_raise(monkeypatch, tmp_path, caplog):
    """The notifier must never fail a publish."""
    monkeypatch.delenv(telegram.TOKEN_ENV, raising=False)
    monkeypatch.delenv(telegram.CHAT_ENV, raising=False)
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({"schema": 1, "signals": [_signal()]}))
    monkeypatch.setattr(telegram, "previous_signals", lambda p: [])
    assert telegram.notify("csl", load_league("csl"), signals_path=str(path)) == 0


def test_an_unreachable_telegram_does_not_raise(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.RequestException("no route to host")

    monkeypatch.setattr(requests, "post", boom)
    assert telegram.send("t", "c", "hello") is False


def test_a_league_with_notifications_off_sends_nothing(monkeypatch):
    """Constructed rather than read from a shipped league, so flipping a league's
    setting cannot make this test lie about the behaviour it guards."""
    from dataclasses import replace

    from betmodel.config.schema import NotifyConfig

    config = replace(load_league("ligamx"), notify=NotifyConfig(telegram=False))
    monkeypatch.setattr(
        telegram, "previous_signals", lambda p: pytest.fail("should not read a baseline")
    )
    assert telegram.notify("ligamx", config) == 0


def test_both_shipped_leagues_currently_alert():
    """Recorded deliberately: the league that fires had alerts off until now."""
    assert load_league("csl").notify.telegram is True
    assert load_league("ligamx").notify.telegram is True


# --------------------------------------------------------------------------- #
# the message
# --------------------------------------------------------------------------- #

def test_the_price_and_the_book_appear_together():
    """Separating them lets the reader take the right side at the wrong price."""
    message = telegram.format_message(load_league("ligamx"), _signal())
    assert "Duel" in message and "3.80" in message


def test_every_league_shows_its_kickoff_in_the_same_timezone(monkeypatch):
    """One inbox, one timezone.

    Showing each league in its own zone answers a question nobody asked. This
    kickoff is 19:00 in Mexico City, which sounds like an evening; the reader is
    in London, where it is two in the morning. Only the second is actionable.
    """
    monkeypatch.setenv(display.ENV, "Europe/London")
    mex = telegram.format_message(load_league("ligamx"), _signal())
    assert "08-29 02:00" in mex
    assert "19:00" not in mex

    csl = telegram.format_message(load_league("csl"), _signal(book="onexbet"))
    # Same zone label in both, whatever the league.
    assert display.label() in mex and display.label() in csl


def test_the_display_timezone_follows_the_reader_not_the_league(monkeypatch):
    monkeypatch.setenv(display.ENV, "Asia/Shanghai")
    message = telegram.format_message(load_league("ligamx"), _signal())
    assert "08-29 09:00" in message, "01:00Z is 09:00 in Shanghai"


def test_fair_odds_are_labelled_as_a_reference_not_a_floor():
    """The signal required a materially higher bar than EV above zero, and
    calling this a floor implies the band between them is bettable."""
    message = telegram.format_message(load_league("ligamx"), _signal())
    assert "Fair odds" in message


def test_alternate_books_are_listed_with_their_own_prices():
    signal = _signal()
    signal["bet"]["books"] = ["duel", "betano"]
    signal["quotes"].append({"book": "betano", "side": "home", "odds": 3.5,
                             "ev": 0.12, "captured_at": None, "last_update": None,
                             "proof": "observed"})
    message = telegram.format_message(load_league("ligamx"), signal)
    assert "备选" in message and "3.50" in message


# --------------------------------------------------------------------------- #
# the anchor gate
# --------------------------------------------------------------------------- #

def _unanchored(fixture_id="MEX-1", odds=6.65, ev=0.494):
    return {
        "fixture_id": fixture_id,
        "kickoff_utc": "2026-08-29T01:00:00Z",
        "home_team": "Dalian Yingbo", "away_team": "Qingdao Hainiu",
        "model": {"home": 0.55, "draw": 0.23, "away": 0.22, "method": "raw"},
        "quotes": [{"book": "duel", "side": "away", "odds": odds, "ev": ev,
                    "captured_at": "2026-08-28T12:00:00Z", "last_update": None,
                    "proof": "window"}],
        "best": {"away": {"book": "duel", "odds": odds, "ev": ev}},
        "judged": {"side": "away", "book": "duel", "odds": odds, "ev": ev,
                   "proof": "window", "captured_at": "2026-08-28T12:00:00Z"},
        "bet": None,
        "state": "unanchored",
    }


def test_an_uncalibrated_edge_is_never_pushed(monkeypatch, tmp_path):
    """Silence is the correct output, and a caveat is not a cheaper alternative.

    The backtest that validates this strategy ran on true opening prices, so a row
    calibrated on anything else is an untested variant rather than a weaker
    version of the same thing. A message saying so would still be a message about
    a fake signal, and the cost of that is a decision made with more noise in it.
    """
    payload = {"schema": 1, "league": "csl", "generated_at": "2026-08-29T00:00:00Z",
               "signals": [_unanchored()]}
    path = tmp_path / "signals.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(telegram, "previous_signals", lambda _p: [])

    assert telegram.notify("csl", load_league("csl"), signals_path=str(path)) == 0


def test_an_unanchored_row_is_never_treated_as_a_bet():
    """`bet` is null on these, which is what keeps the two paths from crossing."""
    assert telegram.new_signals([_unanchored()], []) == []


# --------------------------------------------------------------------------- #
# withdrawal
# --------------------------------------------------------------------------- #

def _no_longer_firing(ev=0.15, odds=5.6, state=""):
    """The same fixture as `_signal()`, published again without a bet."""
    row = _signal()
    row["bet"] = None
    row["state"] = state
    row["best"] = {"home": {"book": "duel", "odds": odds, "ev": ev}}
    return row


def test_a_signal_that_stops_firing_is_withdrawn():
    """It reached you when it fired, so it has to reach you when it stops."""
    out = telegram.withdrawn_signals([_no_longer_firing()], [_signal()])
    assert len(out) == 1


def test_a_signal_still_firing_is_not_withdrawn():
    assert telegram.withdrawn_signals([_signal()], [_signal()]) == []


def test_a_fixture_that_has_left_the_file_is_not_a_withdrawal():
    """It kicked off. That is not news, and announcing it would train you to
    ignore the channel."""
    assert telegram.withdrawn_signals([], [_signal()]) == []


def test_a_withdrawal_is_announced_once_and_not_again():
    """Self-deduplicating: once published, the next run's baseline has no bet."""
    gone = _no_longer_firing()
    assert telegram.withdrawn_signals([gone], [gone]) == []


def test_the_reason_separates_a_moved_line_from_a_changed_model():
    """The two call for different reactions, so the message must tell them apart."""
    config = load_league("csl")
    before = _signal()

    same_price = telegram.withdrawal_reason(
        config, before, _no_longer_firing(odds=before["bet"]["odds"]))
    assert "模型更新" in same_price and "赔率未动" in same_price

    moved = telegram.withdrawal_reason(config, before, _no_longer_firing(odds=4.10))
    assert "赔率从" in moved and "4.10" in moved


def test_losing_the_anchor_is_named_as_the_reason():
    reason = telegram.withdrawal_reason(
        load_league("csl"), _signal(), _no_longer_firing(state="unanchored"))
    assert "校准" in reason


def test_the_withdrawal_message_cannot_be_read_as_a_new_bet():
    message = telegram.format_withdrawal_message(
        load_league("csl"), _signal(), _no_longer_firing())
    assert "信号撤回" in message
    assert "BET 信号" not in message
    assert "原方向" in message and "原 EV" in message


def test_a_pick_that_switches_sides_withdraws_the_old_one_and_fires_the_new():
    """Both messages are correct: one bet is off and a different one is on."""
    before = _signal()                      # home
    after = _signal(book="duel")
    after["bet"]["side"] = "away"
    after["best"] = {"home": {"book": "duel", "odds": 5.6, "ev": 0.05}}

    assert len(telegram.withdrawn_signals([after], [before])) == 1
    assert len(telegram.new_signals([after], [before])) == 1


def test_a_thin_evidence_withdrawal_says_so_rather_than_blaming_the_edge():
    """The edge is intact; the club is the problem.

    The fall-through said "EV +0.213, below the threshold +0.100", which is not
    merely unhelpful -- it is arithmetically false, and it points the reader at
    the wrong thing to check.
    """
    after = _no_longer_firing(ev=0.213, state="thin_evidence")
    reason = telegram.withdrawal_reason(load_league("ligamx"), _signal(), after)
    assert "样本不足" in reason
    assert "低于阈值" not in reason


def test_an_edge_that_survives_is_not_reported_as_a_threshold_miss():
    after = _no_longer_firing(ev=0.40)
    reason = telegram.withdrawal_reason(load_league("csl"), _signal(), after)
    assert "低于阈值" not in reason and "+0.400" in reason


def test_an_edge_that_really_fell_still_names_the_threshold():
    after = _no_longer_firing(ev=0.05)
    reason = telegram.withdrawal_reason(load_league("csl"), _signal(), after)
    assert "低于阈值" in reason
