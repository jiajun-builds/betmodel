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


def test_the_kickoff_is_shown_in_league_local_time():
    """A display concern, and the one place local time belongs."""
    message = telegram.format_message(load_league("ligamx"), _signal())
    assert "08-28 19:00" in message, "01:00Z is 19:00 the previous day in Mexico City"


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
