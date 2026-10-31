"""Alerts survive a renamed fixture and a failed send.

Two ways the alert channel went wrong without anything failing:

* Fixing the round-0 ids (2026-10-31) renamed every live signal. Compared by
  exact ``fixture_id`` against a baseline holding the old names, the first run
  after the fix would have re-sent every live bet and retracted each one too.
* A send Telegram refused was gone for good. The workflow commits the published
  tree regardless, the next run reads that commit as its baseline, and the bet
  looks already alerted.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from betmodel.config import load_league
from betmodel.notify import telegram
from betmodel.signals.engine import make_fixture_id

CSL = load_league("csl")


def _row(home, away, round_label, *, side="home", book="duel", fires=True,
         day=date(2026, 11, 8), odds=3.8, ev=0.25):
    """A published signals row with the id the engine would give it."""
    return {
        "fixture_id": make_fixture_id(CSL.code, CSL.season, round_label or "0",
                                      day, home, away),
        "kickoff_utc": f"{day.isoformat()}T07:30:00Z",
        "round": round_label,
        "home_team": home, "away_team": away,
        "model": {"home": 0.4, "draw": 0.3, "away": 0.3, "method": "market_anchor"},
        "quotes": [{"book": book, "side": side, "odds": odds, "ev": ev,
                    "captured_at": None, "last_update": None, "proof": "observed"}],
        "best": {side: {"book": book, "odds": odds, "ev": ev}},
        "judged": {"side": side, "book": book, "odds": odds, "ev": ev,
                   "proof": "observed", "captured_at": None},
        "bet": ({"side": side, "book": book, "odds": odds, "ev": ev, "books": [book]}
                if fires else None),
        "state": "bet" if fires else "",
    }


PAIRS = [("Beijing Guoan", "Chongqing Tonglianglong"),
         ("Dalian Yingbo", "Tianjin Jinmen Tiger"),
         ("Shandong Taishan", "Chengdu Rongcheng")]


# --------------------------------------------------------------------------- #
# the round-0 transition
# --------------------------------------------------------------------------- #

def test_the_round_is_not_part_of_what_makes_two_rows_the_same_fixture():
    old, new = _row(*PAIRS[0], ""), _row(*PAIRS[0], "30")
    assert old["fixture_id"] != new["fixture_id"]
    assert telegram.fixture_key(old) == telegram.fixture_key(new)
    assert telegram.fixture_key(new) == (
        "CSL-2026-2026-11-08-beijing-guoan-chongqing-tonglianglong"
    )


def test_a_different_matchday_is_still_a_different_fixture():
    """A rescheduled match is one match on two dates, and CLAUDE.md says the date
    is part of the key on purpose. Dropping the round must not drop that."""
    a = _row(*PAIRS[0], "30")
    b = _row(*PAIRS[0], "30", day=date(2026, 11, 9))
    assert telegram.fixture_key(a) != telegram.fixture_key(b)


def test_renaming_every_live_signal_sends_nothing():
    """The first run after the fix: baseline under round 0, current under the real
    round, same fixtures, same bets."""
    previous = [_row(*PAIRS[0], ""), _row(*PAIRS[1], "", side="away"),
                _row(*PAIRS[2], "", fires=False)]
    current = [_row(*PAIRS[0], "30"), _row(*PAIRS[1], "30", side="away"),
               _row(*PAIRS[2], "30", fires=False)]
    assert telegram.new_signals(current, previous) == []
    assert telegram.withdrawn_signals(current, previous) == []


def test_a_genuinely_new_fixture_still_alerts_across_the_rename():
    previous = [_row(*PAIRS[0], "")]
    current = [_row(*PAIRS[0], "30"), _row(*PAIRS[1], "30")]
    fresh = telegram.new_signals(current, previous)
    assert [s["home_team"] for s in fresh] == ["Dalian Yingbo"]


def test_a_genuinely_withdrawn_fixture_still_withdraws_across_the_rename():
    previous = [_row(*PAIRS[0], ""), _row(*PAIRS[1], "")]
    current = [_row(*PAIRS[0], "30"), _row(*PAIRS[1], "30", fires=False)]
    out = telegram.withdrawn_signals(current, previous)
    assert [(b["home_team"], a["fixture_id"]) for b, a in out] == [
        ("Dalian Yingbo", current[1]["fixture_id"])]


def test_a_better_book_still_alerts_across_the_rename():
    previous = [_row(*PAIRS[0], "", book="duel")]
    current = [_row(*PAIRS[0], "30", book="onexbet")]
    assert len(telegram.new_signals(current, previous)) == 1


# --------------------------------------------------------------------------- #
# end to end through notify(): the outbox
# --------------------------------------------------------------------------- #

class _Run:
    """One notify() run against a given baseline, with Telegram stubbed."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.outbox = tmp_path / "alert_outbox.json"
        self.delivered: list[str] = []
        monkeypatch.setenv(telegram.TOKEN_ENV, "t")
        monkeypatch.setenv(telegram.CHAT_ENV, "c")

    def __call__(self, current, previous, *, telegram_up=True, dry_run=False):
        path = self.tmp_path / "signals.json"
        path.write_text(json.dumps({"schema": 1, "signals": current}), encoding="utf-8")
        self.monkeypatch.setattr(telegram, "previous_signals", lambda _p: previous)

        def send(_token, _chat, text, **_kw):
            if telegram_up:
                self.delivered.append(text)
            return telegram_up

        self.monkeypatch.setattr(telegram, "send", send)
        return telegram.notify("csl", CSL, signals_path=str(path),
                               outbox_path=str(self.outbox), dry_run=dry_run)

    def pending(self):
        if not self.outbox.exists():
            return []
        return json.loads(self.outbox.read_text(encoding="utf-8"))["pending"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    return _Run(tmp_path, monkeypatch)


def test_the_rename_sends_nothing_end_to_end(run):
    previous = [_row(*PAIRS[0], ""), _row(*PAIRS[1], "")]
    current = [_row(*PAIRS[0], "30"), _row(*PAIRS[1], "30")]
    assert run(current, previous) == 0
    assert run.delivered == []


def test_a_failed_send_is_retried_on_the_next_run(run):
    signals = [_row(*PAIRS[0], "30")]
    # Run 1: the bet is new and Telegram is down.
    assert run(signals, [], telegram_up=False) == 0
    assert [e["kind"] for e in run.pending()] == ["bet"]
    # Run 2: the workflow committed run 1's signals regardless, so the baseline
    # already holds the bet. Only the outbox says it never arrived.
    assert run(signals, signals) == 1
    assert "Beijing Guoan" in run.delivered[0]
    assert run.pending() == []
    # Run 3: delivered, so never again.
    assert run(signals, signals) == 0
    assert len(run.delivered) == 1


def test_a_delivered_alert_is_not_resent(run):
    signals = [_row(*PAIRS[0], "30")]
    assert run(signals, []) == 1
    assert run.pending() == []
    assert run(signals, signals) == 0
    assert len(run.delivered) == 1


def test_a_retry_survives_the_rename(run):
    """Run 1 fails under the old id; run 2 is the first under the new one."""
    assert run([_row(*PAIRS[0], "")], [], telegram_up=False) == 0
    assert run([_row(*PAIRS[0], "30")], [_row(*PAIRS[0], "")]) == 1


def test_an_unsent_bet_that_stopped_firing_is_neither_sent_nor_retracted(run):
    """It was never seen, so there is nothing to bet and nothing to take back."""
    fired = [_row(*PAIRS[0], "30")]
    assert run(fired, [], telegram_up=False) == 0
    assert run([_row(*PAIRS[0], "30", fires=False)], fired) == 0
    assert run.delivered == []
    assert run.pending() == []


def test_an_unsent_upgrade_that_stops_firing_still_owes_the_withdrawal(run):
    """The first book's alert landed and the reader is holding it."""
    at_duel = [_row(*PAIRS[0], "30", book="duel")]
    at_onexbet = [_row(*PAIRS[0], "30", book="onexbet")]
    assert run(at_duel, []) == 1
    assert run(at_onexbet, at_duel, telegram_up=False) == 0
    assert run([_row(*PAIRS[0], "30", fires=False)], at_onexbet) == 1
    assert "信号撤回" in run.delivered[-1]


def test_an_unsent_bet_is_resent_at_today_s_price_not_the_old_one(run):
    run([_row(*PAIRS[0], "30", odds=3.80)], [], telegram_up=False)
    now = [_row(*PAIRS[0], "30", odds=3.95)]
    assert run(now, [_row(*PAIRS[0], "30", odds=3.80)]) == 1
    assert "3.95" in run.delivered[0]


def test_a_failed_withdrawal_is_retried_on_the_next_run(run):
    fired = [_row(*PAIRS[0], "30")]
    stopped = [_row(*PAIRS[0], "30", fires=False)]
    assert run(stopped, fired, telegram_up=False) == 0
    assert [e["kind"] for e in run.pending()] == ["withdrawal"]
    assert run(stopped, stopped) == 1
    assert "信号撤回" in run.delivered[0]
    assert run(stopped, stopped) == 0


def test_a_failed_withdrawal_for_a_match_that_kicked_off_is_dropped(run):
    fired = [_row(*PAIRS[0], "30")]
    stopped = [_row(*PAIRS[0], "30", fires=False)]
    run(stopped, fired, telegram_up=False)
    assert run([], stopped) == 0
    assert run.pending() == []


def test_one_failure_among_several_keeps_only_that_one(run, monkeypatch):
    signals = [_row(*PAIRS[0], "30"), _row(*PAIRS[1], "30")]
    path = run.tmp_path / "signals.json"
    path.write_text(json.dumps({"schema": 1, "signals": signals}), encoding="utf-8")
    monkeypatch.setattr(telegram, "previous_signals", lambda _p: [])
    monkeypatch.setattr(telegram, "send",
                        lambda _t, _c, text, **_k: "Beijing Guoan" in text)
    assert telegram.notify("csl", CSL, signals_path=str(path),
                           outbox_path=str(run.outbox)) == 1
    assert [e["home_team"] for e in run.pending()] == ["Dalian Yingbo"]


def test_a_dry_run_leaves_the_outbox_alone(run):
    run([_row(*PAIRS[0], "30")], [], telegram_up=False)
    before = run.outbox.read_text(encoding="utf-8")
    run([_row(*PAIRS[0], "30")], [_row(*PAIRS[0], "30")], dry_run=True)
    assert run.outbox.read_text(encoding="utf-8") == before


def test_an_emptied_outbox_is_kept_as_a_file_not_deleted(run):
    """The commit action stages only paths that exist. A deleted outbox would never
    reach main, and the next run would check out the old one and send again."""
    signals = [_row(*PAIRS[0], "30")]
    run(signals, [], telegram_up=False)
    run(signals, signals)
    assert run.outbox.exists()
    assert run.pending() == []
