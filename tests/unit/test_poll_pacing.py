"""Each book is polled at its own declared interval.

The interval sat in every league's YAML and was never read: `polled_books`
tested it for truthiness and threw the number away, so a book declared at 180
minutes was polled exactly as often as one declared at 10. Every open tick
therefore spent a request against the monthly-budget account at the same rate as
against the daily-budget one, which is the wrong way round.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betmodel.config import ConfigError, load_league
from betmodel.config.schema import BookConfig
from betmodel.odds import capture_open as co

CSL = load_league("csl")


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 8, 30, hour, minute, tzinfo=timezone.utc)


def test_the_anchor_comes_due_only_on_its_own_multiple():
    due = co.books_due(CSL, "theoddsapi", _at(3, 0))
    assert [b.key for b in due] == ["pinnacle"]
    assert co.books_due(CSL, "theoddsapi", _at(3, 10)) == ()
    assert co.books_due(CSL, "theoddsapi", _at(4, 0)) == ()


def test_the_bet_books_come_due_far_more_often():
    assert {b.key for b in co.books_due(CSL, "oddsapiio", _at(3, 10))} == {"onexbet", "duel"}
    assert co.books_due(CSL, "oddsapiio", _at(3, 5)) == ()


def test_midnight_is_a_multiple_of_everything():
    # The modulus is minutes since midnight, so 00:00 must be a valid slot for
    # every interval or the first tick of the day would be skipped.
    assert co.books_due(CSL, "theoddsapi", _at(0, 0))
    assert co.books_due(CSL, "oddsapiio", _at(0, 0))


def test_a_days_worth_of_ticks_matches_the_declared_rate():
    """The whole point: the expensive book is polled a fraction as often."""
    counts = {}
    for provider in ("oddsapiio", "theoddsapi"):
        n = sum(
            1 for h in range(24) for m in range(0, 60, 5)
            if co.books_due(CSL, provider, _at(h, m))
        )
        counts[provider] = n
    assert counts["oddsapiio"] == 144
    assert counts["theoddsapi"] == 8


def _book(**kw):
    base = dict(key="x", provider="oddsapiio", role="bet", legacy_prefix="x_open")
    base.update(kw)
    return BookConfig.parse(base, "odds.books[0]")


def test_an_interval_that_does_not_divide_the_day_is_refused():
    # 50 minutes would fire at a different wall-clock time every day and skip
    # the slot straddling midnight.
    with pytest.raises(ConfigError, match="divide 1440"):
        _book(poll_interval_minutes=50)


def test_an_interval_finer_than_the_timer_is_refused():
    # The timer fires every five minutes; a 3-minute book would never come due.
    with pytest.raises(ConfigError, match="multiple of the"):
        _book(poll_interval_minutes=3)


def test_a_valid_interval_is_accepted():
    assert _book(poll_interval_minutes=15).poll_interval_minutes == 15


def test_a_manual_run_can_ignore_the_pacing(monkeypatch, tmp_path):
    """The clock must not silence the tool you reach for when something is wrong.

    Verifying the credential change hours before a matchday, a dispatched capture
    ran at 11:01, declined every provider because 11:01 is not a multiple of 10 or
    180, and reported success having done nothing. A scheduled tick skipping its
    turn is the design; a human asking for one and getting a no-op is not.
    """
    from betmodel.odds import capture_open as module

    seen = []

    def _fake_pending(league, config, *, books, now, **kw):
        seen.append(tuple(b.key for b in books))
        return []

    monkeypatch.setattr(module, "pending_fixtures", _fake_pending)
    at_an_odd_minute = _at(11, 1)

    module.capture_opens(
        "csl", CSL, now=at_an_odd_minute, dry_run=True,
        history_path=str(tmp_path / "h.csv"), fixtures_path=str(tmp_path / "f.csv"),
    )
    assert seen == [], "an ordinary tick at 11:01 is due for nothing"

    module.capture_opens(
        "csl", CSL, now=at_an_odd_minute, dry_run=True, ignore_schedule=True,
        history_path=str(tmp_path / "h.csv"), fixtures_path=str(tmp_path / "f.csv"),
    )
    assert seen, "an override must poll despite the clock"
    assert {b for group in seen for b in group} == {"onexbet", "duel", "pinnacle"}
