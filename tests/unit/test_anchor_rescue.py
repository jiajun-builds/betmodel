"""The on-demand anchor fetch.

The anchor gate is only tolerable if the anchor usually arrives in time, and the
anchor's own poll interval is three hours. This is what closes that gap, so what
matters is that it spends a request exactly when an edge is stranded and never
otherwise.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from betmodel.config import load_league
from betmodel.config.schema import ConfigError, DebiasConfig
from betmodel.signals import anchor_rescue
from betmodel.signals.engine import STATE_BET, STATE_UNANCHORED


class _Signal:
    def __init__(self, state, home="A", away="B", ev=0.3):
        self.state, self.home_team, self.away_team, self.ev = state, home, away, ev


class _Spy:
    """Stands in for `capture_opens`, recording how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, league, config, **kwargs):
        self.calls.append((league, kwargs))
        return {"rows": 1}


@pytest.fixture
def spy():
    return _Spy()


def _rescue(monkeypatch, spy, signals, league="csl", config=None):
    monkeypatch.setattr(anchor_rescue, "build_signals", lambda *a, **k: signals)
    return anchor_rescue.rescue(league, config or load_league(league), capture=spy)


def test_a_stranded_edge_buys_one_anchor_fetch(monkeypatch, spy):
    stats = _rescue(monkeypatch, spy, [_Signal(STATE_UNANCHORED)])
    assert stats["stranded"] == 1 and stats["fetched"] == 1
    assert len(spy.calls) == 1


def test_nothing_stranded_spends_nothing(monkeypatch, spy):
    stats = _rescue(monkeypatch, spy, [_Signal(STATE_BET), _Signal("")])
    assert stats == {"stranded": 0, "fetched": 0}
    assert spy.calls == [], "an idle tick must not touch a metered provider"


def test_five_stranded_fixtures_still_cost_one_request(monkeypatch, spy):
    """The provider returns every event, so the cost does not scale with them."""
    stats = _rescue(monkeypatch, spy, [_Signal(STATE_UNANCHORED) for _ in range(5)])
    assert stats["stranded"] == 5
    assert len(spy.calls) == 1


def test_it_asks_for_the_anchor_and_ignores_the_anchor_schedule(monkeypatch, spy):
    """The schedule is precisely what has to be overridden: the interval not
    having come round is the reason the fixture is stranded."""
    _rescue(monkeypatch, spy, [_Signal(STATE_UNANCHORED)])
    _, kwargs = spy.calls[0]
    anchor = load_league("csl").odds.book("pinnacle")
    assert kwargs["providers"] == (anchor.provider,)
    assert kwargs["ignore_schedule"] is True


def test_a_league_that_asked_for_no_anchor_is_left_alone(monkeypatch, spy):
    config = load_league("csl")
    no_anchor = replace(
        config, signals=replace(config.signals, debias=DebiasConfig(method="none"))
    )
    stats = _rescue(monkeypatch, spy, [_Signal(STATE_UNANCHORED)], config=no_anchor)
    assert stats == {"stranded": 0, "fetched": 0}
    assert spy.calls == []


def test_an_anchor_book_that_is_not_in_the_book_list_is_refused_by_the_config():
    """The rescue has a defensive branch for this; the config makes it
    unreachable, which is the stronger guarantee and the one worth pinning.

    It matters because the rescue resolves the anchor by key at request time: a
    book renamed in the list without the de-bias following would otherwise mean
    silently fetching nothing, forever, for every stranded edge.
    """
    config = load_league("csl")
    with pytest.raises(ConfigError):
        replace(
            config,
            signals=replace(
                config.signals,
                debias=DebiasConfig(method="market_anchor", lam=1.0, anchor_book="gone"),
            ),
        )


def test_a_refused_fetch_is_not_reported_as_a_fetch(monkeypatch, spy):
    """The failure this whole branch exists to stop being silent.

    An account under its quota floor refuses every call while the workflow step
    still exits green, so the numbers a tick returns look exactly like a tick that
    found nothing. Reporting `fetched: 1` here would hide it one layer further up.
    """
    def refused(league, config, **kwargs):
        return {"pending": 4, "captured": 0, "appended": 0, "requests": 0, "refused": 1}

    monkeypatch.setattr(anchor_rescue, "build_signals",
                        lambda *a, **k: [_Signal(STATE_UNANCHORED)])
    stats = anchor_rescue.rescue("csl", load_league("csl"), capture=refused)
    assert stats["fetched"] == 0, "nothing was fetched, so nothing was fetched"
    assert stats["refused"] == 1
    assert stats["stranded"] == 1
