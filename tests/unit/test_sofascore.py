"""SofaScore field semantics.

Every case here is a real way the upstream data misleads. They run offline: the
network behaviour is covered by the G0 workflow, this is about what we do with a
payload once we have it.
"""

from __future__ import annotations

import pytest

from betmodel.providers import sofascore as sf


def _client(strategy="halves_sum", periods=None):
    c = sf.SofascoreClient(xg_strategy=strategy, pause=0.0)
    c.event_xg_by_period = lambda event_id: dict(periods or {})  # type: ignore[method-assign]
    return c


# --------------------------------------------------------------------------- #
# xG period selection
# --------------------------------------------------------------------------- #

def test_halves_sum_adds_the_two_regulation_halves():
    c = _client(periods={"ALL": (1.39, 0.5), "1ST": (0.35, 0.25), "2ND": (1.03, 0.25)})
    # Deliberately not the ALL rollup, which disagrees at the second decimal.
    assert c.event_xg(1) == (1.38, 0.5)


def test_zero_filled_halves_fall_back_to_the_rollup():
    """The failure this exists to prevent.

    Chinese Super League publishes both half keys as literal 0.0 and carries the
    real figure only in ALL. Summing them yields a plausible-looking zero, which
    would silently turn xG off and leave a goals-only model that still runs.
    """
    c = _client(periods={"ALL": (3.27, 1.61), "1ST": (0.0, 0.0), "2ND": (0.0, 0.0)})
    assert c.event_xg(1) == (3.27, 1.61)


def test_a_genuine_goalless_half_sum_is_not_overridden():
    """Zero halves with a zero rollup really are zero, and must stay zero."""
    c = _client(periods={"ALL": (0.0, 0.0), "1ST": (0.0, 0.0), "2ND": (0.0, 0.0)})
    assert c.event_xg(1) == (0.0, 0.0)


def test_missing_halves_use_the_rollup():
    c = _client(periods={"ALL": (2.0, 1.0)})
    assert c.event_xg(1) == (2.0, 1.0)


def test_nothing_published_is_none_not_zero():
    assert _client(periods={}).event_xg(1) == (None, None)


def test_extra_time_is_never_added_in():
    """xG must stay on the same ninety-minute clock as the scoreline."""
    c = _client(periods={"1ST": (0.5, 0.2), "2ND": (0.5, 0.3), "ET1": (9.0, 9.0)})
    assert c.event_xg(1) == (1.0, 0.5)


def test_all_first_prefers_the_rollup():
    c = _client("all_first", {"ALL": (3.27, 1.61), "1ST": (0.0, 0.0), "2ND": (0.0, 0.0)})
    assert c.event_xg(1) == (3.27, 1.61)


def test_unknown_strategy_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown xg strategy"):
        sf.SofascoreClient(xg_strategy="whatever")


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #

def test_goals_read_regulation_not_the_running_total():
    """current includes shootout and extra-time goals; the model forecasts ninety."""
    event = {
        "homeScore": {"normaltime": 2, "current": 11},
        "awayScore": {"normaltime": 1, "current": 9},
    }
    assert sf.event_goals(event) == (2, 1)


def test_goals_fall_back_to_current_when_regulation_is_absent():
    event = {"homeScore": {"current": 1}, "awayScore": {"current": 0}}
    assert sf.event_goals(event) == (1, 0)


# --------------------------------------------------------------------------- #
# round labels
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "event,expected",
    [
        ({"roundInfo": {"round": 21}}, "21"),
        ({"roundInfo": {"round": 28, "name": "Semifinals"}}, "Semifinals"),
        ({"roundInfo": None, "tournament": {"slug": "liga-mx-apertura-play-in"}}, "Play-in"),
        ({}, ""),
    ],
)
def test_round_label(event, expected):
    assert sf.round_label(event) == expected


def test_all_first_refuses_the_rollup_once_extra_time_appears():
    """The setting assumes a competition that cannot go to extra time.

    Measured on the Liga MX Apertura 25/26 final: ALL (2.36, 0.66), halves
    (2.68, 0.38), extra time (0.30, 0.05). The rollup is neither the ninety
    minutes nor the full match, so it cannot be put on the same clock as the
    scoreline. If a league configured all_first gains a knockout stage, falling
    back to the halves is the only safe answer.
    """
    c = _client("all_first", {
        "ALL": (2.36, 0.66), "1ST": (0.35, 0.13), "2ND": (2.33, 0.25),
        "ET1": (0.1, 0.0), "ET2": (0.2, 0.05),
    })
    assert c.event_xg(1) == (2.68, 0.38)


def test_all_first_still_answers_when_extra_time_leaves_no_usable_halves():
    """Degraded but not silent: the warning is the product here."""
    c = _client("all_first", {
        "ALL": (1.5, 0.5), "1ST": (0.0, 0.0), "2ND": (0.0, 0.0), "ET1": (0.2, 0.1),
    })
    assert c.event_xg(1) == (1.5, 0.5)
