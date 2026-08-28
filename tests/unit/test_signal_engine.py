"""The signal decision.

The frozen baseline covers one league's firing path but not the other's, whose
best row sat at 0.196 against a 0.20 threshold. So the state machine is exercised
directly here, with quotes constructed to sit either side of each bar.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betmodel.config import load_league
from betmodel.fixtures.upcoming import Fixture
from betmodel.signals import debias
from betmodel.signals.engine import Best, Quote, _decide, fixture_id


def _quote(book, side, odds, ev, proof="observed"):
    return Quote(book=book, side=side, odds=odds, ev=ev, proof=proof)


def _decide_with(league, quotes):
    config = load_league(league)
    best: dict[str, Best] = {}
    for side in ("home", "draw", "away"):
        same = [q for q in quotes if q.side == side]
        if same:
            winner = max(same, key=lambda q: q.odds)
            best[side] = Best(winner.book, winner.odds, winner.ev)
    return _decide(config, quotes, best)


# --------------------------------------------------------------------------- #
# the bars
# --------------------------------------------------------------------------- #

def test_edge_above_the_threshold_fires():
    pick, state, ev, books = _decide_with("csl", [_quote("onexbet", "home", 3.0, 0.25)])
    assert (pick, state, books) == ("home", "bet", ("onexbet",))
    assert ev == 0.25


def test_edge_at_the_threshold_does_not_fire():
    """Strictly greater. The frozen CSL board's best row sat at 0.196 against a
    0.20 bar and correctly published nothing."""
    assert _decide_with("csl", [_quote("onexbet", "home", 3.0, 0.20)])[1] == ""


def test_a_long_shot_over_the_cap_is_surfaced_but_not_bettable():
    pick, state, _, books = _decide_with("csl", [_quote("onexbet", "home", 9.0, 0.40)])
    assert (pick, state) == ("home", "odds_cap")
    assert books == (), "a greyed row must show no book to bet with"


def test_state_is_bet_exactly_when_a_book_is_named():
    """The invariant that keeps every displayed logo a bet you should place."""
    for quotes in ([_quote("onexbet", "home", 3.0, 0.25)],
                   [_quote("onexbet", "home", 9.0, 0.40)],
                   [_quote("onexbet", "home", 3.0, 0.05)]):
        _, state, _, books = _decide_with("csl", quotes)
        assert (state == "bet") == bool(books)


# --------------------------------------------------------------------------- #
# pick first, then cap
# --------------------------------------------------------------------------- #

def test_adding_a_book_can_remove_a_bet():
    """Deliberate and inherited. The pick is chosen on the best price and only
    then checked against the cap, so a better price that is untouchable greys the
    row rather than falling back to a worse one. The long-shot tail is the
    least-edge slice; a row whose best price cannot be taken is not a row to bet.
    """
    alone = _decide_with("csl", [_quote("duel", "home", 6.5, 0.30)])
    assert alone[1] == "bet"

    with_second = _decide_with("csl", [
        _quote("duel", "home", 6.5, 0.30),
        _quote("onexbet", "home", 8.0, 0.60),
    ])
    assert with_second[1] == "odds_cap"


def test_the_highest_ev_side_wins_not_the_highest_odds():
    pick, _, _, _ = _decide_with("csl", [
        _quote("onexbet", "home", 3.0, 0.25),
        _quote("onexbet", "away", 6.0, 0.50),
    ])
    assert pick == "away"


def test_a_league_that_does_not_bet_the_draw_never_picks_it():
    pick, state, _, _ = _decide_with("csl", [_quote("onexbet", "draw", 4.0, 0.90)])
    assert (pick, state) == ("", "")


def test_a_league_that_bets_the_draw_can_pick_it():
    pick, state, _, _ = _decide_with("ligamx", [_quote("duel", "draw", 4.0, 0.30)])
    assert (pick, state) == ("draw", "bet")


# --------------------------------------------------------------------------- #
# the provenance gate
# --------------------------------------------------------------------------- #

def test_an_unproven_price_cannot_fire_where_proof_is_required():
    assert load_league("ligamx").signals.require_price_proof is True
    assert _decide_with("ligamx", [_quote("duel", "home", 3.0, 0.30, proof="")])[1] == ""


def test_there_is_no_fallback_to_a_worse_but_proven_side():
    """Choosing a bet on the strength of its paperwork rather than its edge."""
    pick, state, _, _ = _decide_with("ligamx", [
        _quote("duel", "home", 3.0, 0.40, proof=""),
        _quote("duel", "away", 2.0, 0.20, proof="observed"),
    ])
    assert (pick, state) == ("", "")


def test_proof_is_not_required_where_the_league_has_no_evidence_yet():
    """One league has no unpriced-observation history at all, so requiring proof
    there would publish nothing."""
    assert load_league("csl").signals.require_price_proof is False
    assert _decide_with("csl", [_quote("onexbet", "home", 3.0, 0.30, proof="")])[1] == "bet"


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #

def test_the_fixture_id_uses_the_local_matchday_not_the_utc_one():
    """A 19:00 kickoff in a UTC-6 league falls on the next UTC day, and naming
    the fixture by that day names a day nobody played on."""
    config = load_league("ligamx")
    fixture = Fixture(
        home="Necaxa", away="Cruz Azul",
        kickoff=datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc), round="6",
    )
    assert fixture_id(config, fixture) == "MEX-apertura-2026-6-2026-08-28-necaxa-cruz-azul"


def test_accents_do_not_reach_an_identifier():
    config = load_league("ligamx")
    fixture = Fixture(
        home="Club América", away="León",
        kickoff=datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc), round="6",
    )
    assert fixture_id(config, fixture).endswith("club-america-leon")


# --------------------------------------------------------------------------- #
# de-bias
# --------------------------------------------------------------------------- #

def test_full_anchoring_replaces_the_draw_with_the_market_and_keeps_the_sum():
    config = load_league("csl").signals.debias
    assert config.lam == 1.0
    probabilities, method = debias.apply((0.45, 0.25, 0.30), config,
                                         anchor_odds=(2.1, 3.3, 3.4))
    assert method == "market_anchor"
    assert probabilities[1] == pytest.approx(1 / 3.3 / (1 / 2.1 + 1 / 3.3 + 1 / 3.4))
    assert sum(probabilities) == pytest.approx(1.0)


def test_home_and_away_keep_their_relative_strength():
    config = load_league("csl").signals.debias
    raw = (0.45, 0.25, 0.30)
    out, _ = debias.apply(raw, config, anchor_odds=(2.1, 3.3, 3.4))
    assert out[0] / out[2] == pytest.approx(raw[0] / raw[2])


def test_a_fixture_with_no_anchor_ships_the_raw_grid():
    """The normal state for a fixture no anchor book has opened. Not a failure,
    and the method label records which path ran."""
    raw = (0.45, 0.25, 0.30)
    out, method = debias.apply(raw, load_league("csl").signals.debias, anchor_odds=None)
    assert (out, method) == (raw, "raw")
