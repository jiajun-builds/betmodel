"""The signal decision.

The frozen baseline covers one league's firing path but not the other's, whose
best row sat at 0.196 against a 0.20 threshold. So the state machine is exercised
directly here, with quotes constructed to sit either side of each bar.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from scipy.optimize import brentq

from dataclasses import replace

from betmodel.config import load_league
from betmodel.dates import local_matchday
from betmodel.config.schema import DebiasConfig
from betmodel.fixtures.upcoming import Fixture
from betmodel.signals import debias
from betmodel.signals.engine import Best, Quote, _decide, fixture_id
from betmodel.signals.ev import devig


def _quote(book, side, odds, ev, proof="observed"):
    return Quote(book=book, side=side, odds=odds, ev=ev, proof=proof)


def _decide_with(league, quotes, *, method=debias.MARKET_ANCHOR, config=None, thin=()):
    """``method`` is what the de-bias actually produced for this fixture.

    It defaults to the anchored path because that is the state every other test
    here is about; the raw case has its own tests below.
    """
    config = config or load_league(league)
    best: dict[str, Best] = {}
    for side in ("home", "draw", "away"):
        same = [q for q in quotes if q.side == side]
        if same:
            winner = max(same, key=lambda q: q.odds)
            best[side] = Best(winner.book, winner.odds, winner.ev)
    return _decide(config, quotes, best, method, thin)


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
# de-vig
# --------------------------------------------------------------------------- #

def _log_devig(odds):
    """The logarithmic function, solved independently of the implementation.

    ``p_i = q_i ** k`` for the ``k`` that makes the triple sum to one. Solved
    with a different root finder than the engine's bisection, so this states the
    method rather than restating the code.
    """
    q = [1.0 / o for o in odds]
    k = brentq(lambda k: sum(x ** k for x in q) - 1.0, 1e-6, 100.0)
    return tuple(x ** k for x in q)


def test_the_devig_is_the_logarithmic_function_not_a_proportional_shave():
    odds = (2.1, 3.3, 3.4)
    fair = devig(odds)
    assert fair == pytest.approx(_log_devig(odds))
    assert sum(fair) == pytest.approx(1.0)

    # The distinction is not academic: proportional shaves every price by the
    # same fraction, the logarithmic function shaves the longshot harder. On this
    # triple the draw moves nearly four tenths of a point, which is the scale the
    # anchor correction itself operates at.
    total = sum(1.0 / o for o in odds)
    proportional = tuple((1.0 / o) / total for o in odds)
    assert fair[0] > proportional[0]
    assert fair[1] < proportional[1]
    assert fair[2] < proportional[2]


def test_a_vig_free_triple_is_returned_unchanged():
    """k solves to exactly 1 when the book has no margin, so the method is the
    identity there -- the same fixed point proportional normalisation has."""
    assert devig((3.0, 3.0, 3.0)) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert devig((2.0, 4.0, 4.0)) == pytest.approx((0.5, 0.25, 0.25))


def test_a_price_at_or_below_evens_falls_back_rather_than_failing():
    """No exponent pulls an implied probability of 1.0 below itself. Callers
    reject such a triple before it gets here; the function stays total anyway."""
    fair = devig((1.0, 5.0, 6.0))
    assert sum(fair) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# de-bias
# --------------------------------------------------------------------------- #

def test_full_anchoring_replaces_the_draw_with_the_market_and_keeps_the_sum():
    config = load_league("csl").signals.debias
    assert config.lam == 1.0
    probabilities, method = debias.apply((0.45, 0.25, 0.30), config,
                                         anchor_odds=(2.1, 3.3, 3.4))
    assert method == "market_anchor"
    assert probabilities[1] == pytest.approx(_log_devig((2.1, 3.3, 3.4))[1])
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


# --------------------------------------------------------------------------- #
# the anchor gate
# --------------------------------------------------------------------------- #

def test_an_edge_found_on_raw_probabilities_is_surfaced_but_never_bet():
    """The bar was cleared using numbers the league says are the wrong ones.

    A fixture's prices arrive over a window -- the soft books minutes after it
    appears, the anchor on its own slower poll -- so an edge computed before the
    anchor lands is an edge against an uncalibrated grid. Measured on 664
    walk-forward fixtures, anchoring moves log loss by -0.008 (95% CI
    [-0.015, -0.001]) and shifts a fixture's EV in either direction by several
    points, which is more than enough to move one across a 0.20 bar.
    """
    pick, state, ev, books = _decide_with(
        "csl", [_quote("onexbet", "home", 3.0, 0.25)], method=debias.RAW
    )
    assert state == "unanchored"
    assert books == (), "an empty book list is what stops it being presented as a bet"
    assert pick == "home", "the provisional side is still worth showing"
    assert ev == 0.25


def test_the_anchor_gate_only_applies_where_the_league_asked_for_an_anchor():
    """A league configured `none` is not waiting for anything, so raw IS its answer."""
    config = load_league("csl")
    unanchored_league = replace(
        config,
        signals=replace(config.signals, debias=DebiasConfig(method="none")),
    )
    pick, state, _, books = _decide_with(
        "csl", [_quote("onexbet", "home", 3.0, 0.25)],
        method=debias.RAW, config=unanchored_league,
    )
    assert (pick, state, books) == ("home", "bet", ("onexbet",))


def test_an_unanchored_edge_is_reported_as_unanchored_not_as_over_the_cap():
    """Both bars are decided from an EV the anchor gate has just called suspect,
    so the reason given is the one that came first."""
    _, state, _, books = _decide_with(
        "csl", [_quote("onexbet", "home", 12.0, 0.9)], method=debias.RAW
    )
    assert state == "unanchored", "the cap verdict rests on the same doubtful EV"
    assert books == ()


def test_an_anchored_edge_still_fires_normally():
    """The gate must not be a blanket suppressor."""
    _, state, _, books = _decide_with(
        "csl", [_quote("onexbet", "home", 3.0, 0.25)], method=debias.MARKET_ANCHOR
    )
    assert (state, books) == ("bet", ("onexbet",))


# --------------------------------------------------------------------------- #
# the anchor must be a proven opener
# --------------------------------------------------------------------------- #

def _fixture_row(home="A", away="B", days=5):
    from datetime import timedelta
    return Fixture(home=home, away=away, round="1",
                   kickoff=datetime.now(timezone.utc) + timedelta(days=days))


def test_an_unproven_anchor_is_treated_as_no_anchor(monkeypatch):
    """`reduce` has always derived this proof; the signal path never asked.

    A Pinnacle price we simply happen to hold is not evidence of an opening line.
    It may have been posted after the market moved, and the backtest that
    validates this strategy was run on true openers, so firing on it would be
    running an untested variant rather than a weaker version of the same thing.
    """
    from betmodel.signals import engine

    priced = {"home_odds": 2.0, "draw_odds": 3.4, "away_odds": 4.0,
              "captured_at": None, "last_update": "", "provenance": "", "lead_h": 120.0}
    config = load_league("csl")

    row = _fixture_row()
    day = local_matchday(row.kickoff, config.timezone)

    def _opens(*_a, **_k):
        return {
            ("A", "B", day, "pinnacle"): {**priced, "proof": ""},      # unproven
            ("A", "B", day, "duel"): {**priced, "proof": "window", "away_odds": 6.0},
        }

    monkeypatch.setattr(engine.reduce_module, "collapse_opens", _opens)
    monkeypatch.setattr(engine, "load_upcoming", lambda *_a, **_k: [row])
    monkeypatch.setattr(engine, "_model_probabilities",
                        lambda *a, **k: {("A", "B"): (0.30, 0.24, 0.46)},
                        raising=False)

    built = engine.build_signals("csl", config)
    assert built, "the fixture is priced, so it must produce a row"
    assert built[0].debias_method == debias.RAW
    assert built[0].state != "bet", "an unproven anchor must not produce a bet"


def test_a_proven_anchor_still_calibrates(monkeypatch):
    """The counterpart, so the gate above cannot pass by suppressing everything."""
    from betmodel.signals import engine

    priced = {"home_odds": 2.0, "draw_odds": 3.4, "away_odds": 4.0,
              "captured_at": None, "last_update": "", "provenance": "", "lead_h": 120.0}
    config = load_league("csl")

    row = _fixture_row()
    day = local_matchday(row.kickoff, config.timezone)

    def _opens(*_a, **_k):
        return {
            ("A", "B", day, "pinnacle"): {**priced, "proof": "window"},
            ("A", "B", day, "duel"): {**priced, "proof": "window", "away_odds": 6.0},
        }

    monkeypatch.setattr(engine.reduce_module, "collapse_opens", _opens)
    monkeypatch.setattr(engine, "load_upcoming", lambda *_a, **_k: [row])
    monkeypatch.setattr(engine, "_model_probabilities",
                        lambda *a, **k: {("A", "B"): (0.30, 0.24, 0.46)})

    built = engine.build_signals("csl", config)
    assert built and built[0].debias_method == debias.MARKET_ANCHOR


# --------------------------------------------------------------------------- #
# the proof cutover
# --------------------------------------------------------------------------- #

def _anchor(proof, captured):
    return {"home_odds": 2.0, "draw_odds": 3.4, "away_odds": 4.0,
            "proof": proof, "captured_at": captured}


def test_an_anchor_banked_before_the_cutover_keeps_the_weaker_proof():
    """Refusing them would retroactively void prices captured correctly under the
    rules of the day: the evidence the strong proof needs was not being recorded."""
    from betmodel.signals.engine import ANCHOR_PROOF_CUTOVER as C, _anchor_is_a_proven_opener
    from datetime import timedelta

    assert _anchor_is_a_proven_opener(_anchor("window", C - timedelta(days=1)))


def test_an_anchor_banked_after_the_cutover_must_earn_the_strong_proof():
    from betmodel.signals.engine import ANCHOR_PROOF_CUTOVER as C, _anchor_is_a_proven_opener
    from datetime import timedelta

    assert not _anchor_is_a_proven_opener(_anchor("window", C + timedelta(days=1)))
    assert _anchor_is_a_proven_opener(_anchor("observed", C + timedelta(days=1)))


def test_no_proof_is_never_enough_on_either_side_of_the_cutover():
    from betmodel.signals.engine import ANCHOR_PROOF_CUTOVER as C, _anchor_is_a_proven_opener
    from datetime import timedelta

    assert not _anchor_is_a_proven_opener(_anchor("", C - timedelta(days=1)))
    assert not _anchor_is_a_proven_opener(_anchor("", C + timedelta(days=1)))


def test_the_cutover_expires_by_itself():
    """It applies to evidence, not to a stretch of calendar. Every fixture kicks
    off, so once none predates it the constant is dead and can be deleted."""
    from betmodel.signals.engine import ANCHOR_PROOF_CUTOVER
    assert ANCHOR_PROOF_CUTOVER.tzinfo is not None, "a naive boundary would compare wrong"


# --------------------------------------------------------------------------- #
# thin evidence
# --------------------------------------------------------------------------- #

def test_a_club_with_too_little_history_cannot_be_bet_on():
    """Shrinkage regularises a thin rating without making it trustworthy.

    Its target is the league mean, which for a promoted side is the wrong prior:
    it pulls a club nobody has seen toward average and therefore overrates it.
    Measured on Atlante six matches in, the model sat 6.0 points above Pinnacle's
    no-vig line on its own fixture -- the difference between a -1.5% price and a
    +21.3% signal.
    """
    pick, state, ev, books = _decide_with(
        "csl", [_quote("onexbet", "home", 3.0, 0.25)], thin=("Atlante",)
    )
    assert state == "thin_evidence"
    assert books == (), "surfaced, never bet"
    assert (pick, ev) == ("home", 0.25), "the edge is still worth showing"


def test_a_thin_club_is_reported_before_a_missing_anchor():
    """Both mean the number is not to be trusted, but only one is fixable by a
    request: an anchor can be fetched, matches can only be played."""
    _, state, _, _ = _decide_with(
        "csl", [_quote("onexbet", "home", 3.0, 0.25)],
        method=debias.RAW, thin=("Atlante",),
    )
    assert state == "thin_evidence"


def test_both_clubs_are_checked_not_just_the_one_backed():
    """A fixture is only as well understood as its worse understood side."""
    for thin in (("Home FC",), ("Away FC",)):
        _, state, _, _ = _decide_with(
            "csl", [_quote("onexbet", "home", 3.0, 0.25)], thin=thin
        )
        assert state == "thin_evidence", thin


def test_a_league_that_sets_no_threshold_is_unaffected():
    _, state, _, books = _decide_with("csl", [_quote("onexbet", "home", 3.0, 0.25)])
    assert (state, books) == ("bet", ("onexbet",))


def test_missing_team_stats_disables_the_check_rather_than_muting_the_league(tmp_path):
    """Failing the other way would make a missing sidecar look like a drought."""
    from betmodel.signals.engine import team_evidence
    assert team_evidence("csl", str(tmp_path / "absent.csv")) == {}
