"""Gate G4: the canonical contract is what it claims to be.

The published tree is served straight from the repository, so a bad file is live
the moment it is committed. Validation therefore runs before writing, and this
gate checks the properties a consumer is entitled to assume.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from betmodel.config import available_leagues, load_all, load_league
from betmodel.publish import contract, public
from betmodel.signals.engine import build_signals, make_fixture_id

AT = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def published():
    payloads = {}
    for league in available_leagues():
        config = load_league(league)
        signals = build_signals(league, config, now=AT)
        payloads[league] = {
            "signals": public.signals_payload(config, signals, AT),
            "results": public.results_payload(league, config, AT),
        }
    payloads["index"] = public.index_payload(load_all(), AT)
    return payloads


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #

def test_the_manifest_lists_every_league_found_on_disk(published):
    """Adding a league must not need a consumer change, which is only true if
    the manifest is discovered rather than written."""
    listed = {entry["id"] for entry in published["index"]["leagues"]}
    assert listed == set(available_leagues())


def test_the_manifest_names_every_file_a_consumer_needs(published):
    for entry in published["index"]["leagues"]:
        assert set(entry["files"]) == set(contract.LEAGUE_FILES)


def test_the_manifest_publishes_each_threshold(published):
    """A consumer that hardcodes a threshold cannot follow a league that changes
    one, and one already had: a board carried a bar three times smaller than the
    producer's."""
    for entry in published["index"]["leagues"]:
        assert entry["ev_min"] == load_league(entry["id"]).signals.ev_min


# --------------------------------------------------------------------------- #
# one shape for every league
# --------------------------------------------------------------------------- #

def test_signal_records_have_the_same_field_set_in_every_league(published):
    shapes = {
        league: set(payload["signals"]["signals"][0])
        for league, payload in published.items()
        if league != "index" and payload["signals"]["signals"]
    }
    assert len(set(map(frozenset, shapes.values()))) == 1, shapes


def test_expected_value_is_a_fraction_in_every_league(published):
    """The old contract expressed it two ways, which is why the board downstream
    guesses the scale from a median of the payload."""
    for league, payload in published.items():
        if league == "index":
            continue
        values = [q["ev"] for s in payload["signals"]["signals"] for q in s["quotes"]]
        assert values, league
        assert max(abs(v) for v in values) < 5.0, league


def test_every_timestamp_is_utc(published):
    for league, payload in published.items():
        if league == "index":
            continue
        for signal in payload["signals"]["signals"]:
            assert signal["kickoff_utc"].endswith("Z"), league


# --------------------------------------------------------------------------- #
# internal consistency
# --------------------------------------------------------------------------- #

def test_a_firing_row_always_names_a_book_to_bet_with(published):
    for league, payload in published.items():
        if league == "index":
            continue
        for signal in payload["signals"]["signals"]:
            fires = signal["state"] == contract.STATE_BET
            assert fires == (signal["bet"] is not None), f"{league} {signal['fixture_id']}"
            if fires:
                assert signal["bet"]["books"]


def test_the_judged_price_is_published_even_when_nothing_fires(published):
    """The distinction the old shape expressed by which field you read. A row
    that did not fire still says which price it was judged against."""
    quiet = [
        s for league, payload in published.items() if league != "index"
        for s in payload["signals"]["signals"]
        if s["state"] != contract.STATE_BET and s["quotes"]
    ]
    assert quiet, "no non-firing rows with quotes, so this proves nothing"
    assert all(s["judged"] is not None for s in quiet)


def test_results_carry_a_score_exactly_when_they_are_played(published):
    for league, payload in published.items():
        if league == "index":
            continue
        for result in payload["results"]["results"]:
            played = result["status"] == "played"
            assert played == (result["home_goals"] is not None), league


def test_signals_and_results_build_identity_the_same_way():
    """Otherwise the join between a signal and its outcome breaks silently, and
    the whole point of publishing results is settling signals."""
    config = load_league("ligamx")
    from datetime import date
    built = make_fixture_id(
        config.code, config.season, "6", date(2026, 8, 28), "Necaxa", "Cruz Azul"
    )
    assert built == "MEX-apertura-2026-6-2026-08-28-necaxa-cruz-azul"


def test_a_knockout_stage_name_survives_into_the_identifier():
    """Ties are filed under names, not numbers, and "28" says far less."""
    config = load_league("ligamx")
    from datetime import date
    built = make_fixture_id(
        config.code, config.season, "Quarterfinals", date(2026, 12, 1), "Toluca", "Leon"
    )
    assert "quarterfinals" in built


# --------------------------------------------------------------------------- #
# the validators refuse what would mislead
# --------------------------------------------------------------------------- #

def test_a_probability_triple_that_does_not_sum_to_one_is_refused():
    with pytest.raises(contract.ContractError, match="sum to"):
        contract.validate_signals({
            "schema": 1, "signals": [{
                "fixture_id": "x", "kickoff_utc": "2026-08-28T12:00:00Z",
                "model": {"home": 0.5, "draw": 0.3, "away": 0.3},
                "quotes": [], "state": "", "bet": None,
            }],
        })


def test_percentage_point_expected_value_is_refused():
    with pytest.raises(contract.ContractError, match="percentage points"):
        contract.validate_signals({
            "schema": 1, "signals": [{
                "fixture_id": "x", "kickoff_utc": "2026-08-28T12:00:00Z",
                "model": {"home": 0.5, "draw": 0.2, "away": 0.3},
                "quotes": [{"side": "home", "odds": 2.0, "ev": 21.07}],
                "state": "", "bet": None,
            }],
        })


def test_a_duplicate_fixture_makes_a_file_self_contradictory_and_is_refused():
    row = {
        "fixture_id": "x", "kickoff_utc": "2026-08-28T12:00:00Z",
        "model": {"home": 0.5, "draw": 0.2, "away": 0.3},
        "quotes": [], "state": "", "bet": None,
    }
    with pytest.raises(contract.ContractError, match="duplicate"):
        contract.validate_signals({"schema": 1, "signals": [row, dict(row)]})


def test_a_naive_timestamp_is_refused():
    with pytest.raises(contract.ContractError, match="UTC"):
        contract.validate_signals({
            "schema": 1, "signals": [{
                "fixture_id": "x", "kickoff_utc": "2026-08-28T12:00:00",
                "model": {"home": 0.5, "draw": 0.2, "away": 0.3},
                "quotes": [], "state": "", "bet": None,
            }],
        })
