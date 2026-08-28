"""Gate G2: replaying the capture history reproduces the match table.

The capture history is append-only and irreplaceable, and the match table is what
every downstream measurement is computed from. So the reducer has two duties that
are easy to state and easy to get wrong: it must never overwrite, and it must
never promote a price into a column it does not belong in.

The league that already had a reducer gives the strong form of the gate: replay
its whole history and nothing should change, because everything the reducer would
write is already there and identical. The league that never had one gives the
weak form: every write must land in a blank.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from betmodel import paths
from betmodel.config import load_league
from betmodel.odds import capture_store, provenance
from betmodel.odds import reduce as rd


def _records(league):
    return rd.build_records(league, load_league(league))


def _copy_matches(league, tmp_path) -> str:
    target = str(tmp_path / f"{league}-matches.csv")
    shutil.copy2(paths.for_league(league).matches_csv, target)
    return target


# --------------------------------------------------------------------------- #
# the strong form
# --------------------------------------------------------------------------- #

def test_replaying_an_already_reduced_history_changes_nothing():
    """Liga MX has been reduced continuously, so a replay must be a no-op.

    Anything else means the merged reducer disagrees with the one that produced
    the committed data.
    """
    stats = rd.merge("ligamx", _records("ligamx"), dry_run=True)
    assert stats["written"] == 0, (
        "the reducer would write values the pre-merge one did not; "
        f"stats={stats}"
    )
    assert stats["kept"] > 0, "nothing matched at all, so the replay proved nothing"


# --------------------------------------------------------------------------- #
# the weak form, and the invariants that hold for both
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_no_existing_value_is_ever_overwritten(league, tmp_path):
    """Values in the match table were assembled from several sources and
    repaired by hand. A reducer that clobbered them would undo work that cannot
    be redone."""
    target = _copy_matches(league, tmp_path)
    before = pd.read_csv(target)
    rd.merge(league, _records(league), create_columns=True, matches_path=target)
    after = pd.read_csv(target)

    for column in before.columns:
        was_set = before[column].notna()
        if not was_set.any():
            continue
        assert after.loc[was_set, column].equals(before.loc[was_set, column]), (
            f"{league}: reduce modified existing values in {column}"
        )


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_reducing_twice_writes_nothing_the_second_time(league, tmp_path):
    target = _copy_matches(league, tmp_path)
    first = rd.merge(league, _records(league), create_columns=True, matches_path=target)
    second = rd.merge(league, _records(league), create_columns=True, matches_path=target)
    assert second["written"] == 0, f"not idempotent: {first} then {second}"


def test_placeholder_prices_never_become_opening_lines():
    """The corruption this gate exists to prevent.

    Twenty rows in the CSL history are labelled 'open' but hold the current line,
    written when a predicted window elapsed unfilled. Eight are on a book the
    model bets. Letting them through would put mid-market prices into the one
    series the project's positive result rests on.
    """
    config = load_league("csl")
    history = capture_store.load_history(paths.for_league("csl").capture_history_csv)
    prepared = rd._prepared(history, config)
    opens = prepared[prepared["snapshot_type"] == "open"]
    excluded = opens[~opens["_provenance"].map(provenance.is_opening_price)]
    assert len(excluded) == 20, "the contaminated set changed; re-check the history"

    for record in _records("csl"):
        assert record["open_provenance"] != provenance.BACKFILLED


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_an_unproven_open_is_measured_but_not_written(league):
    """Rows that fail the proof stay in the history as price observations. They
    just never reach the match table."""
    unproven = [r for r in _records(league) if not r["open_trusted"]]
    assert unproven, f"{league}: no unproven opens, so this proves nothing"
    for record in unproven:
        assert record["open_h"] is None
        assert record["open_d"] is None
        assert record["open_a"] is None


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_a_close_taken_too_far_from_kickoff_is_rejected(league):
    """A price captured hours out is not a close, and storing one as if it were
    poisons every closing-line-value number computed afterwards."""
    for record in _records(league):
        if record["close_h"] is not None:
            assert 0 <= record["close_lead_h"] <= rd.MAX_CLOSE_LEAD_HOURS


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_every_captured_book_now_has_somewhere_to_go(league):
    """Captured data with no column is silent waste.

    Before D5 was applied, CSL was missing nine columns and 147 captured values
    had nowhere to land. Both match tables now carry a column for every book the
    league captures and reduces.
    """
    assert rd.missing_columns(league, _records(league)) == []


def test_reporting_a_missing_column_still_works(tmp_path):
    """The report is what made the waste visible, so it keeps a test of its own."""
    target = _copy_matches("ligamx", tmp_path)
    trimmed = pd.read_csv(target).drop(columns=["duel_open_h"])
    trimmed.to_csv(target, index=False)
    stats = rd.merge("ligamx", _records("ligamx"), dry_run=True, matches_path=target)
    assert stats["no_column"] > 0
