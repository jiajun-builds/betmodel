"""The two tracked, irreproducible stores.

An opening line exists only while the book shows it and no provider sells opener
history, so a bug here loses data permanently rather than causing a rerun.
"""

from __future__ import annotations

import pandas as pd
import pytest

from betmodel.odds import capture_store as cs
from betmodel.odds import capture_watch as cw


def _row(**over):
    base = {
        "event_id": "e1", "commence_time": "2026-09-01T12:00:00Z",
        "api_home_team": "A FC", "api_away_team": "B FC",
        "home_team": "A", "away_team": "B",
        "home_odds": "2.10", "draw_odds": "3.30", "away_odds": "3.40",
        "bookmaker": "duel", "market": "h2h", "regions": "oddsapiio",
        "last_update": "2026-08-20T10:00:00Z", "fetched_at": "2026-08-20T10:00:05Z",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# capture store
# --------------------------------------------------------------------------- #

def test_both_shipped_histories_use_the_shared_schema():
    """The merge's luckiest fact: no conversion was needed to combine them."""
    for league in ("csl", "ligamx"):
        history = cs.load_history(cs.history_path(league))
        assert list(history.columns) == cs.HISTORY_COLUMNS


def test_an_unmoved_line_is_not_appended_twice(tmp_path):
    """last_update is the book's own 'the line moved' stamp, so re-polling an
    unmoved price must be a no-op rather than a duplicate row."""
    path = str(tmp_path / "history.csv")
    rows = pd.DataFrame([_row()])
    _, first = cs.append_snapshots(rows, path=path, snapshot_type="open")
    # A later poll: we fetched again, the book did not move.
    _, second = cs.append_snapshots(
        pd.DataFrame([_row(fetched_at="2026-08-20T10:30:00Z")]),
        path=path, snapshot_type="open",
    )
    assert (first, second) == (1, 0)


def test_a_moved_line_is_appended(tmp_path):
    path = str(tmp_path / "history.csv")
    cs.append_snapshots(pd.DataFrame([_row()]), path=path, snapshot_type="open")
    _, appended = cs.append_snapshots(
        pd.DataFrame([_row(last_update="2026-08-20T11:00:00Z", home_odds="2.05")]),
        path=path, snapshot_type="open",
    )
    assert appended == 1


def test_the_same_price_can_be_both_an_open_and_a_close(tmp_path):
    """snapshot_type is in the key on purpose: the two answer different questions."""
    path = str(tmp_path / "history.csv")
    cs.append_snapshots(pd.DataFrame([_row()]), path=path, snapshot_type="open")
    _, appended = cs.append_snapshots(
        pd.DataFrame([_row()]), path=path, snapshot_type="close"
    )
    assert appended == 1


def test_an_idle_tick_leaves_the_file_untouched(tmp_path):
    """The republish job is gated on whether a commit appended anything, so an
    idle tick that rewrote the file would trigger a pointless rebuild."""
    path = str(tmp_path / "history.csv")
    cs.append_snapshots(pd.DataFrame([_row()]), path=path, snapshot_type="open")
    before = open(path, "rb").read()
    cs.append_snapshots(pd.DataFrame([_row()]), path=path, snapshot_type="open")
    assert open(path, "rb").read() == before


def test_an_unknown_snapshot_type_is_refused(tmp_path):
    with pytest.raises(ValueError, match="snapshot_type"):
        cs.append_snapshots(
            pd.DataFrame([_row()]), path=str(tmp_path / "h.csv"), snapshot_type="guess"
        )


def test_values_are_stored_as_written(tmp_path):
    """Numeric coercion on read would reformat the key and defeat dedup."""
    path = str(tmp_path / "history.csv")
    cs.append_snapshots(
        pd.DataFrame([_row(event_id="0012340", home_odds="2.10")]),
        path=path, snapshot_type="open",
    )
    stored = cs.load_history(path).iloc[0]
    assert stored["event_id"] == "0012340", "a leading zero must survive"
    assert stored["home_odds"] == "2.10", "a trailing zero must survive"


# --------------------------------------------------------------------------- #
# capture watch
# --------------------------------------------------------------------------- #

def test_only_the_first_unpriced_sighting_is_kept(tmp_path):
    """That one bounds how early we were watching. Keeping every tick would grow
    the file by every unpriced fixture forever and prove nothing extra."""
    path = str(tmp_path / "watch.csv")
    first = cw.record_unpriced(
        [("A", "B", "duel")], path=path, observed_at="2026-08-20T09:00:00Z"
    )
    second = cw.record_unpriced(
        [("A", "B", "duel")], path=path, observed_at="2026-08-20T10:00:00Z"
    )
    assert (first, second) == (1, 0)
    assert cw.watched_before(path)[("A", "B", "duel")].isoformat().startswith("2026-08-20T09")


def test_repeats_within_one_batch_collapse(tmp_path):
    path = str(tmp_path / "watch.csv")
    added = cw.record_unpriced(
        [("A", "B", "duel"), ("A", "B", "duel")], path=path, observed_at="2026-08-20T09:00:00Z"
    )
    assert added == 1


def test_seeing_a_fixture_unpriced_proves_the_next_price_is_the_opener():
    """The strong proof, and the only one that covers a fixture already inside
    the lookahead when capture began."""
    watched = {("A", "B", "duel"): pd.Timestamp("2026-08-20T09:00:00Z")}
    proof = cw.opener_proof(
        home="A", away="B", bookmaker="duel",
        captured_at=pd.Timestamp("2026-08-20T10:00:00Z"),
        kickoff=pd.Timestamp("2026-08-25T12:00:00Z"),
        watched=watched, since=pd.Timestamp("2026-08-19T00:00:00Z"),
        horizon=pd.Timedelta(days=14),
    )
    assert proof == cw.OBSERVED


def test_a_fixture_that_became_observable_after_capture_began_gets_the_window_proof():
    proof = cw.opener_proof(
        home="A", away="B", bookmaker="duel",
        captured_at=pd.Timestamp("2026-08-20T10:00:00Z"),
        kickoff=pd.Timestamp("2026-09-30T12:00:00Z"),
        watched={}, since=pd.Timestamp("2026-08-01T00:00:00Z"),
        horizon=pd.Timedelta(days=14),
    )
    assert proof == cw.WINDOW


def test_a_price_from_a_book_that_may_already_have_been_quoting_proves_nothing():
    proof = cw.opener_proof(
        home="A", away="B", bookmaker="duel",
        captured_at=pd.Timestamp("2026-08-20T10:00:00Z"),
        kickoff=pd.Timestamp("2026-08-22T12:00:00Z"),
        watched={}, since=pd.Timestamp("2026-08-19T00:00:00Z"),
        horizon=pd.Timedelta(days=14),
    )
    assert proof == cw.NONE


def test_watching_since_uses_the_whole_history_not_just_the_opens():
    """A close tick is also a moment we looked. Taking the minimum over a subset
    would date the start of watching later than it was."""
    history = pd.DataFrame({
        "fetched_at": ["2026-08-01T00:00:00Z", "2026-08-15T00:00:00Z"],
        "snapshot_type": ["close", "open"],
    })
    assert cw.watching_since(history).isoformat().startswith("2026-08-01")
