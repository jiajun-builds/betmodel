"""The opener proof, and the gap that used to slip through it.

An unpriced sighting only rules out an earlier price if nobody stopped looking in
between. The original test asked whether we had *ever* seen the book unpriced
before the price we hold, which assumes continuous capture and says nothing when
that assumption breaks. It broke: an exhausted API account refused every Pinnacle
call for two days, and the old test would still have certified whatever was
fetched afterwards as an opening line.
"""

from __future__ import annotations

import pandas as pd
import pytest

from betmodel.odds import capture_watch as cw

KEY = ("A", "B", "pinnacle")
KICKOFF = pd.Timestamp("2026-09-05T12:00:00Z")
HOUR = pd.Timedelta(hours=1)


def _proof(*, seen, last, captured, max_gap=HOUR * 2.5, since=None, horizon=None):
    return cw.opener_proof(
        home="A", away="B", bookmaker="pinnacle",
        captured_at=captured, kickoff=KICKOFF,
        watched={KEY: seen} if seen is not None else {},
        since=since, horizon=horizon or pd.Timedelta(days=21),
        watched_last={KEY: last} if last is not None else {},
        max_gap=max_gap,
    )


def test_a_recent_sighting_proves_the_opener():
    at = pd.Timestamp("2026-08-29T10:00:00Z")
    assert _proof(seen=at, last=at, captured=at + HOUR) == cw.OBSERVED


def test_a_stale_sighting_proves_nothing():
    """The exact shape of the incident: seen unpriced, then nobody looked."""
    at = pd.Timestamp("2026-08-29T10:00:00Z")
    assert _proof(seen=at, last=at, captured=at + pd.Timedelta(days=2)) != cw.OBSERVED


def test_one_missed_tick_does_not_void_the_proof():
    """A runner delay or a cancelled overlapping tick is not an outage."""
    at = pd.Timestamp("2026-08-29T10:00:00Z")
    assert _proof(seen=at, last=at, captured=at + HOUR * 2) == cw.OBSERVED


def test_a_sighting_that_kept_being_refreshed_proves_the_opener():
    """First seen days ago, still confirmed unpriced minutes before the price."""
    first = pd.Timestamp("2026-08-27T10:00:00Z")
    last = pd.Timestamp("2026-08-29T10:00:00Z")
    assert _proof(seen=first, last=last, captured=last + HOUR) == cw.OBSERVED


def test_a_caller_that_passes_no_gap_keeps_the_old_behaviour():
    """Books nobody polls for opens have no cadence to measure a gap against."""
    at = pd.Timestamp("2026-08-29T10:00:00Z")
    assert _proof(seen=at, last=None, captured=at + pd.Timedelta(days=9),
                  max_gap=None) == cw.OBSERVED


def test_a_row_written_before_the_column_existed_withholds_the_proof():
    """Absent is not recent. Failing closed is the whole point."""
    at = pd.Timestamp("2026-08-29T10:00:00Z")
    assert _proof(seen=at, last=None, captured=at + HOUR) != cw.OBSERVED


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #

def test_the_last_sighting_moves_while_the_first_stays_put(tmp_path):
    path = str(tmp_path / "watch.csv")
    cw.record_unpriced([("A", "B", "pinnacle")], path=path,
                       observed_at="2026-08-27T10:00:00Z")
    cw.record_unpriced([("A", "B", "pinnacle")], path=path,
                       observed_at="2026-08-29T10:00:00Z")

    assert cw.watched_before(path)[KEY] == pd.Timestamp("2026-08-27T10:00:00Z")
    assert cw.watched_until(path)[KEY] == pd.Timestamp("2026-08-29T10:00:00Z")


def test_re_sighting_a_pair_does_not_add_a_row(tmp_path):
    """One row per pair, or the file grows by every unpriced fixture every tick."""
    path = str(tmp_path / "watch.csv")
    for stamp in ("2026-08-27T10:00:00Z", "2026-08-28T10:00:00Z", "2026-08-29T10:00:00Z"):
        cw.record_unpriced([("A", "B", "pinnacle")], path=path, observed_at=stamp)
    assert len(cw.load_watch(path)) == 1
