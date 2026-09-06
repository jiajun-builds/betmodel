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


def test_one_row_s_precision_does_not_decide_whether_the_others_parse(tmp_path):
    """The bug that cost Necaxa v Puebla its anchor on 2026-09-06.

    Rows written before `dates.stamp` existed carry microseconds. Read without an
    explicit format, pandas infers one from the first row and coerces the rest to
    NaT -- so the fourteen Pinnacle rows, all of them stamped in the same
    pre-`stamp` tick, vanished together. The engine then saw no unpriced sighting,
    withheld `observed`, treated a genuine opener as absent and published
    `unanchored` on a fixture whose edge cleared the bar.

    Both orderings, because the one that survives is whichever pandas happens to
    see first, and that is not a property to depend on.
    """
    coarse = "2026-08-27T10:00:00Z"
    fine = "2026-08-29T10:01:39.918058Z"
    for name, (first, second) in {
        "coarse first": ((coarse, "X"), (fine, "Y")),
        "fine first": ((fine, "Y"), (coarse, "X")),
    }.items():
        path = str(tmp_path / f"{name}.csv")
        pd.DataFrame(
            [
                {"home_team": "A", "away_team": away, "bookmaker": "pinnacle",
                 "first_seen_unpriced_at": at, "last_seen_unpriced_at": at}
                for at, away in (first, second)
            ]
        ).to_csv(path, index=False)
        assert len(cw.watched_before(path)) == 2, name
        assert len(cw.watched_until(path)) == 2, name


def test_a_value_nothing_can_parse_withholds_a_proof_rather_than_raising(tmp_path):
    """One unreadable row must not take the league's other captures down with it."""
    path = str(tmp_path / "watch.csv")
    pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "bookmaker": "pinnacle",
             "first_seen_unpriced_at": "not a time", "last_seen_unpriced_at": ""},
            {"home_team": "C", "away_team": "D", "bookmaker": "pinnacle",
             "first_seen_unpriced_at": "2026-08-27T10:00:00Z",
             "last_seen_unpriced_at": "2026-08-29T10:00:00Z"},
        ]
    ).to_csv(path, index=False)
    seen = cw.watched_before(path)
    assert ("A", "B", "pinnacle") not in seen
    assert ("C", "D", "pinnacle") in seen
