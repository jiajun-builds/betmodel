"""The round a fixture's identity carries.

From the merge to 2026-10-31 every published signal id carried round 0: the
upcoming-fixtures file names its column ``Round`` and the reader asked for
``round`` and ``Wk``. ``results.json`` built the same fixture's id from the real
round, so no signal ever joined its outcome -- and nothing failed, because two
ids that disagree are just two ids.
"""

from __future__ import annotations

import csv

import pytest

from betmodel.config import load_league
from betmodel.fixtures import sync
from betmodel.fixtures.upcoming import load_upcoming
from betmodel.signals.engine import fixture_id, make_fixture_id


def _write(tmp_path, columns, rows) -> str:
    path = tmp_path / "upcoming_fixtures.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_the_round_is_read_from_the_column_the_fixtures_stage_writes(tmp_path):
    path = _write(tmp_path, sync.UPCOMING_COLUMNS, [{
        "Season": "2026", "Round": "30", "Date": "2026-11-08", "Time": "07:30",
        "Home": "Beijing Guoan", "Away": "Chongqing Tonglianglong",
        "kickoff_utc": "2026-11-08T07:30:00Z",
    }])
    [fixture] = load_upcoming(path)
    assert fixture.round == "30"
    assert fixture_id(load_league("csl"), fixture) == (
        "CSL-2026-30-2026-11-08-beijing-guoan-chongqing-tonglianglong"
    )


@pytest.mark.parametrize("column", ["round", "Wk"])
def test_the_pre_merge_spellings_are_still_accepted(tmp_path, column):
    path = _write(tmp_path, ["Home", "Away", column, "kickoff_utc"], [{
        "Home": "A", "Away": "B", column: "7", "kickoff_utc": "2026-11-08T07:30:00Z",
    }])
    assert load_upcoming(path)[0].round == "7"


def test_a_signal_id_now_joins_the_result_id_for_the_same_match(tmp_path):
    """The join this was broken for. Liga MX on purpose: a 19:00 kickoff there is
    the next UTC day, so the local matchday is part of what has to agree."""
    config = load_league("ligamx")
    path = _write(tmp_path, sync.UPCOMING_COLUMNS, [{
        "Season": "Apertura 2026", "Round": "15", "Date": "2026-10-30",
        "Time": "01:00", "Home": "FC Juarez", "Away": "Queretaro",
        "kickoff_utc": "2026-10-31T01:00:00Z",
    }])
    [fixture] = load_upcoming(path)
    from datetime import date
    # What publish.public.results_payload builds from matches.csv's Round and the
    # kickoff's local date for the same match.
    result_id = make_fixture_id(config.code, config.season, "15", date(2026, 10, 30),
                                "FC Juarez", "Queretaro")
    assert fixture_id(config, fixture) == result_id
