"""Fixture sync.

Every test here guards a way this stage can destroy data rather than merely fail.
It is the only stage that writes to the match history, which several seasons of
research and every model fit read.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from betmodel import paths
from betmodel.config import load_league
from betmodel.fixtures import sync


def _history(tmp_path, rows) -> str:
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _base_row(**over):
    row = {
        "Country": "Mexico", "League": "Liga MX", "Season": "Clausura 2026",
        "Round": 15, "Date": "2026-04-22", "Time": "04:05",
        "Home": "Monterrey", "Away": "Puebla",
        "HG": 2.0, "AG": 1.0, "Res": "H",
    }
    row.update(over)
    return row


def _match(**over):
    values = {
        "season": "Clausura 2026", "round": "15",
        # 03:05Z is 21:05 the previous day in Mexico City, and 04:05 in UK summer
        # time. All three spellings of this one kickoff exist in the history.
        "kickoff": datetime(2026, 4, 22, 3, 5, tzinfo=timezone.utc),
        "home": "Monterrey", "away": "Puebla", "home_goals": 2, "away_goals": 1,
    }
    values.update(over)
    return sync.Match(**values)


def _run(tmp_path, matches, monkeypatch, **kwargs):
    config = load_league("ligamx")
    path = _history(tmp_path, [_base_row()])
    monkeypatch.setattr(sync, "fetch", lambda cfg: matches)

    class Redirect:
        matches_csv = path
        upcoming_fixtures_csv = str(tmp_path / "upcoming.csv")
        fixtures_meta_json = str(tmp_path / "meta.json")

        def ensure_dirs(self):
            pass

    monkeypatch.setattr(paths, "for_league", lambda league: Redirect())
    monkeypatch.setattr(sync.paths, "for_league", lambda league: Redirect())
    stats = sync.sync("ligamx", config, **kwargs)
    return stats, path


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #

def test_a_row_stored_in_another_timezone_is_still_recognised(tmp_path, monkeypatch):
    """The bug this exists to prevent.

    One history carries league-local, UK-local and UTC dates depending on which
    source backfilled each row. Keying on the date reported 98 additions where
    the answer was none, which would have duplicated a third of a season.
    """
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    stats, _ = _run(tmp_path, [_match(), upcoming], monkeypatch)
    assert stats["added"] == 0, "the UK-stamped row was not recognised"


def test_a_genuinely_new_pairing_is_added(tmp_path, monkeypatch):
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    new = _match(home="Atlas", away="Tigres UANL",
                 kickoff=datetime(2026, 4, 25, 3, 0, tzinfo=timezone.utc))
    stats, _ = _run(tmp_path, [_match(), new, upcoming], monkeypatch)
    assert stats["added"] == 1


def test_a_pairing_far_outside_the_tolerance_is_a_different_match(tmp_path, monkeypatch):
    """The same clubs meet twice a season. A month apart is not the same match."""
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    reverse = _match(kickoff=datetime(2026, 2, 22, 3, 5, tzinfo=timezone.utc))
    stats, _ = _run(tmp_path, [reverse, upcoming], monkeypatch)
    assert stats["added"] == 1


# --------------------------------------------------------------------------- #
# never destroy
# --------------------------------------------------------------------------- #

def test_the_explicit_kickoff_is_recorded_so_the_ambiguity_stops_here(tmp_path, monkeypatch):
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    stats, path = _run(tmp_path, [_match(), upcoming], monkeypatch)
    assert stats["stamped"] == 1
    stored = pd.read_csv(path)[sync.KICKOFF_COLUMN].iloc[0]
    assert stored == "2026-04-22T03:05:00Z"


def test_the_original_date_and_time_are_left_alone(tmp_path, monkeypatch):
    """Superseded, not corrected: several seasons of research reference them, and
    rewriting them would silently move rows other work is keyed to."""
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    _, path = _run(tmp_path, [_match(), upcoming], monkeypatch)
    row = pd.read_csv(path).iloc[0]
    assert row["Date"] == "2026-04-22" and row["Time"] == "04:05"


def test_a_result_already_recorded_is_never_overwritten(tmp_path, monkeypatch):
    """Forward-only. A provider that briefly disagrees must not rewrite history."""
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    _, path = _run(tmp_path, [_match(home_goals=9, away_goals=9), upcoming], monkeypatch)
    row = pd.read_csv(path).iloc[0]
    assert (row["HG"], row["AG"], row["Res"]) == (2.0, 1.0, "H")


def test_no_row_is_ever_removed(tmp_path, monkeypatch):
    """An upstream that briefly forgets a season would otherwise delete years of
    results, and the next model fit would train on the remains."""
    upcoming = _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )
    _, path = _run(tmp_path, [upcoming], monkeypatch)
    assert len(pd.read_csv(path)) == 1


def test_an_empty_upcoming_list_is_refused(tmp_path, monkeypatch):
    """Every timing decision downstream reads that file, so an outage that
    emptied it would stop opening-line capture silently."""
    with pytest.raises(RuntimeError, match="no upcoming fixture"):
        _run(tmp_path, [_match()], monkeypatch)


def test_an_empty_upcoming_list_can_be_allowed_at_season_end(tmp_path, monkeypatch):
    stats, _ = _run(tmp_path, [_match()], monkeypatch, allow_empty_upcoming=True)
    assert stats["upcoming"] == 0


def test_a_provider_returning_nothing_leaves_every_file_alone(tmp_path, monkeypatch):
    stats, path = _run(tmp_path, [], monkeypatch)
    assert stats == {"fetched": 0, "updated": 0, "added": 0, "upcoming": 0}
    assert len(pd.read_csv(path)) == 1
