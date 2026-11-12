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


def _run(tmp_path, matches, monkeypatch, *, rows=None, **kwargs):
    config = load_league("ligamx")
    path = _history(tmp_path, rows if rows is not None else [_base_row()])
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


# --------------------------------------------------------------------------- #
# only a final score is a result
# --------------------------------------------------------------------------- #

def _future():
    return _match(
        kickoff=datetime.now(timezone.utc) + timedelta(days=3),
        home="Toluca", away="Leon", home_goals=None, away_goals=None,
    )


def _unscored_row():
    return _base_row(HG=None, AG=None, Res=None)


def _with_unscored():
    """A played row first, as in every real history, so the result columns load
    with the dtypes they really have."""
    return [_base_row(Home="Atlas", Away="Tigres UANL", Date="2026-03-01"), _unscored_row()]


def test_a_score_from_a_match_still_in_play_is_not_recorded(tmp_path, monkeypatch):
    """Seven CSL rows were written 6-70 minutes after kickoff, and since a stored
    result is never revisited each kept its live score for good."""
    live = _match(home_goals=0, away_goals=0, finished=False)
    stats, path = _run(tmp_path, [live, _future()], monkeypatch, rows=_with_unscored())
    row = pd.read_csv(path).iloc[1]
    assert stats["not_final"] == 1 and stats["updated"] == 0
    assert pd.isna(row["HG"]) and pd.isna(row["Res"])


def test_the_final_score_is_recorded_once_the_provider_says_finished(tmp_path, monkeypatch):
    done = _match(home_goals=0, away_goals=2, finished=True)
    stats, path = _run(tmp_path, [done, _future()], monkeypatch, rows=_with_unscored())
    row = pd.read_csv(path).iloc[1]
    assert stats["updated"] == 1
    assert (row["HG"], row["AG"], row["Res"]) == (0.0, 2.0, "A")


def test_a_live_score_is_not_added_as_a_new_match_either(tmp_path, monkeypatch):
    live = _match(home="Atlas", away="Tigres UANL", home_goals=1, away_goals=0,
                  finished=False)
    stats, path = _run(tmp_path, [live, _future()], monkeypatch)
    assert stats["added"] == 0 and len(pd.read_csv(path)) == 1


def test_without_a_status_a_score_waits_until_well_after_kickoff(tmp_path, monkeypatch):
    kickoff = datetime(2026, 4, 22, 3, 5, tzinfo=timezone.utc)
    scored = _match(home_goals=1, away_goals=0, finished=None)

    early = kickoff + timedelta(minutes=40)
    stats, _ = _run(tmp_path, [scored, _future()], monkeypatch,
                    rows=_with_unscored(), now=early)
    assert stats["updated"] == 0 and stats["not_final"] == 1

    late = kickoff + sync.RESULT_SETTLE
    stats, path = _run(tmp_path, [scored, _future()], monkeypatch,
                       rows=_with_unscored(), now=late)
    assert stats["updated"] == 1
    assert pd.read_csv(path).iloc[1]["Res"] == "H"


@pytest.mark.parametrize("status, expected", [
    ("Match Finished", True), ("FT", True), ("AET", True), ("PEN", True),
    ("finished", True),
    ("1H", False), ("HT", False), ("2H", False), ("Not Started", False),
    ("inprogress", False), ("notstarted", False), ("postponed", False),
    (None, None), ("", None), ("something new", None),
])
def test_provider_statuses_are_read_for_both_providers(status, expected):
    """TheSportsDB's strStatus and SofaScore's status.type. An unknown spelling
    falls back to kickoff age rather than blocking every result."""
    assert sync.status_finished(status) is expected


def test_the_last_rounds_results_are_written_even_when_nothing_is_upcoming(
    tmp_path, monkeypatch,
):
    """At season end the final round's results arrive in a response with no
    fixture left in it. The guard used to raise before writing anything, so
    those results were thrown away on every run."""
    upcoming_path = tmp_path / "upcoming.csv"
    upcoming_path.write_text("kept\n")
    done = _match(home_goals=3, away_goals=1, finished=True)
    with pytest.raises(RuntimeError, match="no upcoming fixture"):
        _run(tmp_path, [done], monkeypatch, rows=_with_unscored())
    row = pd.read_csv(tmp_path / "matches.csv").iloc[1]
    assert (row["HG"], row["AG"], row["Res"]) == (3.0, 1.0, "H")
    assert upcoming_path.read_text() == "kept\n", "the upcoming list was overwritten"
    assert not (tmp_path / "meta.json").exists(), "the fixtures were stamped as refreshed"


# --------------------------------------------------------------------------- #
# the end of a season
# --------------------------------------------------------------------------- #

def _stored(tmp_path, fixtures):
    """The upcoming list as the previous run left it."""
    rows = [{"Season": "Clausura 2026", "Round": "17", "Date": k.strftime("%Y-%m-%d"),
             "Time": k.strftime("%H:%M"), "Home": h, "Away": a,
             "kickoff_utc": k.strftime("%Y-%m-%dT%H:%M:%SZ")} for h, a, k in fixtures]
    pd.DataFrame(rows, columns=sync.UPCOMING_COLUMNS).to_csv(
        tmp_path / "upcoming.csv", index=False)


LAST_KICKOFF = datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc)
AFTER = LAST_KICKOFF + timedelta(days=4)


def _last_round(**over):
    values = {"home": "Toluca", "away": "Leon", "kickoff": LAST_KICKOFF,
              "home_goals": 1, "away_goals": 1, "finished": True}
    values.update(over)
    return _match(**values)


def test_a_finished_season_empties_the_list_without_failing(tmp_path, monkeypatch):
    """The CSL's last round was 2026-11-08, and from the next day every refresh
    failed on the guard. A stage that fails daily for a whole off-season is one
    whose real failures nobody reads."""
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF)])
    stats, _ = _run(tmp_path, [_match(), _last_round()], monkeypatch, now=AFTER)
    assert stats["season_complete"] == 1 and stats["upcoming"] == 0
    left = pd.read_csv(tmp_path / "upcoming.csv")
    assert list(left.columns) == sync.UPCOMING_COLUMNS and left.empty
    assert (tmp_path / "meta.json").exists()


def test_the_empty_list_reads_back_as_no_fixtures(tmp_path, monkeypatch):
    """Every stage downstream loads this file. Empty must mean nothing to do,
    not a missing column."""
    from betmodel.fixtures.upcoming import load_upcoming

    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF)])
    _run(tmp_path, [_match(), _last_round()], monkeypatch, now=AFTER)
    assert load_upcoming(str(tmp_path / "upcoming.csv")) == []


def test_a_finished_season_stays_finished_on_the_next_run(tmp_path, monkeypatch):
    """The second run finds the empty list the first one wrote. That is the
    decision already made, not a lost schedule."""
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF)])
    _run(tmp_path, [_match(), _last_round()], monkeypatch, now=AFTER)
    stats, _ = _run(tmp_path, [_match(), _last_round()], monkeypatch,
                    now=AFTER + timedelta(days=1))
    assert stats["season_complete"] == 1


def test_an_outage_that_drops_the_remaining_fixtures_is_still_refused(
    tmp_path, monkeypatch,
):
    """The guard's original purpose. The list was waiting on a fixture the
    response does not have, so the schedule went missing; it did not run out."""
    later = LAST_KICKOFF + timedelta(days=7)
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF), ("Atlas", "Puebla", later)])
    with pytest.raises(RuntimeError, match="Atlas v Puebla"):
        _run(tmp_path, [_match(), _last_round()], monkeypatch, now=AFTER)
    assert len(pd.read_csv(tmp_path / "upcoming.csv")) == 2


def test_a_stored_fixture_still_in_the_future_is_not_a_finished_season(
    tmp_path, monkeypatch,
):
    future = AFTER + timedelta(days=2)
    _stored(tmp_path, [("Toluca", "Leon", future)])
    with pytest.raises(RuntimeError, match="not among the finished matches"):
        _run(tmp_path, [_match(), _last_round(kickoff=future)], monkeypatch, now=AFTER)


def test_a_postponed_match_without_a_new_date_keeps_the_season_open(
    tmp_path, monkeypatch,
):
    """Unplayed and in the past: it will be replayed on a date nobody has
    published yet, and that is exactly when the list must not be emptied."""
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF)])
    postponed = _match(home="Atlas", away="Puebla", kickoff=LAST_KICKOFF,
                       home_goals=None, away_goals=None, finished=False)
    with pytest.raises(RuntimeError, match="not final: Atlas v Puebla"):
        _run(tmp_path, [_match(), _last_round(), postponed], monkeypatch, now=AFTER)


def test_a_last_round_still_in_play_is_not_a_finished_season(tmp_path, monkeypatch):
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF)])
    live = _last_round(finished=False)
    with pytest.raises(RuntimeError, match="not final"):
        _run(tmp_path, [_match(), live], monkeypatch,
             now=LAST_KICKOFF + timedelta(minutes=50))


def test_a_rescheduled_last_match_is_matched_within_the_date_tolerance(
    tmp_path, monkeypatch,
):
    """Providers nudge kickoffs; the stored kickoff and the final one need only
    be the same match."""
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF - timedelta(hours=20))])
    stats, _ = _run(tmp_path, [_match(), _last_round()], monkeypatch, now=AFTER)
    assert stats["season_complete"] == 1


def test_a_dry_run_reports_a_finished_season_and_writes_nothing(tmp_path, monkeypatch):
    _stored(tmp_path, [("Toluca", "Leon", LAST_KICKOFF)])
    stats, _ = _run(tmp_path, [_match(), _last_round()], monkeypatch, now=AFTER,
                    dry_run=True)
    assert stats["season_complete"] == 1
    assert len(pd.read_csv(tmp_path / "upcoming.csv")) == 1
