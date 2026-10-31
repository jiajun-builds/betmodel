"""Team-name mapping.

Four upstreams spell the same club four ways. The failures this guards against
are silent ones: a price filed under the wrong club, or a request that asks for a
name the provider has never heard of.
"""

from __future__ import annotations

import pytest

from betmodel import teams


# --------------------------------------------------------------------------- #
# the shipped mappings
# --------------------------------------------------------------------------- #

def test_both_leagues_have_the_same_namespace_columns():
    """A new league should not invent its own column names."""
    csl = teams.for_league("csl")
    mex = teams.for_league("ligamx")
    for ns in ("match_team", "sofascore_team", "theoddsapi_team", "oddsapiio_team"):
        assert ns in csl.namespaces, ns
        assert ns in mex.namespaces, ns


@pytest.mark.parametrize(
    "standard,spelling",
    [
        # The ones a hand transcription got wrong, which is why they are pinned.
        ("Leon", "Club Leon"),
        ("Necaxa", "Club Necaxa"),
        ("Puebla", "Club Puebla"),
        ("Santos Laguna", "Club Santos Laguna"),
        ("Tijuana", "Club Tijuana de Caliente"),
        ("Toluca", "Deportivo Toluca FC"),
        ("UNAM Pumas", "Pumas UNAM"),
    ],
)
def test_oddsapiio_spellings_are_exact(standard, spelling):
    """An outgoing request uses this string verbatim; a near miss is refused."""
    assert teams.for_league("ligamx").spelling(standard, "oddsapiio_team") == spelling


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Club América", "Club America"),
        ("Club America", "Club America"),
        ("León", "Leon"),
        ("Querétaro", "Queretaro"),
        ("FC Juárez", "FC Juarez"),
        ("MAZATLÁN", "Mazatlan"),      # never listed in the old accent dictionary
        ("  santos   laguna  ", "Santos Laguna"),
    ],
)
def test_accents_and_case_fold_at_lookup(name, expected):
    """Replaces a dictionary that had to list both spellings of every club.

    That approach worked until an upstream added a club nobody remembered to
    double-enter. Folding covers the pairs that were never written down.
    """
    assert teams.for_league("ligamx").to_standard(name) == expected


def test_an_unknown_name_is_none_not_a_guess():
    assert teams.for_league("csl").to_standard("Nottingham Forest") is None


def test_require_standard_names_the_file_to_edit():
    """An unmapped club must stop the run: dropping the row loses a capture that
    cannot be taken again, and guessing files it under the wrong match."""
    with pytest.raises(teams.TeamMappingError, match="team_name_mapping.csv"):
        teams.for_league("csl").require_standard("Nottingham Forest", source="odds feed")


# --------------------------------------------------------------------------- #
# a dropped name says so, once
# --------------------------------------------------------------------------- #

def _unmapped_lines(caplog, name):
    return [r.getMessage() for r in caplog.records
            if r.levelname == "WARNING" and name in r.getMessage()]


def test_an_unmapped_name_is_reported_once_per_name_and_source(monkeypatch, caplog):
    """Once, because a name fails on every row it appears in -- every book of
    every event -- and a warning repeated per row is one nobody reads."""
    monkeypatch.setattr(teams, "_reported_unmapped", set(), raising=False)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            teams.report_unmapped("csl", "odds-api.io event listing", "Wuhan Yangtze")
        teams.report_unmapped("csl", "odds-api.io event listing", "wuhan  yangtze")
        teams.report_unmapped("csl", "SofaScore xG events", "Wuhan Yangtze")
    lines = _unmapped_lines(caplog, "Wuhan Yangtze")
    assert len(lines) == 2, lines
    assert "odds-api.io event listing" in lines[0]
    assert "SofaScore xG events" in lines[1]


def test_an_odds_listing_that_mostly_matches_still_names_the_club_it_dropped(
        monkeypatch, caplog):
    """The nothing-matched warning named unmapped clubs; a listing where every
    other fixture matched dropped one club's matches without a word."""
    from datetime import datetime, timedelta, timezone

    from betmodel.config import load_league
    from betmodel.fixtures.upcoming import Fixture
    from betmodel.odds import capture_open as co
    from betmodel.providers import oddsapiio

    class _Client:
        requests = 0

        def upcoming_events(self, _days):
            self.requests += 1
            return [{"id": "1", "home": "Wuhan Three Towns",
                     "away": "Henan Songshan Longmen"},
                    {"id": "2", "home": "Shanghai Port", "away": "Qingdao Hainiu FC II"}]

        def iter_batches(self, ids):
            yield list(ids)

        def multi_odds(self, ids, books):
            self.requests += 1
            return []

    monkeypatch.setattr(teams, "_reported_unmapped", set(), raising=False)
    monkeypatch.setattr(oddsapiio, "OddsApiIoClient", lambda *a, **k: _Client())
    config = load_league("csl")
    kickoff = datetime(2026, 9, 3, 11, 0, tzinfo=timezone.utc)
    pending = [co.Pending(Fixture(home="Wuhan Three Towns", away="Henan Songshan Longmen",
                                  round="1", kickoff=kickoff),
                          co.books_for(config, "oddsapiio"))]
    with caplog.at_level("WARNING"):
        co._capture_oddsapiio("csl", config, pending,
                              now=kickoff - timedelta(days=6), dry_run=False)
    lines = _unmapped_lines(caplog, "Qingdao Hainiu FC II")
    assert len(lines) == 1 and "odds-api.io" in lines[0]
    assert not _unmapped_lines(caplog, "Shanghai Port"), "only the side that failed"


def test_a_close_slate_names_an_unmapped_club_once_not_once_per_book(monkeypatch, caplog):
    from betmodel.config import load_league
    from betmodel.odds import capture_close as cc

    def _book(key):
        return {"key": key, "markets": [{"key": "h2h", "outcomes": [
            {"name": "Beijing FC", "price": 2.1}, {"name": "Draw", "price": 3.3},
            {"name": "Nowhere United", "price": 3.4}]}]}

    events = [{"id": "e1", "commence_time": "2026-08-28T12:10:00Z",
               "home_team": "Beijing FC", "away_team": "Nowhere United",
               "bookmakers": [_book("pinnacle"), _book("betfair_ex_eu"),
                              _book("marathonbet")]}]
    monkeypatch.setattr(teams, "_reported_unmapped", set(), raising=False)
    with caplog.at_level("WARNING"):
        rows = cc.extract_rows("csl", load_league("csl"), events,
                               {("Beijing Guoan", "Nowhere United", "2026-08-28")},
                               "2026-08-28T11:55:00Z")
    assert rows == [], "which rows are dropped is unchanged"
    lines = _unmapped_lines(caplog, "Nowhere United")
    assert len(lines) == 1 and "close" in lines[0], lines


def test_an_xg_event_dropped_for_its_name_says_so(monkeypatch, caplog):
    """A dropped xG event leaves the match's xG blank, which reads as "not
    published yet", and its blended target blank, which takes it out of the fit."""
    from betmodel.config import load_league
    from betmodel.providers import sofascore
    from betmodel.xg import sync

    def _event(i, home, away):
        return {"id": i, "homeTeam": {"name": home}, "awayTeam": {"name": away},
                "startTimestamp": 1_788_000_000 + i}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def current_season_id(self, _tournament):
            return 1

        def season_events(self, _tournament, _season, _kind):
            return [_event(1, "Shanghai Port", "Ghost City FC"),
                    _event(2, "Ghost City FC", "Shanghai Port"),
                    _event(3, "Shanghai Port", "Beijing Guoan")]

    monkeypatch.setattr(teams, "_reported_unmapped", set(), raising=False)
    monkeypatch.setattr(sofascore, "SofascoreClient", _Client)
    with caplog.at_level("WARNING"):
        _, events = sync._events(load_league("csl"))
    assert [e["id"] for e in events] == [3], "which events are dropped is unchanged"
    lines = _unmapped_lines(caplog, "Ghost City FC")
    assert len(lines) == 1 and "SofaScore" in lines[0], lines


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #

def _write(tmp_path, text):
    p = tmp_path / "team_name_mapping.csv"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_two_clubs_claiming_one_spelling_is_fatal(tmp_path):
    """The corruption this exists to prevent.

    If two canonical clubs both answer to one upstream spelling, every price for
    one silently lands on the other. No error, no empty column, wrong money.
    """
    path = _write(tmp_path, "theoddsapi_team,standard_team\nAtletico,Atletico Madrid\nAtletico,Atletico Bilbao\n")
    with pytest.raises(teams.TeamMappingError, match="claimed by both|repeats canonical"):
        teams.load(path, league="test")


def test_an_accent_only_collision_is_also_caught(tmp_path):
    """Folding makes these meet, so the check has to see them as one."""
    path = _write(tmp_path, "match_team,standard_team\nLeón,Club A\nLeon,Club B\n")
    with pytest.raises(teams.TeamMappingError, match="claimed by both"):
        teams.load(path, league="test")


def test_a_missing_canonical_column_is_fatal(tmp_path):
    path = _write(tmp_path, "match_team,odds_team\nA,B\n")
    with pytest.raises(teams.TeamMappingError, match="standard_team"):
        teams.load(path, league="test")


def test_an_empty_file_is_fatal_not_an_empty_mapping(tmp_path):
    """The pre-merge loader returned empty dictionaries on any failure, which
    looks like working software until two upstreams disagree."""
    path = _write(tmp_path, "match_team,standard_team\n")
    with pytest.raises(teams.TeamMappingError, match="empty"):
        teams.load(path, league="test")


def test_a_missing_file_is_fatal(tmp_path):
    with pytest.raises(teams.TeamMappingError, match="no team mapping"):
        teams.load(str(tmp_path / "nope.csv"), league="test")


def test_asking_for_an_absent_namespace_is_an_error_not_none(tmp_path):
    path = _write(tmp_path, "match_team,standard_team\nA,Alpha\n")
    m = teams.load(path, league="test")
    with pytest.raises(teams.TeamMappingError, match="no namespace"):
        m.spelling("Alpha", "oddsapiio_team")


def test_a_fixture_dropped_for_its_name_says_so(monkeypatch, caplog):
    """The fixtures stage drops a club it cannot name from both the history and
    the upcoming list, which looks exactly like a match not yet published."""
    from types import SimpleNamespace

    from betmodel.fixtures import sync

    monkeypatch.setattr(teams, "_reported_unmapped", set(), raising=False)
    config = SimpleNamespace(id="csl")
    with caplog.at_level("WARNING"):
        sync._report_unmapped(config, "TheSportsDB events",
                              ("Wuhan Yangtze", None), ("Shanghai Port", "Shanghai Port"))
        sync._report_unmapped(config, "TheSportsDB events", ("", None))
    lines = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(lines) == 1 and "Wuhan Yangtze" in lines[0]
