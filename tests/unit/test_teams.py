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
