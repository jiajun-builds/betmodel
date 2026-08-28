"""Closing-line capture.

A missed close is unrecoverable at any price, and a price stored as a close that
was not one poisons every closing-line-value number computed afterwards. Both
failure modes are silent, so both get a test.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pandas as pd

from betmodel.config import load_league
from betmodel.odds import capture_close as cc
from betmodel.odds import capture_store

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _fixtures(tmp_path, rows):
    path = tmp_path / "upcoming.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Home", "Away", "kickoff_utc"])
        writer.writeheader()
        for home, away, kickoff in rows:
            writer.writerow({"Home": home, "Away": away,
                             "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z")})
    return str(path)


def _history(tmp_path, rows):
    path = str(tmp_path / "history.csv")
    if rows:
        capture_store.append_snapshots(pd.DataFrame(rows), path=path, snapshot_type="close")
    return path


def _close_row(home, away, book, kickoff, fetched):
    return {
        "event_id": f"{home}{away}", "commence_time": kickoff.isoformat().replace("+00:00", "Z"),
        "api_home_team": home, "api_away_team": away, "home_team": home, "away_team": away,
        "home_odds": "2.1", "draw_odds": "3.3", "away_odds": "3.4",
        "bookmaker": book, "market": "h2h", "regions": "theoddsapi",
        "last_update": fetched.isoformat().replace("+00:00", "Z"),
        "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
    }


def _due(tmp_path, fixtures, history, league="ligamx"):
    return cc.fixtures_in_window(
        league, load_league(league), now=NOW,
        fixtures_path=_fixtures(tmp_path, fixtures),
        history_path=_history(tmp_path, history),
    )


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #

def test_a_fixture_inside_the_window_is_due(tmp_path):
    assert len(_due(tmp_path, [("A", "B", NOW + timedelta(minutes=10))], [])) == 1


def test_a_fixture_further_out_than_the_window_is_not(tmp_path):
    """The window is a budget decision: every extra minute is another tick's
    worth of the monthly allowance for every fixture."""
    window = load_league("ligamx").odds.close.window_minutes
    assert _due(tmp_path, [("A", "B", NOW + timedelta(minutes=window + 5))], []) == []


def test_a_kicked_off_fixture_is_not_due(tmp_path):
    assert _due(tmp_path, [("A", "B", NOW - timedelta(minutes=1))], []) == []


def test_the_window_is_configuration_and_now_the_same_for_both():
    """It used to differ, because one league's trigger was unreliable and needed
    the wider window. That reason is gone: both fire from the same Worker on the
    same cadence, so both take the tight window. The coupling between the two is
    guarded in tests/unit/test_orchestration.py, which is where the trigger
    cadence lives."""
    windows = {load_league(l).odds.close.window_minutes for l in ("csl", "ligamx")}
    assert windows == {15.0}


# --------------------------------------------------------------------------- #
# finalisation stops the spending
# --------------------------------------------------------------------------- #

def test_a_fixture_whose_close_landed_in_the_target_band_stops_spending(tmp_path):
    kickoff = NOW + timedelta(minutes=8)
    landed = kickoff - timedelta(minutes=5)  # inside the 10-minute target band
    assert _due(
        tmp_path, [("A", "B", kickoff)],
        [_close_row("A", "B", "pinnacle", kickoff, landed)],
    ) == []


def test_a_close_from_the_wrong_book_does_not_finalise_a_fixture(tmp_path):
    """Only the first book in close.books decides. A fixture is done when the
    book that matters has closed, not when any book has."""
    kickoff = NOW + timedelta(minutes=8)
    landed = kickoff - timedelta(minutes=5)
    due = _due(tmp_path, [("A", "B", kickoff)],
               [_close_row("A", "B", "matchbook", kickoff, landed)])
    assert len(due) == 1


def test_an_early_close_does_not_finalise_and_the_fixture_is_recaptured(tmp_path):
    """Re-capturing until the target band keeps the latest price, not the first."""
    kickoff = NOW + timedelta(minutes=15)
    early = kickoff - timedelta(minutes=45)  # outside the target band
    due = _due(tmp_path, [("A", "B", kickoff)],
               [_close_row("A", "B", "pinnacle", kickoff, early)])
    assert len(due) == 1


# --------------------------------------------------------------------------- #
# the discard
# --------------------------------------------------------------------------- #

def test_prices_for_fixtures_outside_the_window_are_discarded():
    """One call returns the whole slate. Storing a price captured hours before
    kickoff as a close once put thirteen lines a median thirty-five hours out
    into the closing columns, at a 105.5% overround against 103.4% for real ones.
    """
    events = [{
        "id": "e1", "commence_time": "2026-08-28T12:10:00Z",
        "home_team": "Beijing FC", "away_team": "Shanghai Port",
        "bookmakers": [{"key": "pinnacle", "markets": [{
            "key": "h2h", "last_update": "2026-08-28T11:55:00Z",
            "outcomes": [{"name": "Beijing FC", "price": 2.1},
                         {"name": "Shanghai Port", "price": 3.4},
                         {"name": "Draw", "price": 3.3}]}]}],
    }]
    wanted_none = cc.extract_rows("csl", events, set(), "2026-08-28T11:55:00Z")
    assert wanted_none == [], "a fixture we are not in-window for must be dropped"

    wanted = {("Beijing Guoan", "Shanghai Port")}
    rows = cc.extract_rows("csl", events, wanted, "2026-08-28T11:55:00Z")
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Beijing Guoan", "names are canonicalised"


def test_a_book_quoting_only_two_ways_is_not_a_1x2_close():
    events = [{
        "id": "e1", "commence_time": "2026-08-28T12:10:00Z",
        "home_team": "Beijing FC", "away_team": "Shanghai Port",
        "bookmakers": [{"key": "pinnacle", "markets": [{
            "key": "h2h", "outcomes": [{"name": "Beijing FC", "price": 1.5},
                                       {"name": "Shanghai Port", "price": 2.5}]}]}],
    }]
    assert cc.extract_rows(
        "csl", events, {("Beijing Guoan", "Shanghai Port")}, "2026-08-28T11:55:00Z"
    ) == []
