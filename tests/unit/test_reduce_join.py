"""The reducer's join from a captured fixture to its row in the match table.

``Date`` carries whichever timezone wrote the row (D12). From Apertura 2026 it is
Liga MX's local day, so a record labelled by its UTC day missed every kickoff
after 18:00 local, and 17 of 20 September closes never reached the table.
"""

from __future__ import annotations

import pandas as pd

from betmodel import paths
from betmodel.config import load_league
from betmodel.odds import reduce as rd

TZ = "America/Mexico_City"


def _table():
    return pd.DataFrame([
        # 21:00 local on the 4th is 03:00 UTC on the 5th.
        {"Date": "2026-09-04", "Home": "FC Juarez", "Away": "Pachuca",
         "kickoff_utc": "2026-09-05T03:00:00Z"},
        # An older row with no kickoff, written in UTC days.
        {"Date": "2024/03/10", "Home": "FC Juarez", "Away": "Pachuca",
         "kickoff_utc": None},
    ])


def test_an_evening_kickoff_is_found_by_its_local_matchday():
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2026/09/05",
                        "2026-09-04", TZ) == 0


def test_without_the_matchday_it_is_missed_as_before():
    """What the reducer did until now, kept as the fallback."""
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2026/09/05") is None


def test_a_different_matchday_is_not_matched():
    """A rescheduled meeting must not be filed under the replacement's row."""
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2026/09/03",
                        "2026-09-02", TZ) is None


def test_a_row_without_a_kickoff_still_matches_on_its_date():
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2024/03/10",
                        "2024-03-09", TZ) == 1


# --------------------------------------------------------------------------- #
# the proof the table trusts
# --------------------------------------------------------------------------- #

CAPTURED = "2026-08-12T09:03:11Z"


def _proof_files(tmp_path, monkeypatch, last_unpriced):
    history = tmp_path / "history.csv"
    pd.DataFrame([{
        "event_id": "oddsapiio:1", "commence_time": "2026-08-18T01:00:00Z",
        "api_home_team": "Club Necaxa", "api_away_team": "Club Leon",
        "home_team": "Necaxa", "away_team": "Leon",
        "home_odds": 2.24, "draw_odds": 3.60, "away_odds": 3.15,
        "bookmaker": "duel", "market": "h2h", "regions": "oddsapiio",
        "last_update": "2026-08-12T07:54:29.495Z", "fetched_at": CAPTURED,
        "snapshot_type": "open", "target_round": 4,
        "capture_reason": f"odds-api.io first-seen open price @ {CAPTURED}",
    }]).to_csv(history, index=False)
    watch = tmp_path / "watch.csv"
    pd.DataFrame([{
        "home_team": "Necaxa", "away_team": "Leon", "bookmaker": "duel",
        "first_seen_unpriced_at": "2026-08-09T09:00:00Z",
        "last_seen_unpriced_at": last_unpriced,
    }]).to_csv(watch, index=False)

    class Redirect:
        capture_history_csv = str(history)
        capture_watch_csv = str(watch)

    monkeypatch.setattr(paths, "for_league", lambda league: Redirect())
    return rd.build_records("ligamx", load_league("ligamx"))


def test_the_table_refuses_an_open_whose_last_unpriced_sighting_is_stale(
    tmp_path, monkeypatch,
):
    """The same gap test the engine applies. Without it nine Liga MX duel opens
    the engine refused were trusted here, and seven reached matches.csv."""
    (record,) = _proof_files(tmp_path, monkeypatch, "2026-08-10T09:00:00Z")
    assert not record["open_trusted"] and record["open_h"] is None


def test_the_table_trusts_an_open_watched_right_up_to_its_capture(
    tmp_path, monkeypatch,
):
    (record,) = _proof_files(tmp_path, monkeypatch, "2026-08-12T08:55:00Z")
    assert record["open_proof"] == "observed" and record["open_h"] == 2.24
