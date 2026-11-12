"""The League One sync in the xG stage (D42).

It owns its file outright, so what is pinned is what would corrupt it or cost the
primary something: a rerun that duplicates, a revision that is never picked up, a
past season dropped because it is no longer listed, a placeholder zero stored as
a measurement, and a provider failure that takes the stage -- and the CSL's
freshly synced xG -- down with it.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from betmodel import cli, paths
from betmodel.config import load_league
from betmodel.models.auxiliary import COLUMNS
from betmodel.providers.sofascore import SofascoreUnavailable
from betmodel.xg import auxiliary_sync

SOURCE = load_league("csl").model.auxiliary[0]


def _event(event_id, day, home="Shijiazhuang Gongfu", away="Dalian Yingbo FC",
           goals=(2, 1), status="finished"):
    start = datetime(2026, 10, day, 7, 30, tzinfo=timezone.utc)
    return {
        "id": event_id, "startTimestamp": int(start.timestamp()),
        "status": {"type": status}, "roundInfo": {"round": day},
        "season": {"year": "2026"},
        "homeTeam": {"name": home, "id": 100 + event_id},
        "awayTeam": {"name": away, "id": 200 + event_id},
        "homeScore": {"normaltime": goals[0]}, "awayScore": {"normaltime": goals[1]},
    }


class FakeClient:
    def __init__(self, events, xg, fail=None):
        self.events, self.xg, self.fail = events, dict(xg), fail
        self.fetched: list[int] = []

    def current_season_id(self, tournament):
        assert tournament == 782
        if self.fail:
            raise self.fail
        return 90905

    def season_events(self, tournament, season_id, kind):
        assert (season_id, kind) == (90905, "last")
        return list(self.events)

    def event_xg(self, event_id):
        self.fetched.append(event_id)
        return self.xg.get(event_id, (None, None))


@pytest.fixture
def league_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "data_dir", lambda: str(tmp_path))
    (tmp_path / "csl").mkdir()
    return tmp_path / "csl"


def _run(client, **kwargs):
    return auxiliary_sync.sync("csl", load_league("csl"), SOURCE, client=client, **kwargs)


def _table(league_dir):
    return pd.read_csv(league_dir / "league_one_matches.csv", dtype=str, keep_default_na=False)


def test_a_rerun_adds_nothing_and_rewrites_nothing(league_dir):
    client = FakeClient([_event(1, 1), _event(2, 2)], {1: (1.5, 0.5), 2: (0.9, 1.1)})
    first = _run(client)
    assert first["added"] == 2 and first["written"] == 2
    table = _table(league_dir)
    assert list(table.columns) == list(COLUMNS)
    assert table["event_id"].tolist() == ["1", "2"]
    assert table.loc[0, ["HG", "AG", "HxG", "AxG", "Season", "Round", "Date", "Time"]].tolist() \
        == ["2", "1", "1.5", "0.5", "2026", "1", "2026-10-01", "07:30"]
    assert table.loc[0, "kickoff_utc"].startswith("2026-10-01T07:30:00")

    path = league_dir / "league_one_matches.csv"
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    again = _run(FakeClient([_event(1, 1), _event(2, 2), _event(2, 2)],
                            {1: (1.5, 0.5), 2: (0.9, 1.1)}))
    assert again["added"] == 0 and again["written"] == 0
    assert path.read_bytes() == before and path.stat().st_mtime_ns == stamp


def test_a_revision_inside_the_review_window_is_picked_up_and_older_rows_are_left(league_dir):
    _run(FakeClient([_event(1, 1), _event(2, 20)], {1: (1.5, 0.5), 2: (0.9, 1.1)}))
    client = FakeClient([_event(1, 1), _event(2, 20)], {1: (9.9, 9.9), 2: (1.3, 1.1)})
    stats = _run(client)
    assert client.fetched == [2], "a match outside the review window was refetched"
    assert stats["revised"] == 1
    table = _table(league_dir).set_index("event_id")
    assert table.loc["2", ["HxG", "AxG"]].tolist() == ["1.3", "1.1"]
    assert table.loc["1", ["HxG", "AxG"]].tolist() == ["1.5", "0.5"]


def test_rows_no_longer_listed_are_kept_and_the_file_stays_in_kickoff_order(league_dir):
    """Every past season is absent from the current season's listing. Dropping
    what is not listed would delete the backfill on the first run."""
    old = {c: "" for c in COLUMNS} | {
        "event_id": "777", "Season": "2025", "Round": "30", "Date": "2025-11-01",
        "Time": "07:00", "kickoff_utc": "2025-11-01T07:00:00+00:00",
        "home_sofascore": "A", "away_sofascore": "B", "home_team_id": "1",
        "away_team_id": "2", "HG": "0", "AG": "0", "HxG": "0.4", "AxG": "0.3"}
    pd.DataFrame([old]).to_csv(league_dir / "league_one_matches.csv", index=False)
    _run(FakeClient([_event(5, 3), _event(4, 2)], {4: (1.0, 1.0), 5: (2.0, 0.5)}))
    table = _table(league_dir)
    assert table["event_id"].tolist() == ["777", "4", "5"]
    assert table.loc[0, ["HxG", "AxG"]].tolist() == ["0.4", "0.3"]


def test_an_all_zero_xg_is_stored_blank_and_a_stored_placeholder_is_cleared(league_dir):
    _run(FakeClient([_event(1, 1)], {1: (0.0, 0.0)}))
    assert _table(league_dir).loc[0, ["HxG", "AxG"]].tolist() == ["", ""]

    table = _table(league_dir)
    table.loc[0, ["HxG", "AxG"]] = ["0.0", "0.0"]          # stored before it was understood
    table.to_csv(league_dir / "league_one_matches.csv", index=False)
    stats = _run(FakeClient([_event(1, 1)], {1: (0.0, 0.0)}))
    assert stats["cleared"] == 1
    assert _table(league_dir).loc[0, ["HxG", "AxG"]].tolist() == ["", ""]

    _run(FakeClient([_event(1, 1)], {1: (1.2, 0.4)}))
    assert _table(league_dir).loc[0, ["HxG", "AxG"]].tolist() == ["1.2", "0.4"]


def test_an_unfinished_event_is_not_added(league_dir):
    _run(FakeClient([_event(1, 1, status="postponed"), _event(2, 2)], {2: (1.0, 1.0)}))
    assert _table(league_dir)["event_id"].tolist() == ["2"]


def test_a_provider_failure_is_loud_returns_and_touches_nothing(league_dir, caplog):
    _run(FakeClient([_event(1, 1)], {1: (1.5, 0.5)}))
    path = league_dir / "league_one_matches.csv"
    before = path.read_bytes()
    failing = FakeClient([], {}, fail=SofascoreUnavailable("403 from the edge"))
    with caplog.at_level(logging.ERROR, logger="betmodel.xg.auxiliary_sync"):
        stats = _run(failing)
    assert stats["failed"] == 1
    assert "league_one sync failed" in caplog.text and "unchanged" in caplog.text
    assert path.read_bytes() == before


def test_dry_run_and_limit_are_honoured(league_dir):
    client = FakeClient([_event(1, 1), _event(2, 2), _event(3, 3)],
                        {1: (1.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 1.0)})
    _run(client, dry_run=True)
    assert not (league_dir / "league_one_matches.csv").exists()
    limited = FakeClient(client.events, client.xg)
    stats = _run(limited, limit=2)
    assert stats["fetched"] == 2 and len(limited.fetched) == 2
    assert stats["added"] == 3      # rows are still recorded; their xG comes next run


def test_the_xg_stage_survives_a_failed_auxiliary_and_keeps_the_primary(monkeypatch, league_dir):
    """Invariant 8: a League One outage costs League One, never the CSL."""
    calls = []
    monkeypatch.setattr("betmodel.xg.sync.sync",
                        lambda league, config, **kw: calls.append(league) or {"written": 3})
    monkeypatch.setattr(auxiliary_sync, "_client",
                        lambda config: FakeClient([], {}, fail=RuntimeError("boom")))
    args = argparse.Namespace(dry_run=False, limit=None)
    assert cli._xg("csl", args) == 0
    assert calls == ["csl"]
    assert not (league_dir / "league_one_matches.csv").exists()


def test_ligamx_runs_no_auxiliary_sync(monkeypatch):
    monkeypatch.setattr("betmodel.xg.sync.sync", lambda league, config, **kw: {})
    monkeypatch.setattr(auxiliary_sync, "sync",
                        lambda *a, **k: pytest.fail("ligamx has no auxiliary"))
    assert cli._xg("ligamx", argparse.Namespace(dry_run=False, limit=None)) == 0


# --------------------------------------------------------------------------- #
# a blank is asked about for a month, not for ever
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 11, 12, 6, 0, tzinfo=timezone.utc)


def test_a_blank_older_than_a_month_is_no_longer_asked_about(league_dir):
    """League One 2026 rounds 1-7 have no xG and never got any. Asking about
    all fifty blanks every day cost three minutes of each refresh."""
    _run(FakeClient([_event(1, 1), _event(2, 30)], {2: (0.9, 1.1)}), now=NOW)
    client = FakeClient([_event(1, 1), _event(2, 30)], {1: (1.0, 1.0), 2: (0.9, 1.1)})
    stats = _run(client, now=NOW)
    assert 1 not in client.fetched
    assert stats["abandoned"] == 1
    assert _table(league_dir).set_index("event_id").loc["1", "HxG"] == ""


def test_a_recent_blank_is_still_asked_about_every_run(league_dir):
    early = datetime(2026, 10, 15, tzinfo=timezone.utc)
    _run(FakeClient([_event(1, 1), _event(2, 30)], {2: (0.9, 1.1)}), now=early)
    client = FakeClient([_event(1, 1), _event(2, 30)], {1: (1.0, 1.0), 2: (0.9, 1.1)})
    stats = _run(client, now=early)
    assert 1 in client.fetched and stats["abandoned"] == 0
    assert _table(league_dir).set_index("event_id").loc["1", "HxG"] == "1"


def test_a_row_new_to_the_file_is_asked_once_however_old(league_dir):
    """A first sync or a backfill adds months-old matches. Each still gets the
    one request that tells us whether it has xG."""
    client = FakeClient([_event(1, 1), _event(2, 30)], {1: (1.0, 1.0), 2: (0.9, 1.1)})
    stats = _run(client, now=NOW)
    assert sorted(client.fetched) == [1, 2] and stats["abandoned"] == 0
    assert _table(league_dir).set_index("event_id").loc["1", "HxG"] == "1"


def test_the_review_window_still_wins_over_age(league_dir):
    """Off-season, the latest match is itself over a month old. It is still
    inside the review window, so a late revision is still picked up."""
    _run(FakeClient([_event(1, 30)], {}), now=NOW + pd.Timedelta(days=60))
    client = FakeClient([_event(1, 30)], {1: (1.4, 0.2)})
    stats = _run(client, now=NOW + pd.Timedelta(days=60))
    assert client.fetched == [1] and stats["abandoned"] == 0


def test_giving_up_can_be_turned_off(league_dir):
    _run(FakeClient([_event(1, 1), _event(2, 30)], {2: (0.9, 1.1)}), now=NOW)
    client = FakeClient([_event(1, 1), _event(2, 30)], {1: (1.0, 1.0), 2: (0.9, 1.1)})
    stats = _run(client, now=NOW, give_up_days=None)
    assert 1 in client.fetched and stats["abandoned"] == 0
