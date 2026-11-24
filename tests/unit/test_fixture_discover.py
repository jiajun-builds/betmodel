"""Fixture discovery: adding newly scheduled fixtures between daily refreshes.

The upcoming list decides what the opening-price capture polls, and it is read by
every timing decision downstream. Discovery runs every half hour, so each test
here pins a way it could damage that list rather than merely fail to help it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from betmodel.config import load_league
from betmodel.fixtures import discover as d
from betmodel.fixtures import sync

# 10:30Z, a discovery slot for a 30-minute interval.
NOW = datetime(2026, 11, 24, 10, 31, 15, tzinfo=timezone.utc)


def _match(home, away, kickoff, *, round_="Quarterfinals", goals=None):
    return sync.Match(season="Apertura 2026", round=round_, kickoff=kickoff,
                      home=home, away=away,
                      home_goals=None if goals is None else goals[0],
                      away_goals=None if goals is None else goals[1])


def _stored(path, rows):
    pd.DataFrame(rows, columns=sync.UPCOMING_COLUMNS).to_csv(path, index=False)


def _run(tmp_path, monkeypatch, scheduled, *, stored=None, now=NOW, **kwargs):
    path = tmp_path / "upcoming.csv"
    if stored is not None:
        _stored(path, stored)
    asked = []

    def fake(config, source, kinds=("last", "next")):
        asked.append(kinds)
        return scheduled

    monkeypatch.setattr(sync, "_from_sofascore", fake)
    stats = d.discover("ligamx", load_league("ligamx"), now=now,
                       upcoming_path=str(path), **kwargs)
    return stats, path, asked


QF1 = _match("Pachuca", "Guadalajara", datetime(2026, 11, 25, 1, 0, tzinfo=timezone.utc))
QF2 = _match("Toluca", "Atlas", datetime(2026, 11, 26, 1, 0, tzinfo=timezone.utc))


def test_a_newly_scheduled_knockout_tie_is_added(tmp_path, monkeypatch):
    """The Apertura 2026 quarterfinals reached the list up to a day after they
    were set, by which time every opener was gone."""
    stored = [sync.upcoming_row(QF1, load_league("ligamx"))]
    stats, path, asked = _run(tmp_path, monkeypatch, [QF1, QF2], stored=stored)
    assert stats == {"scheduled": 2, "added": 1}
    assert asked == [("next",)], "only the scheduled events, never the history"
    left = pd.read_csv(path, dtype=str)
    assert list(left.columns) == sync.UPCOMING_COLUMNS
    assert list(zip(left.Home, left.Away)) == [("Pachuca", "Guadalajara"), ("Toluca", "Atlas")]
    assert left.iloc[1].Round == "Quarterfinals"
    assert left.iloc[1].kickoff_utc == "2026-11-26T01:00:00Z"


def test_it_never_removes_or_rewrites_a_row(tmp_path, monkeypatch):
    """A row the provider no longer lists, even one already played, is the full
    refresh's to drop. Discovery that rewrote the list would be a second sync
    with none of the first one's guards."""
    config = load_league("ligamx")
    played = _match("Queretaro", "Necaxa", datetime(2026, 11, 23, 1, 10, tzinfo=timezone.utc),
                    round_="17")
    stored = [sync.upcoming_row(played, config), sync.upcoming_row(QF1, config)]
    stats, path, _ = _run(tmp_path, monkeypatch, [QF2], stored=stored)
    assert stats["added"] == 1
    left = pd.read_csv(path, dtype=str)
    assert ("Queretaro", "Necaxa") in set(zip(left.Home, left.Away))
    assert len(left) == 3


def test_nothing_new_writes_nothing(tmp_path, monkeypatch, caplog):
    """And says so: the tick spent requests, so it must not look like one that
    never asked."""
    config = load_league("ligamx")
    _stored(tmp_path / "upcoming.csv", [sync.upcoming_row(QF1, config)])
    before = (tmp_path / "upcoming.csv").read_bytes()
    with caplog.at_level("INFO"):
        stats, path, _ = _run(tmp_path, monkeypatch, [QF1])
    assert stats == {"scheduled": 1, "added": 0}
    assert path.read_bytes() == before
    assert "none new" in caplog.text


def test_a_rescheduled_tie_is_a_new_fixture(tmp_path, monkeypatch):
    """A fixture is a pairing on a matchday. The replacement date must be polled
    while unpriced, or its opener can never be proven."""
    config = load_league("ligamx")
    moved = _match("Pachuca", "Guadalajara", QF1.kickoff + timedelta(days=2))
    stats, path, _ = _run(tmp_path, monkeypatch, [moved],
                          stored=[sync.upcoming_row(QF1, config)])
    assert stats["added"] == 1
    assert len(pd.read_csv(path)) == 2


def test_a_kickoff_nudged_within_the_day_is_not_a_new_fixture(tmp_path, monkeypatch):
    config = load_league("ligamx")
    nudged = _match("Pachuca", "Guadalajara", QF1.kickoff + timedelta(minutes=15))
    stats, _, _ = _run(tmp_path, monkeypatch, [nudged],
                       stored=[sync.upcoming_row(QF1, config)])
    assert stats["added"] == 0


def test_a_started_or_played_match_is_not_added(tmp_path, monkeypatch):
    started = _match("Monterrey", "Club America", NOW - timedelta(minutes=30))
    played = _match("Leon", "Cruz Azul", NOW + timedelta(days=1), goals=(1, 0))
    stats, path, _ = _run(tmp_path, monkeypatch, [started, played], stored=[])
    assert stats == {"scheduled": 0, "added": 0}


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    stats, path, _ = _run(tmp_path, monkeypatch, [QF1], stored=[], dry_run=True)
    assert stats["added"] == 1
    assert pd.read_csv(path).empty


def test_off_its_slot_it_asks_nothing(tmp_path, monkeypatch):
    """Half-hourly, not every five minutes: a residential proxy is paid for."""
    off = NOW.replace(minute=36)
    stats, _, asked = _run(tmp_path, monkeypatch, [QF1], stored=[], now=off)
    assert (stats, asked) == ({"scheduled": 0, "added": 0}, [])
    _, _, asked = _run(tmp_path, monkeypatch, [QF1], stored=[], now=off,
                       ignore_schedule=True)
    assert asked == [("next",)]


def test_the_slot_tolerates_the_runner_s_startup_lag():
    config = load_league("ligamx")
    assert d.due(config, NOW.replace(minute=30, second=0))
    assert d.due(config, NOW.replace(minute=34, second=59))
    assert not d.due(config, NOW.replace(minute=35))


def test_a_league_without_discovery_is_left_alone(tmp_path, monkeypatch):
    """The CSL's schedule is on TheSportsDB and has no knockout stage."""
    monkeypatch.setattr(sync, "_from_sofascore",
                        lambda *a, **k: pytest.fail("must not be asked"))
    stats = d.discover("csl", load_league("csl"), now=NOW, ignore_schedule=True,
                       upcoming_path=str(tmp_path / "upcoming.csv"))
    assert stats == {"scheduled": 0, "added": 0}
    assert not (tmp_path / "upcoming.csv").exists()


def test_an_interval_that_does_not_fit_the_timer_is_refused():
    from dataclasses import replace

    config = load_league("ligamx")
    source = config.sources["fixtures"]
    bad = replace(source, params={**source.params, d.INTERVAL_KEY: 7})
    with pytest.raises(ValueError, match="multiple of 5"):
        d.interval(replace(config, sources={**config.sources, "fixtures": bad}))
