"""Opening-line capture: the pending set and the per-book gate.

The gate is the whole defence of the opening-line series. Two books do not open
together, so a request made because one is missing also returns the other's
*current* price, and storing that as an open would overwrite a genuine opening
line with a mid-market one. Dedup cannot help: one book stamps every fixture
with a shared feed-refresh timestamp, so its last_update moves regardless.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from betmodel.config import load_league
from betmodel.odds import capture_open as co
from betmodel.odds import capture_store
from betmodel.providers import oddsapiio

NOW = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)


def _fixtures(tmp_path, rows):
    path = tmp_path / "upcoming.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Home", "Away", "kickoff_utc"])
        writer.writeheader()
        for home, away, kickoff in rows:
            writer.writerow({
                "Home": home, "Away": away,
                "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
            })
    return str(path)


def _history(tmp_path, rows):
    path = str(tmp_path / "history.csv")
    if rows:
        capture_store.append_snapshots(
            pd.DataFrame(rows), path=path, snapshot_type="open"
        )
    return path


def _open_row(home, away, book):
    return {
        "event_id": f"{home}-{away}-{book}", "commence_time": "2026-09-01T12:00:00Z",
        "api_home_team": home, "api_away_team": away,
        "home_team": home, "away_team": away,
        "home_odds": "2.1", "draw_odds": "3.3", "away_odds": "3.4",
        "bookmaker": book, "market": "h2h", "regions": "oddsapiio",
        "last_update": "2026-08-20T10:00:00Z", "fetched_at": "2026-08-20T10:00:05Z",
    }


def _pending(tmp_path, fixtures, history, league="csl"):
    config = load_league(league)
    return co.pending_fixtures(
        league, config,
        books=co.books_for(config, "oddsapiio"),
        now=NOW,
        fixtures_path=_fixtures(tmp_path, fixtures),
        history_path=_history(tmp_path, history),
    )


# --------------------------------------------------------------------------- #
# the per-book gate
# --------------------------------------------------------------------------- #

def test_a_book_that_already_has_an_open_is_not_asked_again(tmp_path):
    """The invariant that stops a mid-market price overwriting an opener."""
    pending = _pending(
        tmp_path,
        [("A", "B", NOW + timedelta(days=3))],
        [_open_row("A", "B", "duel")],
    )
    assert len(pending) == 1
    assert [b.key for b in pending[0].missing] == ["onexbet"], (
        "duel already opened this fixture and must not be requested again"
    )


def test_a_fixture_with_every_book_captured_is_not_pending_at_all(tmp_path):
    pending = _pending(
        tmp_path,
        [("A", "B", NOW + timedelta(days=3))],
        [_open_row("A", "B", "duel"), _open_row("A", "B", "onexbet")],
    )
    assert pending == []


# --------------------------------------------------------------------------- #
# what is pending
# --------------------------------------------------------------------------- #

def test_a_started_match_is_dropped(tmp_path):
    """Its pre-match line is gone, and keeping it pending burns a request every
    tick forever."""
    assert _pending(tmp_path, [("A", "B", NOW - timedelta(hours=1))], []) == []


def test_a_fixture_beyond_the_lookahead_is_not_pending_yet(tmp_path):
    """Without this, a league that publishes its whole season keeps every fixture
    pending forever and spends the cap on matches months away."""
    far = NOW + timedelta(days=load_league("csl").odds.open.lookahead_days + 5)
    assert _pending(tmp_path, [("A", "B", far)], []) == []


def test_pending_is_ordered_soonest_first(tmp_path):
    """An overflowing set must spend its budget on the lines opening now."""
    pending = _pending(tmp_path, [
        ("C", "D", NOW + timedelta(days=9)),
        ("A", "B", NOW + timedelta(days=2)),
        ("E", "F", NOW + timedelta(days=5)),
    ], [])
    assert [p.fixture.home for p in pending] == ["A", "E", "C"]


# --------------------------------------------------------------------------- #
# spending
# --------------------------------------------------------------------------- #

def test_an_idle_tick_spends_nothing(tmp_path):
    """Most ticks are idle, so this is the property the whole budget rests on."""
    config = load_league("csl")
    stats = co.capture_opens(
        "csl", config, providers=("oddsapiio",),
        now=NOW,
        fixtures_path=_fixtures(tmp_path, [("A", "B", NOW - timedelta(days=1))]),
        history_path=_history(tmp_path, []),
        watch_path=str(tmp_path / "watch.csv"),
    )
    assert stats == {"pending": 0, "captured": 0, "appended": 0, "unpriced": 0,
                     "requests": 0, "refused": 0}


def test_a_dry_run_decides_without_spending(tmp_path):
    """This provider bills per request, including the event listing, so "decide
    only" has to mean touching neither endpoint."""
    stats = co.capture_opens(
        "csl", load_league("csl"), providers=("oddsapiio",), dry_run=True, now=NOW,
        fixtures_path=_fixtures(tmp_path, [("A", "B", NOW + timedelta(days=2))]),
        history_path=_history(tmp_path, []),
        watch_path=str(tmp_path / "watch.csv"),
    )
    assert stats["pending"] == 1
    assert stats["requests"] == 0


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #

def test_each_league_names_its_own_odds_api_io_key():
    """A key is entitled to a fixed, small number of bookmakers, and the two
    leagues bet three distinct books between them, so they cannot share one."""
    creds = {
        lg: load_league(lg).odds.providers["oddsapiio"].require("credential")
        for lg in ("csl", "ligamx")
    }
    assert creds["csl"] != creds["ligamx"]
    assert oddsapiio.key_env_for(creds["csl"]) == "ODDS_API_IO_KEY_CSL"
    assert oddsapiio.key_env_for(creds["ligamx"]) == "ODDS_API_IO_KEY_LIGAMX"


def test_a_named_credential_falls_back_to_the_shared_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_IO_KEY_CSL", raising=False)
    monkeypatch.setenv("ODDS_API_IO_KEY", "shared")
    assert oddsapiio.api_key("csl") == "shared"


def test_an_account_under_its_floor_is_a_refusal_not_an_empty_tick(tmp_path, monkeypatch):
    """The two look identical in the numbers, and only one of them is fine.

    A refused tick returns no rows and spends no requests, exactly like a tick
    with nothing to do, and the workflow step exits green either way. That is how
    the CSL anchor stopped being captured for three days without anything
    noticing. `refused` is what tells them apart.
    """
    def _refuse(*args, **kwargs):
        raise co.QuotaRefused("THE_ODDS_API_KEY_CSL has 48 requests left, floor is 50")

    monkeypatch.setattr(co, "_capture_theoddsapi", _refuse)
    config = load_league("csl")
    stats = co.capture_opens(
        "csl", config, providers=("theoddsapi",),
        now=NOW, ignore_schedule=True,
        fixtures_path=_fixtures(tmp_path, [("A", "B", NOW + timedelta(days=2))]),
        history_path=_history(tmp_path, []),
        watch_path=str(tmp_path / "watch.csv"),
    )
    assert stats["refused"] == 1
    assert stats["captured"] == 0 and stats["requests"] == 0


def test_one_provider_running_out_does_not_take_the_other_down(tmp_path, monkeypatch):
    """The soft books and the anchor are different accounts. Losing both because
    one is spent would turn a degraded tick into a dead one."""
    def _refuse(*args, **kwargs):
        raise co.QuotaRefused("out of credit")

    seen = []

    def _soft(league, config, pending, **kwargs):
        seen.append("oddsapiio")
        return [], [], 1

    monkeypatch.setattr(co, "_capture_theoddsapi", _refuse)
    monkeypatch.setattr(co, "_capture_oddsapiio", _soft)
    stats = co.capture_opens(
        "csl", load_league("csl"), providers=("theoddsapi", "oddsapiio"),
        now=NOW, ignore_schedule=True,
        fixtures_path=_fixtures(tmp_path, [("A", "B", NOW + timedelta(days=2))]),
        history_path=_history(tmp_path, []),
        watch_path=str(tmp_path / "watch.csv"),
    )
    assert seen == ["oddsapiio"], "the healthy provider still ran"
    assert stats["refused"] == 1 and stats["requests"] == 1
