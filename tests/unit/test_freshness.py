"""The xG staleness alert.

It guards a failure that is silent by construction: the merge never erases, so a
feed that stops leaves the previous values in place and every downstream stage
rebuilds green on stale data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from betmodel.config import load_league
from betmodel.xg import freshness


def _matches(tmp_path, rows) -> str:
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _row(date, hg=1, ag=0, hxg=1.2, axg=0.8):
    return {"Date": date, "Home": f"H{date}", "Away": f"A{date}",
            "HG": hg, "AG": ag, "HxG": hxg, "AxG": axg}


def test_a_healthy_feed_is_not_stale(tmp_path):
    path = _matches(tmp_path, [_row("2026-08-20"), _row("2026-08-27")])
    reading = freshness.measure("csl", matches_path=path)
    assert reading["gap_days"] == 0 and reading["stranded"] == 0


def test_a_dead_feed_strands_a_whole_round(tmp_path):
    """What a stopped fetcher actually looks like."""
    rows = [_row("2026-08-10")] + [
        _row(d, hxg=None, axg=None) for d in
        ("2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23")
    ]
    reading = freshness.measure("csl", matches_path=_matches(tmp_path, rows))
    assert reading["stranded"] == 4
    assert reading["gap_days"] == 13
    assert freshness.is_stale(reading, stale_days=3, min_missing=3)


def test_one_uncovered_match_does_not_raise_an_alert(tmp_path):
    """A provider does not cover every fixture, and alerting on a single gap
    trains the reader to ignore the channel."""
    rows = [_row("2026-08-20"), _row("2026-08-27", hxg=None, axg=None)]
    reading = freshness.measure("csl", matches_path=_matches(tmp_path, rows))
    assert reading["stranded"] == 1
    assert not freshness.is_stale(reading, stale_days=3, min_missing=3)


def test_a_recent_gap_alone_does_not_alert(tmp_path):
    """Both conditions are required. Today's matches have no xG yet by design."""
    rows = [_row("2026-08-25")] + [
        _row(d, hxg=None, axg=None) for d in ("2026-08-26", "2026-08-27", "2026-08-27")
    ]
    reading = freshness.measure("csl", matches_path=_matches(tmp_path, rows))
    assert reading["stranded"] == 3
    assert not freshness.is_stale(reading, stale_days=3, min_missing=3), \
        "two days behind is not stale, however many matches are waiting"


def test_the_comparison_is_against_results_not_the_xg_feeds_own_state(tmp_path):
    """When a fetcher dies the xG side freezes whole, so a self-consistency check
    on it alone sees nothing wrong. Results come from a provider that keeps
    running, and the gap between the two is what makes the failure visible."""
    rows = [_row("2026-08-01")] + [
        _row(d, hxg=None, axg=None)
        for d in ("2026-08-20", "2026-08-21", "2026-08-22")
    ]
    reading = freshness.measure("csl", matches_path=_matches(tmp_path, rows))
    assert reading["xg_to"] == "2026-08-01"
    assert reading["results_to"] == "2026-08-22"


def test_no_xg_at_all_is_stale(tmp_path):
    rows = [_row(d, hxg=None, axg=None) for d in ("2026-08-20", "2026-08-21", "2026-08-22")]
    reading = freshness.measure("csl", matches_path=_matches(tmp_path, rows))
    assert reading["xg_to"] is None and reading["stranded"] == 3


def test_a_missing_token_does_not_fail_the_run(tmp_path, monkeypatch):
    """A monitor that can fail the run it monitors is worse than no monitor."""
    monkeypatch.delenv(freshness.TOKEN_ENV, raising=False)
    monkeypatch.delenv(freshness.CHAT_ENV, raising=False)
    rows = [_row("2026-08-01")] + [
        _row(d, hxg=None, axg=None) for d in ("2026-08-20", "2026-08-21", "2026-08-22")
    ]
    reading = freshness.check(
        "csl", load_league("csl"), matches_path=_matches(tmp_path, rows)
    )
    assert reading["stale"] is True


def test_the_alert_names_both_frontiers(tmp_path, monkeypatch):
    sent = {}
    monkeypatch.setattr(freshness, "send",
                        lambda t, c, text: sent.setdefault("text", text) or True)
    monkeypatch.setenv(freshness.TOKEN_ENV, "t")
    monkeypatch.setenv(freshness.CHAT_ENV, "c")
    rows = [_row("2026-08-01")] + [
        _row(d, hxg=None, axg=None) for d in ("2026-08-20", "2026-08-21", "2026-08-22")
    ]
    freshness.check("csl", load_league("csl"), matches_path=_matches(tmp_path, rows))
    assert "2026-08-22" in sent["text"] and "2026-08-01" in sent["text"]
