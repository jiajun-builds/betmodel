"""Odds provider behaviour, especially where a mistake costs quota.

Both providers meter requests, so the expensive errors are the ones that spend
allowance to learn nothing: retrying an entitlement refusal, splitting a slate
across calls, or asking for a bookmaker under the wrong spelling.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betmodel.config import load_league
from betmodel.providers import http, oddsapiio as oio, theoddsapi as toa


# --------------------------------------------------------------------------- #
# The Odds API
# --------------------------------------------------------------------------- #

def test_credentials_map_to_separate_environment_variables():
    """Two accounts, split by role: open polling is continuous, closing is bursty."""
    assert toa.key_env_for("default") == "THE_ODDS_API_KEY"
    assert toa.key_env_for("opens") == "THE_ODDS_API_KEY_OPENS"
    assert toa.key_env_for("closes") == "THE_ODDS_API_KEY_CLOSES"


def test_a_named_credential_falls_back_to_the_shared_key(monkeypatch):
    """A single-account setup must work with no extra configuration."""
    monkeypatch.delenv("THE_ODDS_API_KEY_OPENS", raising=False)
    monkeypatch.setenv("THE_ODDS_API_KEY", "shared")
    assert toa.api_key("opens") == "shared"


def test_a_missing_key_is_a_loud_failure(monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("THE_ODDS_API_KEY_OPENS", raising=False)
    with pytest.raises(RuntimeError, match="no The Odds API key"):
        toa.api_key("opens")


def test_unknown_remaining_does_not_block_a_run():
    """Older API behaviour omits the header; that is not an outage."""
    assert toa.Quota(remaining=None, used=None, sport_available=True).below(50) is False
    assert toa.Quota(remaining=10, used=1, sport_available=True).below(50) is True
    assert toa.Quota(remaining=90, used=1, sport_available=True).below(50) is False


def test_prices_are_flattened_from_the_bookmaker_level():
    """last_update read from the wrong nesting level is the classic bug here."""
    events = [{
        "id": "e1", "commence_time": "2026-08-28T12:00:00Z",
        "home_team": "A", "away_team": "B",
        "bookmakers": [{
            "key": "pinnacle", "last_update": "2026-08-27T00:00:00Z",
            "markets": [{
                "key": "h2h", "last_update": "2026-08-28T09:00:00Z",
                "outcomes": [{"name": "A", "price": 2.1},
                             {"name": "B", "price": 3.4},
                             {"name": "Draw", "price": 3.3}],
            }],
        }],
    }]
    rows = list(toa.iter_prices(events))
    assert len(rows) == 1
    row = rows[0]
    assert row["bookmaker"] == "pinnacle"
    assert row["last_update"] == "2026-08-28T09:00:00Z", "market level, not book level"
    assert row["outcomes"] == {"A": 2.1, "B": 3.4, "Draw": 3.3}


def test_an_event_with_no_bookmakers_yields_nothing():
    assert list(toa.iter_prices([{"id": "e1", "bookmakers": []}])) == []


# --------------------------------------------------------------------------- #
# odds-api.io
# --------------------------------------------------------------------------- #

def test_an_entitlement_refusal_is_distinct_and_not_retried(monkeypatch):
    """A 403 means a bookmaker outside the plan; retrying burns the daily budget."""
    monkeypatch.setenv(oio.KEY_ENV, "k")
    c = oio.OddsApiIoClient(["some-league"])
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise http.HttpError("refused", status=403)

    object.__setattr__(c._http, "get", boom)
    with pytest.raises(oio.EntitlementError, match="outside this plan"):
        c.get("odds/multi")
    assert calls["n"] == 1


def test_batches_are_capped_rather_than_silently_truncated(monkeypatch):
    monkeypatch.setenv(oio.KEY_ENV, "k")
    c = oio.OddsApiIoClient(["l"])
    with pytest.raises(ValueError, match="at most 10"):
        c.multi_odds(list(range(11)), [oio.Book("Duel", "duel")])


def test_an_empty_batch_costs_no_request(monkeypatch):
    monkeypatch.setenv(oio.KEY_ENV, "k")
    c = oio.OddsApiIoClient(["l"])
    object.__setattr__(c, "get", lambda *a, **k: pytest.fail("should not spend"))
    assert c.multi_odds([], [oio.Book("Duel", "duel")]) == []


def test_an_absent_book_is_unpriced_not_an_error():
    """This absence is the whole basis of opening-line capture."""
    quoted = {"bookmakers": {"Duel": [{"name": "ML", "odds": [
        {"home": 2.1, "draw": 3.3, "away": 3.4}]}]}}
    assert oio.extract_ml(quoted, oio.Book("Duel", "duel")) is not None
    assert oio.extract_ml(quoted, oio.Book("Betano UK", "betano")) is None


def test_a_partial_price_is_treated_as_unpriced():
    quoted = {"bookmakers": {"Duel": [{"name": "ML", "odds": [
        {"home": 2.1, "draw": None, "away": 3.4}]}]}}
    assert oio.extract_ml(quoted, oio.Book("Duel", "duel")) is None


def test_book_spelling_comes_from_config_not_from_the_key():
    """"Betano" is a 403 where "Betano UK" is the entitled book."""
    books = oio.books_from_config(load_league("ligamx").odds.books)
    assert [(b.provider_name, b.key) for b in books] == [
        ("Betano UK", "betano"), ("Duel", "duel")
    ]


def test_only_this_provider_s_books_are_requested_from_it():
    """Pinnacle is on the other provider and must never enter this call."""
    keys = [b.key for b in oio.books_from_config(load_league("csl").odds.books)]
    assert keys == ["onexbet", "duel"]
    assert "pinnacle" not in keys


def test_timestamps_are_rfc3339_because_a_bare_date_is_rejected():
    got = oio._rfc3339(datetime(2026, 8, 28, 7, 0, tzinfo=timezone.utc))
    assert got == "2026-08-28T07:00:00Z"
    naive = oio._rfc3339(datetime(2026, 8, 28, 7, 0))
    assert naive.endswith("Z")
