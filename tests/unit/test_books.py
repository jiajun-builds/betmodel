"""Published column names.

These are a contract, not a convenience: the downstream board derives its own
column vocabulary from them, so a prefix that drifts blanks every price it names
without raising anything.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from betmodel.config import load_league
from betmodel.odds import books


def test_the_frozen_prefix_produces_the_names_the_board_reads():
    book = load_league("csl").odds.book("onexbet")
    assert books.odds_column(book, "home") == "onexbet_open_home_odds"
    assert books.ev_column(book, "away") == "onexbet_open_away_ev"
    assert books.last_update_column(book) == "onexbet_open_last_update"


def test_a_derived_prefix_produces_the_derived_names():
    book = load_league("csl").odds.book("duel")
    assert books.odds_column(book, "draw") == "duel_open_draw_odds"


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_declared_order_survives_into_column_order(league):
    """Order breaks ties when two books quote the same best price, and sets the
    published column order."""
    config = load_league(league)
    columns = books.per_book_columns(config)
    first, second = (b.effective_legacy_prefix for b in config.odds.bet_books)
    assert columns[0].startswith(first)
    assert any(c.startswith(second) for c in columns[7:])


def test_every_generated_name_exists_in_the_frozen_payload():
    """The strongest available check: the board is reading these today."""
    payload = json.loads(
        pathlib.Path("tests/golden/csl/published/upcoming_market_comparison.json").read_text()
    )
    present = set(payload["rows"][0])
    generated = set(books.per_book_columns(load_league("csl")))
    generated |= {books.best_odds_column(s) for s in books.SIDES}
    generated |= {books.best_ev_column(s) for s in books.SIDES}
    generated |= {books.best_book_column(s) for s in books.SIDES}
    assert generated <= present, f"not published today: {sorted(generated - present)}"


def test_an_invalid_side_is_refused():
    book = load_league("csl").odds.book("duel")
    with pytest.raises(ValueError, match="side must be"):
        books.odds_column(book, "over")
