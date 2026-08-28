"""Column names for a bookmaker's published prices.

The registry itself is the league config; this is the one place its book entries
become actual column names. That matters because the names are a contract, not a
convenience: the downstream board derives its own column vocabulary from them,
so a prefix that drifts blanks every price it names, silently and without an
error anywhere.

One book's prefix does not follow the derived pattern and never will. It is
frozen at its historical spelling because every stored row and the board's
vocabulary already use it. The config schema refuses a deviation that does not
carry a written reason, and this module simply obeys whatever the config says.
"""

from __future__ import annotations

from betmodel.config.schema import BookConfig, LeagueConfig

SIDES = ("home", "draw", "away")


def odds_column(book: BookConfig, side: str) -> str:
    _check_side(side)
    return f"{book.effective_legacy_prefix}_{side}_odds"


def ev_column(book: BookConfig, side: str) -> str:
    _check_side(side)
    return f"{book.effective_legacy_prefix}_{side}_ev"


def last_update_column(book: BookConfig) -> str:
    """When this book's price was last stamped.

    Named off the key rather than the prefix, matching the published shape:
    ``onexbet_open_last_update``, not ``onexbet_open_open_last_update``.
    """
    return f"{book.key}_open_last_update"


def columns_for(book: BookConfig) -> list[str]:
    """Every column this book contributes, in published order."""
    return (
        [odds_column(book, s) for s in SIDES]
        + [ev_column(book, s) for s in SIDES]
        + [last_update_column(book)]
    )


def best_odds_column(side: str) -> str:
    _check_side(side)
    return f"best_open_{side}_odds"


def best_ev_column(side: str) -> str:
    _check_side(side)
    return f"best_open_{side}_ev"


def best_book_column(side: str) -> str:
    _check_side(side)
    return f"best_open_{side}_book"


def per_book_columns(config: LeagueConfig) -> list[str]:
    """All bet-book columns for a league, in declared order.

    Order is load-bearing: it sets published column order and breaks ties when
    two books quote the same best price.
    """
    out: list[str] = []
    for book in config.odds.bet_books:
        out += columns_for(book)
    return out


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
