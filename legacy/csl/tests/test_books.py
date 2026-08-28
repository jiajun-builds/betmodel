"""Tests for the bet-book registry.

The one that matters is ``test_keys_match_capture_vocabulary``. The registry's ``key``
is joined against the ``bookmaker`` column of the capture history, which is written by
``oddsapi_io.Book.stored_key``. If those two ever drift — ``"Duel"`` vs ``"duel"`` is
the obvious way — ``load_open_snapshots`` filters to zero rows and pandas merges an
**empty frame**, producing an all-NaN column. Nothing raises. The board simply shows
that book as permanently unpriced, forever, and the only other signal is the zero-count
warning in ``run()``. So the parity is pinned here rather than left to a code review.

This test file may import ``oddsapi_io``; ``books.py`` itself must not, since it has to
stay stdlib-only for ``signal_alert``.

Runnable either way::

    pytest tests/test_books.py          # if pytest is installed
    python tests/test_books.py          # no pytest needed
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.odds.books import BET_BOOKS, BOOK_BY_KEY, DUEL_BOOK, ONEXBET_BOOK, SIDES  # noqa: E402
from csl.odds.oddsapi_io import CAPTURE_BOOKS  # noqa: E402


def test_keys_match_capture_vocabulary() -> None:
    """Every bet book must be spelled exactly as the capture layer stores it.

    A mismatch is silent: empty filter -> empty merge -> all-NaN column -> a book that
    looks like it never posts a price.
    """
    assert {b.key for b in BET_BOOKS} == {b.stored_key for b in CAPTURE_BOOKS}


def test_prefix_follows_the_key() -> None:
    """dashboard/app.js derives column names from `key` alone; prefix must agree."""
    for book in BET_BOOKS:
        assert book.prefix == f"{book.key}_open", book


def test_onexbet_columns_are_the_frozen_legacy_names() -> None:
    """These exact strings are the archive contract and the 1xBet-only reconstruction.

    Renaming any of them silently drops the dashboard's bet price, EV and signal —
    `load_open_snapshots` and `app.js` both match on the literal string.
    """
    assert ONEXBET_BOOK.columns == [
        "onexbet_open_home_odds",
        "onexbet_open_draw_odds",
        "onexbet_open_away_odds",
        "onexbet_open_home_ev",
        "onexbet_open_draw_ev",
        "onexbet_open_away_ev",
        "onexbet_open_last_update",
    ]


def test_duel_columns_do_not_collide_with_onexbet() -> None:
    assert set(DUEL_BOOK.columns).isdisjoint(ONEXBET_BOOK.columns)
    assert DUEL_BOOK.odds_col("home") == "duel_open_home_odds"
    assert DUEL_BOOK.ev_col("away") == "duel_open_away_ev"


def test_registry_is_consistent() -> None:
    assert BOOK_BY_KEY == {b.key: b for b in BET_BOOKS}
    assert len(BOOK_BY_KEY) == len(BET_BOOKS), "duplicate key"
    # 1xBet first: the incumbent must win an exact best-price tie so `signal_book`
    # cannot flip between runs and re-fire the Telegram alert.
    assert BET_BOOKS[0] is ONEXBET_BOOK
    assert SIDES == ("home", "draw", "away")


def test_every_book_has_a_usable_link_and_logo_stem() -> None:
    """The logo path is ./assets/{key}.png and the key must be filesystem-safe.

    Lowercase is not cosmetic: macOS is case-insensitive but GitHub Pages serves from
    Linux, so a capitalized stem 404s ONLY in production.
    """
    root = os.path.join(os.path.dirname(__file__), "..", "dashboard", "assets")
    for book in BET_BOOKS:
        assert book.key == book.key.lower(), book.key
        assert book.url.startswith("https://"), book
        assert os.path.isfile(os.path.join(root, f"{book.key}.png")), (
            f"missing dashboard/assets/{book.key}.png"
        )


def _run_all() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
