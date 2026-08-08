"""The books we bet into, and the column vocabulary each one owns.

Why this is its own module
--------------------------
``signal_alert`` needs a book's label, URL and column names to write a Telegram
message. It is deliberately **stdlib-only** (``csv`` + ``requests``), and importing
``export_upcoming_market_comparison`` to reach the registry would drag pandas *and*
the Dixon-Coles fitter (``csl.models.dc``) into a notifier that must stay light and
must never fail the pipeline. So the registry lives here with **no third-party
imports**, and ``export_upcoming_market_comparison`` re-exports it.

Bet books vs the anchor
-----------------------
These are the books whose **opening** price we would actually take. Pinnacle is NOT
one of them: its open feeds the λ draw anchor
(``export_upcoming_market_comparison.ANCHOR_BOOKMAKER``) and is never a bet price.

Why two books at all (2026-08-08)
---------------------------------
Duel joined 1xBet because *per-outcome best price* is free money, not because it is
"the cheaper book". Measured on the three fixtures both quoted the day it was wired
in, Duel's much lower overround (3.00% vs 5.41%) is concentrated almost entirely in
the **draw** — which ``SIGNAL_ALLOW_DRAW=False`` means we never bet:

    home  Duel 2 - 1xBet 1     draw  Duel 3 - 1xBet 0     away  Duel 1 - 1xBet 2

On home/away — the only sides that can fire a signal — it is 3-3. Picking a book by
its headline overround would therefore have taken the worse price about half the
time. Taking the better price *per outcome* was worth +2.06% mean (max +5.66%) on
those sides, which near breakeven is roughly +2pp of EV against a 2.61pp vig wall.
"""

from __future__ import annotations

from dataclasses import dataclass

SIDES = ("home", "draw", "away")


@dataclass(frozen=True)
class BetBook:
    """One book we would take a price at, and the columns it owns.

    ``key`` is the single source of identity and is reused in four places, which is
    what keeps a third book a config change rather than surgery:
      * the ``bookmaker`` value in the capture history (``oddsapi_io.Book.stored_key``)
      * the token written to ``signal_book`` / ``signal_books``
      * the column-name root, via ``prefix``
      * the logo filename stem, ``dashboard/assets/{key}.png``

    ``prefix`` is stored rather than derived **only** because 1xBet's columns are the
    frozen legacy name ``onexbet_open_*``, which every pre-existing row, the archive
    CSV and the reconstruction of 1xBet-only performance all depend on. The invariant
    ``prefix == f"{key}_open"`` is asserted below — ``app.js`` derives its column
    names from ``key`` alone, so a book that broke it would read as permanently
    unpriced on the board with no error anywhere.
    """

    key: str
    prefix: str
    label: str
    url: str

    def odds_col(self, side: str) -> str:
        return f"{self.prefix}_{side}_odds"

    def ev_col(self, side: str) -> str:
        return f"{self.prefix}_{side}_ev"

    @property
    def last_update_col(self) -> str:
        return f"{self.prefix}_last_update"

    @property
    def columns(self) -> list[str]:
        """The 7 columns this book owns, in archive order: 3 odds, 3 EV, last_update."""
        return (
            [self.odds_col(s) for s in SIDES]
            + [self.ev_col(s) for s in SIDES]
            + [self.last_update_col]
        )


ONEXBET_BOOK = BetBook(
    key="onexbet",
    # Frozen legacy prefix — predates the registry. Do not "fix" to onexbet_open's
    # sibling naming; the archive CSV and every captured row depend on it.
    prefix="onexbet_open",
    label="1xBet",
    url="https://1xbetjap.com/en/line/football/58043-china-super-league",
)

DUEL_BOOK = BetBook(
    key="duel",
    prefix="duel_open",
    label="Duel",
    # duel.com, NOT duel.bet. A `?bt-path=` deep link — pass it through HTML-escaping
    # unchanged and never append a further `?`/`&` to it.
    url="https://duel.com/sports?bt-path=/soccer/china/chinese-super-league-1669818818899349504",
)

# ORDER IS LOAD-BEARING in three places: column order in both CSVs, left-to-right
# display order on the board, and — most importantly — the best-price tie-break.
# 1xBet first means the incumbent wins an exact tie, which keeps `signal_book`
# deterministic; that field is part of the alert dedup key, so a tie-break that could
# flip would re-alert on every run.
BET_BOOKS: tuple[BetBook, ...] = (ONEXBET_BOOK, DUEL_BOOK)

BOOK_BY_KEY: dict[str, BetBook] = {b.key: b for b in BET_BOOKS}

assert all(b.prefix == f"{b.key}_open" for b in BET_BOOKS), (
    "dashboard/app.js builds column names as `${book.key}_open_{side}_{odds,ev}`; a "
    "BetBook whose prefix does not follow that pattern renders as permanently unpriced."
)
assert len(BOOK_BY_KEY) == len(BET_BOOKS), "duplicate BetBook.key"
