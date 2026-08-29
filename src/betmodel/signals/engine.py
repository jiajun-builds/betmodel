"""Turn model probabilities and captured prices into signals.

One engine for every league. What differed between the two merged pipelines was
never the logic, it was five settings: which books may be bet, how much edge is
enough, whether the draw is bettable, whether a long-shot cap applies, and
whether a price has to be provably an opener.

The record it produces carries the general form, a quote per (book, side), and
the two projections of it that the pre-merge boards displayed fall out
afterwards. One published a column per book, the other a single composite best
price; both are views of the same quotes.

Two rules are worth stating because they are easy to get subtly wrong.

**The pick is chosen on the best price, then the cap is applied.** Not the
reverse. So adding a second book can remove a bet that the first would have
fired alone, when the better price is over the long-shot cap. That ordering is
deliberate and inherited: the long-shot tail is the least-edge slice, and a row
where the best available price is untouchable is not a row to bet at a worse one.

**A price must be one you could have taken.** Expected value against a price
nobody offered is a number about nothing, so a book with no opening line
contributes no quote at all rather than a null one.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from betmodel import paths
from betmodel.config.schema import LeagueConfig
from betmodel.fixtures.upcoming import Fixture, load_upcoming
from betmodel.odds import reduce as reduce_module
from betmodel.signals import debias as debias_module
from betmodel.signals.ev import expected_value

log = logging.getLogger(__name__)

#: A fixture with no pick, as published. The two pipelines spelled the empty
#: state differently; this is the internal one and the legacy exporters render it.
NO_PICK = ""

STATE_BET = "bet"
STATE_ODDS_CAP = "odds_cap"
STATE_NONE = ""


def slugify(value: str) -> str:
    """Lowercase, accent-free, hyphenated. Used only for identifiers."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", stripped.lower()).strip("-")


def make_fixture_id(code: str, season: str, round_label: str, local_date, home: str, away: str) -> str:
    """Assemble an identifier from its parts.

    Shared so signals and results cannot construct the same fixture's identity
    differently, which would silently break the join between them.

    The round may be a stage name rather than a number: knockout ties are filed
    under names like "Quarterfinals", and "28" says far less about a match.
    """
    return "-".join([
        code,
        slugify(season),
        slugify(str(round_label)) or "0",
        local_date.strftime("%Y-%m-%d"),
        slugify(home),
        slugify(away),
    ])


def fixture_id(config: LeagueConfig, fixture: Fixture) -> str:
    """Stable identifier, and the join key across every published file.

    Derived rather than stored, so there is exactly one construction of a
    fixture's identity and no file can disagree with another about it.

    The date is the league's **local** matchday, not the UTC one. A 19:00 kickoff
    in a UTC-6 league falls on the next UTC day, and calling that its date names
    a fixture by a day nobody played on. The mistake hides in a league whose
    kickoffs sit in the middle of the UTC day and appears the moment one does
    not, which is the worst way for it to appear.
    """
    local = fixture.kickoff.astimezone(ZoneInfo(config.timezone))
    return make_fixture_id(
        config.code, config.season, fixture.round or "0", local,
        fixture.home, fixture.away,
    )


@dataclass(frozen=True)
class Quote:
    """One book's price for one outcome, and what we know about it."""

    book: str
    side: str
    odds: float
    ev: float
    captured_at: str = ""
    last_update: str = ""
    proof: str = ""

    @property
    def is_proven_opener(self) -> bool:
        return bool(self.proof)


@dataclass(frozen=True)
class Best:
    """The best price available on one outcome, and who is offering it."""

    book: str
    odds: float
    ev: float


@dataclass(frozen=True)
class Signal:
    """Everything decided about one fixture."""

    fixture_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    round: str
    probabilities: tuple[float, float, float]
    #: Before de-bias. Carried because one published file uses these and another
    #: uses the corrected ones, so the same fixture appears with two different
    #: probabilities and nothing in either file says which is which.
    raw_probabilities: tuple[float, float, float]
    debias_method: str
    quotes: tuple[Quote, ...]
    best: dict[str, Best]
    pick: str
    state: str
    ev: float | None
    #: Highest-EV bettable side BEFORE any threshold. Distinct from ``pick``,
    #: which is empty unless the edge cleared. Provenance and price age describe
    #: the price we would have taken, so they follow this rather than the pick:
    #: a row that did not fire still says which price it was judged on.
    top_side: str = ""
    books: tuple[str, ...] = ()
    anchor_odds: tuple[float, float, float] | None = None
    anchor_last_update: str = ""
    anchor_captured_at: str = ""

    @property
    def fires(self) -> bool:
        return self.state == STATE_BET

    @property
    def top_quote(self) -> "Quote | None":
        """The quote behind :attr:`top_side`, whether or not it fired."""
        if not self.top_side or self.top_side not in self.best:
            return None
        book = self.best[self.top_side].book
        return next(
            (q for q in self.quotes if q.side == self.top_side and q.book == book), None
        )

    def quotes_for(self, book: str) -> dict[str, Quote]:
        return {q.side: q for q in self.quotes if q.book == book}


def _model_probabilities(
    league: str, path: str | None = None
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """1X2 per ordered pairing, from the fitted simulations.

    Read rather than refitted. One pipeline refitted inside its exporter, which
    meant a published probability could differ from the one the same model wrote
    minutes earlier for no reason a reader could see.
    """
    frame = pd.read_csv(path or paths.for_league(league).simulations_csv)
    return {
        (str(row["Home Team"]), str(row["Away Team"])): (
            float(row["Home Win Probability"]),
            float(row["Draw Probability"]),
            float(row["Away Win Probability"]),
        )
        for _, row in frame.iterrows()
    }


def _side_index(side: str) -> int:
    return {"home": 0, "draw": 1, "away": 2}[side]


def build_signals(
    league: str,
    config: LeagueConfig,
    *,
    now: datetime | None = None,
    simulations_path: str | None = None,
    fixtures_path: str | None = None,
    history_path: str | None = None,
    watch_path: str | None = None,
) -> list[Signal]:
    """Every upcoming fixture that some book has priced.

    A fixture nobody has priced is left out entirely rather than published with
    empty odds columns. There is nothing to decide about it, and a row that
    cannot fire is noise on a board whose job is to show what can.
    """
    now = now or datetime.now(timezone.utc)
    model = _model_probabilities(league, simulations_path)
    opens = reduce_module.collapse_opens(
        league, config, history_path=history_path, watch_path=watch_path
    )
    anchor_key = config.signals.debias.anchor_book

    signals: list[Signal] = []
    fixtures = fixtures_path or paths.for_league(league).upcoming_fixtures_csv
    for fixture in load_upcoming(fixtures):
        if fixture.kickoff <= now:
            continue
        raw = model.get(fixture.key)
        if raw is None:
            # A promoted side appears in a fixture list before it appears in any
            # training window. Skipping is correct: league-average strength for a
            # club nobody has seen is a fabricated probability.
            log.debug("no model probabilities for %s", fixture.label)
            continue

        anchor = opens.get((fixture.home, fixture.away, anchor_key)) if anchor_key else None
        anchor_odds = (
            (anchor["home_odds"], anchor["draw_odds"], anchor["away_odds"])
            if anchor else None
        )
        probabilities, method = debias_module.apply(
            raw, config.signals.debias, anchor_odds=anchor_odds
        )

        quotes: list[Quote] = []
        for book in config.odds.bet_books:
            record = opens.get((fixture.home, fixture.away, book.key))
            if record is None:
                continue  # no price is not a price
            for side in ("home", "draw", "away"):
                odds = record[f"{side}_odds"]
                value = expected_value(probabilities[_side_index(side)], odds)
                if value is None:
                    continue
                quotes.append(Quote(
                    book=book.key, side=side, odds=float(odds), ev=value,
                    captured_at=_iso(record["captured_at"]),
                    last_update=record["last_update"], proof=record["proof"],
                ))

        best: dict[str, Best] = {}
        for side in ("home", "draw", "away"):
            candidates = [q for q in quotes if q.side == side]
            if not candidates:
                continue
            # Ties break on declared book order, which is why bet_books order is
            # load-bearing: max() keeps the first maximal element.
            winner = max(candidates, key=lambda q: q.odds)
            best[side] = Best(book=winner.book, odds=winner.odds, ev=winner.ev)

        if not quotes and anchor_odds is None:
            continue  # nobody has priced it; there is nothing to decide

        allowed = {s: b for s, b in best.items() if s in config.signals.sides}
        top_side = max(allowed, key=lambda s: allowed[s].ev) if allowed else ""

        pick, state, pick_ev, books = _decide(config, quotes, best)
        signals.append(Signal(
            fixture_id=fixture_id(config, fixture),
            home_team=fixture.home, away_team=fixture.away,
            kickoff=fixture.kickoff, round=fixture.round,
            probabilities=probabilities, raw_probabilities=raw, debias_method=method,
            quotes=tuple(quotes), best=best,
            pick=pick, state=state, ev=pick_ev, top_side=top_side, books=books,
            anchor_odds=anchor_odds,
            anchor_last_update=(anchor or {}).get("last_update", ""),
            anchor_captured_at=_iso((anchor or {}).get("captured_at")),
        ))
    return signals


def _decide(
    config: LeagueConfig, quotes: list[Quote], best: dict[str, Best]
) -> tuple[str, str, float | None, tuple[str, ...]]:
    """Pick the side, then decide whether it is bettable."""
    signals_config = config.signals
    candidates = {s: b for s, b in best.items() if s in signals_config.sides}
    if not candidates:
        return NO_PICK, STATE_NONE, None, ()

    pick = max(candidates, key=lambda s: candidates[s].ev)
    chosen = candidates[pick]
    if chosen.ev <= signals_config.ev_min:
        return NO_PICK, STATE_NONE, None, ()

    # A price must be provably the book's opener where the league requires it.
    # Deliberately no fallback to a lower-EV side with a proven price: that would
    # be choosing a bet on the strength of its paperwork.
    if signals_config.require_price_proof:
        winner = next(
            (q for q in quotes if q.side == pick and q.book == chosen.book), None
        )
        if winner is None or not winner.is_proven_opener:
            return NO_PICK, STATE_NONE, None, ()

    if signals_config.odds_cap is not None and chosen.odds > signals_config.odds_cap:
        # Surfaced but not bettable. signal_books stays empty so the invariant
        # holds: a displayed book is always a bet you should actually place.
        return pick, STATE_ODDS_CAP, chosen.ev, ()

    clearing = tuple(
        q.book for q in quotes
        if q.side == pick
        and q.ev > signals_config.ev_min
        and (signals_config.odds_cap is None or q.odds <= signals_config.odds_cap)
    )
    order = [b.key for b in config.odds.bet_books]
    clearing = tuple(sorted(set(clearing), key=order.index))
    return pick, STATE_BET, chosen.ev, clearing


def _iso(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    return str(value)
