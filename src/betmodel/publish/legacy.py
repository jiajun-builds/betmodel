"""The pre-merge JSON shapes. No longer published; kept as the gate-G3 fixture.

The board reads ``public/index.json`` now, so the compatibility tree is deleted
and nothing renders these shapes into ``public/`` any more. Nothing should ever be
written against them again.

This module survives for the reason its own docstring always gave for valuing it:
rendering the old shapes from the merged engine is the only way to compare its
output against what the two pipelines actually published, field by field, against
a frozen baseline. That comparison is gate G3, and it was worth more than the
compatibility even while the compatibility was load-bearing.

Deleting it would trade a standing regression test on the signal engine for 222
lines. The migration it proved is finished; the protection it gives is not.

Two leagues, two shapes, and the differences are not cosmetic:

* one publishes a column per book, the other a single composite best price;
* one expresses expected value as a fraction, the other as percentage points;
* they use ``kickoff_at`` and ``match_time`` with **opposite** conventions. One
  puts local time in ``kickoff_at`` and UTC in ``match_time``; the other does the
  reverse. Both parse to the right instant, so nothing downstream broke, but the
  fields mean opposite things. The canonical contract has one unambiguous UTC
  field instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from betmodel.config.schema import LeagueConfig
from betmodel.odds import books as books_module
from betmodel.signals.engine import Signal

SIDES = ("home", "draw", "away")

#: What the two boards spell an absent pick as.
EMPTY_PICK = {"per_book": None, "composite": "none"}
EMPTY_STATE = {"per_book": None, "composite": "none"}


def _iso_z(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _ev_out(value: float | None, unit: str) -> float | None:
    """Expected value in whichever unit this league's board expects.

    The engine is fractions throughout. Percentage points exist only here, at the
    moment of writing a legacy file, which is why the board downstream has to
    guess the scale from the payload.
    """
    if value is None:
        return None
    return round(value * 100, 3) if unit == "percent" else value


#: The envelope timestamp's zone, which the two shapes also disagree about: one
#: stamps it in league-local time, the other in UTC. Part of the same inversion
#: as kickoff_at and match_time. The canonical contract is UTC everywhere.
_META_ZONE = {"per_book": "local", "composite": "utc"}


def _meta(config: LeagueConfig, generated_at: datetime) -> dict:
    zone = _META_ZONE[config.publish.legacy_contract]
    stamped = (
        generated_at.astimezone(timezone.utc) if zone == "utc"
        else generated_at.astimezone(ZoneInfo(config.timezone))
    )
    return {
        "competition_code": config.code,
        "season": config.season,
        "updated_at": stamped.isoformat(timespec="seconds"),
    }


def _per_book_row(config: LeagueConfig, signal: Signal, fetched_at: str) -> dict:
    """The shape that shows every book's price side by side."""
    local = signal.kickoff.astimezone(ZoneInfo(config.timezone))
    row: dict = {
        "fixture_id": signal.fixture_id,
        "round": int(signal.round) if str(signal.round).isdigit() else signal.round,
        # Local, with offset. The sibling shape puts UTC here.
        "kickoff_at": local.isoformat(timespec="seconds"),
        "home_team": signal.home_team,
        "away_team": signal.away_team,
        # UTC time of day. The sibling shape puts local time here.
        "match_time": signal.kickoff.astimezone(timezone.utc).strftime("%H:%M"),
        "home_win_prob": signal.probabilities[0],
        "draw_prob": signal.probabilities[1],
        "away_win_prob": signal.probabilities[2],
        "debias_method": signal.debias_method,
    }

    unit = config.publish.legacy_ev_unit
    for book in config.odds.bet_books:
        quotes = signal.quotes_for(book.key)
        for side in SIDES:
            row[books_module.odds_column(book, side)] = (
                quotes[side].odds if side in quotes else None
            )
        for side in SIDES:
            row[books_module.ev_column(book, side)] = (
                _ev_out(quotes[side].ev, unit) if side in quotes else None
            )

    for side in SIDES:
        best = signal.best.get(side)
        row[books_module.best_odds_column(side)] = best.odds if best else None
    for side in SIDES:
        best = signal.best.get(side)
        row[books_module.best_ev_column(side)] = _ev_out(best.ev, unit) if best else None
    for side in SIDES:
        best = signal.best.get(side)
        row[books_module.best_book_column(side)] = best.book if best else None

    row["signal_pick"] = signal.pick or EMPTY_PICK["per_book"]
    row["signal_state"] = signal.state or EMPTY_STATE["per_book"]
    best = signal.best.get(signal.pick) if signal.pick else None
    row["signal_book"] = best.book if (best and signal.fires) else None
    # "|"-joined, never a list: a list would not survive the scalar cleaning the
    # exporter applies, and the board splits on the pipe.
    row["signal_books"] = "|".join(signal.books) if signal.books else None

    for book in config.odds.bet_books:
        quotes = signal.quotes_for(book.key)
        stamp = next((q.last_update for q in quotes.values() if q.last_update), None)
        row[books_module.last_update_column(book)] = stamp
    row["fetched_at"] = fetched_at
    return row


def _composite_row(config: LeagueConfig, signal: Signal, fetched_at: str) -> dict:
    """The shape that shows one best price and a reference alongside it."""
    unit = config.publish.legacy_ev_unit
    local = signal.kickoff.astimezone(ZoneInfo(config.timezone))
    row: dict = {
        "fixture_id": signal.fixture_id,
        "home_team": signal.home_team,
        "away_team": signal.away_team,
        "round": int(signal.round) if str(signal.round).isdigit() else signal.round,
        # Local time of day, and UTC in kickoff_at. The sibling shape is reversed.
        "match_time": local.strftime("%H:%M"),
        "kickoff_at": _iso_z(signal.kickoff),
        "home_win_prob": signal.probabilities[0],
        "draw_prob": signal.probabilities[1],
        "away_win_prob": signal.probabilities[2],
    }
    for side in SIDES:
        best = signal.best.get(side)
        row[f"{side}_odds"] = best.odds if best else None
    for side in SIDES:
        best = signal.best.get(side)
        row[f"{side}_book"] = best.book if best else None
    for side in SIDES:
        best = signal.best.get(side)
        row[f"{side}_ev"] = _ev_out(best.ev, unit) if best else None

    row["signal_pick"] = signal.pick or EMPTY_PICK["composite"]
    row["signal_state"] = signal.state or EMPTY_STATE["composite"]

    # Two different notions of "the book", and the original shape distinguishes
    # them, so this does too.
    #
    # Provenance and price age describe the price the row was JUDGED on: the
    # best-EV side whether or not it cleared. A row that did not fire still says
    # which price it was judged against, and that is the point of publishing it.
    judged = signal.top_quote
    # bookmaker and last_update describe the price to actually BET. When nothing
    # fires there is no such book, and the field falls back to naming everyone
    # who contributed a price.
    fired = signal.top_quote if signal.fires else None

    contributing = sorted({q.book for q in signal.quotes})
    row["bookmaker"] = fired.book if fired else ("+".join(contributing) or None)
    row["price_proof"] = (judged.proof if judged else "") or ""

    anchor = signal.anchor_odds
    for index, side in enumerate(SIDES):
        row[f"pinnacle_{side}_odds"] = anchor[index] if anchor else None
    row["pinnacle_last_update"] = signal.anchor_last_update or None
    # Was the current-price fetch time. The current price is retired, so this is
    # when the anchor's OPENING price was captured. See docs/DECISIONS.md D1.
    row["pinnacle_fetched_at"] = signal.anchor_captured_at or None

    captured = judged.captured_at if judged else ""
    row["price_captured_at"] = captured or None
    row["price_age_h"] = _age_hours(captured, fetched_at)
    row["last_update"] = (fired.last_update if fired else None) or None
    row["fetched_at"] = fetched_at
    return row


def _age_hours(captured_at: str, fetched_at: str) -> float | None:
    """How stale the published price was at export time.

    Frozen at export, so it ages wrongly if a consumer caches the payload. Kept
    because the board reads it; the canonical contract publishes the capture
    timestamp instead and lets the reader do the arithmetic.
    """
    if not captured_at or not fetched_at:
        return None
    try:
        start = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((end - start).total_seconds() / 3600.0, 1)


RENDERERS = {"per_book": _per_book_row, "composite": _composite_row}


def _match_date(config: LeagueConfig, signal: Signal) -> str:
    """The local matchday, the same one the identifier is built from."""
    return signal.kickoff.astimezone(ZoneInfo(config.timezone)).strftime("%Y-%m-%d")


def _kickoff_field(config: LeagueConfig, signal: Signal) -> str:
    """Whichever spelling of the kickoff this league's shape expects.

    The two shapes disagree: one carries league-local with an offset, the other
    UTC. Reproduced rather than reconciled, because the board reads it. The
    canonical contract has one UTC field.
    """
    if config.publish.legacy_contract == "per_book":
        return signal.kickoff.astimezone(ZoneInfo(config.timezone)).isoformat(timespec="seconds")
    return _iso_z(signal.kickoff)


def upcoming_fixtures(
    config: LeagueConfig, signals: list[Signal], *, generated_at: datetime | None = None
) -> dict:
    """The fixture list, in the pre-merge shape.

    Fetched by the board for one league's shape and optional for the other's, so
    it is produced for both rather than conditionally: a file the consumer asks
    for and does not get is a failed fetch it has to tolerate.
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    return {
        "meta": _meta(config, generated_at),
        "rows": [{
            "fixture_id": s.fixture_id,
            "round": int(s.round) if str(s.round).isdigit() else s.round,
            "match_date": _match_date(config, s),
            "match_time": (
                s.kickoff.astimezone(timezone.utc).strftime("%H:%M")
                if config.publish.legacy_contract == "per_book"
                else s.kickoff.astimezone(ZoneInfo(config.timezone)).strftime("%H:%M")
            ),
            "kickoff_at": _kickoff_field(config, s),
            "home_team": s.home_team,
            "away_team": s.away_team,
        } for s in signals],
    }


def match_predictions(
    config: LeagueConfig, signals: list[Signal], *, generated_at: datetime | None = None
) -> dict:
    """Model probabilities and the fair price each implies.

    Fair odds are 1/p: the price at which expected value is exactly zero. A
    reference point against a quoted price, never a betting floor, since a signal
    requires a materially higher bar than zero expected value.

    **These are the RAW probabilities, before the market-anchored correction.**
    The signal file publishes the corrected ones, so the same fixture appears in
    two files with two different probabilities, differing by about half a
    percentage point where the correction applies, and nothing in either file
    says which is which. Reproduced because the board reads it. The canonical
    contract publishes one set with the method that produced it named alongside.
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    meta = _meta(config, generated_at)
    meta["model_name"] = config.model_name
    meta["model_version"] = config.model_version
    rows = []
    for s in signals:
        exact = s.raw_probabilities
        # Published to six places, but the fair price is taken from the unrounded
        # value. Inverting a rounded probability magnifies the rounding, and the
        # pre-merge exporter did it this way.
        home, draw, away = (round(p, 6) for p in exact)
        rows.append({
            "fixture_id": s.fixture_id,
            "round": int(s.round) if str(s.round).isdigit() else s.round,
            "match_date": _match_date(config, s),
            "kickoff_at": _kickoff_field(config, s),
            "home_team": s.home_team,
            "away_team": s.away_team,
            "home_win_prob": home, "draw_prob": draw, "away_win_prob": away,
            "home_win_fair_odds": round(1.0 / exact[0], 4) if exact[0] else None,
            "draw_fair_odds": round(1.0 / exact[1], 4) if exact[1] else None,
            "away_win_fair_odds": round(1.0 / exact[2], 4) if exact[2] else None,
        })
    return {"meta": meta, "rows": rows}


def market_comparison(
    config: LeagueConfig, signals: list[Signal], *, generated_at: datetime | None = None
) -> dict:
    """The signal payload in this league's pre-merge shape."""
    generated_at = generated_at or datetime.now(timezone.utc)
    fetched_at = _iso_z(generated_at)
    render = RENDERERS[config.publish.legacy_contract]
    return {
        "meta": _meta(config, generated_at),
        "rows": [render(config, s, fetched_at) for s in signals],
    }
