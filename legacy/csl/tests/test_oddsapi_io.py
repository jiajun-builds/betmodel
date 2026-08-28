"""Tests for the odds-api.io opening-line capture (1xBet + Duel).

The migration's whole premise is that a row produced from a *different* provider is
indistinguishable, downstream, from one The Odds API produced. Nothing else in the
pipeline was changed to accommodate it, so these tests pin the contracts that silently
carry that weight:

* **Row shape and vocabulary.** ``event_to_row`` must emit exactly ``OUTPUT_COLUMNS``
  with ``bookmaker="onexbet"`` — The Odds API's key, not odds-api.io's ``"1xbet"``.
  ``export_upcoming_market_comparison`` matches that string exactly (its
  ``load_open_snapshots`` filters ``hist["bookmaker"] == bookmaker``), so a drifted key
  would not raise anywhere: the fixture would just quietly lose its bet price, EV and
  signal. Pinned by ``test_row_matches_output_columns`` / ``test_row_uses_theoddsapi_vocabulary``.

* **Pending logic.** With no predicted window left, "pending" is the only thing standing
  between an idle tick and burning the ~500/day request budget. Pinned by the
  ``test_pending_*`` cases.

* **The per-book gate** (added with Duel, 2026-08-08). Pending is tracked per
  ``(fixture, book)``. If it collapsed back to per-fixture, a fixture with a banked 1xBet
  open but no Duel open would still be requested — and 1xBet's *current* price would be
  written as ``snapshot_type="open"``, silently replacing an opening line with a
  mid-market one. Nothing downstream would raise. Pinned by
  ``test_pending_tracks_each_book_independently`` and
  ``test_pending_drops_fixture_only_when_every_book_has_an_open``.

The unmapped-team case gets its own test because this module deliberately *differs*
from ``fetch_pinnacle_spreads.extract_rows``, which raises on an unknown club. Here a
supplementary source must never take the dashboard publish down, so it returns None.

Fixtures below are trimmed copies of real payloads (Shandong Taishan vs Tianjin Jinmen
Tiger — the fixture whose lost open motivated the migration — from 2026-08-02, plus the
two-book response shape observed 2026-08-08).

Runnable either way::

    pytest tests/test_oddsapi_io.py          # if pytest is installed
    python tests/test_oddsapi_io.py          # no pytest needed
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.odds.fetch_oddsapiio_opens import (  # noqa: E402
    _select_books,
    load_upcoming,
    match_events_to_pending,
    pending_fixtures,
)
from csl.odds.fetch_pinnacle_spreads import OUTPUT_COLUMNS, TeamMapping  # noqa: E402
from csl.odds.oddsapi_io import (  # noqa: E402
    CAPTURE_BOOKS,
    DUEL,
    ONEXBET,
    PROVIDER_TAG,
    TARGET_BOOKMAKER_KEY,
    event_to_row,
    extract_ml,
)
from csl.odds.snapshot_store import HISTORY_COLUMNS  # noqa: E402

NOW = datetime(2026, 8, 2, 13, 0, tzinfo=timezone.utc)

EVENT = {
    "id": 68995314,
    "home": "Shandong Taishan FC",
    "away": "Tianjin Jinmen Tiger",
    "date": "2026-08-09T14:00:00Z",
    "status": "pending",
    "bookmakers": {
        "1xbet": [
            {"name": "ML",
             "odds": [{"home": "1.74", "draw": "4", "away": "4.34"}],
             "updatedAt": "2026-08-02T12:39:07.12Z"},
            {"name": "Both Teams To Score",
             "odds": [{"yes": "1.55", "no": "2.297"}],
             "updatedAt": "2026-08-02T12:39:07.12Z"},
        ]
    },
}

# The real "line not posted yet" shape: /odds/multi returns the event with no markets.
EVENT_NO_PRICE = {
    "id": 68995316,
    "home": "Shandong Taishan FC",
    "away": "Qingdao Hainiu FC",
    "date": "2026-08-14T11:35:00Z",
    "bookmakers": {},
}

# `bookmakers=1xbet,Duel` response shape: both books under one event, keyed by
# odds-api.io's own casing ("1xbet" lowercase, "Duel" capitalised).
EVENT_BOTH_BOOKS = {
    "id": 68995314,
    "home": "Shandong Taishan FC",
    "away": "Tianjin Jinmen Tiger",
    "date": "2026-08-09T14:00:00Z",
    "bookmakers": {
        "1xbet": [
            {"name": "ML",
             "odds": [{"home": "1.74", "draw": "4", "away": "4.34"}],
             "updatedAt": "2026-08-02T12:39:07.12Z"},
        ],
        "Duel": [
            {"name": "ML",
             "odds": [{"home": "1.76", "draw": "4.10", "away": "4.60"}],
             "updatedAt": "2026-08-08T12:05:43.851Z"},
            {"name": "Totals", "odds": [{"over": "1.9", "under": "1.9"}]},
        ],
    },
}

# Only one of the two books quotes it — the routine mixed state, not an error.
EVENT_ONLY_DUEL = {
    "id": 68995320,
    "home": "Qingdao West Coast FC",
    "away": "Chongqing Tonglianglong FC",
    "date": "2026-08-14T12:00:00Z",
    "bookmakers": {
        "Duel": [
            {"name": "ML",
             "odds": [{"home": "2.22", "draw": "3.95", "away": "3.05"}],
             "updatedAt": "2026-08-08T14:29:33.6Z"},
        ],
    },
}

MAPPING = TeamMapping(
    odds_to_standard={},
    standard_to_standard={"Shandong Taishan": "Shandong Taishan",
                          "Tianjin Jinmen Tiger": "Tianjin Jinmen Tiger",
                          "Qingdao Hainiu": "Qingdao Hainiu",
                          "Qingdao West Coast": "Qingdao West Coast",
                          "Chongqing Tonglianglong": "Chongqing Tonglianglong"},
    match_to_standard={},
    oddsapiio_to_standard={"Shandong Taishan FC": "Shandong Taishan",
                           "Tianjin Jinmen Tiger": "Tianjin Jinmen Tiger",
                           "Qingdao Hainiu FC": "Qingdao Hainiu",
                           "Qingdao West Coast FC": "Qingdao West Coast",
                           "Chongqing Tonglianglong FC": "Chongqing Tonglianglong"},
)

FIXTURES_HEADER = "Wk,Date,Time,Home,Away"


def _fixtures_csv(path: str, rows: list[tuple[str, str, str, str, str]]) -> str:
    """Write an upcoming-fixtures CSV from ``(wk, date, time, home, away)`` tuples."""
    lines = [FIXTURES_HEADER] + [",".join(r) for r in rows]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _history_csv(path: str, rows: list[dict]) -> str:
    """Write a capture-history CSV with the real 17-column header."""
    lines = [",".join(HISTORY_COLUMNS)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in HISTORY_COLUMNS))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- rows


def test_row_matches_output_columns() -> None:
    """The row must be exactly the history store's 14 base columns — no more, no less."""
    row = event_to_row(EVENT, extract_ml(EVENT), MAPPING, fetched_at="2026-08-02T13:00:00Z")
    assert row is not None
    assert list(row) == list(OUTPUT_COLUMNS), list(row)


def test_row_uses_theoddsapi_vocabulary() -> None:
    """bookmaker/market must use The Odds API's strings so downstream needs no change."""
    row = event_to_row(EVENT, extract_ml(EVENT), MAPPING, fetched_at="2026-08-02T13:00:00Z")
    assert row["bookmaker"] == TARGET_BOOKMAKER_KEY == "onexbet"
    assert row["market"] == "h2h"
    # Provenance lives in `regions`, which has no odds-api.io meaning.
    assert row["regions"] == PROVIDER_TAG
    # Namespaced id: odds-api.io ids must not collide with The Odds API's in the dedup key.
    assert row["event_id"] == "oddsapiio:68995314"


def test_row_uses_duel_vocabulary() -> None:
    """Duel stores under its own key. A collision with "onexbet" would corrupt both books'
    opens in one store, and `load_open_snapshots` filters on exactly this string."""
    row = event_to_row(EVENT_BOTH_BOOKS, extract_ml(EVENT_BOTH_BOOKS, DUEL), MAPPING,
                       fetched_at="2026-08-08T15:00:00Z", book=DUEL)
    assert row is not None
    assert row["bookmaker"] == DUEL.stored_key == "duel"
    assert row["bookmaker"] != TARGET_BOOKMAKER_KEY
    assert list(row) == list(OUTPUT_COLUMNS)
    assert (row["home_odds"], row["draw_odds"], row["away_odds"]) == (1.76, 4.10, 4.60)


def test_both_books_share_an_event_id_but_differ_in_the_dedup_key() -> None:
    """One odds-api.io event id serves both books; `bookmaker` is what separates the rows.

    The store's dedup key is (event_id, bookmaker, last_update, snapshot_type) — if these
    two rows were equal on it, the second book's open would be silently dropped.
    """
    onexbet_row = event_to_row(EVENT_BOTH_BOOKS, extract_ml(EVENT_BOTH_BOOKS, ONEXBET),
                               MAPPING, fetched_at="2026-08-08T15:00:00Z", book=ONEXBET)
    duel_row = event_to_row(EVENT_BOTH_BOOKS, extract_ml(EVENT_BOTH_BOOKS, DUEL),
                            MAPPING, fetched_at="2026-08-08T15:00:00Z", book=DUEL)
    assert onexbet_row["event_id"] == duel_row["event_id"]
    key = ("event_id", "bookmaker", "last_update")
    assert tuple(onexbet_row[k] for k in key) != tuple(duel_row[k] for k in key)


def test_row_normalizes_teams_and_prices() -> None:
    row = event_to_row(EVENT, extract_ml(EVENT), MAPPING, fetched_at="2026-08-02T13:00:00Z")
    assert (row["home_team"], row["away_team"]) == ("Shandong Taishan", "Tianjin Jinmen Tiger")
    assert (row["api_home_team"], row["api_away_team"]) == ("Shandong Taishan FC",
                                                            "Tianjin Jinmen Tiger")
    # Prices arrive as strings and must be coerced; "4" must not stay an int-ish string.
    assert (row["home_odds"], row["draw_odds"], row["away_odds"]) == (1.74, 4.0, 4.34)
    assert row["last_update"] == "2026-08-02T12:39:07.12Z"
    assert row["fetched_at"] == "2026-08-02T13:00:00Z"


def test_unmapped_team_returns_none_instead_of_raising() -> None:
    """A supplementary source must not take the pipeline down over one unknown club."""
    stranger = dict(EVENT, home="Some New Club FC")
    assert event_to_row(stranger, extract_ml(stranger), MAPPING,
                        fetched_at="2026-08-02T13:00:00Z") is None


# ------------------------------------------------------------------------ markets


def test_extract_ml_picks_the_ml_market() -> None:
    assert extract_ml(EVENT) == (1.74, 4.0, 4.34, "2026-08-02T12:39:07.12Z")


def test_extract_ml_returns_none_when_unpriced() -> None:
    """No 1X2 price yet is the normal pre-open state, not an error."""
    assert extract_ml(EVENT_NO_PRICE) is None
    assert extract_ml({"bookmakers": {"1xbet": [{"name": "Totals", "odds": [{}]}]}}) is None
    assert extract_ml({}) is None


def test_extract_ml_reads_the_requested_book() -> None:
    """One response carries both books; each must yield its OWN price, not the other's."""
    assert extract_ml(EVENT_BOTH_BOOKS, ONEXBET) == (1.74, 4.0, 4.34, "2026-08-02T12:39:07.12Z")
    assert extract_ml(EVENT_BOTH_BOOKS, DUEL) == (1.76, 4.10, 4.60, "2026-08-08T12:05:43.851Z")


def test_extract_ml_is_none_for_a_book_absent_from_the_response() -> None:
    """/odds/multi omits a book that does not quote the event — routine, not an error."""
    assert extract_ml(EVENT_ONLY_DUEL, ONEXBET) is None
    assert extract_ml(EVENT_ONLY_DUEL, DUEL) == (2.22, 3.95, 3.05, "2026-08-08T14:29:33.6Z")


# ------------------------------------------------------------------------ pending


def _pending(fixture_rows, history_rows, now=NOW):
    with tempfile.TemporaryDirectory() as tmp:
        target = _fixtures_csv(os.path.join(tmp, "up.csv"), fixture_rows)
        history = _history_csv(os.path.join(tmp, "hist.csv"), history_rows)
        return pending_fixtures(now, target_path=target, history_path=history)


def _pending_labels(fixture_rows, history_rows, now=NOW):
    return [p.label for p in _pending(fixture_rows, history_rows, now)]


def _open_row(home, away, bookmaker):
    return {"home_team": home, "away_team": away,
            "bookmaker": bookmaker, "snapshot_type": "open"}


def test_pending_includes_fixture_without_any_open() -> None:
    pending = _pending(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")], []
    )
    assert [p.label for p in pending] == ["Shandong Taishan vs Tianjin Jinmen Tiger"]
    assert pending[0].missing == CAPTURE_BOOKS


def test_pending_tracks_each_book_independently() -> None:
    """A banked 1xBet open must NOT stop the chase for Duel's — nor make 1xBet eligible again.

    ``missing`` is what the capture loop iterates. If it still contained 1xBet here, the
    next tick would write 1xBet's current price over the opening line already stored.
    """
    pending = _pending(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        [_open_row("Shandong Taishan", "Tianjin Jinmen Tiger", "onexbet")],
    )
    assert [p.label for p in pending] == ["Shandong Taishan vs Tianjin Jinmen Tiger"]
    assert pending[0].missing == (DUEL,)


def test_pending_tracks_each_book_independently_other_way() -> None:
    pending = _pending(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        [_open_row("Shandong Taishan", "Tianjin Jinmen Tiger", "duel")],
    )
    assert pending[0].missing == (ONEXBET,)


def test_pending_drops_fixture_only_when_every_book_has_an_open() -> None:
    labels = _pending_labels(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        [_open_row("Shandong Taishan", "Tianjin Jinmen Tiger", "onexbet"),
         _open_row("Shandong Taishan", "Tianjin Jinmen Tiger", "duel")],
    )
    assert labels == []


def test_pending_ignores_other_books_opens() -> None:
    """A Pinnacle open must not stop us chasing these books — different provider now."""
    pending = _pending(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        [_open_row("Shandong Taishan", "Tianjin Jinmen Tiger", "pinnacle")],
    )
    assert [p.label for p in pending] == ["Shandong Taishan vs Tianjin Jinmen Tiger"]
    assert pending[0].missing == CAPTURE_BOOKS


def test_pending_excludes_started_fixture() -> None:
    """Past kickoff the pre-match line is gone; keeping it pending would burn requests forever."""
    labels = _pending_labels(
        [("21", "2026-08-01", "12:00", "Shandong Taishan", "Tianjin Jinmen Tiger")], []
    )
    assert labels == []


def test_pending_is_ordered_by_kickoff() -> None:
    """Two books widened the pending set past one batch; the soonest opens must win it."""
    labels = _pending_labels(
        [("24", "2026-08-20", "12:00", "Qingdao West Coast", "Chongqing Tonglianglong"),
         ("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger"),
         ("23", "2026-08-14", "11:35", "Shandong Taishan", "Qingdao Hainiu")],
        [],
    )
    assert labels == ["Shandong Taishan vs Tianjin Jinmen Tiger",
                      "Shandong Taishan vs Qingdao Hainiu",
                      "Qingdao West Coast vs Chongqing Tonglianglong"]


def test_load_upcoming_reads_round_and_utc_kickoff() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = _fixtures_csv(
            os.path.join(tmp, "up.csv"),
            [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        )
        (fixture,) = load_upcoming(target)
    assert fixture.round == "22"
    assert fixture.kickoff == datetime(2026, 8, 9, 14, 0, tzinfo=timezone.utc)
    assert fixture.key == ("shandong taishan", "tianjin jinmen tiger")


# ------------------------------------------------------------------------ matching


def test_match_events_maps_provider_spellings() -> None:
    """odds-api.io's names must resolve to the repo's standard names before matching."""
    with tempfile.TemporaryDirectory() as tmp:
        target = _fixtures_csv(
            os.path.join(tmp, "up.csv"),
            [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        )
        pending = pending_fixtures(NOW, target_path=target,
                                   history_path=_history_csv(os.path.join(tmp, "h.csv"), []))
    matched = match_events_to_pending([EVENT], pending, MAPPING)
    assert len(matched) == 1
    event, entry = matched[0]
    assert event["id"] == 68995314
    assert entry.fixture.round == "22"
    # The books still owed an open travel with the match — the capture loop reads them
    # off this, never off CAPTURE_BOOKS.
    assert entry.missing == CAPTURE_BOOKS


def test_match_events_skips_non_pending_and_unmapped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = _fixtures_csv(
            os.path.join(tmp, "up.csv"),
            [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        )
        pending = pending_fixtures(NOW, target_path=target,
                                   history_path=_history_csv(os.path.join(tmp, "h.csv"), []))
    # A real event for a fixture we are not chasing, plus one with an unknown club.
    others = [dict(EVENT, id=1, home="Qingdao Hainiu FC", away="Shandong Taishan FC"),
              dict(EVENT, id=2, home="Totally Unknown FC")]
    assert match_events_to_pending(others, pending, MAPPING) == []


# ------------------------------------------------------------------- book selection


def test_select_books_defaults_to_every_entitled_book() -> None:
    assert _select_books(None) == CAPTURE_BOOKS
    assert _select_books("") == CAPTURE_BOOKS


def test_select_books_is_case_insensitive() -> None:
    assert _select_books("duel") == (DUEL,)
    assert _select_books("1xbet, Duel") == (ONEXBET, DUEL)


def test_select_books_rejects_an_unknown_name() -> None:
    """A typo must fail loudly — silently capturing fewer books looks exactly like
    "that book never posts a price", which is unfalsifiable from the logs."""
    for bad in ("fanduel", "pinnacle", "Duel bet"):
        try:
            _select_books(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error, keep going
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
