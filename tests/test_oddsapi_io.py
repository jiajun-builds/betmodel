"""Tests for the odds-api.io 1xBet opening-line capture.

The migration's whole premise is that a row produced from a *different* provider is
indistinguishable, downstream, from one The Odds API produced. Nothing else in the
pipeline was changed to accommodate it, so these tests pin the two contracts that
silently carry that weight:

* **Row shape and vocabulary.** ``event_to_row`` must emit exactly ``OUTPUT_COLUMNS``
  with ``bookmaker="onexbet"`` — The Odds API's key, not odds-api.io's ``"1xbet"``.
  ``export_upcoming_market_comparison`` matches that string exactly (its
  ``load_open_snapshots`` filters ``hist["bookmaker"] == bookmaker``), so a drifted key
  would not raise anywhere: the fixture would just quietly lose its bet price, EV and
  signal. Pinned by ``test_row_matches_output_columns`` / ``test_row_uses_theoddsapi_vocabulary``.

* **Pending logic.** With no predicted window left, "pending" is the only thing standing
  between an idle tick and burning the ~500/day request budget. Pinned by the
  ``test_pending_*`` cases.

The unmapped-team case gets its own test because this module deliberately *differs*
from ``fetch_pinnacle_spreads.extract_rows``, which raises on an unknown club. Here a
supplementary source must never take the dashboard publish down, so it returns None.

Fixtures below are trimmed copies of real 2026-08-02 payloads (Shandong Taishan vs
Tianjin Jinmen Tiger — the fixture whose lost open motivated the migration).

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

from csl.odds.fetch_onexbet_open import (  # noqa: E402
    load_upcoming,
    match_events_to_pending,
    pending_fixtures,
)
from csl.odds.fetch_pinnacle_spreads import OUTPUT_COLUMNS, TeamMapping  # noqa: E402
from csl.odds.oddsapi_io import (  # noqa: E402
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

MAPPING = TeamMapping(
    odds_to_standard={},
    standard_to_standard={"Shandong Taishan": "Shandong Taishan",
                          "Tianjin Jinmen Tiger": "Tianjin Jinmen Tiger",
                          "Qingdao Hainiu": "Qingdao Hainiu"},
    match_to_standard={},
    oddsapiio_to_standard={"Shandong Taishan FC": "Shandong Taishan",
                           "Tianjin Jinmen Tiger": "Tianjin Jinmen Tiger",
                           "Qingdao Hainiu FC": "Qingdao Hainiu"},
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


# ------------------------------------------------------------------------ pending


def _pending_labels(fixture_rows, history_rows, now=NOW):
    with tempfile.TemporaryDirectory() as tmp:
        target = _fixtures_csv(os.path.join(tmp, "up.csv"), fixture_rows)
        history = _history_csv(os.path.join(tmp, "hist.csv"), history_rows)
        return [f.label for f in pending_fixtures(now, target_path=target, history_path=history)]


def test_pending_includes_fixture_without_onexbet_open() -> None:
    labels = _pending_labels(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")], []
    )
    assert labels == ["Shandong Taishan vs Tianjin Jinmen Tiger"]


def test_pending_excludes_fixture_with_onexbet_open() -> None:
    labels = _pending_labels(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        [{"home_team": "Shandong Taishan", "away_team": "Tianjin Jinmen Tiger",
          "bookmaker": "onexbet", "snapshot_type": "open"}],
    )
    assert labels == []


def test_pending_ignores_other_books_opens() -> None:
    """A Pinnacle open must not stop us chasing the 1xBet price — different provider now."""
    labels = _pending_labels(
        [("22", "2026-08-09", "14:00", "Shandong Taishan", "Tianjin Jinmen Tiger")],
        [{"home_team": "Shandong Taishan", "away_team": "Tianjin Jinmen Tiger",
          "bookmaker": "pinnacle", "snapshot_type": "open"}],
    )
    assert labels == ["Shandong Taishan vs Tianjin Jinmen Tiger"]


def test_pending_excludes_started_fixture() -> None:
    """Past kickoff the pre-match line is gone; keeping it pending would burn requests forever."""
    labels = _pending_labels(
        [("21", "2026-08-01", "12:00", "Shandong Taishan", "Tianjin Jinmen Tiger")], []
    )
    assert labels == []


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
    event, fixture = matched[0]
    assert event["id"] == 68995314
    assert fixture.round == "22"


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
