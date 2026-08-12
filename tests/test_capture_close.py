"""Window and finality tests for the closing-line capture.

Two bugs these exist to prevent:

1. Spending. The window is the only thing standing between a 5-minute tick and
   the monthly allowance. A 60-minute window costs ~468 of 500 credits/month at
   Liga MX's 39 fixtures; 20 minutes costs ~156. A fixture that never finalises
   re-spends on every tick until kickoff.

2. Storing a price captured hours out as a "close". That already happened once:
   13 lines captured a median 35h before kickoff landed in pinnacle_close_* and
   carried 105.5% overround against 103.4% for real closes, quietly poisoning
   every CLV number computed against them.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from ligamx.odds.capture_close import (
    CLOSE_TARGET_MINUTES,
    DEFAULT_WINDOW_MINUTES,
    extract_rows,
    finalised_fixtures,
    fixtures_in_window,
)
from ligamx.odds.capture_store import append_snapshots

KICKOFF = datetime(2026, 8, 15, 23, 0, tzinfo=timezone.utc)


class _Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.target = os.path.join(self.dir.name, "upcoming.csv")
        self.history = os.path.join(self.dir.name, "history.csv")
        pd.DataFrame([{
            "Season": "Apertura 2026", "round": "4",
            "Date": "2026-08-15", "Time": "17:00",
            "Home": "Atlante", "Away": "Toluca",
            "kickoff_utc": KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sofa_event_id": "1",
        }]).to_csv(self.target, index=False)

    def tearDown(self):
        self.dir.cleanup()

    def _window(self, now, **kw):
        return fixtures_in_window(now, target_path=self.target,
                                  history_path=self.history, **kw)

    def _store_close(self, fetched_at, bookmaker="pinnacle"):
        append_snapshots(pd.DataFrame([{
            "event_id": "abc", "commence_time": KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_home_team": "Atlante FC", "api_away_team": "Toluca",
            "home_team": "Atlante", "away_team": "Toluca",
            "home_odds": "6.4", "draw_odds": "4.5", "away_odds": "1.47",
            "bookmaker": bookmaker, "market": "h2h", "regions": "theoddsapi",
            "last_update": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }]), snapshot_type="close", path=self.history)


class TestWindow(_Base):
    def test_outside_the_window_spends_nothing(self):
        self.assertEqual(self._window(KICKOFF - timedelta(hours=3)), [])

    def test_just_before_the_window_opens_is_still_excluded(self):
        edge = KICKOFF - timedelta(minutes=DEFAULT_WINDOW_MINUTES + 1)
        self.assertEqual(self._window(edge), [])

    def test_inside_the_window_is_captured(self):
        got = self._window(KICKOFF - timedelta(minutes=8))
        self.assertEqual([f.label for f in got], ["Atlante vs Toluca"])

    def test_kickoff_itself_closes_the_window(self):
        """At kickoff the fixture leaves the pre-match feed; anything returned
        after this point is an in-play price, not a close."""
        self.assertEqual(self._window(KICKOFF), [])
        self.assertEqual(self._window(KICKOFF + timedelta(minutes=5)), [])

    def test_default_window_stays_inside_the_monthly_budget(self):
        """39 fixtures/month x (window / 5-min tick) credits must clear 500."""
        ticks_per_fixture = DEFAULT_WINDOW_MINUTES / 5.0
        self.assertLess(39 * ticks_per_fixture, 500)


class TestFinality(_Base):
    def test_a_close_inside_the_target_band_finalises_the_fixture(self):
        """The spend stops here. Without it every remaining tick re-bills."""
        self._store_close(KICKOFF - timedelta(minutes=CLOSE_TARGET_MINUTES - 2))
        self.assertEqual(self._window(KICKOFF - timedelta(minutes=4)), [])

    def test_a_close_captured_too_early_does_not_finalise(self):
        """Captured at the top of the window, still 18 minutes out -- keep going,
        the point is to get nearer to kickoff."""
        self._store_close(KICKOFF - timedelta(minutes=18))
        got = self._window(KICKOFF - timedelta(minutes=4))
        self.assertEqual([f.label for f in got], ["Atlante vs Toluca"])

    def test_an_exchange_close_does_not_finalise(self):
        """Pinnacle is the benchmark and alone gates the spend; Betfair riding
        along free must not end the capture."""
        self._store_close(KICKOFF - timedelta(minutes=2), bookmaker="betfair_ex_eu")
        got = self._window(KICKOFF - timedelta(minutes=1))
        self.assertEqual([f.label for f in got], ["Atlante vs Toluca"])

    def test_a_post_kickoff_capture_does_not_finalise(self):
        """Negative lead means the row is in-play, not a close."""
        self._store_close(KICKOFF + timedelta(minutes=3))
        self.assertEqual(finalised_fixtures(self.history), set())


class TestExtractRows(_Base):
    def _event(self, home="Atlante FC", away="Toluca", draw=4.57):
        return {
            "id": "abc", "commence_time": "2026-08-15T23:00:00Z",
            "home_team": home, "away_team": away,
            "bookmakers": [{
                "key": "pinnacle", "last_update": "2026-08-15T22:52:00Z",
                "markets": [{"key": "h2h", "last_update": "2026-08-15T22:52:00Z",
                             "outcomes": [{"name": home, "price": 6.41},
                                          {"name": away, "price": 1.47},
                                          {"name": "Draw", "price": draw}]}],
            }],
        }

    def test_in_window_fixture_produces_a_row(self):
        rows = extract_rows([self._event()], {("atlante", "toluca")}, "2026-08-15T22:52:00Z")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["home_team"], "Atlante")
        self.assertEqual(rows[0]["home_odds"], 6.41)

    def test_out_of_window_fixtures_in_the_same_response_are_dropped(self):
        """One /odds call returns the whole slate. Only the in-window fixtures
        may be stored -- this is the 35h-lead bug's actual guard."""
        self.assertEqual(extract_rows([self._event()], set(), "2026-08-15T22:52:00Z"), [])

    def test_a_book_missing_the_draw_is_skipped(self):
        rows = extract_rows([self._event(draw=None)], {("atlante", "toluca")},
                            "2026-08-15T22:52:00Z")
        self.assertEqual(rows, [])

    def test_unmapped_team_is_skipped_not_raised(self):
        rows = extract_rows([self._event(home="Some New Club")], {("atlante", "toluca")},
                            "2026-08-15T22:52:00Z")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
