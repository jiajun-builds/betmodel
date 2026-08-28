"""Regression tests for the opening-line capture gate.

The bug these exist to prevent: overwriting a banked opening line with a
mid-market price. Betano UK and Duel do not open together (measured 2026-08-12:
6/10 and 9/10 of the same slate), so most requests are made on one book's behalf
while the other's *current* price rides along in the response. If the writer
iterates the full book list instead of `missing`, that current price is stored as
snapshot_type="open" and the real opener is gone.

Dedup does not catch it: Duel stamps every fixture with a single shared
feed-refresh updatedAt, so its last_update moves whether or not its price did.
The gate tested here is the only defence.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from ligamx.odds.capture_store import append_snapshots
from ligamx.odds.fetch_oddsapiio_opens import (
    DEFAULT_LOOKAHEAD_DAYS,
    match_events_to_pending,
    pending_fixtures,
)
from ligamx.odds.oddsapi_io import BETANO, DUEL

NOW = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)

FIXTURES = [
    # (home, away, days from NOW)
    ("Atlante", "Toluca", 3),
    ("Monterrey", "FC Juarez", 4),
    ("Leon", "Monterrey", 30),      # outside any sane lookahead
    ("Pachuca", "Puebla", -1),      # already kicked off
]


class _Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.target = os.path.join(self.dir.name, "upcoming.csv")
        self.history = os.path.join(self.dir.name, "history.csv")
        rows = [{
            "Season": "Apertura 2026", "round": "4",
            "Date": (NOW + timedelta(days=d)).strftime("%Y-%m-%d"), "Time": "17:00",
            "Home": h, "Away": a,
            "kickoff_utc": (NOW + timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sofa_event_id": "1",
        } for h, a, d in FIXTURES]
        pd.DataFrame(rows).to_csv(self.target, index=False)

    def tearDown(self):
        self.dir.cleanup()

    def _bank_open(self, home, away, bookmaker, **over):
        row = {
            "event_id": f"oddsapiio:{abs(hash((home, away, bookmaker))) % 10**8}",
            "commence_time": "2026-08-15T23:00:00Z",
            "api_home_team": home, "api_away_team": away,
            "home_team": home, "away_team": away,
            "home_odds": "2.0", "draw_odds": "3.4", "away_odds": "3.8",
            "bookmaker": bookmaker, "market": "h2h", "regions": "oddsapiio",
            "last_update": "2026-08-10T19:38:34.147Z",
            "fetched_at": "2026-08-10T20:00:00Z",
        }
        row.update(over)
        append_snapshots(pd.DataFrame([row]), snapshot_type="open", path=self.history)

    def _pending(self, **kw):
        kw.setdefault("target_path", self.target)
        kw.setdefault("history_path", self.history)
        return pending_fixtures(NOW, **kw)


class TestPendingSet(_Base):
    def test_empty_history_means_every_book_owes_every_fixture(self):
        pending = self._pending()
        self.assertEqual([p.label for p in pending],
                         ["Atlante vs Toluca", "Monterrey vs FC Juarez"])
        for p in pending:
            self.assertEqual(p.missing, (BETANO, DUEL))

    def test_kicked_off_and_far_future_fixtures_are_excluded(self):
        """A started match's pre-match line is gone; a fixture 30 days out would
        burn the request cap every tick on a line no book has posted."""
        labels = [p.label for p in self._pending()]
        self.assertNotIn("Pachuca vs Puebla", labels)
        self.assertNotIn("Leon vs Monterrey", labels)

    def test_pending_is_ordered_soonest_kickoff_first(self):
        """An overflowing set must spend its batch on the lines opening now."""
        pending = self._pending()
        self.assertEqual([p.fixture.kickoff for p in pending],
                         sorted(p.fixture.kickoff for p in pending))

    def test_lookahead_widening_pulls_in_the_far_fixture(self):
        labels = [p.label for p in self._pending(lookahead_days=45)]
        self.assertIn("Leon vs Monterrey", labels)

    def test_default_lookahead_sits_outside_betanos_median_open(self):
        """Betano's measured median open is T-293h (~12.2d). A lookahead inside
        that would make the fixture pending only after it had already opened."""
        self.assertGreater(DEFAULT_LOOKAHEAD_DAYS * 24, 293)


class TestPerBookGate(_Base):
    def test_one_book_banked_leaves_only_the_other_missing(self):
        """THE gate. Betano's open is banked, so the fixture stays pending for
        Duel -- but Betano must not be in `missing`, or the next tick writes
        Betano's current price over the opener."""
        self._bank_open("Atlante", "Toluca", "betano")
        entry = next(p for p in self._pending() if p.label == "Atlante vs Toluca")
        self.assertEqual(entry.missing, (DUEL,))
        self.assertNotIn(BETANO, entry.missing)

    def test_fixture_drops_out_once_every_book_has_opened(self):
        self._bank_open("Atlante", "Toluca", "betano")
        self._bank_open("Atlante", "Toluca", "duel")
        self.assertNotIn("Atlante vs Toluca", [p.label for p in self._pending()])

    def test_a_close_row_does_not_satisfy_the_open_gate(self):
        """Only snapshot_type=open counts. A close stored for the same fixture
        must not make it look like the opener was captured."""
        row = pd.DataFrame([{
            "event_id": "x", "commence_time": "", "api_home_team": "Atlante",
            "api_away_team": "Toluca", "home_team": "Atlante", "away_team": "Toluca",
            "home_odds": "2.0", "draw_odds": "3.4", "away_odds": "3.8",
            "bookmaker": "betano", "market": "h2h", "regions": "oddsapiio",
            "last_update": "2026-08-15T22:50:00Z", "fetched_at": "2026-08-15T22:52:00Z",
        }])
        append_snapshots(row, snapshot_type="close", path=self.history)
        entry = next(p for p in self._pending() if p.label == "Atlante vs Toluca")
        self.assertEqual(entry.missing, (BETANO, DUEL))

    def test_book_selection_narrows_the_gate(self):
        entry = next(p for p in self._pending(books=(DUEL,))
                     if p.label == "Atlante vs Toluca")
        self.assertEqual(entry.missing, (DUEL,))


class TestEventMatching(_Base):
    def _event(self, home, away, ident=1):
        return {"id": ident, "home": home, "away": away,
                "date": "2026-08-15T23:00:00Z"}

    def test_provider_spellings_match_repo_fixtures(self):
        """odds-api.io says 'Atlante FC' / 'Deportivo Toluca FC'; the fixtures CSV
        says 'Atlante' / 'Toluca'. API_TO_STANDARD is what bridges them."""
        matched = match_events_to_pending(
            [self._event("Atlante FC", "Deportivo Toluca FC")], self._pending())
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][1].label, "Atlante vs Toluca")

    def test_unmappable_name_is_skipped_not_raised(self):
        """One unknown club must not take down the tick and cost every other
        fixture its opening line."""
        matched = match_events_to_pending(
            [self._event("Some New Club FC", "Deportivo Toluca FC")], self._pending())
        self.assertEqual(matched, [])

    def test_reversed_fixture_does_not_match(self):
        """Home and away are not interchangeable -- Liga MX plays both legs."""
        matched = match_events_to_pending(
            [self._event("Deportivo Toluca FC", "Atlante FC")], self._pending())
        self.assertEqual(matched, [])


if __name__ == "__main__":
    unittest.main()
