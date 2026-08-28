"""Dedup regression tests for the odds-capture history store.

The bug these exist to prevent: a capture loop that re-appends the same unmoved
line on every tick. At a 5-minute cadence that is ~288 duplicate rows per fixture
per day, and it makes "earliest fetched_at wins" -- the rule the reducer uses to
pick an opening line -- meaningless.
"""

import os
import tempfile
import unittest

import pandas as pd

from ligamx.odds.capture_store import (
    HISTORY_COLUMNS,
    append_snapshots,
    load_history,
)


def _row(**over):
    base = {
        "event_id": "72055084", "commence_time": "2026-08-15T23:00:00Z",
        "api_home_team": "Atlante FC", "api_away_team": "Deportivo Toluca FC",
        "home_team": "Atlante", "away_team": "Toluca",
        "home_odds": "6.5", "draw_odds": "4.2", "away_odds": "1.44",
        "bookmaker": "betano", "market": "h2h", "regions": "oddsapiio",
        "last_update": "2026-08-10T19:38:34.147Z",
        "fetched_at": "2026-08-12T09:00:00Z",
    }
    base.update(over)
    return base


class TestCaptureStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "history.csv")

    def tearDown(self):
        self.dir.cleanup()

    def _append(self, rows, **kw):
        kw.setdefault("snapshot_type", "open")
        return append_snapshots(pd.DataFrame(rows), path=self.path, **kw)

    def test_missing_file_reads_as_empty_schema(self):
        df = load_history(self.path)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), HISTORY_COLUMNS)

    def test_first_append_writes_all_rows(self):
        _, n = self._append([_row()])
        self.assertEqual(n, 1)
        self.assertEqual(len(load_history(self.path)), 1)

    def test_repolling_an_unmoved_line_appends_nothing(self):
        """The core guarantee. fetched_at changes; the key must ignore it."""
        self._append([_row()])
        _, n = self._append([_row(fetched_at="2026-08-12T09:05:00Z")])
        self.assertEqual(n, 0)
        self.assertEqual(len(load_history(self.path)), 1)

    def test_moved_line_is_a_new_row(self):
        self._append([_row()])
        _, n = self._append([_row(last_update="2026-08-12T10:00:00Z",
                                  home_odds="6.2")])
        self.assertEqual(n, 1)
        self.assertEqual(len(load_history(self.path)), 2)

    def test_two_books_sharing_a_last_update_both_land(self):
        """Duel and Betano can carry the same feed timestamp; bookmaker is in the key."""
        _, n = self._append([_row(bookmaker="betano"), _row(bookmaker="duel")])
        self.assertEqual(n, 2)

    def test_open_and_close_of_the_same_line_coexist(self):
        self._append([_row()], snapshot_type="open")
        _, n = self._append([_row()], snapshot_type="close")
        self.assertEqual(n, 1)

    def test_duplicates_within_one_batch_collapse(self):
        _, n = self._append([_row(), _row()])
        self.assertEqual(n, 1)

    def test_numeric_looking_ids_survive_a_round_trip(self):
        """dtype=str on read is what stops 72055084 becoming 72055084.0 and
        re-appending as a new key on the next tick."""
        self._append([_row(event_id="72055084")])
        self._append([_row(event_id="72055084")])
        df = load_history(self.path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["event_id"], "72055084")

    def test_idle_tick_leaves_the_file_untouched(self):
        """The workflow's `appended` output keys off a clean tree; a no-op append
        that rewrote the file would trigger empty commits every tick."""
        self._append([_row()])
        before = os.path.getmtime(self.path)
        os.utime(self.path, (before - 10, before - 10))
        stamped = os.path.getmtime(self.path)
        self._append([_row()])
        self.assertEqual(os.path.getmtime(self.path), stamped)

    def test_invalid_snapshot_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self._append([_row()], snapshot_type="opening")


if __name__ == "__main__":
    unittest.main()
