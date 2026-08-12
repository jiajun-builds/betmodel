"""Tests for reducing the capture history into MEX_ligamx.csv.

Two things must hold, and both protect data that cannot be reconstructed:

1. Nothing is overwritten. pinnacle_close_* holds 856 values spliced from three
   sources and repaired by hand after a leakage bug; betano_open_* holds 461
   rows entered by hand. A reducer that clobbered either would destroy work with
   no way back.

2. An "open" only reaches the table if we were watching before the book priced
   it. Otherwise it is a mid-market price wearing an opener's label, and mixing
   those into the Betano series would corrupt the one positive-EV result the
   project has.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import pandas as pd

from ligamx.odds.capture_store import append_snapshots
from ligamx.odds.reduce_capture_history import build_records, merge

KICKOFF = datetime(2026, 9, 20, 23, 0, tzinfo=timezone.utc)
LOOKAHEAD = 14


class _Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.history = os.path.join(self.dir.name, "history.csv")
        self.table = os.path.join(self.dir.name, "MEX_ligamx.csv")
        cols = {
            "Date": "2026/09/20", "Home": "Atlante", "Away": "Toluca",
            "HG": "1", "AG": "0",
            "betano_open_h": "", "betano_open_d": "", "betano_open_a": "",
            "duel_open_h": "", "duel_open_d": "", "duel_open_a": "",
            "pinnacle_close_h": "", "pinnacle_close_d": "", "pinnacle_close_a": "",
        }
        pd.DataFrame([cols]).to_csv(self.table, index=False)
        self._patch = mock.patch("ligamx.odds.reduce_capture_history.paths.ligamx_data_csv",
                                 return_value=self.table)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.dir.cleanup()

    def _store(self, *, snapshot_type, bookmaker, fetched_at, odds=("2.0", "3.4", "3.8")):
        append_snapshots(pd.DataFrame([{
            "event_id": f"{bookmaker}:1", "commence_time": KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_home_team": "Atlante FC", "api_away_team": "Toluca",
            "home_team": "Atlante", "away_team": "Toluca",
            "home_odds": odds[0], "draw_odds": odds[1], "away_odds": odds[2],
            "bookmaker": bookmaker, "market": "h2h", "regions": "oddsapiio",
            "last_update": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }]), snapshot_type=snapshot_type, path=self.history)

    def _records(self):
        return build_records(history_path=self.history, lookahead_days=LOOKAHEAD)

    def _table(self):
        return pd.read_csv(self.table, dtype=str, keep_default_na=False)


class TestTrustGate(_Base):
    def test_open_seen_from_before_the_fixture_was_observable_is_trusted(self):
        """Capture began 30 days out; the fixture became pending at 14 days out,
        so no price can have escaped us."""
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30))
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=12))
        rec = self._records()[0]
        self.assertTrue(rec["open_trusted"])
        self.assertEqual(rec["open_h"], 2.0)

    def test_open_from_a_history_that_started_too_late_is_held_back(self):
        """Capture began 10 days out, inside the 14-day lookahead: the book could
        have opened before we ever looked."""
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=10))
        rec = self._records()[0]
        self.assertFalse(rec["open_trusted"])
        self.assertIsNone(rec["open_h"])
        self.assertAlmostEqual(rec["open_lead_h"], 240.0, places=1)

    def test_untrusted_open_still_reports_its_lead_for_audit(self):
        self._store(snapshot_type="open", bookmaker="duel",
                    fetched_at=KICKOFF - timedelta(hours=86))
        self.assertIsNotNone(self._records()[0]["open_lead_h"])

    def test_earliest_capture_wins_not_the_latest(self):
        """The opener is the first price seen; a later, better-looking price is
        still not the opening line."""
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30), odds=("2.0", "3.4", "3.8"))
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=13), odds=("9.9", "9.9", "9.9"))
        self.assertEqual(self._records()[0]["open_h"], 2.0)


class TestCloseGate(_Base):
    def _open_history(self):
        """Anchor the history early so the trust gate is not what's under test."""
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30))

    def test_close_near_kickoff_is_taken(self):
        self._open_history()
        self._store(snapshot_type="close", bookmaker="pinnacle",
                    fetched_at=KICKOFF - timedelta(minutes=8), odds=("6.4", "4.5", "1.47"))
        rec = next(r for r in self._records() if r["prefix"] == "pinnacle")
        self.assertEqual(rec["close_h"], 6.4)

    def test_close_captured_too_early_is_dropped(self):
        """The 35h-lead bug: a price that far out has a materially wider margin
        and is not a close."""
        self._open_history()
        self._store(snapshot_type="close", bookmaker="pinnacle",
                    fetched_at=KICKOFF - timedelta(hours=35))
        rec = next(r for r in self._records() if r["prefix"] == "pinnacle")
        self.assertIsNone(rec["close_h"])
        self.assertAlmostEqual(rec["close_lead_h"], 35.0, places=1)

    def test_in_play_capture_is_dropped(self):
        self._open_history()
        self._store(snapshot_type="close", bookmaker="pinnacle",
                    fetched_at=KICKOFF + timedelta(minutes=30))
        rec = next(r for r in self._records() if r["prefix"] == "pinnacle")
        self.assertIsNone(rec["close_h"])

    def test_latest_close_wins(self):
        self._open_history()
        self._store(snapshot_type="close", bookmaker="pinnacle",
                    fetched_at=KICKOFF - timedelta(minutes=18), odds=("6.0", "4.0", "1.5"))
        self._store(snapshot_type="close", bookmaker="pinnacle",
                    fetched_at=KICKOFF - timedelta(minutes=4), odds=("6.4", "4.5", "1.47"))
        rec = next(r for r in self._records() if r["prefix"] == "pinnacle")
        self.assertEqual(rec["close_h"], 6.4)


class TestMergeNeverOverwrites(_Base):
    def test_blank_cell_is_filled(self):
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30))
        stats = merge(self._records())
        self.assertEqual(stats["written"], 3)
        self.assertEqual(self._table().iloc[0]["betano_open_h"], "2.0")

    def test_existing_value_is_never_clobbered(self):
        """The guarantee. A hand-entered opener outranks anything we capture."""
        df = self._table()
        df.loc[0, ["betano_open_h", "betano_open_d", "betano_open_a"]] = ["1.11", "2.22", "3.33"]
        df.to_csv(self.table, index=False)

        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30))
        stats = merge(self._records())

        self.assertEqual(stats["written"], 0)
        self.assertEqual(stats["kept"], 3)
        self.assertEqual(list(self._table().iloc[0][
            ["betano_open_h", "betano_open_d", "betano_open_a"]]), ["1.11", "2.22", "3.33"])

    def test_held_back_open_writes_nothing(self):
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=10))
        self.assertEqual(merge(self._records())["written"], 0)
        self.assertEqual(self._table().iloc[0]["betano_open_h"], "")

    def test_a_book_with_no_close_column_is_counted_not_crashed(self):
        """betano/duel are opening-line sources; there are no betano_close_*
        columns and there should not be."""
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30))
        self._store(snapshot_type="close", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(minutes=5))
        stats = merge(self._records())
        self.assertEqual(stats["no_column"], 3)

    def test_unplayed_fixture_is_skipped_without_error(self):
        self._store(snapshot_type="open", bookmaker="betano",
                    fetched_at=KICKOFF - timedelta(days=30))
        recs = self._records()
        recs[0]["date"] = "2027/01/01"  # no such row in the table
        self.assertEqual(merge(recs)["no_row"], 1)


if __name__ == "__main__":
    unittest.main()
