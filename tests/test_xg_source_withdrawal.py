"""Tests for the rule that a withdrawn source may not erase stored xG.

SofaScore stopped serving xG for older Liga MX matches. Measured 2026-08-23: 678
of 1057 fetched events return 0/0, every one of them 2024 or older, and single
spaced re-queries return 0/0 too -- so it is the source dropping the data, not a
throttled bulk fetch. `MEX_ligamx.csv` still holds the real values, collected when
SofaScore did serve them, and no provider sells that history back.

Before this, reconcile() filed those rows under XG_DIFF, which made
`verify-xg --fix-xg-diffs` an instruction to overwrite two thirds of the model's
input feature with zeros. The flag is gated and documents itself as a deliberate
one-off migration, so nothing stopped a future run from taking it at its word.

Two independent defences, and both are tested here because they fail differently:

1. reconcile() classifies these as SOURCE_WITHDRAWN, so they never reach the fix
   loop -- and the audit's XG_DIFF count goes back to meaning something (7 real
   method divergences, not 685).
2. apply_fix() refuses a 0/0 write over a stored non-zero value regardless of how
   the row was classified, so removing or breaking (1) still cannot destroy data.
"""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from ligamx import paths
from ligamx.xg import verify_xg

# Mirrors the real MEX_ligamx.csv header (name and order), so a schema change
# there surfaces here rather than in a silent KeyError at audit time.
COLUMNS = [
    "Country", "League", "Season", "Round", "Date", "Time", "Home", "Away",
    "HxG", "AxG", "HG", "AG", "HExpG+", "AExpG+", "Res",
]


def _row(home, away, date, hxg, axg, hg=1, ag=0):
    return {
        "Country": "Mexico", "League": "Liga MX", "Season": "Apertura 2024",
        "Round": "1", "Date": date, "Time": "19:00", "Home": home, "Away": away,
        "HxG": hxg, "AxG": axg, "HG": hg, "AG": ag,
        "HExpG+": "", "AExpG+": "", "Res": "H",
    }


def _record(home, away, date, hxg, axg, hg=1, ag=0):
    return {
        "event_id": 1, "date": date, "season": "Apertura 2024", "round": "1",
        "home": home, "away": away, "hxg": hxg, "axg": axg, "hg": hg, "ag": ag,
    }


class TestSourceWithdrawal(unittest.TestCase):
    def setUp(self):
        fd, self.csv = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self.patcher = mock.patch.object(paths, "ligamx_data_csv", lambda: self.csv)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        os.unlink(self.csv)

    def _write(self, rows):
        pd.DataFrame(rows, columns=COLUMNS).to_csv(self.csv, index=False)

    def test_zero_from_source_is_not_a_disagreement(self):
        """A 0/0 record against stored xG is SOURCE_WITHDRAWN, never XG_DIFF."""
        self._write([_row("Leon", "Toluca", "2024-01-13", "1.4", "0.9")])
        res = verify_xg.reconcile([_record("Leon", "Toluca", "2024-01-13", 0.0, 0.0)])

        self.assertEqual(len(res["withdrawn"]), 1)
        self.assertEqual(res["xg_diff"], [])
        self.assertEqual(res["no_xg"], [])

    def test_real_divergence_is_still_a_disagreement(self):
        """The guard must not swallow a genuine numeric difference."""
        self._write([_row("Leon", "Toluca", "2024-01-13", "1.4", "0.9")])
        res = verify_xg.reconcile([_record("Leon", "Toluca", "2024-01-13", 1.7, 0.95)])

        self.assertEqual(len(res["xg_diff"]), 1)
        self.assertEqual(res["withdrawn"], [])

    def test_blank_csv_still_accepts_a_real_value(self):
        """A row we have nothing for is a fill, not a withdrawal."""
        self._write([_row("Leon", "Toluca", "2024-01-13", "0", "0")])
        res = verify_xg.reconcile([_record("Leon", "Toluca", "2024-01-13", 1.7, 0.95)])

        self.assertEqual(len(res["no_xg"]), 1)
        self.assertEqual(res["withdrawn"], [])

    def test_fix_refuses_zeros_even_when_misclassified(self):
        """Defence 2: the write guard holds without help from the classifier."""
        self._write([_row("Leon", "Toluca", "2024-01-13", "1.4", "0.9")])
        res = verify_xg.reconcile([_record("Leon", "Toluca", "2024-01-13", 0.0, 0.0)])

        # Simulate a future edit that drops the SOURCE_WITHDRAWN branch.
        res["xg_diff"] = res["withdrawn"]
        res["withdrawn"] = []

        stats = verify_xg.apply_fix(res, add_missing=False, correct_xg=True,
                                    fill_meta=False, resolve_meta_conflicts=False)

        self.assertEqual(stats["xg_refused"], 1)
        self.assertEqual(stats["xg_corrected"], 0)
        df = pd.read_csv(self.csv, dtype=str, keep_default_na=False)
        self.assertEqual(df.at[0, "HxG"], "1.4")
        self.assertEqual(df.at[0, "AxG"], "0.9")

    def test_fix_still_applies_a_real_correction(self):
        """The guard is about zeros only; a genuine correction must still land."""
        self._write([_row("Leon", "Toluca", "2024-01-13", "1.4", "0.9")])
        res = verify_xg.reconcile([_record("Leon", "Toluca", "2024-01-13", 1.7, 0.95)])

        stats = verify_xg.apply_fix(res, add_missing=False, correct_xg=True,
                                    fill_meta=False, resolve_meta_conflicts=False)

        self.assertEqual(stats["xg_corrected"], 1)
        self.assertEqual(stats["xg_refused"], 0)
        df = pd.read_csv(self.csv, dtype=str, keep_default_na=False)
        self.assertEqual(df.at[0, "HxG"], "1.7")


if __name__ == "__main__":
    unittest.main()
