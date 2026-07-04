import os
import unittest

import pandas as pd

import config
from src.data_updater import DataUpdater
from src.xg_calculator import XGCalculator


class TestXGReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.tmp_path = os.path.join(cls.project_root, "tests", "tmp_MEX_ligamx.csv")
        cls.original_data_file = config.DATA_FILE
        config.DATA_FILE = "tests/tmp_MEX_ligamx.csv"

    @classmethod
    def tearDownClass(cls):
        config.DATA_FILE = cls.original_data_file
        if os.path.exists(cls.tmp_path):
            os.remove(cls.tmp_path)

    def setUp(self):
        self.columns = [
            "Country", "League", "Season", "round", "Date", "Time", "Home", "Away",
            "HG", "AG", "HxG", "AxG", "HExpG+", "AExpG+", "Res", "PSCH", "PSCD", "PSCA",
        ]
        pd.DataFrame(columns=self.columns).to_csv(self.tmp_path, index=False)
        self.updater = DataUpdater()
        self.calc = XGCalculator()

    def _write_rows(self, rows):
        pd.DataFrame(rows, columns=self.columns).to_csv(self.tmp_path, index=False)

    def _read_df(self):
        return pd.read_csv(self.tmp_path)

    def test_formula_weights(self):
        home = self.calc.calculate_hext(hg=2, hxg=1.0)
        away = self.calc.calculate_aext(ag=1, axg=0.8)
        self.assertAlmostEqual(home, 1.45, places=9)  # 0.55*1.0 + 0.45*2
        self.assertAlmostEqual(away, 0.89, places=9)  # 0.55*0.8 + 0.45*1

    def test_upsert_add_new_row(self):
        match = {
            "home_team": "Atlas",
            "away_team": "Pachuca",
            "home_goals": 2,
            "away_goals": 1,
            "commence_time": "2026-04-20T02:00:00",
            "HxG": 1.3,
            "AxG": 0.9,
        }
        stats = self.updater.upsert_matches([match])
        self.assertEqual(stats, {"added": 1, "updated": 0, "unchanged": 0, "skipped": 0})

        df = self._read_df()
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.loc[0, "HxG"], 1.3, places=9)
        self.assertAlmostEqual(df.loc[0, "AxG"], 0.9, places=9)
        self.assertAlmostEqual(df.loc[0, "HExpG+"], 1.615, places=9)
        self.assertAlmostEqual(df.loc[0, "AExpG+"], 0.945, places=9)

    def test_upsert_updates_existing_changed_xg(self):
        self._write_rows([{
            "Country": "Mexico", "League": "Liga MX", "Season": "2025/2026", "round": "",
            "Date": "2026/04/20", "Time": "02:00", "Home": "Atlas", "Away": "Pachuca",
            "HG": 2, "AG": 1, "HxG": 1.0, "AxG": 0.7, "HExpG+": 1.55, "AExpG+": 0.865,
            "Res": "H", "PSCH": "", "PSCD": "", "PSCA": "",
        }])

        match = {
            "home_team": "Atlas",
            "away_team": "Pachuca",
            "home_goals": 2,
            "away_goals": 1,
            "commence_time": "2026-04-20T02:00:00",
            "HxG": 1.5,
            "AxG": 1.2,
        }
        stats = self.updater.upsert_matches([match])
        self.assertEqual(stats, {"added": 0, "updated": 1, "unchanged": 0, "skipped": 0})

        df = self._read_df()
        self.assertAlmostEqual(df.loc[0, "HxG"], 1.5, places=9)
        self.assertAlmostEqual(df.loc[0, "AxG"], 1.2, places=9)
        self.assertAlmostEqual(df.loc[0, "HExpG+"], 1.725, places=9)
        self.assertAlmostEqual(df.loc[0, "AExpG+"], 1.11, places=9)

    def test_upsert_noop_when_unchanged(self):
        self._write_rows([{
            "Country": "Mexico", "League": "Liga MX", "Season": "2025/2026", "round": "",
            "Date": "2026/04/20", "Time": "02:00", "Home": "Atlas", "Away": "Pachuca",
            "HG": 2, "AG": 1, "HxG": 1.5, "AxG": 1.2, "HExpG+": 1.725, "AExpG+": 1.11,
            "Res": "H", "PSCH": "", "PSCD": "", "PSCA": "",
        }])

        match = {
            "home_team": "Atlas",
            "away_team": "Pachuca",
            "home_goals": 2,
            "away_goals": 1,
            "commence_time": "2026-04-20T02:00:00",
            "HxG": 1.5,
            "AxG": 1.2,
        }
        stats = self.updater.upsert_matches([match])
        self.assertEqual(stats, {"added": 0, "updated": 0, "unchanged": 1, "skipped": 0})

    def test_upsert_does_not_overwrite_nonzero_with_zero(self):
        self._write_rows([{
            "Country": "Mexico", "League": "Liga MX", "Season": "2025/2026", "round": "",
            "Date": "2026/04/20", "Time": "02:00", "Home": "Atlas", "Away": "Pachuca",
            "HG": 2, "AG": 1, "HxG": 1.5, "AxG": 1.2, "HExpG+": 1.725, "AExpG+": 1.11,
            "Res": "H", "PSCH": "", "PSCD": "", "PSCA": "",
        }])

        match = {
            "home_team": "Atlas",
            "away_team": "Pachuca",
            "home_goals": 2,
            "away_goals": 1,
            "commence_time": "2026-04-20T02:00:00",
            "HxG": 0.0,
            "AxG": 0.0,
        }
        stats = self.updater.upsert_matches([match])
        self.assertEqual(stats, {"added": 0, "updated": 0, "unchanged": 0, "skipped": 1})

        df = self._read_df()
        self.assertAlmostEqual(df.loc[0, "HxG"], 1.5, places=9)
        self.assertAlmostEqual(df.loc[0, "AxG"], 1.2, places=9)

    def test_recalculate_all_expected_goals(self):
        self._write_rows([
            {
                "Country": "Mexico", "League": "Liga MX", "Season": "2025/2026", "round": "",
                "Date": "2026/04/20", "Time": "02:00", "Home": "Atlas", "Away": "Pachuca",
                "HG": 2, "AG": 1, "HxG": 1.3, "AxG": 0.9, "HExpG+": 99.0, "AExpG+": 99.0,
                "Res": "H", "PSCH": "", "PSCD": "", "PSCA": "",
            },
            {
                "Country": "Mexico", "League": "Liga MX", "Season": "2025/2026", "round": "",
                "Date": "2026/04/21", "Time": "04:00", "Home": "Toluca", "Away": "Monterrey",
                "HG": 1, "AG": 2, "HxG": 0.7, "AxG": 1.4, "HExpG+": 88.0, "AExpG+": 88.0,
                "Res": "A", "PSCH": "", "PSCD": "", "PSCA": "",
            },
        ])

        rows_recalculated = self.updater.recalculate_all_expected_goals()
        self.assertEqual(rows_recalculated, 2)

        df = self._read_df()
        self.assertAlmostEqual(df.loc[0, "HExpG+"], 1.615, places=9)
        self.assertAlmostEqual(df.loc[0, "AExpG+"], 0.945, places=9)
        self.assertAlmostEqual(df.loc[1, "HExpG+"], 0.835, places=9)
        self.assertAlmostEqual(df.loc[1, "AExpG+"], 1.67, places=9)


if __name__ == "__main__":
    unittest.main()
