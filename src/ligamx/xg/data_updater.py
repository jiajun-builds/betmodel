"""
Data Updater - updates MEX_ligamx.csv with new results, avoiding duplicates.
"""

import os
import pandas as pd
from typing import List, Dict
from datetime import datetime

from ligamx import config, paths
from ligamx.xg.xg_calculator import XGCalculator


class DataUpdater:
    def __init__(self):
        # Kept as a (possibly relative) config value so tests can redirect it;
        # resolved to an absolute path against the repo root via _abs_path().
        self.data_file = config.DATA_FILE
        self.xg_calculator = XGCalculator()

    def _abs_path(self) -> str:
        """Resolve self.data_file against the repo root (not the CWD)."""
        if os.path.isabs(self.data_file):
            return self.data_file
        return os.path.join(paths.project_root(), self.data_file)

    def _map_team_name(self, team_name: str) -> str:
        """Map Sofascore team names to standard team names for CSV."""
        return config.SOFASCORE_TO_STANDARD.get(team_name, team_name)

    def load_existing(self) -> pd.DataFrame:
        """Load existing CSV data."""
        df = pd.read_csv(self._abs_path())
        return df

    def get_existing_keys(self) -> set:
        """Get set of (Date, Home, Away) tuples to detect duplicates."""
        df = self.load_existing()
        keys = set()
        for _, row in df.iterrows():
            date = self.normalize_date(str(row["Date"]))
            if date:
                keys.add((date, str(row["Home"]), str(row["Away"])))
        return keys

    def normalize_date(self, date_str: str) -> str:
        """Normalize date to CSV_DATE_FORMAT (YYYY/MM/DD), accepting slash or dash input."""
        date_str = str(date_str).strip()
        if not date_str:
            return ""
        formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y/%m/d", "%Y/m/%d", "%Y/m/d"]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime(config.CSV_DATE_FORMAT)
            except ValueError:
                pass
        return date_str

    def format_date(self, date_str: str) -> str:
        """Convert an ISO YYYY-MM-DD date to CSV_DATE_FORMAT (YYYY/MM/DD)."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime(config.CSV_DATE_FORMAT)
        except ValueError:
            return self.normalize_date(date_str)

    def create_row(self, match: Dict) -> Dict:
        """Create a CSV row from Odds API match data."""
        commence = match.get("commence_time", "")
        date_part = commence[:10] if commence else ""

        home_team = self._map_team_name(match.get("home_team", ""))
        away_team = self._map_team_name(match.get("away_team", ""))

        row = {
            "Country": "Mexico",
            "League": "Liga MX",
            "Season": match.get("season", "2025/2026"),
            "Round": match.get("round", ""),
            "Date": self.format_date(date_part),
            "Time": commence[11:16] if len(commence) > 16 else "",
            "Home": home_team,
            "Away": away_team,
            "HxG": match.get("HxG", 0),
            "AxG": match.get("AxG", 0),
            "HG": match.get("home_goals", 0),
            "AG": match.get("away_goals", 0),
            "HExpG+": match.get("HExpG+", 0),
            "AExpG+": match.get("AExpG+", 0),
            "Res": self._result_from_score(
                match.get("home_goals", 0),
                match.get("away_goals", 0)
            ),
        }
        # Odds columns (CHN schema) are populated later by the odds/capture layer.
        for col in config.ODDS_COLUMNS:
            row[col] = ""
        return row

    def _result_from_score(self, hg: int, ag: int) -> str:
        """Determine result: H (home win), A (away win), D (draw)."""
        if hg > ag:
            return "H"
        elif ag > hg:
            return "A"
        else:
            return "D"

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Convert value to float safely, handling empty strings and NaN."""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value, default: int = 0) -> int:
        """Convert value to int safely, handling empty strings and NaN."""
        try:
            if pd.isna(value):
                return default
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by Date using slash format with a dash-format fallback."""
        date_col = df["Date"].astype(str).str.strip()
        parsed = pd.to_datetime(date_col, format="%Y/%m/%d", errors="coerce")
        missing = parsed.isna()
        if missing.any():
            parsed_dash = pd.to_datetime(date_col[missing], format="%Y-%m-%d", errors="coerce")
            parsed.loc[missing] = parsed_dash

        df["_date_sort"] = parsed
        df = df.sort_values("_date_sort", na_position="last").reset_index(drop=True)
        return df.drop("_date_sort", axis=1)

    def _should_skip_zero_overwrite(self, existing_hxg: float, existing_axg: float, new_hxg: float, new_axg: float) -> bool:
        """Prevent overwriting existing non-zero xG with transient 0/0 scrape misses."""
        epsilon = config.XG_COMPARE_EPSILON
        new_is_zero = abs(new_hxg) <= epsilon and abs(new_axg) <= epsilon
        existing_non_zero = abs(existing_hxg) > epsilon or abs(existing_axg) > epsilon
        return new_is_zero and existing_non_zero

    def _has_xg_change(self, old_hxg: float, old_axg: float, new_hxg: float, new_axg: float) -> bool:
        """Return True if xG values changed beyond epsilon."""
        epsilon = config.XG_COMPARE_EPSILON
        return abs(old_hxg - new_hxg) > epsilon or abs(old_axg - new_axg) > epsilon

    def upsert_matches(self, matches: List[Dict]) -> Dict[str, int]:
        """Insert new matches or update existing rows when xG changes."""
        stats = {"added": 0, "updated": 0, "unchanged": 0, "skipped": 0}

        if not matches:
            return stats

        df = self.load_existing()
        key_to_index = {}
        for idx, row in df.iterrows():
            date = self.normalize_date(str(row.get("Date", "")))
            home = str(row.get("Home", ""))
            away = str(row.get("Away", ""))
            if date and home and away:
                key_to_index[(date, home, away)] = idx

        for match in matches:
            commence = match.get("commence_time", "")
            date_part = commence[:10] if commence else ""
            formatted_date = self.format_date(date_part)
            home_team = self._map_team_name(match.get("home_team", ""))
            away_team = self._map_team_name(match.get("away_team", ""))

            if not formatted_date or not home_team or not away_team:
                stats["skipped"] += 1
                continue

            key = (self.normalize_date(formatted_date), home_team, away_team)
            new_hxg = self._safe_float(match.get("HxG", 0))
            new_axg = self._safe_float(match.get("AxG", 0))

            existing_idx = key_to_index.get(key)
            if existing_idx is None:
                # For brand-new rows, require non-zero xG so we don't add placeholder rows.
                if abs(new_hxg) <= config.XG_COMPARE_EPSILON and abs(new_axg) <= config.XG_COMPARE_EPSILON:
                    stats["skipped"] += 1
                    continue
                new_row = self.create_row(match)
                hg = self._safe_int(new_row.get("HG", match.get("home_goals", 0)))
                ag = self._safe_int(new_row.get("AG", match.get("away_goals", 0)))
                recalculated = self.xg_calculator.calculate_match_xg(hg=hg, ag=ag, hxg=new_hxg, axg=new_axg)
                new_row["HxG"] = new_hxg
                new_row["AxG"] = new_axg
                new_row["HExpG+"] = recalculated["HExpG+"]
                new_row["AExpG+"] = recalculated["AExpG+"]
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                key_to_index[key] = len(df) - 1
                stats["added"] += 1
                continue

            old_hxg = self._safe_float(df.at[existing_idx, "HxG"])
            old_axg = self._safe_float(df.at[existing_idx, "AxG"])

            if self._should_skip_zero_overwrite(old_hxg, old_axg, new_hxg, new_axg):
                stats["skipped"] += 1
                continue

            if not self._has_xg_change(old_hxg, old_axg, new_hxg, new_axg):
                stats["unchanged"] += 1
                continue

            hg = self._safe_int(df.at[existing_idx, "HG"], self._safe_int(match.get("home_goals", 0)))
            ag = self._safe_int(df.at[existing_idx, "AG"], self._safe_int(match.get("away_goals", 0)))
            recalculated = self.xg_calculator.calculate_match_xg(hg=hg, ag=ag, hxg=new_hxg, axg=new_axg)

            df.at[existing_idx, "HxG"] = new_hxg
            df.at[existing_idx, "AxG"] = new_axg
            df.at[existing_idx, "HExpG+"] = recalculated["HExpG+"]
            df.at[existing_idx, "AExpG+"] = recalculated["AExpG+"]
            stats["updated"] += 1

        if stats["added"] > 0 or stats["updated"] > 0:
            df = df[
                (df["Date"].astype(str).str.strip() != "") &
                (df["Home"].astype(str).str.strip() != "") &
                (df["Away"].astype(str).str.strip() != "")
            ]
            df = self._sort_by_date(df)
            df.to_csv(self._abs_path(), index=False)

        return stats

    def recalculate_all_expected_goals(self) -> int:
        """Recalculate HExpG+ and AExpG+ for all rows in the CSV."""
        df = self.load_existing()

        if df.empty:
            return 0

        recalculated_home = []
        recalculated_away = []

        for _, row in df.iterrows():
            hg = self._safe_int(row.get("HG", 0))
            ag = self._safe_int(row.get("AG", 0))
            hxg = self._safe_float(row.get("HxG", 0))
            axg = self._safe_float(row.get("AxG", 0))
            recalculated = self.xg_calculator.calculate_match_xg(hg=hg, ag=ag, hxg=hxg, axg=axg)
            recalculated_home.append(recalculated["HExpG+"])
            recalculated_away.append(recalculated["AExpG+"])

        df["HExpG+"] = recalculated_home
        df["AExpG+"] = recalculated_away

        df.to_csv(self._abs_path(), index=False)
        return len(df)

    def add_matches(self, matches: List[Dict]) -> int:
        """Add new matches to CSV, skipping duplicates. Returns count of added rows."""
        return self.upsert_matches(matches)["added"]
