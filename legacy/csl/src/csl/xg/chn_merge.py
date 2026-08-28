"""
Merge SofaScore xG (data/raw_data/xg_data.csv by default) into CHN_Super League.csv
in the same data/raw_data folder by default.

Updates existing rows only (left-join semantics): HxG, AxG from home_xg, away_xg;
Round from round. Match key: normalized Date + Home + Away.

Rows present only in xg_data are ignored (no append).

Usage (仓库根目录，PYTHONPATH=src):
    python -m csl.xg.chn_merge
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

from csl.date_utils import format_date_only_series, parse_date_only_series
from csl.paths import data_output_dir, data_raw_dir

# Same raw_data folder as csl.xg.xg_pipeline OUTPUT_DIR
_DEFAULT_CHINA_DATA_RAW = data_raw_dir()
_TEAM_MAPPING_PATH = os.path.join(data_output_dir(), "CHN_team_name_mapping.csv")

DEFAULT_LEAGUE_CSV = os.path.join(_DEFAULT_CHINA_DATA_RAW, "CHN_Super League.csv")
DEFAULT_XG_CSV = os.path.join(_DEFAULT_CHINA_DATA_RAW, "xg_data.csv")
DEFAULT_BACKUP_DIR = os.path.join(_DEFAULT_CHINA_DATA_RAW, "backups")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _normalize_date_column(series: pd.Series) -> pd.Series:
    """Parse mixed date strings to normalized midnight timestamps for joining."""
    return parse_date_only_series(series)


def _strip_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _load_team_alias_mapping(csv_path: str) -> dict[str, str]:
    if not os.path.isfile(csv_path):
        log.warning("Team mapping file not found: %s", csv_path)
        return {}

    df = pd.read_csv(csv_path)
    if "standard_team" not in df.columns:
        log.warning("Team mapping file missing 'standard_team': %s", csv_path)
        return {}

    alias_columns = [col for col in ("sofa_team", "match_team", "odds_team", "standard_team") if col in df.columns]
    mapping: dict[str, str] = {}
    duplicate_aliases: set[str] = set()

    for _, row in df.iterrows():
        standard_raw = row.get("standard_team", "")
        if pd.isna(standard_raw):
            continue
        standard = str(standard_raw).strip()
        if not standard:
            continue
        seen_aliases: set[str] = set()
        for col in alias_columns:
            alias_raw = row.get(col, "")
            if pd.isna(alias_raw):
                continue
            alias = str(alias_raw).strip()
            if not alias or alias in seen_aliases:
                continue
            seen_aliases.add(alias)
            if alias in mapping and mapping[alias] != standard:
                duplicate_aliases.add(alias)
            mapping[alias] = standard

    if duplicate_aliases:
        log.warning("Duplicate team aliases in mapping; used last row for: %s", sorted(duplicate_aliases))

    log.info("Loaded %d team aliases from %s", len(mapping), csv_path)
    return mapping


def _normalize_team_series(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    cleaned = _strip_team(series)
    if not mapping:
        return cleaned
    return cleaned.map(mapping).fillna(cleaned)


def _coerce_round_column(series: pd.Series) -> pd.Series:
    """Normalize Round to nullable Int64 (handles legacy 'Regular Season - N' text)."""
    as_text = series.astype("string")
    legacy = as_text.str.extract(r"Regular Season\s*-\s*(\d+)", expand=False)
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(legacy.fillna(numeric), errors="coerce").round().astype("Int64")


def _build_xg_lookup(xg: pd.DataFrame, team_aliases: dict[str, str]) -> tuple[pd.DataFrame, int]:
    """
    Prepare xg side: columns d, H, A, round, home_xg, away_xg.
    Duplicate (d,H,A) keeps last row; returns duplicate key count for logging.
    """
    required = ("date", "home_team", "away_team", "round", "home_xg", "away_xg")
    for col in required:
        if col not in xg.columns:
            log.error("xg_data.csv missing column %r", col)
            sys.exit(1)

    x = xg[list(required)].copy()
    x["d"] = _normalize_date_column(x["date"])
    x["H"] = _normalize_team_series(x["home_team"], team_aliases)
    x["A"] = _normalize_team_series(x["away_team"], team_aliases)

    dup_before = x.duplicated(subset=["d", "H", "A"], keep=False).sum()
    x = x.drop_duplicates(subset=["d", "H", "A"], keep="last")
    return x, int(dup_before)


def merge_xg_into_league(
    league_path: str,
    xg_path: str,
    *,
    backup_dir: str | None,
    dry_run: bool,
) -> pd.DataFrame:
    if not os.path.isfile(league_path):
        log.error("League file not found: %s", league_path)
        sys.exit(1)
    if not os.path.isfile(xg_path):
        log.error("xg_data file not found: %s", xg_path)
        sys.exit(1)

    league = pd.read_csv(league_path)
    for col in ("Date", "Home", "Away", "HxG", "AxG", "Round"):
        if col not in league.columns:
            log.error("League CSV missing column %r", col)
            sys.exit(1)

    team_aliases = _load_team_alias_mapping(_TEAM_MAPPING_PATH)

    xg = pd.read_csv(xg_path)
    xg_small, dup_keys = _build_xg_lookup(xg, team_aliases)
    if dup_keys:
        log.warning("xg_data had %d rows with duplicate (date, home, away); used last", dup_keys)

    out = league.copy()
    out["_d"] = _normalize_date_column(out["Date"])
    out["_H"] = _normalize_team_series(out["Home"], team_aliases)
    out["_A"] = _normalize_team_series(out["Away"], team_aliases)

    merged = out.merge(
        xg_small[["d", "H", "A", "round", "home_xg", "away_xg"]],
        left_on=["_d", "_H", "_A"],
        right_on=["d", "H", "A"],
        how="left",
        indicator=True,
    )

    matched = merged["_merge"] == "both"
    n_match = int(matched.sum())
    # xg keys with no league row (not merged into any row)
    league_keys = set(zip(out["_d"], out["_H"], out["_A"]))
    xg_keys = set(zip(xg_small["d"], xg_small["H"], xg_small["A"]))
    n_xg_not_in_league = len(xg_keys - league_keys)
    log.info(
        "League rows: %d | xg rows (unique keys): %d | matched updates: %d | xg keys not in league (skipped): %d",
        len(out),
        len(xg_keys),
        n_match,
        n_xg_not_in_league,
    )

    # Apply updates only where merge matched
    out["Round"] = _coerce_round_column(out["Round"])
    round_x = merged.loc[matched, "round"]
    idx = merged.index[matched]
    # Round: always update on key match (Sofa round -> Round)
    out.loc[idx, "Round"] = pd.to_numeric(round_x, errors="coerce").round().astype("Int64")

    # HxG / AxG: update when both xG values are present (align on full merged index)
    both_xg = (
        matched
        & merged["home_xg"].notna()
        & merged["away_xg"].notna()
    )
    idx_xg = merged.index[both_xg]
    out.loc[idx_xg, "HxG"] = merged.loc[both_xg, "home_xg"].astype(float)
    out.loc[idx_xg, "AxG"] = merged.loc[both_xg, "away_xg"].astype(float)

    n_round_only = n_match - int(both_xg.sum())
    if n_round_only:
        log.info(
            "Updated Round only (no full xG pair) on %d rows; HxG/AxG unchanged there",
            n_round_only,
        )

    out.drop(columns=["_d", "_H", "_A"], inplace=True)

    # Canonicalize Date to ISO YYYY-MM-DD on write so a manual edit that
    # reintroduces DD/MM/YYYY (e.g. a spreadsheet re-save) self-heals on the next
    # merge instead of silently reactivating locale-dependent date parsing
    # downstream. Only overwrite rows that parse cleanly; leave anything
    # unrecognized untouched rather than blanking it.
    iso_dates = format_date_only_series(out["Date"])
    parsed_ok = iso_dates != ""
    out["Date"] = iso_dates.where(parsed_ok, out["Date"].astype("string"))

    if dry_run:
        log.info("Dry run: no file written.")
        return out

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(league_path).replace(".csv", "")
        backup_path = os.path.join(backup_dir, f"{base}_backup_{ts}.csv")
        shutil.copy2(league_path, backup_path)
        log.info("Backup saved: %s", backup_path)

    out.to_csv(league_path, index=False)
    log.info("Wrote updated league CSV: %s", league_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge xg_data.csv into CHN_Super League.csv")
    p.add_argument(
        "--league",
        default=DEFAULT_LEAGUE_CSV,
        help="Path to CHN_Super League.csv",
    )
    p.add_argument(
        "--xg",
        default=DEFAULT_XG_CSV,
        help="Path to xg_data.csv from csl.xg.xg_pipeline",
    )
    p.add_argument(
        "--backup-dir",
        default=DEFAULT_BACKUP_DIR,
        help="Directory for league CSV backup (set empty string to skip backup)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute merge but do not write or backup",
    )
    args = p.parse_args()
    backup = args.backup_dir if args.backup_dir else None
    merge_xg_into_league(args.league, args.xg, backup_dir=backup, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
