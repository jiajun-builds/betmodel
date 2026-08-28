"""
Compute HExpG+ and AExpG+ from xG and goals in CHN_Super League.csv.

  HExpG+ = 0.7 * HxG + 0.3 * HG
  AExpG+ = 0.7 * AxG + 0.3 * AG

Overwrites the HExpG+ and AExpG+ columns in place (optional backup).
Default input and backup folder match csl.xg.chn_merge / csl.xg.xg_pipeline (data/raw_data).

Before computing, warns if any row has HxG, AxG, HG, or AG equal to 0 (manual review suggested).

Usage (仓库根目录，PYTHONPATH=src):
    python -m csl.xg.compute_expg
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

from csl.paths import data_raw_dir

# Same raw_data folder as csl.xg.chn_merge defaults
_DEFAULT_CHINA_DATA_RAW = data_raw_dir()
DEFAULT_LEAGUE_CSV = os.path.join(_DEFAULT_CHINA_DATA_RAW, "CHN_Super League.csv")
DEFAULT_BACKUP_DIR = os.path.join(_DEFAULT_CHINA_DATA_RAW, "backups")

WEIGHT_XG = 0.7
WEIGHT_GOALS = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_CHECK_COLS = ("HxG", "AxG", "HG", "AG")
_DISPLAY_COLS = ("Date", "Season", "Round", "Home", "Away", "HxG", "AxG", "HG", "AG")
_MAX_ROWS_LIST = 40


def warn_if_zero_inputs(df: pd.DataFrame) -> None:
    """
    If any of HxG, AxG, HG, AG is exactly 0, log a warning and a table for manual review.
    """
    for col in _CHECK_COLS:
        if col not in df.columns:
            log.error("CSV missing column %r (needed for zero check)", col)
            sys.exit(1)

    n = df.loc[:, list(_CHECK_COLS)].apply(pd.to_numeric, errors="coerce")
    has_zero = (n == 0).any(axis=1)
    if not has_zero.any():
        log.info("Zero check: no rows with HxG/AxG/HG/AG equal to 0.")
        return

    count = int(has_zero.sum())
    log.warning(
        "Zero check: %d row(s) have at least one of HxG, AxG, HG, AG equal to 0. "
        "Please review these rows manually (missing or incorrect data possible).",
        count,
    )

    show_cols = [c for c in _DISPLAY_COLS if c in df.columns]
    bad = df.loc[has_zero, show_cols if show_cols else list(df.columns)].copy()
    if len(bad) > _MAX_ROWS_LIST:
        log.warning("Listing first %d of %d flagged rows:\n%s", _MAX_ROWS_LIST, count, bad.head(_MAX_ROWS_LIST).to_string(index=False))
        log.warning("... and %d more row(s) not shown.", count - _MAX_ROWS_LIST)
    else:
        log.warning("\n%s", bad.to_string(index=False))


def compute_expg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with HExpG+ and AExpG+ recalculated from HxG, AxG, HG, AG."""
    required = ("HxG", "AxG", "HG", "AG", "HExpG+", "AExpG+")
    for col in required:
        if col not in df.columns:
            log.error("CSV missing column %r", col)
            sys.exit(1)

    hx = pd.to_numeric(df["HxG"], errors="coerce")
    ax = pd.to_numeric(df["AxG"], errors="coerce")
    hg = pd.to_numeric(df["HG"], errors="coerce")
    ag = pd.to_numeric(df["AG"], errors="coerce")

    out = df.copy()
    out["HExpG+"] = WEIGHT_XG * hx + WEIGHT_GOALS * hg
    out["AExpG+"] = WEIGHT_XG * ax + WEIGHT_GOALS * ag
    return out


def run(
    league_path: str,
    *,
    backup_dir: str | None,
    dry_run: bool,
) -> pd.DataFrame:
    if not os.path.isfile(league_path):
        log.error("File not found: %s", league_path)
        sys.exit(1)

    df = pd.read_csv(league_path)
    warn_if_zero_inputs(df)
    n_before = len(df)
    out = compute_expg_columns(df)
    log.info(
        "Computed HExpG+ / AExpG+ for %d rows (%g * xG + %g * goals)",
        n_before,
        WEIGHT_XG,
        WEIGHT_GOALS,
    )

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
    log.info("Wrote: %s", league_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Set HExpG+ and AExpG+ from HxG/AxG and HG/AG"
    )
    p.add_argument(
        "--csv",
        default=DEFAULT_LEAGUE_CSV,
        help="Path to CHN_Super League.csv",
    )
    p.add_argument(
        "--backup-dir",
        default=DEFAULT_BACKUP_DIR,
        help="Backup directory (empty string to skip)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute only; do not write or backup",
    )
    args = p.parse_args()
    backup = args.backup_dir if args.backup_dir else None
    run(args.csv, backup_dir=backup, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
