"""Free closing-odds benchmarks from football-data.co.uk.

football-data.co.uk stopped publishing Pinnacle closing 1X2 (PSC*) for Liga MX
after Oct-2025, so pinnacle_close_* in MEX_ligamx.csv is empty for the whole
Polymarket CLV window. This module backfills the surviving free benchmarks —
AvgC (market-average close), MaxC (best-of-market close), BFEC (Betfair exchange
close), B365C (Bet365 close) — by merging the football-data table onto the match
history. Enrichment is analysis-only (eval layer); it does not touch the
production schema or capture pipeline.

The merge is name-mapped (football-data uses its own team spellings) with a
+/-1 day date tolerance to absorb kickoff-date drift between providers.

    python -m ligamx.eval.football_data [--refresh]   # inspect coverage
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests

from ligamx import config, paths

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/MEX.csv"

# football-data team spelling -> our standard (MEX_ligamx.csv) spelling.
# Only names that actually differ need an entry.
FD_TO_STANDARD = {
    "Atl. San Luis": "Atletico San Luis",
    "Club Leon": "Leon",
    "Club Tijuana": "Tijuana",
    "Guadalajara Chivas": "Guadalajara",
    "Juarez": "FC Juarez",
    "Mazatlan FC": "Mazatlan",
}

# Closing-odds columns to pull across (home/draw/away triplets).
FD_CLOSE_COLS = [
    "AvgCH", "AvgCD", "AvgCA",   # market-average close
    "MaxCH", "MaxCD", "MaxCA",   # best-of-market close
    "BFECH", "BFECD", "BFECA",   # Betfair exchange close
    "B365CH", "B365CD", "B365CA",  # Bet365 close
]


def refresh() -> str:
    """Download the latest football-data Liga MX CSV to the local cache."""
    resp = requests.get(FOOTBALL_DATA_URL, headers={"User-Agent": config.HTTP_USER_AGENT}, timeout=30)
    resp.raise_for_status()
    path = paths.football_data_csv()
    with open(path, "wb") as fh:
        fh.write(resp.content)
    return path


def _load_fd() -> pd.DataFrame:
    """Read the cached football-data CSV, normalized (dates parsed, names mapped)."""
    fd = pd.read_csv(paths.football_data_csv(), encoding="utf-8-sig")
    fd["Date"] = pd.to_datetime(fd["Date"], format="%d/%m/%Y", errors="coerce")
    for c in ("Home", "Away"):
        fd[c] = fd[c].replace(FD_TO_STANDARD)
    keep = ["Date", "Home", "Away"] + [c for c in FD_CLOSE_COLS if c in fd.columns]
    return fd[keep].dropna(subset=["Date"])


def load_enriched(refresh_first: bool = False) -> pd.DataFrame:
    """MEX_ligamx.csv joined with football-data closing-odds columns.

    Returns the match history with FD_CLOSE_COLS added (NaN where unmatched).
    Match keys are (Date, Home, Away) with a +/-1 day tolerance on Date.
    """
    if refresh_first:
        refresh()

    fd = _load_fd()
    fd_idx = {(r.Date, r.Home, r.Away): i for i, r in fd.iterrows()}
    keep_cols = [c for c in FD_CLOSE_COLS if c in fd.columns]

    mx = pd.read_csv(paths.ligamx_data_csv())
    mx["Date"] = pd.to_datetime(mx["Date"], format=config.CSV_DATE_FORMAT, errors="coerce")

    rows = []
    for _, r in mx.iterrows():
        hit = None
        for off in (0, 1, -1):
            key = (r["Date"] + pd.Timedelta(days=off), r["Home"], r["Away"])
            if key in fd_idx:
                hit = fd_idx[key]
                break
        rows.append(fd.loc[hit, keep_cols] if hit is not None
                    else pd.Series({c: np.nan for c in keep_cols}))
    add = pd.DataFrame(rows).reset_index(drop=True)
    return pd.concat([mx.reset_index(drop=True), add], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download before reporting")
    args = ap.parse_args()

    df = load_enriched(refresh_first=args.refresh)
    df["Res"] = df["Res"].astype(str).str.strip()
    played = df[df["Res"].isin(["H", "D", "A"])]
    pm = played[played["polymarket_open_h"].notna()]
    print(f"played matches: {len(played)}   |   with polymarket_open: {len(pm)}")
    print("\nfootball-data close coverage (played / of which in Polymarket window):")
    for base in ("AvgCH", "MaxCH", "BFECH", "B365CH"):
        if base in df.columns:
            print(f"  {base[:-1]:<6} {played[base].notna().sum():>5} / {pm[base].notna().sum():>4}")


if __name__ == "__main__":
    main()
