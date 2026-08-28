"""Recompute the blended expected-goals columns (HExpG+/AExpG+) over the whole
match-history CSV. HExpG+ = 0.25*HxG + 0.75*HG (mirror for away).

Also normalizes the Date column back to YYYY/MM/DD, so running this after hand-
editing the CSV (e.g. adding matches in Excel, which rewrites the date format)
repairs both the blended columns and the dates in one pass. Safe to run standalone
via `./scripts/ligamx.sh recompute`."""

from ligamx.xg.data_updater import DataUpdater


def run():
    n = DataUpdater().recalculate_all_expected_goals()
    print(f"Recomputed HExpG+/AExpG+ and normalized dates for {n} rows")


if __name__ == "__main__":
    run()
