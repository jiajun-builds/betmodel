"""Recompute the blended expected-goals columns (HExpG+/AExpG+) over the whole
match-history CSV. HExpG+ = 0.55*HxG + 0.45*HG (mirror for away)."""

from ligamx.xg.data_updater import DataUpdater


def run():
    n = DataUpdater().recalculate_all_expected_goals()
    print(f"Recomputed HExpG+/AExpG+ for {n} rows")


if __name__ == "__main__":
    run()
