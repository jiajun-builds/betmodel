"""Tests for the overall-strength rating and the dashboard strength export.

The central assertion is that ranking by ``calibrated_rating`` reproduces the
ranking by ``expected_points`` — because the old formula, ``attack - defence``,
did not. That difference is a log ratio, so it is blind to the absolute goal
level and cannot tell a high-scoring/leaky club from a low-scoring/tight one with
the same ratio, even though the two do not win the same number of matches.
``test_equal_log_difference_is_not_equal_strength`` pins the exact case.

Runnable either way::

    pytest tests/test_team_strength.py          # if pytest is installed
    python tests/test_team_strength.py          # no pytest needed
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.dashboard.export_dashboard_csv import (  # noqa: E402
    build_team_strength_rankings,
)
from csl.models.strength import (  # noqa: E402
    calibrated_rating,
    expected_points,
    per_match_goals,
    venue_factor,
)

# Roughly the fitted 2026 CSL values, so the tests exercise a realistic regime.
CONST = 0.2468
HOME_ADV = 0.2830

# A synthetic league spanning a plausible spread, mean-centred like a real fit.
FIELD = [
    (0.33, -0.37),
    (0.31, 0.02),
    (0.20, -0.05),
    (0.07, -0.01),
    (0.05, 0.08),
    (-0.04, 0.16),
    (-0.16, -0.08),
    (-0.42, -0.21),
]


def _centred_field():
    atk = np.array([a for a, _ in FIELD])
    dfn = np.array([d for _, d in FIELD])
    return list(zip(atk - atk.mean(), dfn - dfn.mean()))


def test_per_match_goals_matches_the_closed_form():
    """The exported ATT/DEF figure is exp(const + coef) times the venue factor."""
    assert venue_factor(HOME_ADV) == (1.0 + np.exp(HOME_ADV)) / 2.0
    expected = np.exp(CONST + 0.33) * (1.0 + np.exp(HOME_ADV)) / 2.0
    assert abs(per_match_goals(0.33, CONST, HOME_ADV) - expected) < 1e-12

    # A zero coefficient is the league-average reference the legend quotes, and
    # attack and defence share it — the fit mean-centres both.
    avg = per_match_goals(0.0, CONST, HOME_ADV)
    assert abs(avg - 1.4893) < 1e-3, avg


def test_average_club_rates_exactly_zero():
    """0.000 must mean league average, whatever field is supplied."""
    field = _centred_field()
    assert abs(calibrated_rating(0.0, 0.0, CONST, HOME_ADV, field)) < 1e-6
    # Also true against a subset — the rating is defined by the inversion, not by
    # a lucky property of the full field.
    assert abs(calibrated_rating(0.0, 0.0, CONST, HOME_ADV, field[:3])) < 1e-6


def test_rating_ranks_identically_to_expected_points():
    """The defining property: the rating is a monotone map of expected points.

    ``attack - defence`` is not, which is the whole reason this module exists.
    """
    field = _centred_field()
    ratings, points = [], []
    for i, (atk, dfn) in enumerate(field):
        opponents = [p for j, p in enumerate(field) if j != i]
        ratings.append(calibrated_rating(atk, dfn, CONST, HOME_ADV, opponents))
        points.append(expected_points(atk, dfn, CONST, HOME_ADV, opponents))

    order_by_rating = np.argsort(ratings)
    order_by_points = np.argsort(points)
    np.testing.assert_array_equal(
        order_by_rating,
        order_by_points,
        err_msg="rating ordering diverged from the expected-points ordering",
    )


def test_equal_log_difference_is_not_equal_strength():
    """The regression this replaces: the naive difference ties unequal clubs.

    ``attack - defence`` equals ``log(goals_for / goals_against)``, so it sees
    only the *ratio* and is blind to the absolute goal level. But at a fixed
    ratio, playing higher-scoring matches scales up the absolute expected goal
    difference, which amplifies a club's standing in whichever direction it
    already points: an above-average club gains, a below-average club loses.

    Both directions are asserted, because a fix that only got one of them right
    would be a different bug. The below-average case is the visible one on the
    dashboard — a leaky, high-scoring club sitting too high in the table.
    """
    field = _centred_field()

    def pair(gap: float):
        """Two clubs tied on attack - defence, one high-scoring, one low."""
        high = (0.40, 0.40 - gap)
        low = (-0.20, -0.20 - gap)
        assert abs((high[0] - high[1]) - (low[0] - low[1])) < 1e-12
        return high, low

    def rated(club):
        return (
            expected_points(*club, CONST, HOME_ADV, field),
            calibrated_rating(*club, CONST, HOME_ADV, field),
        )

    # Above average: the higher-scoring club is genuinely stronger.
    high, low = pair(0.30)
    (points_high, rating_high), (points_low, rating_low) = rated(high), rated(low)
    assert points_high > points_low, f"{points_high:.4f} vs {points_low:.4f}"
    assert rating_high > rating_low, (
        "above average, the rating must favour the higher-scoring club "
        f"(got {rating_high:.4f} vs {rating_low:.4f})"
    )

    # Below average: the same high-scoring profile is now a liability, and this is
    # the case the naive difference flattered.
    high, low = pair(-0.30)
    (points_high, rating_high), (points_low, rating_low) = rated(high), rated(low)
    assert points_low > points_high, f"{points_low:.4f} vs {points_high:.4f}"
    assert rating_low > rating_high, (
        "below average, the rating must penalise the leakier high-scoring club "
        f"(got {rating_high:.4f} vs {rating_low:.4f})"
    )


# --------------------------------------------------------------------------- #
# export builder
# --------------------------------------------------------------------------- #

def _write_fixture(tmpdir: str) -> tuple[str, str]:
    """A 4-club current season plus one relegated club and one short-history club."""
    clubs = {
        # team: (attack, defence, weighted_matches)
        "Alpha": (0.30, -0.30, 38.0),
        "Bravo": (0.05, -0.05, 37.0),
        "Charlie": (-0.10, 0.10, 37.5),
        # low sample: well under 0.6 x the median of 37.5
        "Delta": (-0.25, 0.25, 14.0),
        # in the model window but not in the current season
        "Echo": (-0.40, 0.35, 20.0),
    }
    stats_path = os.path.join(tmpdir, "CHN_team_stats.csv")
    pd.DataFrame(
        [
            {
                "Team": team,
                "Attack": atk,
                "Defense": dfn,
                "Const": CONST,
                "HomeAdv": HOME_ADV,
                "Matches": int(wm),
                "WeightedMatches": wm,
                "Date": "2026-07-27",
            }
            for team, (atk, dfn, wm) in clubs.items()
        ]
    ).to_csv(stats_path, index=False)

    # Echo only ever played in 2025; everyone else plays in 2026.
    rows = []
    current = ["Alpha", "Bravo", "Charlie", "Delta"]
    for i, home in enumerate(current):
        away = current[(i + 1) % len(current)]
        rows.append(
            {"Season": 2026, "Date": f"2026-03-{i + 1:02d}", "Time": "19:00",
             "Home": home, "Away": away, "Res": "H"}
        )
    rows.append(
        {"Season": 2025, "Date": "2025-05-01", "Time": "19:00",
         "Home": "Echo", "Away": "Alpha", "Res": "A"}
    )
    matches_path = os.path.join(tmpdir, "CHN_Super League.csv")
    pd.DataFrame(rows).to_csv(matches_path, index=False)
    return stats_path, matches_path


def test_builder_output_contract():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path, matches_path = _write_fixture(tmpdir)
        out, league_avg = build_team_strength_rankings(stats_path, matches_path, "2026")

    assert len(out) == 5, "every club in the model window must be listed"
    assert out["rank_overall"].tolist() == [1, 2, 3, 4, 5]
    assert out["overall_rating"].is_monotonic_decreasing
    assert abs(league_avg - per_match_goals(0.0, CONST, HOME_ADV)) < 1e-12

    # ATT/DEF are goals per match now, so strictly positive — this is what the
    # export validator checks, and what a negative would have meant before.
    assert (out["attack_rating"] > 0).all() and (out["defense_rating"] > 0).all()

    flags = out.set_index("team")
    assert not flags.loc["Alpha", "low_sample"]
    assert flags.loc["Delta", "low_sample"], "14 weighted matches vs a median of 37.5"
    assert flags.loc["Echo", "in_current_season"] is np.False_ or not flags.loc["Echo", "in_current_season"]
    assert flags.loc["Alpha", "in_current_season"]

    # Raw coefficients survive alongside the display figures.
    assert abs(flags.loc["Alpha", "attack_coef"] - 0.30) < 1e-12
    assert abs(
        flags.loc["Alpha", "attack_rating"] - per_match_goals(0.30, CONST, HOME_ADV)
    ) < 1e-12


def test_builder_rejects_a_season_with_no_clubs():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path, matches_path = _write_fixture(tmpdir)
        try:
            build_team_strength_rankings(stats_path, matches_path, "1999")
        except ValueError as exc:
            assert "1999" in str(exc)
        else:
            raise AssertionError("expected a ValueError for a season with no matches")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - test runner reports everything
            failures.append((test.__name__, exc))
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
