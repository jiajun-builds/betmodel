"""Gate G1: the model layer reproduces both pre-merge pipelines.

Two checks, deliberately separate, so a porting bug cannot hide behind a
deliberate change:

1. **Faithful port.** Fit with each league's own pre-merge optimiser settings and
   require its frozen coefficients back to floating-point precision. This asks
   only "is the merged fitter the same mathematics".
2. **Applied change.** Fit with the unified settings and bound the movement.
   This measures the decisions in docs/DECISIONS.md rather than assuming them.

The frozen inputs and outputs are in tests/golden/, captured immediately before
the merge with the two source commit SHAs recorded alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betmodel.config import load_league
from betmodel.models import poisson
from betmodel.models.dc import fit_from_csv

GOLDEN = "tests/golden"

#: Which optimiser settings each league's frozen output was produced with.
PRE_MERGE_OPTIONS = {"csl": poisson.LEGACY_CSL, "ligamx": poisson.LEGACY_LIGAMX}

#: Floating-point noise. Anything above this is a real difference.
EXACT = 1e-12

#: The bound D3 claims for the optimiser change. Only CSL moves.
UNIFIED_OPTION_DRIFT = {"csl": 1e-3, "ligamx": EXACT}


def _fit(league, options):
    config = load_league(league)
    return fit_from_csv(f"{GOLDEN}/{league}/inputs/matches.csv", config, options=options)[0]


def _golden_stats(league) -> pd.DataFrame:
    return pd.read_csv(f"{GOLDEN}/{league}/model/team_stats.csv")


def _joined(league, fit) -> pd.DataFrame:
    got = pd.DataFrame(
        {"Team": fit.teams, "attack": fit.model.attack, "defence": fit.model.defence}
    )
    joined = _golden_stats(league).merge(got, on="Team", how="outer", indicator=True)
    unmatched = joined[joined["_merge"] != "both"]["Team"].tolist()
    assert not unmatched, f"{league}: teams present in only one side: {unmatched}"
    return joined


# --------------------------------------------------------------------------- #
# 1. faithful port
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_pre_merge_settings_reproduce_the_frozen_coefficients(league):
    """The merged fitter is the same mathematics as both originals.

    Measured at merge time: 5.6e-16 for CSL, 3.0e-15 for Liga MX.
    """
    joined = _joined(league, _fit(league, PRE_MERGE_OPTIONS[league]))
    assert (joined["Attack"] - joined["attack"]).abs().max() < EXACT
    assert (joined["Defense"] - joined["defence"]).abs().max() < EXACT


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_team_set_is_unchanged(league):
    fit = _fit(league, PRE_MERGE_OPTIONS[league])
    assert set(fit.teams) == set(_golden_stats(league)["Team"])


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_score_equation_holds(league):
    """The check whose absence let a 27% scale error ship in the first place."""
    fit = _fit(league, PRE_MERGE_OPTIONS[league])
    home, away = fit.model.score_equation_ratio()
    assert abs(home - 1.0) < 1e-4
    assert abs(away - 1.0) < 1e-4


def test_shrinkage_is_applied_where_configured_and_not_elsewhere():
    assert _fit("ligamx", PRE_MERGE_OPTIONS["ligamx"]).shrunk is True
    assert _fit("csl", PRE_MERGE_OPTIONS["csl"]).shrunk is False


def test_shrinkage_moves_low_evidence_clubs_and_preserves_the_league_mean():
    fit = _fit("ligamx", PRE_MERGE_OPTIONS["ligamx"])
    moved = np.abs(fit.model.attack - fit.raw_attack)
    assert moved.max() > 0, "shrinkage is enabled but changed nothing"
    # Only the spread is regularised; the overall scoring level is untouched.
    from betmodel.models.dc import effective_n
    train = fit_from_csv(
        f"{GOLDEN}/ligamx/inputs/matches.csv", load_league("ligamx"),
        options=PRE_MERGE_OPTIONS["ligamx"],
    )[1]
    eff = effective_n(fit.teams, fit.weights, train["Home"], train["Away"])
    before = np.average(fit.raw_attack, weights=eff)
    after = np.average(fit.model.attack, weights=eff)
    assert abs(before - after) < 1e-12


# --------------------------------------------------------------------------- #
# 2. applied change (D3)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_unified_optimiser_setting_moves_coefficients_within_the_stated_bound(league):
    joined = _joined(league, _fit(league, poisson.DEFAULT))
    bound = UNIFIED_OPTION_DRIFT[league]
    assert (joined["Attack"] - joined["attack"]).abs().max() < bound
    assert (joined["Defense"] - joined["defence"]).abs().max() < bound


def test_the_unified_setting_is_the_better_converged_one():
    """D3's justification, asserted rather than asserted-in-prose.

    A tighter tolerance has to buy something. On the CSL window it buys a higher
    likelihood, a smaller gradient and a score equation nearer 1.0.
    """
    loose = _fit("csl", poisson.LEGACY_CSL).model
    tight = _fit("csl", poisson.DEFAULT).model
    assert tight.loglikelihood > loose.loglikelihood

    def max_grad(model):
        return float(np.abs(model._objective(model._params)[1]).max())

    assert max_grad(tight) < max_grad(loose)
    off = lambda m: max(abs(r - 1.0) for r in m.score_equation_ratio())
    assert off(tight) < off(loose)


# --------------------------------------------------------------------------- #
# 3. the scoreline grid (D2)
# --------------------------------------------------------------------------- #

#: The grid each league's frozen simulations were produced with.
PRE_MERGE_GRID = {"csl": 9, "ligamx": 15}

#: What D2 costs. Only Liga MX moves; its grid shrank from 16 cells to 10.
GRID_CHANGE_BOUND = {"csl": EXACT, "ligamx": 5e-4}


def _pairing_probabilities(fit, max_goals: int) -> pd.DataFrame:
    """1X2 for every ordered team pairing, which is what the frozen file holds."""
    fit.model.max_goals = max_goals
    rows = []
    for home in fit.teams:
        for away in fit.teams:
            if home == away:
                continue
            home_p, draw_p, away_p = fit.model.predict(home, away).home_draw_away
            rows.append({"Home Team": home, "Away Team": away,
                         "h": home_p, "d": draw_p, "a": away_p})
    return pd.DataFrame(rows)


def _golden_sims(league) -> pd.DataFrame:
    return pd.read_csv(f"{GOLDEN}/{league}/model/simulations.csv").rename(
        columns={
            "Home Win Probability": "h_gold",
            "Draw Probability": "d_gold",
            "Away Win Probability": "a_gold",
        }
    )


def _max_probability_difference(league, max_goals) -> tuple[float, int]:
    fit = _fit(league, PRE_MERGE_OPTIONS[league])
    joined = _golden_sims(league).merge(
        _pairing_probabilities(fit, max_goals), on=["Home Team", "Away Team"]
    )
    worst = max(
        (joined[f"{side}_gold"] - joined[side]).abs().max() for side in ("h", "d", "a")
    )
    return float(worst), len(joined)


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_pre_merge_grid_reproduces_the_frozen_simulations(league):
    """Every ordered pairing, not just the upcoming fixtures.

    Measured at merge time: 4.4e-16 for CSL, 2.3e-15 for Liga MX.
    """
    worst, matched = _max_probability_difference(league, PRE_MERGE_GRID[league])
    assert matched > 0, "no pairings matched; the frozen file changed shape"
    assert worst < EXACT


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_unified_grid_moves_probabilities_within_the_stated_bound(league):
    """D2 reduced Liga MX's grid from 16 cells a side to 10.

    Measured cost: 1.1e-04 on a 1X2 probability, which is about a hundredth of a
    percentage point against a signal threshold of ten percent.
    """
    worst, _ = _max_probability_difference(league, 9)
    assert worst < GRID_CHANGE_BOUND[league]


def test_the_grid_is_the_only_thing_the_grid_setting_changes():
    """Coefficients are fitted before any grid exists, so they must not move."""
    fit = _fit("ligamx", PRE_MERGE_OPTIONS["ligamx"])
    before = np.array(fit.model.attack, copy=True)
    fit.model.max_goals = 9
    fit.model.predict(fit.teams[0], fit.teams[1])
    assert np.array_equal(before, fit.model.attack)


# --------------------------------------------------------------------------- #
# 4. one route to 1X2 (D4)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_three_historical_routes_to_1x2_still_agree(league):
    """Pins the assumption that let D4 collapse them.

    The pre-merge code read home and away from a zero-line Asian handicap and
    subtracted to get the draw. That is only equal to the direct read while the
    library treats a zero-line "win" as push-excluded. If a library upgrade
    changed that convention, the old expression would put the draw at zero
    silently. This test is how we would find out instead.
    """
    fit = _fit(league, PRE_MERGE_OPTIONS[league])
    teams = list(fit.teams)[:8]
    for home in teams:
        for away in teams:
            if home == away:
                continue
            grid = fit.model.predict(home, away)
            direct = fit.model.outcome_probabilities(home, away)

            handicap_home = grid.asian_handicap("home", 0)
            handicap_away = grid.asian_handicap("away", 0)
            assert abs(handicap_home - direct.home) < EXACT
            assert abs(handicap_away - direct.away) < EXACT
            assert abs((1 - handicap_home - handicap_away) - direct.draw) < EXACT

            # The push component of that same call already is the draw.
            assert abs(grid.asian_handicap_probs("home", 0)["push"] - direct.draw) < EXACT

            raw = np.asarray(grid.grid, dtype=float)
            n = raw.shape[0]
            difference = np.subtract.outer(np.arange(n), np.arange(n))
            sums = np.array([
                raw[difference > 0].sum(), raw[difference == 0].sum(),
                raw[difference < 0].sum(),
            ])
            sums = sums / sums.sum()
            assert np.abs(sums - np.array(direct)).max() < EXACT


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_1x2_probabilities_sum_to_one(league):
    """The residual formula guaranteed this by construction; the direct read
    has to earn it, and does, because the grid is normalised."""
    fit = _fit(league, PRE_MERGE_OPTIONS[league])
    teams = list(fit.teams)[:6]
    for home in teams:
        for away in teams:
            if home == away:
                continue
            assert abs(sum(fit.model.outcome_probabilities(home, away)) - 1.0) < EXACT
