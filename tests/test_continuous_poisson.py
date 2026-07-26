"""Regression tests for the continuous-target Poisson fit.

The central assertion is the Poisson **score equation** ``sum(w*lambda) ==
sum(w*y)``. The bug this module exists to fix (penaltyblog silently truncating
non-integer goal targets to integers) violates that identity by 27% — had this
test existed, the bug could never have shipped. Everything else here guards the
same failure mode from a different angle.

Runnable either way::

    pytest tests/test_continuous_poisson.py         # if pytest is installed
    python tests/test_continuous_poisson.py         # no pytest needed
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.models.continuous_poisson import ContinuousPoissonGoalModel  # noqa: E402

TEAMS = [f"Team {c}" for c in "ABCDEFGH"]


def synthetic_league(seed: int = 7, rounds: int = 4):
    """Double round-robin with continuous, deliberately non-integer targets."""
    rng = np.random.default_rng(seed)
    attack = np.linspace(-0.4, 0.4, len(TEAMS))
    defence = np.linspace(0.3, -0.3, len(TEAMS))
    idx = {t: i for i, t in enumerate(TEAMS)}

    home, away, yh, ya = [], [], [], []
    for _ in range(rounds):
        for h, a in itertools.permutations(TEAMS, 2):
            lam_h = np.exp(0.4 + attack[idx[h]] + defence[idx[a]] + 0.25)
            lam_a = np.exp(0.4 + attack[idx[a]] + defence[idx[h]])
            home.append(h)
            away.append(a)
            # gamma noise keeps the targets continuous and strictly positive
            yh.append(rng.gamma(shape=lam_h * 2.0, scale=0.5))
            ya.append(rng.gamma(shape=lam_a * 2.0, scale=0.5))
    weights = rng.uniform(0.4, 1.0, len(home))
    return np.array(yh), np.array(ya), home, away, weights


def fit_synthetic(**overrides):
    yh, ya, home, away, weights = synthetic_league()
    yh = overrides.get("goals_home", yh)
    ya = overrides.get("goals_away", ya)
    return ContinuousPoissonGoalModel(yh, ya, home, away, weights).fit()


def test_score_equation_holds():
    """sum(w*lambda) == sum(w*y): the assertion whose absence caused the bug.

    A truncating fitter returns ~0.733 here on this project's real targets.
    """
    model = fit_synthetic()
    ratio_home, ratio_away = model.score_equation_ratio()
    assert abs(ratio_home - 1.0) < 1e-4, f"home score equation violated: {ratio_home}"
    assert abs(ratio_away - 1.0) < 1e-4, f"away score equation violated: {ratio_away}"

    # ...and in absolute terms, not just as a ratio.
    w = model.weights
    np.testing.assert_allclose((w * model.lambda_home).sum(), (w * model.goals_home).sum(), rtol=1e-4)
    np.testing.assert_allclose((w * model.lambda_away).sum(), (w * model.goals_away).sum(), rtol=1e-4)


def test_targets_are_not_truncated():
    """y and floor(y) must give different fits.

    This is the assertion that actually pins the reported bug. Note the more
    obvious "y vs y+0.4" check is NOT sufficient on its own — see
    test_shifted_targets_move_lambda.
    """
    yh, ya, home, away, weights = synthetic_league()
    base = ContinuousPoissonGoalModel(yh, ya, home, away, weights).fit()
    floored = ContinuousPoissonGoalModel(
        np.floor(yh), np.floor(ya), home, away, weights
    ).fit()

    assert base.lambda_home.mean() > floored.lambda_home.mean() + 0.1, (
        "fit is insensitive to the fractional part of the target — targets are "
        f"being truncated ({base.lambda_home.mean():.4f} vs {floored.lambda_home.mean():.4f})"
    )


def test_shifted_targets_move_lambda():
    """y+0.4 must move lambda by ~0.4.

    Kept as a *weak* companion to the floor test. On continuous targets whose
    fractional parts are roughly uniform, a truncating fitter also shifts by
    ~0.4 (because floor(y+0.4) shifts too), so this check alone would have
    given the bug a clean bill of health. It still catches a fitter that
    ignores the target level entirely.
    """
    yh, ya, home, away, weights = synthetic_league()
    base = ContinuousPoissonGoalModel(yh, ya, home, away, weights).fit()
    shifted = ContinuousPoissonGoalModel(yh + 0.4, ya + 0.4, home, away, weights).fit()

    delta = shifted.lambda_home.mean() - base.lambda_home.mean()
    assert abs(delta - 0.4) < 0.05, f"lambda should track a +0.4 target shift, moved {delta:+.4f}"


def test_parameters_are_identified():
    """attack and defence are centred, so the exported team stats are unique."""
    model = fit_synthetic()
    assert abs(model.attack.sum()) < 1e-8, f"attack not centred: {model.attack.sum()}"
    assert abs(model.defence.sum()) < 1e-8, f"defence not centred: {model.defence.sum()}"
    # _params layout is the contract run_dixon_coles_model relies on
    assert len(model._params) == 2 * model.n_teams + 2
    np.testing.assert_allclose(model._params[: model.n_teams], model.attack)
    np.testing.assert_allclose(
        model._params[model.n_teams : 2 * model.n_teams], model.defence
    )


def test_grid_contract():
    """predict() returns a usable FootballProbabilityGrid for every market."""
    model = fit_synthetic()
    pred = model.predict(TEAMS[0], TEAMS[1])
    grid = np.asarray(pred.grid)

    assert grid.shape == (model.max_goals, model.max_goals)
    np.testing.assert_allclose(grid.sum(), 1.0, atol=1e-9)
    assert (grid >= 0).all()

    home, draw, away = pred.home_draw_away
    np.testing.assert_allclose(home + draw + away, 1.0, atol=1e-9)
    for line in (-2.5, -1.5, -0.5, 0.5, 1.5):
        cover = pred.asian_handicap("home", line)
        assert 0.0 <= cover <= 1.0 and np.isfinite(cover)
    assert 0.0 <= pred.total_goals("over", 2.5) <= 1.0
    # lambdas survive the round-trip through the grid object
    assert pred.home_goal_expectation > pred.away_goal_expectation  # home advantage > 0


def test_unseen_team_raises():
    """Promoted sides hit the test fold before any training fold."""
    model = fit_synthetic()
    for args in ((TEAMS[0], "Newly Promoted FC"), ("Newly Promoted FC", TEAMS[0])):
        try:
            model.predict(*args)
        except KeyError:
            continue
        raise AssertionError(f"predict{args} should raise KeyError for an unseen team")


def test_unfitted_model_refuses_to_predict():
    yh, ya, home, away, weights = synthetic_league()
    model = ContinuousPoissonGoalModel(yh, ya, home, away, weights)
    try:
        model.predict(TEAMS[0], TEAMS[1])
    except ValueError:
        return
    raise AssertionError("predict() on an unfitted model should raise ValueError")


def test_penaltyblog_still_truncates():
    """Canary documenting the upstream behaviour this module works around.

    If this test FAILS, penaltyblog has started honouring non-integer targets —
    good news, and a prompt to re-evaluate whether this module is still needed.
    It is not a defect in this repo.
    """
    import penaltyblog as pb

    yh, ya, home, away, weights = synthetic_league()
    clf = pb.models.PoissonGoalsModel(yh, ya, home, away, weights)
    np.testing.assert_array_equal(
        clf.goals_home,
        np.floor(yh).astype(clf.goals_home.dtype),
        err_msg="penaltyblog no longer truncates non-integer targets — re-check "
        "whether csl.models.continuous_poisson is still required",
    )


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
