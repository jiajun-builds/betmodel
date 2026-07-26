"""Regression tests for the goals-model fit.

The bug these exist to prevent: penaltyblog truncates its goal arguments to
integers, so training on the continuous xG blend silently fit floor(xG) and
deflated every scoring rate by ~30%. It survived years of RPS-driven validation
because RPS barely moves when the whole grid shifts toward low scores. The
scale-invariant test below is the assertion that would have caught it on day one.
"""

import unittest

import numpy as np

from ligamx.models.continuous_poisson import fit_continuous_poisson


def _round_robin(teams):
    """Every ordered (home, away) pair, played twice."""
    home, away = [], []
    for _ in range(2):
        for h in teams:
            for a in teams:
                if h != a:
                    home.append(h)
                    away.append(a)
    return np.array(home, dtype=object), np.array(away, dtype=object)


class TestContinuousPoissonFit(unittest.TestCase):
    def setUp(self):
        self.teams = [f"T{i}" for i in range(6)]
        self.home, self.away = _round_robin(self.teams)
        rng = np.random.default_rng(0)
        # Continuous, xG-like targets: means near 1.6/1.2, plenty of fractional part.
        self.yh = rng.gamma(shape=4.0, scale=0.40, size=len(self.home))
        self.ya = rng.gamma(shape=4.0, scale=0.30, size=len(self.home))
        self.w = rng.uniform(0.3, 1.0, size=len(self.home))

    def _lambdas(self, fit):
        return np.array([fit.lambdas(h, a) for h, a in zip(self.home, self.away)])

    def test_weighted_scale_invariant(self):
        """Score equation for the intercept: sum(w*lambda) == sum(w*y).

        This is the identity the penaltyblog fit violated (it matched the mean of
        the *floored* target instead).
        """
        fit = fit_continuous_poisson(self.yh, self.ya, self.home, self.away, self.w)
        lam = self._lambdas(fit)
        total_y = np.sum(self.w * self.yh) + np.sum(self.w * self.ya)
        total_lam = np.sum(self.w * lam[:, 0]) + np.sum(self.w * lam[:, 1])
        self.assertAlmostEqual(total_lam / total_y, 1.0, places=6)

    def test_home_advantage_score_equation(self):
        """Score equation for home_advantage: the home side balances on its own."""
        fit = fit_continuous_poisson(self.yh, self.ya, self.home, self.away, self.w)
        lam = self._lambdas(fit)
        self.assertAlmostEqual(
            np.sum(self.w * lam[:, 0]) / np.sum(self.w * self.yh), 1.0, places=6)

    def test_target_is_not_truncated(self):
        """Fitted rates track the target mean, not floor(target)'s mean.

        Guards the exact failure mode: on Liga MX, mean(HExpG+)=1.59 but
        mean(floor(HExpG+))=1.12, and the old fit returned 1.12.
        """
        fit = fit_continuous_poisson(self.yh, self.ya, self.home, self.away)
        mean_lam_h = self._lambdas(fit)[:, 0].mean()
        self.assertAlmostEqual(mean_lam_h, self.yh.mean(), places=3)
        floored = np.floor(self.yh).mean()
        self.assertGreater(abs(mean_lam_h - floored), 0.2)

    def test_fractional_part_changes_the_fit(self):
        """y and y+0.4 must give different rates; truncation would collapse them."""
        base = fit_continuous_poisson(self.yh, self.ya, self.home, self.away)
        bumped = fit_continuous_poisson(self.yh + 0.4, self.ya + 0.4, self.home, self.away)
        shift = self._lambdas(bumped)[:, 0].mean() - self._lambdas(base)[:, 0].mean()
        self.assertAlmostEqual(shift, 0.4, places=3)

    def test_recovers_known_parameters(self):
        """With y set to the true lambdas, the MLE is exactly the true parameters."""
        atk = np.array([0.30, 0.15, 0.00, -0.10, -0.15, -0.20])
        dfc = np.array([-0.25, -0.05, 0.10, 0.05, 0.00, 0.15])
        atk, dfc = atk - atk.mean(), dfc - dfc.mean()
        c, ha = np.log(1.25), 0.28
        pos = {t: i for i, t in enumerate(self.teams)}
        hi = np.array([pos[t] for t in self.home])
        ai = np.array([pos[t] for t in self.away])
        yh = np.exp(c + atk[hi] + dfc[ai] + ha)
        ya = np.exp(c + atk[ai] + dfc[hi])

        fit = fit_continuous_poisson(yh, ya, self.home, self.away)
        order = [fit.teams.index(t) for t in self.teams]
        np.testing.assert_allclose(fit.attack[order], atk, atol=1e-4)
        np.testing.assert_allclose(fit.defence[order], dfc, atol=1e-4)
        self.assertAlmostEqual(fit.home_advantage, ha, places=4)
        self.assertAlmostEqual(fit.intercept, c, places=4)

    def test_ratings_are_mean_zero(self):
        """The attack/defence gauge is pinned, so ratings are comparable across fits."""
        fit = fit_continuous_poisson(self.yh, self.ya, self.home, self.away, self.w)
        self.assertAlmostEqual(float(fit.attack.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(fit.defence.mean()), 0.0, places=10)

    def test_grid_is_a_valid_distribution(self):
        """predict() returns a usable penaltyblog grid (downstream API contract)."""
        fit = fit_continuous_poisson(self.yh, self.ya, self.home, self.away, self.w)
        grid = fit.predict(self.teams[0], self.teams[1])
        self.assertAlmostEqual(sum(grid.home_draw_away), 1.0, places=6)
        ah = grid.asian_handicap_probs("home", -0.25)
        self.assertAlmostEqual(ah["win"] + ah["push"] + ah["lose"], 1.0, places=6)
        lam_h, lam_a = fit.lambdas(self.teams[0], self.teams[1])
        self.assertAlmostEqual(grid.home_goal_expectation, lam_h, places=9)
        self.assertAlmostEqual(grid.away_goal_expectation, lam_a, places=9)

    def test_rejects_mismatched_and_negative_input(self):
        with self.assertRaises(ValueError):
            fit_continuous_poisson(self.yh[:-1], self.ya, self.home, self.away)
        with self.assertRaises(ValueError):
            fit_continuous_poisson(-self.yh, self.ya, self.home, self.away)


if __name__ == "__main__":
    unittest.main()
