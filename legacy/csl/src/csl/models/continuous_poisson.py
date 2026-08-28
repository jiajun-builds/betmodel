"""Weighted Poisson pseudo-likelihood fit for **continuous** goal targets.

Why this module exists
----------------------
penaltyblog's ``BaseGoalsModel.__init__`` coerces the goal arrays to a Cython
integer dtype before the likelihood ever runs (v1.11.0,
``penaltyblog/models/base_model.py:114-115``)::

    self.goals_home = np.asarray(goals_home, dtype=cython_long_dtype, order="C")

``np.asarray(1.79, dtype=np.int64) == 1``, so a non-integer target is silently
**truncated toward zero**. This project trains on xG-blend targets
(``HExpG+ = 0.7*HxG + 0.3*HG``, see ``csl.xg.compute_expg``), which are
non-integer by construction, so every fit was made against ``floor(target)``.
The coercion lives in the *base* class, so every penaltyblog model family
inherits it — swapping Poisson/DixonColes/NegBinom/BivariatePoisson cannot help.

Measured on the 18-month production window (n=385, xi=0.001): ``mean(HExpG+)``
= 1.7955 but ``mean(floor(HExpG+))`` = 1.3117, and the fitted ``lambda_home``
came out 1.3195 — tracking the floor mean. The Poisson score-equation ratio
``sum(w*lambda) / sum(w*y)`` was **0.7330**, i.e. lambda was 27% too low. See
``backtest/backtest.md`` for the walk-forward damage report.

What this fits
--------------
The usual log-linear attack/defence/home-advantage mean structure::

    log lambda_h = c + atk_h + def_a + ha
    log lambda_a = c + atk_a + def_h

maximizing the Dixon-Coles-weighted Poisson **pseudo**-likelihood

    sum_i w_i * (y_i * log lambda_i - lambda_i)

which is a valid and consistent estimating equation for continuous non-negative
``y`` (it is the quasi-likelihood whose score is ``sum w (y - lambda) dlog
lambda``, unbiased at the true mean regardless of the true distribution). The
dropped ``-log(y!)`` term does not depend on the parameters, so omitting it
changes nothing but the objective's constant.

Identifiability: the raw parameter vector has exactly two null directions
(shifting the intercept against the attack means, and against the defence
means). Both are removed by centring ``atk`` and ``def`` to mean zero *inside*
the parameter map, which keeps the optimizer unconstrained. The gradient is
projected onto the same subspace so its norm actually reaches zero — otherwise
L-BFGS-B chases a component along a flat direction and stops on ``ftol``
instead of on the gradient.

The fitted object exposes ``teams``/``_params``/``get_params()``/``predict()``
so it is a drop-in for the penaltyblog models at this project's call sites, and
``predict()`` returns a real ``FootballProbabilityGrid`` so every downstream
market (``asian_handicap``, ``home_draw_away``, ``total_goals``, ``.grid``)
keeps working untouched.
"""

from __future__ import annotations

import numpy as np
from penaltyblog.models import FootballProbabilityGrid
from scipy.optimize import minimize
from scipy.stats import poisson

# penaltyblog's ``predict(..., max_goals=10)`` default builds a (10, 10) grid,
# i.e. scorelines 0..9. Matched here so grids — and therefore the draw-delta
# calibration fitted on top of them — stay comparable across the change.
DEFAULT_MAX_GOALS = 10


class ContinuousPoissonGoalModel:
    """Weighted Poisson goals model that is correct for non-integer targets.

    Parameters
    ----------
    goals_home, goals_away
        Per-match target values. **May be continuous** — that is the whole
        point. Must be finite and non-negative.
    teams_home, teams_away
        Per-match team names.
    weights
        Optional per-match weights (e.g. ``dixon_coles_weights``). Defaults to 1.
    max_goals
        Side length of the scoreline grid returned by ``predict``.
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        weights=None,
        max_goals: int = DEFAULT_MAX_GOALS,
    ):
        self.goals_home = np.asarray(goals_home, dtype=np.float64)
        self.goals_away = np.asarray(goals_away, dtype=np.float64)
        home = np.asarray(teams_home, dtype=str)
        away = np.asarray(teams_away, dtype=str)

        n_matches = len(self.goals_home)
        if not (len(self.goals_away) == len(home) == len(away) == n_matches):
            raise ValueError("goals and team arrays must all have the same length")
        if n_matches == 0:
            raise ValueError("no matches supplied")
        if not (np.isfinite(self.goals_home).all() and np.isfinite(self.goals_away).all()):
            raise ValueError("goal targets must be finite (drop or impute NaNs first)")
        if (self.goals_home < 0).any() or (self.goals_away < 0).any():
            raise ValueError("goal targets must be non-negative")

        self.teams = np.sort(np.unique(np.concatenate([home, away])))
        self.n_teams = len(self.teams)
        self._team_index = {t: i for i, t in enumerate(self.teams)}
        self._ih = np.array([self._team_index[t] for t in home])
        self._ia = np.array([self._team_index[t] for t in away])

        if weights is None:
            self.weights = np.ones(n_matches, dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            if len(self.weights) != n_matches:
                raise ValueError("weights must have the same length as the number of matches")

        self.max_goals = int(max_goals)
        self.fitted = False

    # -- parameter map ----------------------------------------------------
    def _unpack(self, params):
        """Split the raw vector and centre attack/defence to mean zero.

        Centring here (rather than as an optimizer constraint) is what makes the
        problem identified while keeping it unconstrained.
        """
        raw_atk = params[: self.n_teams]
        raw_def = params[self.n_teams : 2 * self.n_teams]
        atk = raw_atk - raw_atk.mean()
        dfn = raw_def - raw_def.mean()
        return atk, dfn, params[-2], params[-1]

    def _lambdas(self, atk, dfn, const, home_adv):
        lam_h = np.exp(const + atk[self._ih] + dfn[self._ia] + home_adv)
        lam_a = np.exp(const + atk[self._ia] + dfn[self._ih])
        return lam_h, lam_a

    def _objective(self, params):
        atk, dfn, const, home_adv = self._unpack(params)
        lam_h, lam_a = self._lambdas(atk, dfn, const, home_adv)

        w, yh, ya = self.weights, self.goals_home, self.goals_away
        nll = -(w * (yh * np.log(lam_h) - lam_h + ya * np.log(lam_a) - lam_a)).sum()

        # d/dtheta = -sum w (y - lambda) * dlog(lambda)/dtheta
        res_h = w * (yh - lam_h)
        res_a = w * (ya - lam_a)
        g_atk = np.zeros(self.n_teams)
        g_def = np.zeros(self.n_teams)
        # attack of the home side drives lam_h; attack of the away side drives lam_a
        np.add.at(g_atk, self._ih, -res_h)
        np.add.at(g_atk, self._ia, -res_a)
        # defence of the away side drives lam_h; defence of the home side drives lam_a
        np.add.at(g_def, self._ia, -res_h)
        np.add.at(g_def, self._ih, -res_a)
        # Project onto the mean-zero subspace: the chain rule through the
        # centring in _unpack. Without this the gradient keeps a component along
        # a flat direction and never reaches zero.
        g_atk -= g_atk.mean()
        g_def -= g_def.mean()

        grad = np.empty_like(params)
        grad[: self.n_teams] = g_atk
        grad[self.n_teams : 2 * self.n_teams] = g_def
        grad[-2] = -(res_h + res_a).sum()
        grad[-1] = -res_h.sum()
        return nll, grad

    # -- fitting ----------------------------------------------------------
    def fit(self, maxiter: int = 2000):
        """Fit by L-BFGS-B with the analytic gradient. Returns ``self``."""
        mean_target = max(float(self.goals_home.mean()), 1e-6)
        x0 = np.concatenate([np.zeros(2 * self.n_teams), [np.log(mean_target), 0.1]])

        result = minimize(
            self._objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}
        )
        if not result.success and result.status != 1:  # status 1 == hit maxiter
            raise RuntimeError(f"continuous Poisson fit failed to converge: {result.message}")

        self.attack, self.defence, self.const, self.home_advantage = self._unpack(result.x)
        self.lambda_home, self.lambda_away = self._lambdas(
            self.attack, self.defence, self.const, self.home_advantage
        )
        self._params = np.concatenate(
            [self.attack, self.defence, [self.const, self.home_advantage]]
        )
        self.loglikelihood = -result.fun
        self.n_iter = int(result.nit)
        self.fitted = True
        return self

    def _check_fitted(self):
        if not self.fitted:
            raise ValueError("model has not been fit yet — call fit() first")

    def get_params(self) -> dict:
        self._check_fitted()
        params = {"const": float(self.const), "home_advantage": float(self.home_advantage)}
        params.update({f"attack_{t}": float(v) for t, v in zip(self.teams, self.attack)})
        params.update({f"defence_{t}": float(v) for t, v in zip(self.teams, self.defence)})
        return params

    def appearances(self) -> tuple[np.ndarray, np.ndarray]:
        """``(counts, weighted_counts)`` per team, in ``self.teams`` order.

        How much evidence each team's coefficients actually rest on. Promoted
        sides enter the training window part-way through, so they can carry less
        than half the sample of an established club while their ratings are
        reported on the same scale — the weighted count is what makes that
        visible downstream rather than silently absorbed.
        """
        counts = np.zeros(self.n_teams)
        weighted = np.zeros(self.n_teams)
        for idx in (self._ih, self._ia):
            np.add.at(counts, idx, 1.0)
            np.add.at(weighted, idx, self.weights)
        return counts, weighted

    def predict(self, home_team: str, away_team: str) -> FootballProbabilityGrid:
        """Scoreline grid for a fixture.

        Raises ``KeyError`` for a team absent from the training window — a real
        case here, since promoted sides appear in a walk-forward test fold
        before they appear in any training fold. Callers already guard
        ``predict`` with try/except; that contract is preserved deliberately
        rather than silently substituting league-average strength.
        """
        self._check_fitted()
        for team in (home_team, away_team):
            if team not in self._team_index:
                raise KeyError(f"team not in the training window: {team!r}")

        i_h = self._team_index[home_team]
        i_a = self._team_index[away_team]
        lam_h = float(np.exp(self.const + self.attack[i_h] + self.defence[i_a] + self.home_advantage))
        lam_a = float(np.exp(self.const + self.attack[i_a] + self.defence[i_h]))

        k = np.arange(self.max_goals)
        grid = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        # normalize=True renormalizes away the truncated upper tail (~1e-5 of
        # the mass at these lambdas).
        return FootballProbabilityGrid(grid, lam_h, lam_a)

    def score_equation_ratio(self) -> tuple[float, float]:
        """``(home, away)`` values of ``sum(w*lambda) / sum(w*y)``.

        Both must be 1.0 at a correctly fitted Poisson optimum — this is the
        intercept/home-advantage score equation, and it is the assertion whose
        absence let the truncation bug survive. The truncating fit returns
        ~0.733 here.
        """
        self._check_fitted()
        w = self.weights
        return (
            float((w * self.lambda_home).sum() / (w * self.goals_home).sum()),
            float((w * self.lambda_away).sum() / (w * self.goals_away).sum()),
        )

    def __repr__(self) -> str:
        if not self.fitted:
            return f"ContinuousPoissonGoalModel(n_teams={self.n_teams}, unfitted)"
        rh, ra = self.score_equation_ratio()
        return (
            f"ContinuousPoissonGoalModel(n_teams={self.n_teams}, "
            f"n_matches={len(self.goals_home)}, logLik={self.loglikelihood:.2f}, "
            f"score_eq=({rh:.6f}, {ra:.6f}))"
        )
