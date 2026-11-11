"""Weighted Poisson fit for a CONTINUOUS goal target.

This exists because penaltyblog coerces its goal arrays to an integer dtype in
``BaseGoalsModel.__init__``, before the likelihood ever runs. The training target
here is an xG blend and so is non-integer by construction, which means every fit
was made against ``floor(target)``. Measured on one production window: mean
target 1.7955, mean floored target 1.3117, fitted lambda 1.3195 — tracking the
floor. The Poisson score-equation ratio came out 0.7330, so lambda was 27% too
low. The coercion is in the base class, so no penaltyblog family escapes it.

Every downstream symptom followed from that one scale error: draw probability
several points too high, too little goal-difference tail, and Asian-handicap
cover probabilities light on the home-giving lines.

The model is the usual log-linear form::

    log lambda_home = c + attack_home + defence_away + home_advantage
    log lambda_away = c + attack_away + defence_home

fitted by maximising the Dixon-Coles-weighted Poisson **pseudo** log-likelihood

    sum_m w_m * (y_m * log lambda_m - lambda_m)

which is a consistent estimating equation for the mean at any continuous
``y >= 0``. The dropped ``log(y!)`` does not depend on the parameters.

Identifiability: the raw parameter vector has two null directions, shifting the
intercept against the attack means and against the defence means. Both are
removed by centring attack and defence to mean zero inside the parameter map,
which keeps the optimiser unconstrained. The gradient is projected onto the same
subspace so its norm actually reaches zero; without that, L-BFGS-B chases a
component along a flat direction and stops on the function tolerance instead.

**Auxiliary groups (D42).** Matches from another competition can be pooled in as
evidence. Each match carries a group, 0 for the primary league, and a match in
auxiliary group ``g`` is fitted as::

    log lambda_home = c + gamma_g + attack_home + defence_away + home_advantage + eta_g
    log lambda_away = c + gamma_g + attack_away + defence_home

so the other division keeps its own scoring level and home advantage and cannot
drag the primary's. ``const`` and ``home_advantage`` stay the primary league's,
and every prediction prices a primary-league match. ``gamma_g`` is identified only
through clubs with matches in both groups. With none, the two components are
disconnected and the fit has a flat direction; it moves nothing a primary pairing
can see, so the optimum is still the right one for pricing, and the group's
shifts are simply not meaningful.

With no auxiliary group the parameter vector, the starting point and every
floating-point operation are exactly those of the plain fit. Gate G1 compares to
1e-12, so that is a branch, not an addition of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from penaltyblog.models import FootballProbabilityGrid
from scipy.optimize import minimize
from scipy.stats import poisson


class Outcome(NamedTuple):
    """1X2 probabilities. Sums to 1 because the scoreline grid is normalised."""

    home: float
    draw: float
    away: float


@dataclass(frozen=True)
class FitOptions:
    """Optimiser settings, exposed because the two merged pipelines differed.

    They disagreed on the starting point and on the convergence tolerances. The
    objective is convex, so both should reach the same optimum, but "should" is
    not a thing to assume about a number that reaches a bet. Keeping these
    explicit lets the merge prove it rather than hope.
    """

    maxiter: int = 2000
    ftol: float | None = 1e-12
    gtol: float | None = 1e-10
    #: Start the intercept from the home target mean only, or from the weighted
    #: mean of both sides.
    intercept_from: str = "both"
    home_advantage_start: float = 0.0

    def scipy_options(self) -> dict:
        options: dict = {"maxiter": self.maxiter}
        if self.ftol is not None:
            options["ftol"] = self.ftol
        if self.gtol is not None:
            options["gtol"] = self.gtol
        return options


#: What the two pipelines each used before the merge, so either can be reproduced
#: exactly while the difference is being measured.
LEGACY_CSL = FitOptions(
    maxiter=2000, ftol=None, gtol=None,
    intercept_from="home", home_advantage_start=0.1,
)
LEGACY_LIGAMX = FitOptions(
    maxiter=1000, ftol=1e-12, gtol=1e-10,
    intercept_from="both", home_advantage_start=0.0,
)
DEFAULT = FitOptions()


class ContinuousPoissonModel:
    """Weighted Poisson goals model that is correct for non-integer targets.

    ``max_goals`` is the highest scoreline modelled, so the grid is
    ``(max_goals + 1)`` cells on a side. That is stated because the two merged
    implementations disagreed: one built ``arange(max_goals)`` and the other
    ``arange(max_goals + 1)`` from a field of the same name, so the same number
    meant a 10-cell grid in one and an 11-cell grid in the other.
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        weights=None,
        *,
        max_goals: int = 9,
        options: FitOptions = DEFAULT,
        groups=None,
        pooled: bool | None = None,
    ) -> None:
        self.goals_home = np.asarray(goals_home, dtype=np.float64)
        self.goals_away = np.asarray(goals_away, dtype=np.float64)
        home = np.asarray(teams_home, dtype=str)
        away = np.asarray(teams_away, dtype=str)

        n = len(self.goals_home)
        if not (len(self.goals_away) == len(home) == len(away) == n):
            raise ValueError("goals and team arrays must all have the same length")
        if n == 0:
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
            self.weights = np.ones(n, dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            if len(self.weights) != n:
                raise ValueError("weights must be the same length as the matches")

        # Group 0 is the primary league (and any auxiliary pooled on its level);
        # each other label gets its own intercept and home-advantage shift.
        if groups is None:
            self.groups = np.zeros(n, dtype=np.int64)
        else:
            self.groups = np.asarray(groups, dtype=np.int64)
            if len(self.groups) != n:
                raise ValueError("groups must be the same length as the matches")
            if (self.groups < 0).any():
                raise ValueError("groups must be non-negative; 0 is the primary league")
            if not (self.groups == 0).any():
                raise ValueError("no primary-league match (group 0) supplied")
        self.group_labels = tuple(int(g) for g in np.unique(self.groups) if g != 0)
        self.n_groups = len(self.group_labels)
        position = {g: i + 1 for i, g in enumerate(self.group_labels)}
        self._ig = np.array([position.get(int(g), 0) for g in self.groups], dtype=np.int64)
        #: Matches from outside the primary league are in the fit, on their own
        #: level or on the primary's. Only this relaxes the convergence rule.
        self.pooled = self.n_groups > 0 if pooled is None else bool(pooled)

        self.max_goals = int(max_goals)
        self.options = options
        self.fitted = False

    @property
    def grid_size(self) -> int:
        """Cells per side of the scoreline grid."""
        return self.max_goals + 1

    # --- parameter map ------------------------------------------------------
    def _unpack(self, params):
        raw_atk = params[: self.n_teams]
        raw_def = params[self.n_teams : 2 * self.n_teams]
        return raw_atk - raw_atk.mean(), raw_def - raw_def.mean(), params[-2], params[-1]

    def _shifts(self, params):
        """``(gamma, eta)`` per auxiliary group, each led by the primary's 0."""
        lo, g = 2 * self.n_teams, self.n_groups
        gamma = np.concatenate([[0.0], params[lo : lo + g]])
        eta = np.concatenate([[0.0], params[lo + g : lo + 2 * g]])
        return gamma, eta

    def _lambdas(self, atk, dfn, const, home_adv, shifts=None):
        if self.n_groups == 0:
            lam_h = np.exp(const + atk[self._ih] + dfn[self._ia] + home_adv)
            lam_a = np.exp(const + atk[self._ia] + dfn[self._ih])
            return lam_h, lam_a
        gamma, eta = shifts
        level = const + gamma[self._ig]
        lam_h = np.exp(level + atk[self._ih] + dfn[self._ia] + home_adv + eta[self._ig])
        lam_a = np.exp(level + atk[self._ia] + dfn[self._ih])
        return lam_h, lam_a

    def _objective(self, params):
        atk, dfn, const, home_adv = self._unpack(params)
        shifts = self._shifts(params) if self.n_groups else None
        lam_h, lam_a = self._lambdas(atk, dfn, const, home_adv, shifts)

        w, yh, ya = self.weights, self.goals_home, self.goals_away
        nll = -(w * (yh * np.log(lam_h) - lam_h + ya * np.log(lam_a) - lam_a)).sum()

        res_h = w * (yh - lam_h)
        res_a = w * (ya - lam_a)
        g_atk = np.zeros(self.n_teams)
        g_def = np.zeros(self.n_teams)
        np.add.at(g_atk, self._ih, -res_h)
        np.add.at(g_atk, self._ia, -res_a)
        np.add.at(g_def, self._ia, -res_h)
        np.add.at(g_def, self._ih, -res_a)
        g_atk -= g_atk.mean()
        g_def -= g_def.mean()

        grad = np.empty_like(params)
        grad[: self.n_teams] = g_atk
        grad[self.n_teams : 2 * self.n_teams] = g_def
        grad[-2] = -(res_h + res_a).sum()
        grad[-1] = -res_h.sum()
        if self.n_groups:
            lo, g = 2 * self.n_teams, self.n_groups
            size = g + 1
            grad[lo : lo + g] = -np.bincount(self._ig, res_h + res_a, minlength=size)[1:]
            grad[lo + g : lo + 2 * g] = -np.bincount(self._ig, res_h, minlength=size)[1:]
        return nll, grad

    # --- fitting ------------------------------------------------------------
    def _start(self) -> np.ndarray:
        if self.options.intercept_from == "home":
            mean_target = max(float(self.goals_home.mean()), 1e-6)
        else:
            both = np.concatenate([self.goals_home, self.goals_away])
            w = np.concatenate([self.weights, self.weights])
            mean_target = max(float(np.average(both, weights=w)), 1e-3)
        return np.concatenate([
            np.zeros(2 * self.n_teams + 2 * self.n_groups),
            [np.log(mean_target), self.options.home_advantage_start],
        ])

    #: Largest gradient component at which a line-search stop is still taken as
    #: the optimum, for a pooled fit only. See ``fit``.
    GROUPED_STOP_GRADIENT = 1e-6

    def fit(self):
        result = minimize(
            self._objective, self._start(), jac=True, method="L-BFGS-B",
            options=self.options.scipy_options(),
        )
        if not result.success and result.status != 1:  # 1 == hit maxiter, usable
            # A pooled fit can carry a flat direction (auxiliary clubs with no
            # club in common with the primary), and at these tolerances L-BFGS-B
            # can end such a run on a failed line search while already at the
            # optimum. A small gradient says it is; the score equation in
            # `dc.fit_production_model` then checks it independently. The plain
            # fit keeps its original rule untouched.
            usable = (self.pooled and
                      float(np.abs(result.jac).max()) < self.GROUPED_STOP_GRADIENT)
            if not usable:
                raise RuntimeError(
                    f"continuous Poisson fit failed to converge: {result.message}")

        self.attack, self.defence, self.const, self.home_advantage = self._unpack(result.x)
        if self.n_groups:
            gamma, eta = self._shifts(result.x)
            self.level_shift, self.home_shift = gamma[1:].copy(), eta[1:].copy()
            self.lambda_home, self.lambda_away = self._lambdas(
                self.attack, self.defence, self.const, self.home_advantage, (gamma, eta)
            )
            self._params = np.concatenate([
                self.attack, self.defence, self.level_shift, self.home_shift,
                [self.const, self.home_advantage],
            ])
        else:
            self.level_shift = self.home_shift = np.zeros(0)
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

    # --- outputs ------------------------------------------------------------
    def get_params(self) -> dict:
        self._check_fitted()
        params = {"const": float(self.const), "home_advantage": float(self.home_advantage)}
        params.update({f"attack_{t}": float(v) for t, v in zip(self.teams, self.attack)})
        params.update({f"defence_{t}": float(v) for t, v in zip(self.teams, self.defence)})
        return params

    def group_shifts(self) -> dict[int, tuple[float, float]]:
        """``{group: (level shift, home-advantage shift)}`` for each auxiliary group.

        Relative to the primary league, in log-rate units. Meaningful only for a
        group that shares at least one club with the primary; see the module note.
        """
        self._check_fitted()
        return {g: (float(self.level_shift[i]), float(self.home_shift[i]))
                for i, g in enumerate(self.group_labels)}

    def appearances(self, rows=None) -> tuple[np.ndarray, np.ndarray]:
        """``(counts, weighted_counts)`` per team, in ``self.teams`` order.

        ``rows`` is an optional boolean mask over the matches, to count only some
        of them -- the primary league's, say, when auxiliary matches are pooled.

        How much evidence each team's coefficients rest on. A promoted side
        enters the window part-way through and can carry less than half the
        sample of an established club while its rating is reported on the same
        scale. The weighted count is what makes that visible downstream instead
        of silently absorbed.
        """
        self._check_fitted()
        counts = np.zeros(self.n_teams)
        weighted = np.zeros(self.n_teams)
        if rows is None:
            for idx in (self._ih, self._ia):
                np.add.at(counts, idx, 1.0)
                np.add.at(weighted, idx, self.weights)
            return counts, weighted
        rows = np.asarray(rows, dtype=bool)
        for idx in (self._ih[rows], self._ia[rows]):
            np.add.at(counts, idx, 1.0)
            np.add.at(weighted, idx, self.weights[rows])
        return counts, weighted

    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        self._check_fitted()
        for team in (home_team, away_team):
            if team not in self._team_index:
                raise KeyError(f"team not in the training window: {team!r}")
        i_h, i_a = self._team_index[home_team], self._team_index[away_team]
        lam_h = float(np.exp(
            self.const + self.attack[i_h] + self.defence[i_a] + self.home_advantage
        ))
        lam_a = float(np.exp(self.const + self.attack[i_a] + self.defence[i_h]))
        return lam_h, lam_a

    def predict(self, home_team: str, away_team: str) -> FootballProbabilityGrid:
        """Scoreline grid for a fixture.

        Raises ``KeyError`` for a team absent from the training window. That is a
        real case, not a defensive branch: a promoted side appears in a
        walk-forward test fold before it appears in any training fold. Callers
        guard this deliberately rather than have league-average strength
        substituted for a club nobody has seen.
        """
        lam_h, lam_a = self.lambdas(home_team, away_team)
        k = np.arange(self.grid_size)
        grid = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        return FootballProbabilityGrid(grid, lam_h, lam_a)

    def outcome_probabilities(self, home_team: str, away_team: str) -> Outcome:
        """1X2 for a fixture. The single route to these three numbers.

        Both pre-merge pipelines took a detour: they read the home and away
        probabilities out of a zero-line Asian handicap and recovered the draw by
        subtracting the two from one. That works only because the library defines
        a zero-line "win" as push-excluded, which makes it exactly the outright
        win probability. The library labels that method backward-compatible and
        points elsewhere for anything careful.

        The cost of the detour is not accuracy, it is fragility. The draw is a
        residual, so if the zero-line convention ever became push-adjusted the
        home and away figures would sum to one and the draw would silently become
        zero. The push component of that same call already *is* the draw; it was
        being thrown away and reconstructed.

        A third route existed too, summing the grid directly. All three agree to
        floating point, which is what makes collapsing them safe rather than a
        judgement call.
        """
        home_p, draw_p, away_p = self.predict(home_team, away_team).home_draw_away
        return Outcome(float(home_p), float(draw_p), float(away_p))

    def score_equation_ratio(self, group: int = 0) -> tuple[float, float]:
        """``(home, away)`` values of ``sum(w*lambda) / sum(w*y)``.

        Both are 1.0 at a correctly fitted Poisson optimum. This is the
        intercept and home-advantage score equation, and its absence is what let
        the truncation bug survive unnoticed across five copies of the fit. The
        truncating fit returned about 0.733.

        With auxiliary groups the equation holds within each group separately,
        because each has its own intercept and home shift. ``group`` picks one;
        the default is the primary league's.
        """
        self._check_fitted()
        w = self.weights
        if self.n_groups == 0:
            return (
                float((w * self.lambda_home).sum() / (w * self.goals_home).sum()),
                float((w * self.lambda_away).sum() / (w * self.goals_away).sum()),
            )
        rows = self.groups == group
        w = w[rows]
        return (
            float((w * self.lambda_home[rows]).sum() / (w * self.goals_home[rows]).sum()),
            float((w * self.lambda_away[rows]).sum() / (w * self.goals_away[rows]).sum()),
        )

    def __repr__(self) -> str:
        if not self.fitted:
            return f"ContinuousPoissonModel(n_teams={self.n_teams}, unfitted)"
        rh, ra = self.score_equation_ratio()
        return (
            f"ContinuousPoissonModel(n_teams={self.n_teams}, "
            f"n_matches={len(self.goals_home)}, logLik={self.loglikelihood:.2f}, "
            f"score_eq=({rh:.6f}, {ra:.6f}))"
        )
