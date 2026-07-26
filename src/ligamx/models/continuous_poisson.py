"""Weighted Poisson goals model that accepts a CONTINUOUS training target.

Why this exists: penaltyblog casts its goal arguments to integers before fitting.
Feeding it the xG blend (HExpG+/AExpG+) therefore trains on floor(xG) -- an xG of
1.9 counts as 1, 0.9 counts as 0 -- which deflated the fitted scoring rates by
~30% (lambda_home 1.12 against a target mean of 1.59). Every downstream symptom
followed from that one scale error: draw probability +5.7pp too high, too little
goal-difference tail, Asian-handicap cover probabilities 6-8pp light on the
home-giving lines. Every penaltyblog family is affected -- the truncation happens
outside the likelihood -- so the fix has to be our own fitter.

The model is the standard log-linear attack/defence form,

    log lambda_home = c + attack_home + defence_away + home_advantage
    log lambda_away = c + attack_away + defence_home

fitted by maximizing the Dixon-Coles-weighted Poisson *pseudo* log-likelihood

    sum_m w_m * (y_m * log lambda_m - lambda_m)

which is well defined and consistent for the mean at any continuous y >= 0 (the
dropped log(y!) term does not depend on the parameters). attack and defence are
constrained to mean zero, which fixes the one non-identified direction
(attack += k, defence -= k leaves every lambda unchanged).

predict() returns a penaltyblog FootballProbabilityGrid built from an independent
Poisson scoreline matrix, so every downstream consumer -- asian_handicap_probs(),
home_draw_away, total_goals() -- keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import penaltyblog as pb
from scipy.optimize import minimize
from scipy.stats import poisson

# Scoreline grid size. Liga MX has never produced a side scoring 16 in the
# training history; the tail beyond this is < 1e-9 of the mass.
MAX_GOALS = 15


@dataclass(frozen=True)
class ContinuousPoissonFit:
    """A fitted model: mean-zero team ratings plus the two global terms."""

    teams: tuple[str, ...]
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    intercept: float

    def _idx(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError as exc:
            raise ValueError(f"unknown team: {team!r}") from exc

    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals (home, away) for one fixture."""
        h, a = self._idx(home_team), self._idx(away_team)
        lam_h = np.exp(self.intercept + self.attack[h] + self.defence[a] + self.home_advantage)
        lam_a = np.exp(self.intercept + self.attack[a] + self.defence[h])
        return float(lam_h), float(lam_a)

    def predict(self, home_team: str, away_team: str, max_goals: int = MAX_GOALS):
        """Scoreline grid for one fixture, as a penaltyblog FootballProbabilityGrid."""
        lam_h, lam_a = self.lambdas(home_team, away_team)
        k = np.arange(max_goals + 1)
        grid = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        return pb.models.FootballProbabilityGrid(grid, lam_h, lam_a)


def fit_continuous_poisson(
    goals_home,
    goals_away,
    teams_home,
    teams_away,
    weights=None,
) -> ContinuousPoissonFit:
    """Fit the weighted Poisson model to a continuous (or integer) goal target.

    Parameters mirror the penaltyblog goal-model constructors so this is a
    drop-in replacement at the call sites. ``weights`` are the Dixon-Coles
    recency weights (uniform if omitted).
    """
    yh = np.asarray(goals_home, dtype=float)
    ya = np.asarray(goals_away, dtype=float)
    home = np.asarray(teams_home, dtype=object)
    away = np.asarray(teams_away, dtype=object)
    w = np.ones(len(yh)) if weights is None else np.asarray(weights, dtype=float)

    if not (len(yh) == len(ya) == len(home) == len(away) == len(w)):
        raise ValueError("goals, teams and weights must all be the same length")
    if np.any(yh < 0) or np.any(ya < 0):
        raise ValueError("goal targets must be non-negative")

    teams = tuple(sorted(set(home) | set(away)))
    n = len(teams)
    pos = {t: i for i, t in enumerate(teams)}
    hi = np.array([pos[t] for t in home])
    ai = np.array([pos[t] for t in away])

    def unpack(z):
        # Centering here (rather than as a constraint) pins the one flat
        # direction; the chain rule for it is a mean-subtraction on the gradient.
        atk = z[:n] - z[:n].mean()
        dfc = z[n : 2 * n] - z[n : 2 * n].mean()
        return atk, dfc, z[-2], z[-1]

    def objective(z):
        atk, dfc, c, ha = unpack(z)
        eta_h = c + atk[hi] + dfc[ai] + ha
        eta_a = c + atk[ai] + dfc[hi]
        lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)

        nll = -np.sum(w * (yh * eta_h - lam_h) + w * (ya * eta_a - lam_a))

        # d(nll)/d(eta) = -w * (y - lambda); everything else is the chain rule.
        rh = -w * (yh - lam_h)
        ra = -w * (ya - lam_a)
        g_atk = np.zeros(n)
        g_dfc = np.zeros(n)
        np.add.at(g_atk, hi, rh)   # home side's attack drives lambda_home
        np.add.at(g_atk, ai, ra)
        np.add.at(g_dfc, ai, rh)   # away side's defence drives lambda_home
        np.add.at(g_dfc, hi, ra)
        grad = np.concatenate([
            g_atk - g_atk.mean(),  # project onto the mean-zero subspace
            g_dfc - g_dfc.mean(),
            [rh.sum() + ra.sum()],  # intercept
            [rh.sum()],             # home advantage
        ])
        return nll, grad

    z0 = np.zeros(2 * n + 2)
    z0[-2] = np.log(max(np.average(np.concatenate([yh, ya]), weights=np.concatenate([w, w])), 1e-3))
    res = minimize(objective, z0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10})
    if not res.success and res.status != 1:  # status 1 = hit maxiter, still usable
        raise RuntimeError(f"continuous Poisson fit failed to converge: {res.message}")

    atk, dfc, c, ha = unpack(res.x)
    return ContinuousPoissonFit(teams=teams, attack=atk, defence=dfc,
                                home_advantage=float(ha), intercept=float(c))
