"""Turning fitted attack/defence coefficients into an honest overall strength.

Why this module exists
----------------------
The dashboard used to rank clubs by ``attack_coef - defence_coef``. Under the
log-linear mean structure of ``ContinuousPoissonModel`` that difference is::

    atk - dfn == log(lambda_for / lambda_against)

against an average opponent — a pure **ratio**, blind to the absolute goal level.
So a 2.2-scored/1.6-conceded club and a 1.2/0.9 club score identically, even
though they are not equally good at winning matches.

At a fixed ratio, playing higher-scoring matches scales up the absolute expected
goal difference, and that amplifies a club's standing in whichever direction it
already points. Measured on a realistic field, two clubs tied at ``atk - dfn ==
+0.30`` rate +0.356 and +0.220 (the higher-scoring one is genuinely stronger),
while two tied at ``-0.30`` rate -0.350 and -0.297 (the higher-scoring one is now
*worse*). The naive difference collapses each pair to a single number. The
practical consequence on the dashboard was a leaky, high-scoring club sitting
too high in the lower half of the table.

The quantity that does answer "who is best overall" is expected points per match
against the actual league field. Measured against that ordering on the 2026 CSL
field, the naive difference misorders 3 of 120 pairs (worst error 2 places); the
rating below reproduces it exactly, by construction.

What ``calibrated_rating`` is
-----------------------------
Not a points forecast — a **rating**, on the same scale and of the same
magnitude as the old ``atk - dfn``, so the column reads as it always did. It
answers: *what symmetric rating gap ``r`` (attack ``+r/2``, defence ``-r/2``)
would make a club as strong as this one?* Since it is a monotone reparametrisation
of expected points, equal rating now genuinely means equal strength, and 0 is
exactly league average.

All functions here are pure — they take coefficients and the two fitted scalars,
never a model instance — so they are cheap to test and safe to call from an
export path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import poisson

# Wider than the model's own predict() grid (max_goals, kept there
# so grids stay comparable with the draw-delta calibration fitted on top of
# them). Nothing is layered on these grids, so the tail is simply included:
# at CSL lambdas 10x10 leaves ~1e-4 of the mass out, and the outcome
# probabilities are renormalized below regardless.
MAX_GOALS = 16

# Bracket for the rating inversion. Comfortably wider than any real league
# spread — the 2026 CSL runs about -0.42 to +0.74 — so brentq never has to be
# nursed, but tight enough that a runaway coefficient raises instead of
# returning nonsense.
RATING_BRACKET = (-4.0, 4.0)

# A club's opponents, as ``(attack_coef, defence_coef)`` pairs. Callers build
# this by excluding the rated club **by name** from the season's field — see
# ``calibrated_rating`` for why the exclusion cannot live in here.
Opponents = Sequence[tuple[float, float]]


def venue_factor(home_adv: float) -> float:
    """Average of the home and away multipliers, ``(1 + e^home_adv) / 2``.

    A club plays half its matches at home, so this is what converts a
    venue-free ``exp(const + coef)`` into a per-match figure.
    """
    return (1.0 + np.exp(home_adv)) / 2.0


def per_match_goals(coef: float, const: float, home_adv: float) -> float:
    """Goals per match implied by one coefficient, against an average opponent.

    Pass an attack coefficient to get goals scored, a defence coefficient to get
    goals conceded. ``per_match_goals(0.0, ...)`` is the league-average
    reference the dashboard legend quotes, because attack and defence are both
    mean-centred by the fit.

    The unit is whatever the model was trained on — for this project ``ExpG+``
    (``0.7*xG + 0.3*goals``), not raw goals.
    """
    return float(np.exp(const + coef) * venue_factor(home_adv))


def _outcome_probs(lam_for: float, lam_against: float) -> tuple[float, float]:
    """``(win, draw)`` probabilities for one fixture from a Poisson grid."""
    k = np.arange(MAX_GOALS)
    grid = np.outer(poisson.pmf(k, lam_for), poisson.pmf(k, lam_against))
    grid = grid / grid.sum()
    return float(np.tril(grid, -1).sum()), float(np.trace(grid))


def expected_points(
    atk: float,
    dfn: float,
    const: float,
    home_adv: float,
    opponents: Opponents,
) -> float:
    """Mean league points per match for a club against ``opponents``.

    Plays the club home and away against every ``(atk, dfn)`` pair given. The
    lambda construction mirrors ``ContinuousPoissonModel._lambdas``.

    ``opponents`` should be the clubs actually in the current season, minus the
    rated club: a rating is only meaningful relative to a stated set of
    opponents, and the season's own field is the one a reader has in mind.
    """
    if len(opponents) == 0:
        raise ValueError("no opponents supplied")
    total = 0.0
    for atk_j, dfn_j in opponents:
        for at_home in (True, False):
            adv_for = home_adv if at_home else 0.0
            adv_against = 0.0 if at_home else home_adv
            lam_for = np.exp(const + atk + dfn_j + adv_for)
            lam_against = np.exp(const + atk_j + dfn + adv_against)
            win, draw = _outcome_probs(lam_for, lam_against)
            total += 3.0 * win + draw
    return total / (2 * len(opponents))


def calibrated_rating(
    atk: float,
    dfn: float,
    const: float,
    home_adv: float,
    opponents: Opponents,
) -> float:
    """Overall strength on the attack-minus-defence rating scale.

    The symmetric gap ``r`` such that a club with attack ``+r/2`` and defence
    ``-r/2`` earns the same expected points as this one against the same
    opponents. ``0.0`` is league average; higher is stronger. Monotone in
    ``expected_points``, so ranking by this reproduces the expected-points
    ranking exactly, unlike ``atk - dfn``.

    The club must already be excluded from ``opponents`` by the caller — and by
    name, not by comparing coefficients. Both sides of the inversion have to
    face an identical opponent set or the two expected-points figures are not
    comparable, and a rating built from a 15-opponent target against a
    16-opponent reference would be quietly wrong.
    """
    target = expected_points(atk, dfn, const, home_adv, opponents)

    def gap(r: float) -> float:
        return expected_points(r / 2.0, -r / 2.0, const, home_adv, opponents) - target

    lo, hi = RATING_BRACKET
    if gap(lo) > 0 or gap(hi) < 0:
        raise ValueError(
            f"rating for (atk={atk:.4f}, dfn={dfn:.4f}) falls outside the "
            f"bracket {RATING_BRACKET} — check the fitted coefficients"
        )
    return float(brentq(gap, lo, hi, xtol=1e-8))
