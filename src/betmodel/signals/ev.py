"""Expected value of a price, and the units it is expressed in.

``EV = p * decimal_odds - 1``, and ``odds`` has to be a price you could actually
have taken. That second half is the whole discipline: EV computed against a price
nobody was offering is a number about nothing.

**EV is a fraction everywhere inside the engine.** The two merged pipelines
disagreed on this and the disagreement reached the board, which now guesses the
scale from the payload by taking a median. Percentage points appear only in one
legacy compatibility file, and only at the moment of writing it.
"""

from __future__ import annotations

import math

#: Bisection budget for the logarithmic devig. The loop exits on the tolerance
#: long before this; the cap is only there so a pathological input cannot spin.
_BISECTION_STEPS = 200
#: Width of the exponent bracket at which the search stops. Well below the
#: precision of any quoted price.
_K_TOL = 1e-12
#: Ceiling on the exponent search, reached only by a triple whose margin is far
#: outside anything a real book quotes.
_K_MAX = 1e6


def expected_value(probability: float, odds: float) -> float | None:
    """Fractional EV of backing an outcome at ``odds``.

    ``None`` when either input is unusable, which is common and not an error: a
    book that has not opened has no price, and a fixture outside the training
    window has no probability.
    """
    if probability is None or odds is None:
        return None
    try:
        p = float(probability)
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(p) and math.isfinite(o)):
        return None
    if not 0.0 <= p <= 1.0 or o <= 1.0:
        return None
    return p * o - 1.0


def devig(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    """No-vig probabilities implied by a 1X2 price triple.

    The **logarithmic function** method: the book's margin is assumed to be
    applied as a power of the fair probability, so ``q_i = p_i ** (1 / k)`` and
    the fair grid is recovered as ``p_i = q_i ** k`` for the single ``k`` that
    makes them sum to one. ``k`` is found by bisection; it is unique because
    every ``q_i`` lies in ``(0, 1)``, which makes the sum strictly decreasing
    in ``k``.

    This is not proportional normalisation and does not agree with it. Under
    proportional the margin is a flat fraction of every price, so the favourite
    and the longshot are shaved equally; here the shaving grows with the price,
    which is the direction real books actually load it. The favourite therefore
    comes out higher than proportional gave it and the long legs lower. Across
    the 327 Pinnacle triples in the capture history the draw is never the
    shortest of the three, so its de-vigged probability is always the lower of
    the two -- and that number is the whole output of ``debias.apply``. See D32.

    **One devig method everywhere.** A closing-line-value comparison between two
    devig methods measures the methods, so the engine has exactly one and this
    is it.

    Degenerate input -- a quoted price at or below evens, where ``q_i >= 1`` and
    no power can pull the sum down to one -- falls back to proportional
    normalisation rather than failing. Callers reject such prices before they
    get here; the fallback exists so this function is total.
    """
    inverse = [1.0 / o for o in odds]
    if any(q >= 1.0 for q in inverse):
        total = sum(inverse)
        return tuple(q / total for q in inverse)  # type: ignore[return-value]

    def _sum(k: float) -> float:
        return sum(q ** k for q in inverse)

    # k = 0 gives 3.0, so the low end always overshoots one; push the high end
    # out until it undershoots. A normal 1X2 margin lands k a little above 1.
    lo, hi = 0.0, 1.0
    while _sum(hi) > 1.0 and hi < _K_MAX:
        hi *= 2.0
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if _sum(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= _K_TOL:
            break
    k = 0.5 * (lo + hi)

    fair = [q ** k for q in inverse]
    # Bisection leaves a residue far below the precision of any price; renormalise
    # so the triple sums to exactly one and downstream code never has to care.
    total = sum(fair)
    return tuple(f / total for f in fair)  # type: ignore[return-value]


def overround(odds: tuple[float, float, float]) -> float:
    """The book's margin, as a fraction above 1.0.

    Worth carrying because it is the cheapest tell that a captured price is not
    what it claims to be: real closing lines sit near a book's usual margin, and
    a price taken hours early sits visibly above it.
    """
    return sum(1.0 / o for o in odds) - 1.0
