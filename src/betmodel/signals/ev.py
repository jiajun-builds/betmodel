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

    Proportional normalisation of the inverse prices. Cruder than a power or
    Shin devig, and deliberately the same crude method everywhere, because a
    closing-line-value comparison between two devig methods measures the methods.
    """
    inverse = [1.0 / o for o in odds]
    total = sum(inverse)
    return tuple(i / total for i in inverse)  # type: ignore[return-value]


def overround(odds: tuple[float, float, float]) -> float:
    """The book's margin, as a fraction above 1.0.

    Worth carrying because it is the cheapest tell that a captured price is not
    what it claims to be: real closing lines sit near a book's usual margin, and
    a price taken hours early sits visibly above it.
    """
    return sum(1.0 / o for o in odds) - 1.0
