"""Market-anchored correction of the model's draw probability.

The model prices all three outcomes, but the draw is the one it is worst at, and
the market is unusually good at it. So the draw can be replaced by the anchor
book's no-vig draw, with the freed probability mass returned to home and away
pro-rata so their relative strength is preserved.

The anchor's margin is removed with the **logarithmic function** method --
``ev.devig``, and the only devig in the engine. It is not proportional
normalisation and the two disagree by a few tenths of a point on the draw, which
is the number this whole module exists to set. See D32.

``lam`` controls how far the draw moves. At 1.0 the draw *is* the anchor's no-vig
draw and the model contributes nothing to it, only splitting what remains.

**Or all three are pooled** (``pool_weight``, D38). Replacing only the draw left
the home/away split entirely to the model, and that split is where its error
lives: the side it rates above the market is exactly the side a threshold picks,
and in CSL the market then moved further away by the close. A logarithmic pool,
``p_i ∝ model_i ** w * anchor_i ** (1 - w)``, trusts the model in proportion to
what it has been measured to add over the opener. Pooling the draw as well rather
than keeping it anchored measured indistinguishably on the 2026 hold-out and
fractionally better on every split, which is why the pool covers all three.

**The anchor is the opening price, never the current one.** By the time a current
price exists the bet has either fired on an opening price or it has not, so
anchoring on it would calibrate today's probability against tomorrow's market.
The anchor book is also never a book we bet: letting the price we intend to take
define the probability we price it against is circular, and the config refuses it.
"""

from __future__ import annotations

import math

from betmodel.config.schema import DebiasConfig
from betmodel.signals.ev import devig

Triple = tuple[float, float, float]

#: What produced the published probabilities, carried through to the output so a
#: consumer can tell a market-anchored row from a raw one.
MARKET_ANCHOR = "market_anchor"
RAW = "raw"

#: Lowest model probability the pool will take a logarithm of.
_FLOOR = 1e-12


def apply(
    raw: Triple, config: DebiasConfig, *, anchor_odds: Triple | None
) -> tuple[Triple, str]:
    """Return the corrected probabilities and the method that produced them.

    Falls back to the raw grid when the anchor is missing, which is the normal
    state for a fixture no anchor book has opened yet. That fallback is silent by
    design: it is not a failure, and the method label records which path ran.
    """
    if config.method != MARKET_ANCHOR or anchor_odds is None:
        return raw, RAW
    if any(o is None or float(o) <= 1.0 for o in anchor_odds):
        return raw, RAW

    home, draw, away = (float(x) for x in raw)
    if draw >= 1.0:
        return raw, RAW

    if config.pool_weight is not None:
        return pool(raw, anchor_odds, config.pool_weight), MARKET_ANCHOR

    market_draw = devig(tuple(float(o) for o in anchor_odds))[1]
    new_draw = (1.0 - config.lam) * draw + config.lam * market_draw
    scale = (1.0 - new_draw) / (1.0 - draw)
    return (home * scale, new_draw, away * scale), MARKET_ANCHOR


def pool(raw: Triple, anchor_odds: Triple, weight: float) -> Triple:
    """Logarithmic pool of the model and the anchor's no-vig 1X2.

    ``weight`` is the model's share: 1.0 returns the model, 0.0 the anchor. A
    model probability of exactly zero cannot be pooled in log space and is floored
    far below any price a book would quote.
    """
    market = devig(tuple(float(o) for o in anchor_odds))
    logs = [
        weight * math.log(max(float(m), _FLOOR)) + (1.0 - weight) * math.log(a)
        for m, a in zip(raw, market)
    ]
    top = max(logs)
    weights = [math.exp(x - top) for x in logs]
    total = sum(weights)
    return tuple(x / total for x in weights)  # type: ignore[return-value]
