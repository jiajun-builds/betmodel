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

**The anchor is the opening price, never the current one.** By the time a current
price exists the bet has either fired on an opening price or it has not, so
anchoring on it would calibrate today's probability against tomorrow's market.
The anchor book is also never a book we bet: letting the price we intend to take
define the probability we price it against is circular, and the config refuses it.
"""

from __future__ import annotations

from betmodel.config.schema import DebiasConfig
from betmodel.signals.ev import devig

Triple = tuple[float, float, float]

#: What produced the published probabilities, carried through to the output so a
#: consumer can tell a market-anchored row from a raw one.
MARKET_ANCHOR = "market_anchor"
RAW = "raw"


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

    market_draw = devig(tuple(float(o) for o in anchor_odds))[1]
    new_draw = (1.0 - config.lam) * draw + config.lam * market_draw
    scale = (1.0 - new_draw) / (1.0 - draw)
    return (home * scale, new_draw, away * scale), MARKET_ANCHOR
