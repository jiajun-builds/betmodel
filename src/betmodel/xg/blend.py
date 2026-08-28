"""The continuous scoring target the model is actually fitted against.

    ExpG+ = xg_weight * xG + goals_weight * goals

Not goals, and not xG alone. Goals are the outcome but a noisy one over a single
match; xG is the process but ignores the finishing that decided the result. The
weights are per league and were each chosen by walk-forward search, and they came
out very different, which is information about the two leagues rather than drift.

A match with no xG keeps a blank target rather than falling back to goals. The
fitter would happily accept the fallback and quietly train part of its sample on
a different quantity, which is exactly the kind of thing that does not show up in
an aggregate error metric.
"""

from __future__ import annotations

import logging

import pandas as pd

from betmodel.config.schema import LeagueConfig

log = logging.getLogger(__name__)

SOURCE_COLUMNS = (("HxG", "HG", "HExpG+"), ("AxG", "AG", "AExpG+"))


def recompute(frame: pd.DataFrame, config: LeagueConfig) -> tuple[pd.DataFrame, int]:
    """Rewrite both target columns from the current xG and goals.

    Recomputes every row rather than only the new ones, so a weight change takes
    effect everywhere at once instead of leaving the history fitted against two
    different targets.
    """
    blend = config.model.xg_blend
    frame = frame.copy()
    filled = 0
    for xg_column, goals_column, target in SOURCE_COLUMNS:
        xg = pd.to_numeric(frame[xg_column], errors="coerce")
        goals = pd.to_numeric(frame[goals_column], errors="coerce")
        usable = xg.notna() & goals.notna()
        values = pd.Series(pd.NA, index=frame.index, dtype="object")
        values[usable] = (blend.xg * xg[usable] + blend.goals * goals[usable]).round(6)
        frame[target] = values
        filled = max(filled, int(usable.sum()))
    log.info("blend %.2f/%.2f applied to %d rows", blend.xg, blend.goals, filled)
    return frame, filled
