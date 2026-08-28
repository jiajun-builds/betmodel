"""Fit a league's model and write what the rest of the pipeline reads.

Three artefacts, and the split matters:

``team_stats.csv``
    One row per club: the fitted coefficients plus how much evidence each rests
    on. The evidence counts are there because a promoted side enters the
    training window part-way through and can carry less than half the sample of
    an established club while being reported on the same scale.

``simulations.csv``
    One row per ordered team pairing, not per fixture, so the exporters can look
    up any matchup without refitting. Carries 1X2 and, where configured,
    Asian-handicap cover probabilities. The handicap columns are research input;
    nothing in the production path reads them.

``meta.json``
    When the model was last fitted. Deliberately not touched by odds-only
    refreshes, so a published ``model_updated_at`` keeps pointing at the last
    real fit while the capture loop republishes several times an hour.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from betmodel import paths
from betmodel.config import load_league
from betmodel.config.schema import LeagueConfig
from betmodel.models.dc import ProductionFit, fit_from_csv

log = logging.getLogger(__name__)

TEAM_STATS_COLUMNS = [
    "Team", "Attack", "Defense", "Const", "HomeAdv",
    "Matches", "WeightedMatches", "Date",
]


def team_stats_frame(fit: ProductionFit, as_of: str) -> pd.DataFrame:
    counts, weighted = fit.model.appearances()
    return pd.DataFrame({
        "Team": fit.teams,
        "Attack": fit.model.attack,
        "Defense": fit.model.defence,
        "Const": float(fit.model.const),
        "HomeAdv": float(fit.model.home_advantage),
        "Matches": counts.astype(int),
        "WeightedMatches": weighted,
        "Date": as_of,
    })[TEAM_STATS_COLUMNS]


def simulations_frame(fit: ProductionFit, config: LeagueConfig, as_of: str) -> pd.DataFrame:
    """Every ordered pairing, so an exporter never has to refit to price one."""
    rows = []
    for home in fit.teams:
        for away in fit.teams:
            if home == away:
                continue
            probs = fit.model.predict(home, away)
            # Both pre-merge pipelines derived 1X2 from the zero handicap rather
            # than from the grid aggregate. Kept, because they are equal only
            # while the grid is normalised the same way, and this is the form the
            # frozen output was produced in.
            home_win = probs.asian_handicap("home", 0)
            away_win = probs.asian_handicap("away", 0)
            row = {
                "Date": as_of,
                "Home Team": home,
                "Away Team": away,
                "Home Win Probability": home_win,
                "Draw Probability": 1 - home_win - away_win,
                "Away Win Probability": away_win,
            }
            for line in config.model.ah_lines:
                row[f"Home {line:g}"] = probs.asian_handicap("home", line)
            for line in config.model.ah_lines:
                row[f"Away {line:g}"] = probs.asian_handicap("away", line)
            rows.append(row)
    return pd.DataFrame(rows)


def run_model(league: str, *, write: bool = True) -> ProductionFit:
    """Fit one league and write its three model artefacts."""
    config = load_league(league)
    lp = paths.for_league(league)
    fit, train = fit_from_csv(lp.matches_csv, config)

    as_of = train["Date"].max().strftime("%Y-%m-%d")
    stats = team_stats_frame(fit, as_of)
    sims = simulations_frame(fit, config, as_of)

    if write:
        lp.ensure_dirs()
        stats.to_csv(lp.team_stats_csv, index=False)
        sims.to_csv(lp.simulations_csv, index=False)
        stamped = datetime.now(ZoneInfo(config.timezone)).isoformat(timespec="seconds")
        with open(lp.model_meta_json, "w", encoding="utf-8") as fh:
            json.dump({"model_updated_at": stamped}, fh)
        log.info(
            "%s: wrote %d teams and %d pairings, fitted on %d matches to %s",
            league, len(stats), len(sims), fit.n_matches, as_of,
        )
    return fit
