"""Fit a league's model and write what the rest of the pipeline reads.

Three artefacts, and the split matters:

``team_stats.csv``
    One row per club: the fitted coefficients plus how much evidence each rests
    on. The evidence counts are there because a promoted side enters the
    training window part-way through and can carry less than half the sample of
    an established club while being reported on the same scale.

    Only clubs with a primary-league match are listed, and ``Matches`` and
    ``WeightedMatches`` count primary matches only, at their undiluted weight:
    an auxiliary competition (D42) informs a rating but is never evidence for
    it. What it contributed is in the trailing ``AuxMatches`` and
    ``AuxWeightedMatches``, the latter at the weight it fitted with.

``simulations.csv``
    One row per ordered team pairing, not per fixture, so the exporters can look
    up any matchup without refitting. Carries 1X2 and, where configured,
    Asian-handicap cover probabilities. The handicap columns are research input;
    nothing in the production path reads them, and they come from the fitted
    grid alone: where the league corrects clubs by their market offset (D38) the
    1X2 carries the correction and the handicap columns do not.

``team_offsets.csv``
    Only where ``model.team_offset`` is enabled: each club's fitted offset, so
    the correction behind a published probability can be read rather than
    re-derived.

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
from betmodel.models import auxiliary, team_offset
from betmodel.models.dc import ProductionFit, fit_from_csv

log = logging.getLogger(__name__)

TEAM_STATS_COLUMNS = [
    "Team", "Attack", "Defense", "Const", "HomeAdv",
    "Matches", "WeightedMatches", "Date",
    # Appended, not inserted: a reader that indexes the original columns by
    # position or name is unaffected.
    "AuxMatches", "AuxWeightedMatches",
]


def team_stats_frame(fit: ProductionFit, as_of: str) -> pd.DataFrame:
    counts, weighted = fit.primary_evidence
    aux_counts, aux_weighted = fit.auxiliary_evidence
    return pd.DataFrame({
        "Team": fit.teams,
        "Attack": fit.attack,
        "Defense": fit.defence,
        "Const": float(fit.model.const),
        "HomeAdv": float(fit.model.home_advantage),
        "Matches": counts.astype(int),
        "WeightedMatches": weighted,
        "Date": as_of,
        "AuxMatches": aux_counts.astype(int),
        "AuxWeightedMatches": aux_weighted,
    })[TEAM_STATS_COLUMNS]


def simulations_frame(fit: ProductionFit, config: LeagueConfig, as_of: str,
                      offsets: team_offset.Offsets | None = None) -> pd.DataFrame:
    """Every ordered pairing, so an exporter never has to refit to price one."""
    offsets = offsets or team_offset.Offsets()
    rows = []
    for home in fit.teams:
        for away in fit.teams:
            if home == away:
                continue
            probs = fit.model.predict(home, away)
            outcome = offsets.adjust(fit.model.outcome_probabilities(home, away),
                                     home, away)
            row = {
                "Date": as_of,
                "Home Team": home,
                "Away Team": away,
                "Home Win Probability": outcome.home,
                "Draw Probability": outcome.draw,
                "Away Win Probability": outcome.away,
            }
            for line in config.model.ah_lines:
                row[f"Home {line:g}"] = probs.asian_handicap("home", line)
            for line in config.model.ah_lines:
                row[f"Away {line:g}"] = probs.asian_handicap("away", line)
            rows.append(row)
    return pd.DataFrame(rows)


def team_offsets_frame(offsets: team_offset.Offsets, as_of: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Team": list(offsets.deltas),
        "Offset": list(offsets.deltas.values()),
        "Kappa": offsets.kappa,
        "Residuals": offsets.n_residuals,
        "Date": as_of,
    })


def run_model(league: str, *, write: bool = True,
              today: pd.Timestamp | None = None) -> ProductionFit:
    """Fit one league and write its model artefacts.

    ``today`` dates the team offsets: they are fitted on matches played before
    it, weighted by their age on it. It defaults to the league's local date.
    """
    config = load_league(league)
    lp = paths.for_league(league)
    loaded = auxiliary.load_enabled(league, config)
    fit, train = fit_from_csv(lp.matches_csv, config, auxiliary=loaded)
    for use in fit.auxiliary:
        log.info(
            "%s: %s pooled, %d matches at weight %.2f, %d bridge clubs, "
            "level shift %s, home-advantage shift %s",
            league, use.id, use.rows, use.weight, len(use.bridges),
            "shared" if use.level_shift is None else f"{use.level_shift:+.4f}",
            "shared" if use.home_shift is None else f"{use.home_shift:+.4f}",
        )
    for rows in loaded:
        if not any(use.id == rows.source.id for use in fit.auxiliary):
            log.info("%s: %s pooled no matches; none inside the window with a target",
                     league, rows.source.id)

    as_of = train["Date"].max().strftime("%Y-%m-%d")
    offsets = team_offset.Offsets()
    if config.model.team_offset.enabled:
        if today is None:
            today = pd.Timestamp(datetime.now(ZoneInfo(config.timezone)).date())
        offsets = team_offset.from_history(pd.read_csv(lp.matches_csv), config, today,
                                           auxiliary=loaded)
    stats = team_stats_frame(fit, as_of)
    sims = simulations_frame(fit, config, as_of, offsets)

    if write:
        lp.ensure_dirs()
        stats.to_csv(lp.team_stats_csv, index=False)
        sims.to_csv(lp.simulations_csv, index=False)
        if config.model.team_offset.enabled:
            team_offsets_frame(offsets, pd.Timestamp(today).strftime("%Y-%m-%d")).to_csv(
                lp.team_offsets_csv, index=False)
        stamped = datetime.now(ZoneInfo(config.timezone)).isoformat(timespec="seconds")
        with open(lp.model_meta_json, "w", encoding="utf-8") as fh:
            json.dump({"model_updated_at": stamped}, fh)
        log.info(
            "%s: wrote %d teams and %d pairings, fitted on %d matches to %s",
            league, len(stats), len(sims), fit.n_matches, as_of,
        )
    return fit
