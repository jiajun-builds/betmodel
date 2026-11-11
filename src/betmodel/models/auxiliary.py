"""Auxiliary competitions: another division's matches as evidence for a fit (D42).

China League One is the case this exists for. A promoted club enters the CSL
with no CSL matches, so the model either refuses to price it or rates it on a
handful of games. Its League One season is real evidence about it, and pooling
those matches into the CSL fit -- down-weighted, on League One's own scoring
level and home advantage -- lets it arrive with a rating that rests on something.

What this module guarantees, and the tests state directly:

* **The primary's weighting does not move.** Which auxiliary rows a fit may use
  is decided from the primary training frame alone: on or after its window
  cutoff, on or before its latest match, and in a replay strictly before the day
  being priced. Production and the walk-forward both go through :func:`select`,
  so they cannot disagree about it.
* **One club is one club.** The file stores SofaScore's spelling. Each name is
  resolved through the league's mapping, so a club that has played in both
  divisions is one team to the fitter. That is not cosmetic: those clubs are the
  only thing that identifies the gap between the two divisions. A club the
  mapping does not know keeps its SofaScore name -- most League One clubs have
  never been near the CSL, so that is the normal case and is not reported.
* **A blank target stays blank.** The target is the league's own xG blend,
  computed here at load time so a blend change cannot leave the two tables on
  different targets. A match with no xG has no target and leaves the fit,
  exactly as a primary match does.

A missing file for an enabled auxiliary is a warning, not a failure: the fit goes
ahead without it, which is precisely the fit the league had before it was added.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from betmodel import paths, teams
from betmodel.config.schema import AuxiliarySource, LeagueConfig
from betmodel.dates import parse_date_only_series
from betmodel.xg.blend import recompute

log = logging.getLogger(__name__)

#: The file's schema, in order. See docs/LEAGUE_ONE.md, "Data".
COLUMNS = (
    "event_id", "Season", "Round", "Date", "Time", "kickoff_utc",
    "home_sofascore", "away_sofascore", "home_team_id", "away_team_id",
    "HG", "AG", "HxG", "AxG",
)

#: What the loader cannot do without.
REQUIRED = ("Date", "home_sofascore", "away_sofascore", "HG", "AG", "HxG", "AxG")

TARGET_COLUMNS = ("HExpG+", "AExpG+")


@dataclass(frozen=True)
class AuxiliaryRows:
    """One auxiliary competition's matches, ready to pool into a fit.

    ``frame`` carries ``Date`` as a timestamp, canonical ``Home``/``Away`` and the
    blended target, plus the file's own columns.
    """

    source: AuxiliarySource
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)


def read(path: str, source: AuxiliarySource, config: LeagueConfig,
         mapping: teams.TeamMapping | None = None) -> AuxiliaryRows | None:
    """Load one auxiliary file. ``None``, with a warning, if it is not usable."""
    if not os.path.exists(path):
        log.warning(
            "%s: auxiliary %s is enabled but %s does not exist; fitting without it",
            config.id, source.id, path,
        )
        return None
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        log.error("%s: auxiliary %s file %s lacks columns %s; fitting without it",
                  config.id, source.id, path, missing)
        return None

    frame["Date"] = parse_date_only_series(frame["Date"])
    unparseable = frame["Date"].isna()
    if unparseable.any():
        log.warning("%s: auxiliary %s: %d rows with an unparseable Date left out",
                    config.id, source.id, int(unparseable.sum()))
        frame = frame[~unparseable].copy()

    mapping = mapping if mapping is not None else teams.for_league(config.id)
    frame["Home"] = resolve(frame["home_sofascore"], mapping)
    frame["Away"] = resolve(frame["away_sofascore"], mapping)
    frame = frame[(frame["Home"] != "") & (frame["Away"] != "")]

    frame, _ = recompute(frame, config)
    for column in TARGET_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return AuxiliaryRows(source=source, frame=frame.reset_index(drop=True))


def resolve(names: Iterable, mapping: teams.TeamMapping) -> list[str]:
    """Canonical name where the mapping knows the club, else SofaScore's own.

    Deliberately not `teams.report_unmapped`: an unmapped League One club is the
    expected case, not a mapping gap, and a warning per club per run would bury
    the ones that are.
    """
    out = []
    for name in names:
        raw = "" if pd.isna(name) else str(name).strip()
        out.append((mapping.to_standard(raw) or raw) if raw else "")
    return out


def load_enabled(league: str, config: LeagueConfig,
                 mapping: teams.TeamMapping | None = None) -> list[AuxiliaryRows]:
    """Every enabled auxiliary of a league that could be read."""
    enabled = config.model.enabled_auxiliary
    if not enabled:
        return []
    lp = paths.for_league(league)
    mapping = mapping if mapping is not None else teams.for_league(league)
    loaded = []
    for source in enabled:
        rows = read(lp.auxiliary_csv(source.file), source, config, mapping)
        if rows is not None:
            loaded.append(rows)
    return loaded


def window_cutoff(train: pd.DataFrame, config: LeagueConfig) -> pd.Timestamp:
    """The primary training window's first day.

    ``prepare_training_frame`` records it, because it is computed before rows
    with a blank target are dropped and so cannot always be re-derived from the
    frame it returns. The fallback is for a frame built some other way.
    """
    cutoff = train.attrs.get("window_cutoff")
    if cutoff is not None:
        return pd.Timestamp(cutoff)
    return train["Date"].max() - pd.DateOffset(months=config.model.lookback_months)


def select(rows: AuxiliaryRows, train: pd.DataFrame, config: LeagueConfig, *,
           before=None, target: Sequence[str] = TARGET_COLUMNS) -> AuxiliaryRows:
    """The auxiliary matches a fit on ``train`` may use.

    Inside the primary's window, never after its latest match -- so the
    Dixon-Coles reference date stays the primary's -- with a target, and in a
    replay strictly before the pricing day.
    """
    frame = rows.frame
    keep = (
        (frame["Date"] >= window_cutoff(train, config))
        & (frame["Date"] <= train["Date"].max())
        & frame[list(target)].notna().all(axis=1)
    )
    if before is not None:
        keep &= frame["Date"] < pd.Timestamp(before)
    return AuxiliaryRows(source=rows.source, frame=frame[keep])


def select_all(loaded: Sequence[AuxiliaryRows] | None, train: pd.DataFrame,
               config: LeagueConfig, *, before=None,
               target: Sequence[str] = TARGET_COLUMNS) -> list[AuxiliaryRows]:
    return [select(rows, train, config, before=before, target=target)
            for rows in (loaded or ())]
