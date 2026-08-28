"""Write the canonical contract.

Every file is validated before it is written, not after. A payload that would
mislead a consumer never reaches the tree at all, which matters because the tree
is served straight from the repository and a bad file is live the moment it is
committed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from betmodel import paths
from betmodel.config import load_all
from betmodel.config.schema import LeagueConfig
from zoneinfo import ZoneInfo

from betmodel.dates import parse_date_only_series
from betmodel.fixtures.sync import KICKOFF_COLUMN
from betmodel.publish import contract
from betmodel.signals.engine import Signal, make_fixture_id
from betmodel.fixtures.upcoming import load_upcoming

log = logging.getLogger(__name__)


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=False)
        handle.write("\n")


def _envelope(config: LeagueConfig, generated_at: datetime, key: str, rows: list) -> dict:
    return {
        "schema": contract.SCHEMA_VERSION,
        "league": config.id,
        "generated_at": contract.utc(generated_at),
        key: rows,
    }


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #

def signals_payload(
    config: LeagueConfig, signals: list[Signal], generated_at: datetime
) -> dict:
    rows = []
    for signal in signals:
        judged = signal.top_quote
        best = signal.best.get(signal.pick) if signal.fires else None
        rows.append({
            "fixture_id": signal.fixture_id,
            "kickoff_utc": contract.utc(signal.kickoff),
            "round": signal.round,
            "home_team": signal.home_team,
            "away_team": signal.away_team,
            "model": {
                "home": signal.probabilities[0],
                "draw": signal.probabilities[1],
                "away": signal.probabilities[2],
                "method": signal.debias_method,
            },
            "quotes": [
                {
                    "book": q.book, "side": q.side, "odds": q.odds, "ev": q.ev,
                    "captured_at": q.captured_at or None,
                    "last_update": q.last_update or None,
                    "proof": q.proof or None,
                }
                for q in signal.quotes
            ],
            "best": {
                side: {"book": b.book, "odds": b.odds, "ev": b.ev}
                for side, b in signal.best.items()
            },
            # The price this row was JUDGED on, whether or not it cleared. Named
            # rather than implied, because the pre-merge shape expressed the same
            # distinction by which field you happened to read.
            "judged": (
                {
                    "side": signal.top_side, "book": judged.book, "odds": judged.odds,
                    "ev": judged.ev, "proof": judged.proof or None,
                    "captured_at": judged.captured_at or None,
                }
                if judged else None
            ),
            # The price to actually BET. Absent unless the row fires.
            "bet": (
                {
                    "side": signal.pick, "book": best.book, "odds": best.odds,
                    "ev": best.ev, "books": list(signal.books),
                }
                if best else None
            ),
            "state": signal.state,
        })
    payload = _envelope(config, generated_at, "signals", rows)
    contract.validate_signals(payload)
    return payload


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

def results_payload(
    league: str, config: LeagueConfig, generated_at: datetime, *, window_days: int = 180
) -> dict:
    """Played matches, so a consumer can settle a signal without guessing.

    The absence of this is why a downstream board had to decide a match was over
    from the clock. Facts only: the scoreline and whether it has been played,
    never a settled bet outcome. Settling belongs to whoever placed the bet.
    """
    frame = pd.read_csv(paths.for_league(league).matches_csv)
    frame["MatchDate"] = parse_date_only_series(frame["Date"])

    # Prefer the explicit kickoff where the fixture stage has recorded one. The
    # Date column cannot be trusted to mean a particular timezone: one history
    # carries league-local, UK-local and UTC rows depending on which source
    # backfilled them, and the identifier published here has to agree with the
    # one the signal used or the join between a signal and its outcome breaks.
    zone = ZoneInfo(config.timezone)
    if KICKOFF_COLUMN in frame.columns:
        explicit = pd.to_datetime(frame[KICKOFF_COLUMN], utc=True, errors="coerce")
        local = explicit.dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
        frame["MatchDate"] = local.fillna(frame["MatchDate"])

    cutoff = frame["MatchDate"].max() - pd.Timedelta(days=window_days)
    frame = frame[frame["MatchDate"] >= cutoff]

    rows = []
    for record in frame.itertuples(index=False):
        home_goals = getattr(record, "HG", None)
        away_goals = getattr(record, "AG", None)
        played = pd.notna(home_goals) and pd.notna(away_goals)
        date = record.MatchDate
        raw_round = getattr(record, "Round", None)
        round_label = "" if pd.isna(raw_round) else str(raw_round).strip()
        # A knockout tie is filed under a stage name, not a number.
        if round_label.replace(".0", "").isdigit():
            round_label = str(int(float(round_label)))
        rows.append({
            "fixture_id": make_fixture_id(
                config.code, config.season, round_label or "0", date,
                str(record.Home), str(record.Away),
            ),
            "match_date": date.strftime("%Y-%m-%d"),
            "season": str(getattr(record, "Season", config.season)),
            "round": round_label or None,
            "home_team": str(record.Home),
            "away_team": str(record.Away),
            "home_goals": int(home_goals) if played else None,
            "away_goals": int(away_goals) if played else None,
            "result": (
                "H" if played and home_goals > away_goals
                else "A" if played and home_goals < away_goals
                else "D" if played else None
            ),
            "status": "played" if played else "scheduled",
        })
    payload = _envelope(config, generated_at, "results", rows)
    contract.validate_results(payload)
    return payload


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #

def index_payload(configs: dict[str, LeagueConfig], generated_at: datetime) -> dict:
    """The one address a consumer hardcodes.

    Carries the signal threshold and the validation state per league, so a board
    stops holding its own copy of either. Both had already drifted: one board
    hardcoded a threshold three times smaller than the producer's.
    """
    payload = {
        "schema": contract.SCHEMA_VERSION,
        "generated_at": contract.utc(generated_at),
        "leagues": [
            {
                "id": config.id,
                "name": config.name,
                "code": config.code,
                "season": config.season,
                "timezone": config.timezone,
                "validated": config.publish.validated,
                "caveat": config.publish.caveat or None,
                "ev_min": config.signals.ev_min,
                "allow_draw": config.signals.allow_draw,
                "updated_at": contract.utc(generated_at),
                "files": {name: f"{config.id}/{name}.json" for name in contract.LEAGUE_FILES},
            }
            for config in configs.values()
        ],
    }
    contract.validate_index(payload)
    return payload


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def publish_league(
    league: str,
    config: LeagueConfig,
    signals: list[Signal],
    *,
    generated_at: datetime | None = None,
) -> dict[str, str]:
    """Write one league's canonical files. Returns what it wrote."""
    generated_at = generated_at or datetime.now(timezone.utc)
    lp = paths.for_league(league)
    lp.ensure_dirs()

    written: dict[str, str] = {}
    for name, payload in (
        ("signals", signals_payload(config, signals, generated_at)),
        ("results", results_payload(league, config, generated_at)),
    ):
        path = lp.public_json(name)
        _write(path, payload)
        written[name] = path
    log.info("%s: published %s", league, ", ".join(sorted(written)))
    return written


def publish_index(*, generated_at: datetime | None = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    path = paths.public_index_json()
    _write(path, index_payload(load_all(), generated_at))
    return path
