from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from csl.date_utils import format_date_only_series, parse_date_only_series
from csl.models.strength import calibrated_rating, per_match_goals
from csl.paths import data_dashboard_csv_dir, data_output_dir, data_raw_dir, model_meta_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/London")
COMPETITION_CODE = "CSL"
COMPETITION_NAME = "Chinese Super League"
# The model is csl.models.continuous_poisson.ContinuousPoissonGoalModel: a weighted
# Poisson pseudo-likelihood on xG-blend targets, with no dispersion parameter. The
# name said "Negative Binomial" until v2.8, left over from the family retired by
# the truncation fix — the masthead and MODEL CONTEXT panel both print this, so it
# has to describe the fit that actually produces the ratings.
MODEL_NAME = "Weighted Poisson on xG with Dixon-Coles Time Decay"
MODEL_VERSION = "v2.8"

# A club's weighted match count below this fraction of the league median is
# flagged low-sample in the UI. Promoted sides enter the 18-month window
# part-way through and land around 0.4 of the median, established clubs at 1.0,
# so the boundary is nowhere near either group.
LOW_SAMPLE_FRACTION = 0.6

META_COLUMNS = [
    "competition_code",
    "competition_name",
    "season",
    "updated_at",
    "model_updated_at",
    "timezone",
    "last_completed_match_date",
    "next_fixture_date",
    "matches_played",
    "current_round",
    "total_rounds",
    "model_name",
    "model_version",
    "league_avg_goals",
]

UPCOMING_COLUMNS = [
    "fixture_id",
    "round",
    "match_date",
    "match_time",
    "kickoff_at",
    "home_team",
    "away_team",
]

PREDICTION_COLUMNS = [
    "fixture_id",
    "round",
    "match_date",
    "kickoff_at",
    "home_team",
    "away_team",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "home_win_fair_odds",
    "draw_fair_odds",
    "away_win_fair_odds",
]

# attack_rating/defense_rating are goals per match against an average opponent and
# overall_rating is the calibrated strength rating — NOT the raw fitted coefficients,
# which are carried separately as attack_coef/defense_coef. See csl.models.strength.
# Finished matches, as facts only: the score and who won. Deliberately NOT settled
# outcomes -- whether a bet won depends on which side was backed, at what price and
# on what stake, none of which this pipeline knows. Settlement belongs to whoever
# holds the bet, the same way EV belongs here and not on the board.
RESULTS_COLUMNS = [
    "fixture_id",
    "season",
    "round",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "status",
]

# How far back match_results carries. A trailing window rather than "the current
# season" on purpose: a season-scoped export empties itself the day a new season
# starts, which is exactly when something downstream is still settling last
# season's final round. 180 days is generous enough to backfill a ledger from
# scratch and still bounds the file at roughly one season of rows.
RESULTS_WINDOW_DAYS = 180

STRENGTH_COLUMNS = [
    "rank_overall",
    "team",
    "attack_rating",
    "defense_rating",
    "overall_rating",
    "attack_rank",
    "defense_rank",
    "form",
    "attack_coef",
    "defense_coef",
    "weighted_matches",
    "low_sample",
    "in_current_season",
]


@dataclass(frozen=True)
class ExportPaths:
    matches_csv: str = os.path.join(data_raw_dir(), "CHN_Super League.csv")
    fresh_schedule_csv: str = os.path.join(data_raw_dir(), "chinese_super_league_data.csv")
    fixtures_csv: str = os.path.join(data_raw_dir(), "chn_upcoming_fixtures.csv")
    team_stats_csv: str = os.path.join(data_output_dir(), "CHN_team_stats.csv")
    simulations_csv: str = os.path.join(data_output_dir(), "CHN_team_stats_match_simulations.csv")
    out_dir: str = data_dashboard_csv_dir()

    @property
    def meta_csv(self) -> str:
        return os.path.join(self.out_dir, "dashboard_meta.csv")

    @property
    def upcoming_csv(self) -> str:
        return os.path.join(self.out_dir, "upcoming_fixtures.csv")

    @property
    def predictions_csv(self) -> str:
        return os.path.join(self.out_dir, "match_predictions.csv")

    @property
    def strength_csv(self) -> str:
        return os.path.join(self.out_dir, "team_strength_rankings.csv")

    @property
    def results_csv(self) -> str:
        return os.path.join(self.out_dir, "match_results.csv")


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _slugify(value: str) -> str:
    slug = value.strip().lower().replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def _fair_odds(probability: float) -> float | None:
    if probability <= 0 or math.isnan(probability):
        return None
    return round(1.0 / probability, 4)


def _normalize_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cols = ["home_win_prob", "draw_prob", "away_win_prob"]
    total = out[cols].sum(axis=1)
    bad_total = total <= 0
    if bad_total.any():
        bad_rows = out.loc[bad_total, ["home_team", "away_team"]].to_dict("records")
        raise ValueError(f"Found rows with non-positive probability totals: {bad_rows}")

    for col in cols:
        out[col] = (out[col] / total).round(6)

    rounded_total = out[cols].sum(axis=1)
    adjust_idx = out.index[(rounded_total - 1.0).abs() > 1e-6]
    if len(adjust_idx):
        out.loc[adjust_idx, "away_win_prob"] = (
            1.0 - out.loc[adjust_idx, "home_win_prob"] - out.loc[adjust_idx, "draw_prob"]
        ).round(6)
    return out


def _parse_match_dates(series: pd.Series) -> pd.Series:
    return parse_date_only_series(series)


def _parse_fixture_dates(series: pd.Series) -> pd.Series:
    return parse_date_only_series(series)


def _normalize_season_value(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Season value is missing")
    if isinstance(value, str):
        text = value.strip()
        return text[:-2] if text.endswith(".0") else text
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def _extract_round_number(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+)\s*$", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def _derive_current_season(matches: pd.DataFrame) -> str:
    played = matches.copy()
    played["parsed_date"] = _parse_match_dates(played["Date"])
    played = played[played["parsed_date"].notna()]
    if played.empty:
        raise ValueError("Cannot derive season from matches: no parseable dates found")
    latest_row = played.sort_values("parsed_date").iloc[-1]
    return _normalize_season_value(latest_row["Season"])


def build_round_progress(schedule_path: str, season: str) -> dict[str, int]:
    schedule = pd.read_csv(schedule_path)
    _require_columns(schedule, ["Season", "Round", "Res"], "chinese_super_league_data.csv")

    season_schedule = schedule.copy()
    season_schedule["Season"] = season_schedule["Season"].astype(str).str.strip()
    season_schedule = season_schedule[season_schedule["Season"] == season].copy()
    if season_schedule.empty:
        raise ValueError(f"No schedule rows found for season {season} in chinese_super_league_data.csv")

    season_schedule["round_num"] = _extract_round_number(season_schedule["Round"])
    season_schedule["Res"] = season_schedule["Res"].astype(str).str.strip()
    season_schedule = season_schedule[season_schedule["round_num"].notna()].copy()
    if season_schedule.empty:
        raise ValueError("No parseable round numbers found in chinese_super_league_data.csv")

    played_mask = season_schedule["Res"].isin(["H", "D", "A"])
    matches_played = int(played_mask.sum())
    total_rounds = int(season_schedule["round_num"].max())

    unplayed_rounds = sorted(
        season_schedule.loc[~played_mask, "round_num"].dropna().astype(int).unique().tolist()
    )
    current_round = unplayed_rounds[0] if unplayed_rounds else total_rounds

    return {
        "matches_played": matches_played,
        "current_round": int(current_round),
        "total_rounds": total_rounds,
    }


def build_upcoming_fixtures(fixtures_path: str, season: str, export_now: pd.Timestamp) -> pd.DataFrame:
    src = pd.read_csv(fixtures_path)
    _require_columns(src, ["Wk", "Date", "Time", "Home", "Away"], "chn_upcoming_fixtures.csv")

    out = src.rename(
        columns={
            "Wk": "round",
            "Date": "match_date",
            "Time": "match_time",
            "Home": "home_team",
            "Away": "away_team",
        }
    ).copy()

    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    out["match_date_dt"] = _parse_fixture_dates(out["match_date"])
    export_date = export_now.tz_convert(TZ).normalize().tz_localize(None)
    out = out[out["match_date_dt"].notna()].copy()
    out = out[out["match_date_dt"] >= export_date].copy()

    # Off-season short-circuit: with no rows, `out.apply(..., axis=1)` below
    # returns an empty DataFrame instead of a Series, which then fails to
    # assign back into `out["fixture_id"]`. Return the expected schema
    # directly so downstream consumers see a well-formed empty DataFrame.
    if out.empty:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)

    out["match_date"] = format_date_only_series(out["match_date_dt"])
    out["match_time"] = out["match_time"].astype(str).str.strip().str.slice(0, 5)
    kickoff = pd.to_datetime(
        out["match_date"] + " " + out["match_time"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    # Source kickoff times are UTC (GMT, UK time without DST): localize as UTC then
    # convert to Europe/London so summer fixtures pick up the +1h BST offset.
    out["kickoff_at"] = (
        kickoff.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    out["kickoff_at"] = out["kickoff_at"].str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
    out["fixture_id"] = out.apply(
        lambda row: (
            f"{COMPETITION_CODE}-{season}-{int(row['round'])}-{row['match_date']}"
            f"-{_slugify(row['home_team'])}-{_slugify(row['away_team'])}"
        ),
        axis=1,
    )

    out = out[UPCOMING_COLUMNS].sort_values(["match_date", "match_time", "home_team", "away_team"]).reset_index(drop=True)
    return out


def build_match_predictions(simulations_path: str, upcoming: pd.DataFrame) -> pd.DataFrame:
    sims = pd.read_csv(simulations_path)
    _require_columns(
        sims,
        ["Home Team", "Away Team", "Home Win Probability", "Draw Probability", "Away Win Probability"],
        "CHN_team_stats_match_simulations.csv",
    )

    sims = sims.rename(
        columns={
            "Home Team": "home_team",
            "Away Team": "away_team",
            "Home Win Probability": "home_win_prob",
            "Draw Probability": "draw_prob",
            "Away Win Probability": "away_win_prob",
        }
    ).copy()

    sims = sims[["home_team", "away_team", "home_win_prob", "draw_prob", "away_win_prob"]]
    sims["home_team"] = sims["home_team"].astype(str).str.strip()
    sims["away_team"] = sims["away_team"].astype(str).str.strip()
    dupes = sims.duplicated(subset=["home_team", "away_team"], keep=False)
    if dupes.any():
        duplicated_pairs = sims.loc[dupes, ["home_team", "away_team"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Simulation table has duplicate home/away pairs: {duplicated_pairs}")

    merged = upcoming.merge(sims, on=["home_team", "away_team"], how="left", validate="one_to_one")
    if merged[["home_win_prob", "draw_prob", "away_win_prob"]].isna().any(axis=1).any():
        missing = merged.loc[
            merged[["home_win_prob", "draw_prob", "away_win_prob"]].isna().any(axis=1),
            ["fixture_id", "home_team", "away_team"],
        ].to_dict("records")
        raise ValueError(f"Missing simulation probabilities for fixtures: {missing}")

    merged[["home_win_prob", "draw_prob", "away_win_prob"]] = merged[
        ["home_win_prob", "draw_prob", "away_win_prob"]
    ].apply(pd.to_numeric, errors="coerce")
    out = _normalize_probabilities(merged)

    out["home_win_fair_odds"] = out["home_win_prob"].map(_fair_odds)
    out["draw_fair_odds"] = out["draw_prob"].map(_fair_odds)
    out["away_win_fair_odds"] = out["away_win_prob"].map(_fair_odds)
    out = out[PREDICTION_COLUMNS].copy()
    return out


def _form_token(result: str, is_home: bool) -> str | None:
    mapping = {
        ("H", True): "W",
        ("D", True): "D",
        ("A", True): "L",
        ("H", False): "L",
        ("D", False): "D",
        ("A", False): "W",
    }
    return mapping.get((result, is_home))


def _build_team_form_map(matches: pd.DataFrame) -> dict[str, str]:
    _require_columns(matches, ["Date", "Home", "Away", "Res"], "CHN_Super League.csv")
    played = matches.copy()
    played["parsed_date"] = _parse_match_dates(played["Date"])
    played["Res"] = played["Res"].astype(str).str.strip()
    played = played[played["parsed_date"].notna() & played["Res"].isin(["H", "D", "A"])].copy()
    played = played.sort_values(["parsed_date", "Time"], na_position="last")

    team_results: dict[str, list[str]] = {}
    for _, row in played.iterrows():
        home_team = str(row["Home"]).strip()
        away_team = str(row["Away"]).strip()
        result = str(row["Res"]).strip()

        home_token = _form_token(result, True)
        away_token = _form_token(result, False)
        if home_token is not None:
            team_results.setdefault(home_team, []).append(home_token)
        if away_token is not None:
            team_results.setdefault(away_team, []).append(away_token)

    return {team: ",".join(results[-5:][::-1]) for team, results in team_results.items()}


def _current_season_teams(matches: pd.DataFrame, season: str) -> set[str]:
    """Clubs that appear in ``season``'s fixtures, home or away.

    The model's 18-month training window straddles two seasons, so
    ``CHN_team_stats.csv`` carries relegated clubs alongside the current league —
    18 rows for a 16-team competition. They still belong in the fit as opponents
    that calibrate everyone else, but the panel has to be able to tell them apart.
    """
    _require_columns(matches, ["Season", "Home", "Away"], "CHN_Super League.csv")
    season_rows = matches.copy()
    season_rows["Season"] = season_rows["Season"].astype(str).str.strip()
    season_rows = season_rows[season_rows["Season"] == season]
    if season_rows.empty:
        raise ValueError(f"No matches found for season {season} in CHN_Super League.csv")
    home = season_rows["Home"].dropna().astype(str).str.strip()
    away = season_rows["Away"].dropna().astype(str).str.strip()
    return set(home) | set(away)


def build_team_strength_rankings(
    team_stats_path: str, matches_path: str, season: str
) -> tuple[pd.DataFrame, float]:
    """Strength rankings, plus the league-average goals-per-match reference.

    The reference is returned rather than stored per row because it is a single
    league-wide scalar: it is the value both ``attack_rating`` and
    ``defense_rating`` take for a perfectly average club, so the dashboard legend
    needs it to give the two columns a neutral line. It belongs in the meta.
    """
    stats = pd.read_csv(team_stats_path)
    _require_columns(
        stats,
        ["Team", "Attack", "Defense", "Const", "HomeAdv", "WeightedMatches"],
        "CHN_team_stats.csv",
    )
    matches = pd.read_csv(matches_path)
    form_map = _build_team_form_map(matches)
    season_teams = _current_season_teams(matches, season)

    out = stats.rename(
        columns={
            "Team": "team",
            "Attack": "attack_coef",
            "Defense": "defense_coef",
            "WeightedMatches": "weighted_matches",
        }
    ).copy()
    out["team"] = out["team"].astype(str).str.strip()
    for col in ("attack_coef", "defense_coef", "weighted_matches"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Const and HomeAdv are league-wide, repeated on every row by run_dixon_coles_model.
    const = float(pd.to_numeric(stats["Const"], errors="coerce").iloc[0])
    home_adv = float(pd.to_numeric(stats["HomeAdv"], errors="coerce").iloc[0])
    league_avg_goals = per_match_goals(0.0, const, home_adv)

    out["in_current_season"] = out["team"].isin(season_teams)
    if not out["in_current_season"].any():
        raise ValueError(
            "CHN_team_stats.csv contains no clubs from the current season — "
            "team naming has probably drifted between the model and the fixture list"
        )

    # Ratings are only meaningful against a stated field, and the one a reader has
    # in mind is this season's league. Relegated clubs are still rated — against
    # that same field, which is the comparison that answers "how would they fare
    # here now" — they are simply marked so the panel can grey them out.
    field = {
        row.team: (row.attack_coef, row.defense_coef)
        for row in out.itertuples()
        if row.in_current_season
    }

    def rating_for(team: str, atk: float, dfn: float) -> float:
        # Exclude by name, not by coefficient value: calibrated_rating compares a
        # club against a reference facing the *same* opponents, so the club must
        # not appear in its own opponent list.
        opponents = [pair for name, pair in field.items() if name != team]
        return calibrated_rating(atk, dfn, const, home_adv, opponents)

    out["attack_rating"] = [per_match_goals(v, const, home_adv) for v in out["attack_coef"]]
    out["defense_rating"] = [per_match_goals(v, const, home_adv) for v in out["defense_coef"]]
    out["overall_rating"] = [
        rating_for(row.team, row.attack_coef, row.defense_coef) for row in out.itertuples()
    ]

    # exp() is monotone, so ranking on goals per match orders identically to
    # ranking on the coefficients: more goals scored is better, fewer conceded is better.
    out["attack_rank"] = out["attack_rating"].rank(method="min", ascending=False).astype(int)
    out["defense_rank"] = out["defense_rating"].rank(method="min", ascending=True).astype(int)

    median_weight = out["weighted_matches"].median()
    out["low_sample"] = out["weighted_matches"] < LOW_SAMPLE_FRACTION * median_weight

    out = out.sort_values(["overall_rating", "team"], ascending=[False, True]).reset_index(drop=True)
    out["rank_overall"] = out.index + 1
    out["form"] = out["team"].map(form_map).fillna("")
    out = out[STRENGTH_COLUMNS].copy()
    return out, league_avg_goals


def _read_model_updated_at(default: str) -> str:
    """Return the model's last-fit timestamp from the sidecar, or ``default``."""
    path = model_meta_json()
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh).get("model_updated_at")
        if value:
            return str(value)
        log.warning("%s has no model_updated_at; falling back to export time", path)
    except FileNotFoundError:
        log.warning("Model meta sidecar %s not found; falling back to export time", path)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s); falling back to export time", path, exc)
    return default


def build_dashboard_meta(
    matches: pd.DataFrame,
    upcoming: pd.DataFrame,
    season: str,
    export_now: pd.Timestamp,
    round_progress: dict[str, int],
    league_avg_goals: float,
) -> pd.DataFrame:
    played = matches.copy()
    played["parsed_date"] = _parse_match_dates(played["Date"])
    played["Res"] = played["Res"].astype(str).str.strip()
    played = played[played["parsed_date"].notna() & played["Res"].isin(["H", "D", "A"])].copy()
    if played.empty:
        raise ValueError("No completed matches found for dashboard_meta.csv")

    # Off-season tolerance: when there are no upcoming fixtures in the next 14 days
    # (e.g. between rounds, mid-season break, end of season), the meta is still
    # valuable — strength rankings and last-completed match info are fresh. Emit
    # the meta with next_fixture_date left blank rather than failing Step 5 and
    # leaving the entire dashboard frozen at the previous run's snapshot.
    next_fixture_date = upcoming["match_date"].min() if not upcoming.empty else None

    export_updated_at = export_now.tz_convert(TZ).isoformat(timespec="seconds")
    # model_updated_at is pinned to the last model/full run via the sidecar written by
    # DC_CHN.py; odds-only refreshes re-run this export but never the model, so this
    # stays put while updated_at (export time) advances. Fall back to the export time
    # if the sidecar is missing (e.g. before the first model run has written it).
    model_updated_at = _read_model_updated_at(default=export_updated_at)

    meta = pd.DataFrame(
        [
            {
                "competition_code": COMPETITION_CODE,
                "competition_name": COMPETITION_NAME,
                "season": season,
                "updated_at": export_updated_at,
                "model_updated_at": model_updated_at,
                "timezone": "Europe/London",
                "last_completed_match_date": played["parsed_date"].max().strftime("%Y-%m-%d"),
                "next_fixture_date": next_fixture_date,
                "matches_played": round_progress["matches_played"],
                "current_round": round_progress["current_round"],
                "total_rounds": round_progress["total_rounds"],
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                # Neutral line for the strength panel's ATT/DEF columns: what a
                # perfectly average club scores and concedes per match.
                "league_avg_goals": round(league_avg_goals, 4),
            }
        ],
        columns=META_COLUMNS,
    )
    return meta


def validate_outputs(meta: pd.DataFrame, upcoming: pd.DataFrame, predictions: pd.DataFrame, strength: pd.DataFrame, export_now: pd.Timestamp) -> None:
    if len(meta) != 1:
        raise ValueError("dashboard_meta.csv must contain exactly one row")

    export_date = export_now.tz_convert(TZ).strftime("%Y-%m-%d")
    if not upcoming["fixture_id"].is_unique:
        raise ValueError("upcoming_fixtures.csv fixture_id values must be unique")
    if (upcoming["match_date"] < export_date).any():
        bad = upcoming.loc[upcoming["match_date"] < export_date, ["fixture_id", "match_date"]].to_dict("records")
        raise ValueError(f"upcoming_fixtures.csv contains past matches: {bad}")

    if not predictions["fixture_id"].isin(set(upcoming["fixture_id"])).all():
        raise ValueError("match_predictions.csv contains fixture_id values not found in upcoming_fixtures.csv")

    probs = predictions[["home_win_prob", "draw_prob", "away_win_prob"]]
    if ((probs < 0) | (probs > 1)).any().any():
        raise ValueError("match_predictions.csv contains probabilities outside [0, 1]")
    if not ((probs.sum(axis=1) - 1.0).abs() <= 1e-6).all():
        raise ValueError("match_predictions.csv probability totals are not within tolerance of 1")

    if not strength["team"].is_unique:
        raise ValueError("team_strength_rankings.csv team values must be unique")
    # overall_rating is no longer attack_rating - defense_rating (it is the calibrated
    # strength rating, and the two columns are now goals per match), so the invariants
    # that replace that identity are the ones the panel actually depends on.
    expected_ranks = list(range(1, len(strength) + 1))
    if strength["rank_overall"].tolist() != expected_ranks:
        raise ValueError("team_strength_rankings.csv rank_overall must be 1..N in row order")
    if not strength["overall_rating"].is_monotonic_decreasing:
        raise ValueError("team_strength_rankings.csv rows must be sorted by overall_rating desc")
    goal_cols = strength[["attack_rating", "defense_rating"]]
    if not (goal_cols > 0).all().all():
        raise ValueError(
            "team_strength_rankings.csv attack_rating/defense_rating must be positive "
            "(they are goals per match, not log coefficients)"
        )
    if not strength["in_current_season"].any():
        raise ValueError("team_strength_rankings.csv has no clubs in the current season")


def build_match_results(matches: pd.DataFrame, export_now: pd.Timestamp) -> pd.DataFrame:
    """Finished matches inside the trailing window, keyed by the same fixture_id.

    The id is rebuilt from the row's *own* Season and Round rather than the export's
    current season, so a match played last season keeps the id it had when it was an
    upcoming fixture. That is the whole point of the file: it is what lets a consumer
    join a signal it recorded weeks ago to the score.

    Measured against six months of this repo's own committed fixture snapshots, 92.1%
    of fixtures whose scheduled date has passed join to a result on fixture_id
    exactly. 3.0% are postponed and not yet played, so their absence here is correct.
    The remaining 5.0% are the real weakness: the id embeds the match date, so a match
    moved even by a day gets a different id once played, and the signal recorded
    against the old id never finds its result.

    Those two compound -- a postponed match, once played, is by definition on a new
    date and so lands in the 5%. A consumer should fall back to
    (season, home_team, away_team), the same two-key shape the kickoff join uses.
    Missing a join leaves a bet unsettled rather than settled wrongly, which is the
    survivable direction.
    """
    _require_columns(matches, ["Season", "Round", "Date", "Home", "Away", "HG", "AG", "Res"], "matches CSV")

    out = matches.copy()
    out["parsed_date"] = _parse_match_dates(out["Date"])
    out["result"] = out["Res"].astype(str).str.strip()
    out = out[out["parsed_date"].notna() & out["result"].isin(["H", "D", "A"])].copy()

    cutoff = export_now.tz_convert(TZ).normalize().tz_localize(None) - pd.Timedelta(days=RESULTS_WINDOW_DAYS)
    out = out[out["parsed_date"] >= cutoff].copy()

    if out.empty:
        return pd.DataFrame(columns=RESULTS_COLUMNS)

    out["source_result"] = out["result"]
    out["match_date"] = format_date_only_series(out["parsed_date"])
    out["season"] = out["Season"].map(_normalize_season_value)
    out["round"] = pd.to_numeric(out["Round"], errors="coerce").astype("Int64")
    out["home_team"] = out["Home"].astype(str)
    out["away_team"] = out["Away"].astype(str)
    # Goals are the only numbers here and they are counts. A row whose Res parsed but
    # whose score did not is dropped rather than exported with a null score: a
    # consumer settling on it would read the missing side as zero.
    out["home_goals"] = pd.to_numeric(out["HG"], errors="coerce").astype("Int64")
    out["away_goals"] = pd.to_numeric(out["AG"], errors="coerce").astype("Int64")
    out = out[out["home_goals"].notna() & out["away_goals"].notna()].copy()

    # The score is the primitive fact and `result` is derived from it, so the two can
    # never disagree inside this file. The source CSV's own `Res` column is treated as
    # a second opinion and cross-checked: where it dissents the row is exported
    # `disputed` rather than `finished`, and a consumer settles only on `finished`.
    #
    # Not a hypothetical -- two 2026 rows disagree today (a 3-2 recorded as a draw and
    # a 2-2 recorded as an away win). Picking a winner between the two spellings would
    # mean settling a bet on a guess, which is the one direction that costs money; a
    # row nobody settles costs a warning.
    out["result"] = out.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"] else ("A" if r["home_goals"] < r["away_goals"] else "D"),
        axis=1,
    )
    disputed = out["source_result"] != out["result"]
    out["status"] = "finished"
    out.loc[disputed, "status"] = "disputed"
    if disputed.any():
        log.warning(
            "%d match(es) where Res disagrees with the score, exported as disputed "
            "and not settleable: %s",
            int(disputed.sum()),
            ", ".join(
                f"{r.match_date} {r.home_team} {r.home_goals}-{r.away_goals} {r.away_team} "
                f"(Res={r.source_result}, score says {r.result})"
                for r in out[disputed].itertuples()
            ),
        )

    out["fixture_id"] = out.apply(
        lambda row: (
            f"{COMPETITION_CODE}-{row['season']}-{int(row['round'])}-{row['match_date']}"
            f"-{_slugify(row['home_team'])}-{_slugify(row['away_team'])}"
        ),
        axis=1,
    )

    out = out[RESULTS_COLUMNS].sort_values(["match_date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def validate_results(results: pd.DataFrame) -> None:
    if results.empty:
        return
    if not results["fixture_id"].is_unique:
        dupes = results.loc[results["fixture_id"].duplicated(), "fixture_id"].tolist()
        raise ValueError(f"match_results.csv fixture_id values must be unique: {dupes[:5]}")
    if not results["result"].isin(["H", "D", "A"]).all():
        raise ValueError("match_results.csv result must be one of H/D/A")
    if not results["status"].isin(["finished", "disputed"]).all():
        raise ValueError("match_results.csv status must be one of finished/disputed")
    if (results[["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError("match_results.csv goal counts cannot be negative")
    # `result` is derived from the score, so this can only fail if the derivation is
    # edited into disagreeing with itself. It is the invariant a consumer settles on.
    derived = results.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"] else ("A" if r["home_goals"] < r["away_goals"] else "D"),
        axis=1,
    )
    mismatched = results.loc[derived != results["result"], "fixture_id"].tolist()
    if mismatched:
        raise ValueError(f"match_results.csv result disagrees with the score: {mismatched[:5]}")


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(df))


def run() -> None:
    paths = ExportPaths()
    os.makedirs(paths.out_dir, exist_ok=True)

    export_now = pd.Timestamp.now(tz=TZ)
    matches = pd.read_csv(paths.matches_csv)
    season = _derive_current_season(matches)
    log.info("Exporting dashboard CSV for season %s", season)

    upcoming = build_upcoming_fixtures(paths.fixtures_csv, season, export_now)
    predictions = build_match_predictions(paths.simulations_csv, upcoming)
    strength, league_avg_goals = build_team_strength_rankings(
        paths.team_stats_csv, paths.matches_csv, season
    )
    results = build_match_results(matches, export_now)
    round_progress = build_round_progress(paths.fresh_schedule_csv, season)
    meta = build_dashboard_meta(
        matches, upcoming, season, export_now, round_progress, league_avg_goals
    )

    validate_outputs(meta, upcoming, predictions, strength, export_now)
    validate_results(results)

    write_csv(meta, paths.meta_csv)
    write_csv(upcoming, paths.upcoming_csv)
    write_csv(predictions, paths.predictions_csv)
    write_csv(strength, paths.strength_csv)
    write_csv(results, paths.results_csv)


def main() -> None:
    try:
        run()
    except Exception as exc:  # pragma: no cover - top-level script guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
