from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

from csl.odds.books import BET_BOOKS
from csl.paths import data_dashboard_csv_dir, data_dashboard_json_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportPaths:
    csv_dir: str = data_dashboard_csv_dir()
    json_dir: str = data_dashboard_json_dir()

    @property
    def meta_csv(self) -> str:
        return os.path.join(self.csv_dir, "dashboard_meta.csv")

    @property
    def upcoming_csv(self) -> str:
        return os.path.join(self.csv_dir, "upcoming_fixtures.csv")

    @property
    def predictions_csv(self) -> str:
        return os.path.join(self.csv_dir, "match_predictions.csv")

    @property
    def strength_csv(self) -> str:
        return os.path.join(self.csv_dir, "team_strength_rankings.csv")

    @property
    def market_comparison_csv(self) -> str:
        return os.path.join(self.csv_dir, "upcoming_market_comparison.csv")

    @property
    def results_csv(self) -> str:
        return os.path.join(self.csv_dir, "match_results.csv")

    @property
    def meta_json(self) -> str:
        return os.path.join(self.json_dir, "dashboard_meta.json")

    @property
    def upcoming_json(self) -> str:
        return os.path.join(self.json_dir, "upcoming_fixtures.json")

    @property
    def predictions_json(self) -> str:
        return os.path.join(self.json_dir, "match_predictions.json")

    @property
    def strength_json(self) -> str:
        return os.path.join(self.json_dir, "team_strength_rankings.json")

    @property
    def market_comparison_json(self) -> str:
        return os.path.join(self.json_dir, "upcoming_market_comparison.json")

    @property
    def results_json(self) -> str:
        return os.path.join(self.json_dir, "match_results.json")


# Dashboard v2.8 market-comparison contract: model probabilities + EVERY bet book's
# opening line/EV + the best-price layer + betting signal. No Pinnacle "Now" line —
# the board is the opening-line signal surface (backtest.md §13.4). Order is
# significant: the JSON validator requires each row's keys to equal this list exactly.
#
# Generated from BET_BOOKS rather than written out, so adding a third book is a
# registry edit and not a hand-sync of three column lists. `books` is stdlib-only, so
# this import stays cheap.
#
# Per-book AND best-price fields both ship: app.js renders one odds column per book
# (with the best highlighted) and drives EV, the badge and the logos off the best
# layer. `signal_books` is a "|"-joined STRING — never a list, which would raise in
# _clean_scalar's pd.isna.
MARKET_COMPARISON_FIELDS = [
    # Stable join key across every export, and the one a downstream ledger keys a
    # signal's lifecycle on -- upcoming_fixtures.json and match_predictions.json
    # already carry it, and the comparison CSV has always had it. Emitting it here
    # too means a consumer never has to re-derive a fixture's identity from the
    # (home_team, away_team) pair, which is lossy: team names are display strings,
    # and a rename upstream silently breaks the join.
    "fixture_id",
    # Both sit in the comparison CSV already and both are also on the fixture this
    # row belongs to, so emitting them is not new data -- it removes a join. A
    # consumer previously had to reach into upcoming_fixtures.json on the
    # (home_team, away_team) pair just to date a signal, and a fixture that join
    # missed produced a row with no kickoff at all.
    "round",
    "kickoff_at",
    "home_team",
    "away_team",
    "match_time",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "debias_method",
    *[
        col
        for book in BET_BOOKS
        for col in (
            book.odds_col("home"), book.odds_col("draw"), book.odds_col("away"),
            book.ev_col("home"), book.ev_col("draw"), book.ev_col("away"),
        )
    ],
    *[f"best_open_{side}_{kind}"
      for kind in ("odds", "ev", "book")
      for side in ("home", "draw", "away")],
    "signal_pick",
    "signal_state",
    "signal_book",
    "signal_books",
    *[book.last_update_col for book in BET_BOOKS],
    "fetched_at",
]


# Finished matches, facts only. `result` is derived from the score upstream, never
# from the source CSV's own Res column; a row where those two disagreed is exported
# with status "disputed" and must not be settled on. See build_match_results.
RESULTS_FIELDS = [
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


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    # bool is a subclass of int; keep it untouched if it ever appears.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)
    return value


def _frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({key: _clean_scalar(value) for key, value in row.items()})
    return records


def _load_single_row_csv(path: str, required_columns: list[str], label: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    _require_columns(df, required_columns, label)
    if len(df) != 1:
        raise ValueError(f"{label} must contain exactly one row; got {len(df)}")
    return _frame_to_records(df[required_columns])[0]


def _load_rows_csv(path: str, required_columns: list[str], label: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    _require_columns(df, required_columns, label)
    return _frame_to_records(df[required_columns])


def _build_common_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "competition_code": meta["competition_code"],
        "season": str(meta["season"]),
        "updated_at": meta["updated_at"],
    }


def _write_json(payload: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    row_count = len(payload["rows"]) if "rows" in payload else 1
    log.info("Wrote %s (%d rows)", path, row_count)


def validate_payloads(
    meta_payload: dict[str, Any],
    upcoming_payload: dict[str, Any],
    predictions_payload: dict[str, Any],
    strength_payload: dict[str, Any],
    market_comparison_payload: dict[str, Any],
    results_payload: dict[str, Any],
) -> None:
    for payload_name, payload in (
        ("upcoming_fixtures.json", upcoming_payload),
        ("match_predictions.json", predictions_payload),
        ("team_strength_rankings.json", strength_payload),
        ("upcoming_market_comparison.json", market_comparison_payload),
        ("match_results.json", results_payload),
    ):
        shared_meta = payload.get("meta", {})
        if shared_meta.get("competition_code") != meta_payload["competition_code"]:
            raise ValueError(f"{payload_name} competition_code does not match dashboard_meta.json")
        if shared_meta.get("season") != meta_payload["season"]:
            raise ValueError(f"{payload_name} season does not match dashboard_meta.json")
        if shared_meta.get("updated_at") != meta_payload["updated_at"]:
            raise ValueError(f"{payload_name} updated_at does not match dashboard_meta.json")

    upcoming_ids = {row["fixture_id"] for row in upcoming_payload["rows"]}
    prediction_ids = [row["fixture_id"] for row in predictions_payload["rows"]]
    if len(upcoming_ids) != len(upcoming_payload["rows"]):
        raise ValueError("upcoming_fixtures.json contains duplicate fixture_id values")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("match_predictions.json contains duplicate fixture_id values")
    if not set(prediction_ids).issubset(upcoming_ids):
        raise ValueError("match_predictions.json contains fixture_id values not found in upcoming_fixtures.json")

    market_rows = market_comparison_payload["rows"]
    for row in market_rows:
        row_keys = list(row.keys())
        expected_keys = MARKET_COMPARISON_FIELDS
        if row_keys != expected_keys:
            raise ValueError(
                "upcoming_market_comparison.json rows must contain exactly the approved fields; "
                f"got {row_keys}"
            )

    # Same referential check the predictions payload already gets. Without it
    # `fixture_id` is only a string that looks like a key: a comparison row built
    # from a stale market CSV would still export, and every consumer joining on it
    # would silently drop that fixture instead of failing here.
    result_rows = results_payload["rows"]
    for row in result_rows:
        if list(row.keys()) != RESULTS_FIELDS:
            raise ValueError(
                "match_results.json rows must contain exactly the approved fields; "
                f"got {list(row.keys())}"
            )
    result_ids = [row["fixture_id"] for row in result_rows]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("match_results.json contains duplicate fixture_id values")
    # Deliberately NOT checked against upcoming_fixtures: these are played matches and
    # the upcoming payload holds only future ones, so the two sets are disjoint by
    # construction. The pairing that matters is a *past* signal to its result, which
    # only a consumer holding that signal can make.
    for row in result_rows:
        if row["status"] not in ("finished", "disputed"):
            raise ValueError(f"match_results.json unknown status: {row['status']}")
        if row["result"] not in ("H", "D", "A"):
            raise ValueError(f"match_results.json unknown result: {row['result']}")

    market_ids = [row["fixture_id"] for row in market_rows]
    if len(set(market_ids)) != len(market_ids):
        raise ValueError("upcoming_market_comparison.json contains duplicate fixture_id values")
    if not set(market_ids).issubset(upcoming_ids):
        raise ValueError(
            "upcoming_market_comparison.json contains fixture_id values not found in "
            "upcoming_fixtures.json"
        )


def run() -> None:
    paths = ExportPaths()
    os.makedirs(paths.json_dir, exist_ok=True)

    meta_row = _load_single_row_csv(
        paths.meta_csv,
        [
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
        ],
        "dashboard_meta.csv",
    )
    meta_row["season"] = str(meta_row["season"])

    upcoming_rows = _load_rows_csv(
        paths.upcoming_csv,
        [
            "fixture_id",
            "round",
            "match_date",
            "match_time",
            "kickoff_at",
            "home_team",
            "away_team",
        ],
        "upcoming_fixtures.csv",
    )

    prediction_rows = _load_rows_csv(
        paths.predictions_csv,
        [
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
        ],
        "match_predictions.csv",
    )

    strength_rows = _load_rows_csv(
        paths.strength_csv,
        [
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
        ],
        "team_strength_rankings.csv",
        )

    market_comparison_rows = _load_rows_csv(
        paths.market_comparison_csv,
        MARKET_COMPARISON_FIELDS,
        "upcoming_market_comparison.csv",
    )

    results_rows = _load_rows_csv(
        paths.results_csv,
        RESULTS_FIELDS,
        "match_results.csv",
    )
    # Same coercion dashboard_meta already does: read_csv types a numeric CSL season
    # as an int, while ligamxterminal's is "Apertura 2026". One field of one contract
    # must not be an int in one league and a string in the other.
    for row in results_rows:
        row["season"] = str(row["season"])

    common_meta = _build_common_meta(meta_row)
    meta_payload = meta_row
    upcoming_payload = {"meta": common_meta, "rows": upcoming_rows}
    predictions_payload = {
        "meta": {
            **common_meta,
            "model_name": meta_row["model_name"],
            "model_version": meta_row["model_version"],
        },
        "rows": prediction_rows,
    }
    strength_payload = {"meta": common_meta, "rows": strength_rows}
    market_comparison_payload = {"meta": common_meta, "rows": market_comparison_rows}
    # `season` in the shared meta is the dashboard's current season, while these rows
    # carry their own -- the window is a trailing 180 days and crosses season
    # boundaries by design.
    results_payload = {"meta": common_meta, "rows": results_rows}

    validate_payloads(
        meta_payload,
        upcoming_payload,
        predictions_payload,
        strength_payload,
        market_comparison_payload,
        results_payload,
    )

    _write_json(meta_payload, paths.meta_json)
    _write_json(upcoming_payload, paths.upcoming_json)
    _write_json(predictions_payload, paths.predictions_json)
    _write_json(strength_payload, paths.strength_json)
    _write_json(market_comparison_payload, paths.market_comparison_json)
    _write_json(results_payload, paths.results_json)


def main() -> None:
    try:
        run()
    except Exception as exc:  # pragma: no cover - top-level script guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
