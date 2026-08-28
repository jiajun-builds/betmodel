"""Gate G3, first half: the engine reproduces the frozen signal decisions.

Replays both pre-merge pipelines from their frozen inputs and requires the same
probabilities, the same pick and the same state for every published fixture.

The second half, comparing the whole legacy payload field by field, needs the
compatibility exporter and lands with it. This half already covers the part that
decides money: what fires, on which side, at whose price.

One asymmetry worth naming. The frozen Liga MX board has two firing rows, so its
positive path is covered here. The frozen CSL board has none, its best row
sitting at 0.196 against a 0.20 bar, so CSL's firing path is covered by
constructed cases in the unit tests instead.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from betmodel.config import load_league
from betmodel.signals.engine import build_signals

GOLDEN = "tests/golden"
EXACT = 1e-12

PROBABILITY_FIELDS = ("home_win_prob", "draw_prob", "away_win_prob")


def _frozen_fixtures(league: str, tmp: str) -> str:
    """Adapt the pre-merge fixture file to the post-merge interface.

    The baseline predates the explicit kickoff column, and the loader refuses to
    derive one. Deriving it here rather than editing the baseline keeps the
    frozen bytes frozen; the derivation is exactly what the pre-merge code did,
    reading the Date and Time pair as UTC.
    """
    source = f"{GOLDEN}/{league}/inputs/upcoming_fixtures.csv"
    frame = pd.read_csv(source, encoding="utf-8-sig")
    if "kickoff_utc" not in frame.columns:
        stamps = pd.to_datetime(
            frame["Date"].astype(str) + " " + frame["Time"].astype(str),
            utc=True, errors="coerce",
        )
        frame["kickoff_utc"] = stamps.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    path = os.path.join(tmp, f"{league}-fixtures.csv")
    frame.to_csv(path, index=False)
    return path


def _golden_payload(league: str) -> dict:
    with open(f"{GOLDEN}/{league}/published/upcoming_market_comparison.json") as fh:
        return json.load(fh)


def _replay(league: str) -> tuple[dict, dict]:
    """The frozen payload, and the engine's signals keyed by fixture id."""
    payload = _golden_payload(league)
    published_at: datetime = (
        pd.Timestamp(payload["meta"]["updated_at"]).tz_convert("UTC").to_pydatetime()
    )
    with tempfile.TemporaryDirectory() as tmp:
        signals = build_signals(
            league, load_league(league), now=published_at,
            simulations_path=f"{GOLDEN}/{league}/model/simulations.csv",
            fixtures_path=_frozen_fixtures(league, tmp),
            history_path=f"{GOLDEN}/{league}/inputs/odds_capture_history.csv",
            watch_path=(
                f"{GOLDEN}/{league}/inputs/capture_watch.csv"
                if os.path.exists(f"{GOLDEN}/{league}/inputs/capture_watch.csv") else None
            ),
        )
    return payload, {s.fixture_id: s for s in signals}


def _normalise(value) -> str:
    """The two boards spelled the empty state differently."""
    text = "" if value is None else str(value)
    return "" if text in ("none", "None") else text


@pytest.fixture(scope="module", params=["csl", "ligamx"])
def replay(request):
    payload, signals = _replay(request.param)
    return request.param, payload, signals


def test_every_published_fixture_is_produced_again(replay):
    """Identity is derived, so a mismatch here is a mismatch about what a fixture
    IS, not about what was decided. That is how the local-matchday bug showed up:
    a 19:00 kickoff in a UTC-6 league was being named by the next UTC day.
    """
    league, payload, signals = replay
    missing = [r["fixture_id"] for r in payload["rows"] if r["fixture_id"] not in signals]
    assert not missing, f"{league}: not reproduced: {missing}"


def test_the_engine_publishes_no_fixture_the_baseline_did_not(replay):
    league, payload, signals = replay
    published = {r["fixture_id"] for r in payload["rows"]}
    extra = sorted(set(signals) - published)
    assert not extra, f"{league}: newly published: {extra}"


def test_model_probabilities_match(replay):
    league, payload, signals = replay
    worst = 0.0
    for row in payload["rows"]:
        signal = signals[row["fixture_id"]]
        for index, field in enumerate(PROBABILITY_FIELDS):
            if row.get(field) is None:
                continue
            worst = max(worst, abs(row[field] - signal.probabilities[index]))
    assert worst < EXACT, f"{league}: max probability difference {worst:.3e}"


def test_the_de_bias_method_is_reproduced(replay):
    """One league anchors its draw on the market, the other ships the raw grid."""
    league, payload, signals = replay
    for row in payload["rows"]:
        published = row.get("debias_method")
        if published is None:
            continue
        assert signals[row["fixture_id"]].debias_method == published


def test_the_same_side_is_picked(replay):
    league, payload, signals = replay
    for row in payload["rows"]:
        assert _normalise(row.get("signal_pick")) == _normalise(signals[row["fixture_id"]].pick), \
            f"{league}: {row['fixture_id']}"


def test_the_same_state_is_reached(replay):
    league, payload, signals = replay
    for row in payload["rows"]:
        assert _normalise(row.get("signal_state")) == _normalise(signals[row["fixture_id"]].state), \
            f"{league}: {row['fixture_id']}"


def test_the_positive_path_is_actually_exercised_somewhere(replay):
    """Agreement on nothing-fires is weak evidence on its own."""
    league, payload, _ = replay
    firing = sum(1 for r in payload["rows"] if _normalise(r.get("signal_state")) == "bet")
    if league == "ligamx":
        assert firing >= 1, "the Liga MX baseline should contain firing rows"
    else:
        # Recorded rather than asserted away: CSL's firing path is covered by
        # constructed cases in tests/unit/test_signal_engine.py.
        assert firing == 0
