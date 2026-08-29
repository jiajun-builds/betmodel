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


# --------------------------------------------------------------------------- #
# G3, second half: the whole legacy payload, field by field
# --------------------------------------------------------------------------- #

from betmodel.publish import legacy  # noqa: E402

#: Fields whose source was deliberately removed, recorded as D1. The current
#: price is no longer fetched, so the reference columns it fed carry the anchor's
#: OPENING price instead, and are empty for a league with no anchor opens yet.
D1_RETIRED_NOW_LINE = frozenset({
    "pinnacle_home_odds", "pinnacle_draw_odds", "pinnacle_away_odds",
    "pinnacle_last_update", "pinnacle_fetched_at",
})

#: When the export ran. Not a property of the data.
EXPORT_STAMP = frozenset({"fetched_at"})

ACCEPTED = D1_RETIRED_NOW_LINE | EXPORT_STAMP


def _equal(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(left - right) < 1e-9
    return left == right


def _rendered(league: str) -> tuple[dict, dict]:
    payload, signals = _replay(league)
    published_at = pd.Timestamp(payload["meta"]["updated_at"]).tz_convert("UTC").to_pydatetime()
    rendered = legacy.market_comparison(
        load_league(league), list(signals.values()), generated_at=published_at
    )
    return payload, rendered


@pytest.fixture(scope="module", params=["csl", "ligamx"])
def rendered(request):
    payload, output = _rendered(request.param)
    return request.param, payload, output


def test_the_field_set_and_its_order_are_unchanged(rendered):
    """The board reads these by name, and one prefix is frozen at a historical
    spelling precisely because renaming it blanks every price it names."""
    league, payload, output = rendered
    assert list(output["rows"][0]) == list(payload["rows"][0]), league


def test_the_envelope_matches(rendered):
    league, payload, output = rendered
    assert output["meta"] == payload["meta"], league


def test_every_field_of_every_row_matches_or_is_an_accepted_difference(rendered):
    """The strongest check available: thirty-eight and thirty-one fields, every
    published row, against what the two pipelines actually shipped."""
    league, payload, output = rendered
    ours = {r["fixture_id"]: r for r in output["rows"]}
    unexplained: list[str] = []
    for row in payload["rows"]:
        mine = ours[row["fixture_id"]]
        for field, value in row.items():
            if field in ACCEPTED or _equal(value, mine.get(field)):
                continue
            unexplained.append(
                f"{row['fixture_id']}.{field}: {value!r} != {mine.get(field)!r}"
            )
    assert not unexplained, f"{league}: {len(unexplained)} unexplained:\n  " + \
        "\n  ".join(unexplained[:8])


def test_the_accepted_differences_are_the_ones_recorded_and_no_others(rendered):
    """A shrinking list is fine; a growing one means a decision went unrecorded."""
    league, payload, output = rendered
    ours = {r["fixture_id"]: r for r in output["rows"]}
    differing = {
        field
        for row in payload["rows"]
        for field, value in row.items()
        if not _equal(value, ours[row["fixture_id"]].get(field))
    }
    assert differing <= ACCEPTED, f"{league}: undocumented: {sorted(differing - ACCEPTED)}"


def test_expected_value_is_published_in_the_unit_the_board_expects(rendered):
    """The two boards disagree, which is why the board downstream guesses the
    scale from a median. The canonical contract is a fraction everywhere."""
    league, _, output = rendered
    unit = load_league(league).publish.legacy_ev_unit
    values = [
        v for row in output["rows"] for k, v in row.items()
        if k.endswith("_ev") and isinstance(v, (int, float))
    ]
    assert values
    if unit == "percent":
        assert max(abs(v) for v in values) > 1.5
    else:
        assert max(abs(v) for v in values) < 1.5


# --------------------------------------------------------------------------- #
# the other two payloads the board fetches
# --------------------------------------------------------------------------- #

#: One league's exporter refitted the model inside itself rather than reading the
#: fitted simulations, so its published probabilities differ from the frozen ones
#: in the seventh decimal. That is D10's accepted consequence, and it moves a
#: sixth-decimal rounding on one row.
REFIT_DRIFT = frozenset({"away_win_prob", "away_win_fair_odds"})


@pytest.mark.parametrize("payload_name,builder", [
    ("upcoming_fixtures", "upcoming_fixtures"),
    ("match_predictions", "match_predictions"),
])
def test_the_other_published_payloads_match(rendered, payload_name, builder):
    """The board fetches three files, not one. A file it asks for and does not
    get is a failed fetch it has to tolerate on every poll."""
    league, _, _ = rendered
    payload, signals = _replay(league)
    published_at = pd.Timestamp(payload["meta"]["updated_at"]).tz_convert("UTC").to_pydatetime()
    with open(f"{GOLDEN}/{league}/published/{payload_name}.json") as handle:
        expected = json.load(handle)

    produced = getattr(legacy, builder)(
        load_league(league), list(signals.values()), generated_at=published_at
    )
    assert list(produced["rows"][0]) == list(expected["rows"][0]), "field order"

    ours = {r["fixture_id"]: r for r in produced["rows"]}
    unexplained = []
    for row in expected["rows"]:
        if row["fixture_id"] not in ours:
            continue  # the frozen list runs further ahead than the priced set
        for field, value in row.items():
            if field in REFIT_DRIFT or _equal(value, ours[row["fixture_id"]].get(field)):
                continue
            unexplained.append(f"{row['fixture_id']}.{field}: {value!r}")
    assert not unexplained, f"{league} {payload_name}: {unexplained[:5]}"


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_fair_odds_are_taken_from_the_unrounded_probability(league):
    """Inverting a rounded probability magnifies the rounding.

    Distinguishing the two methods needs a row where they actually disagree at
    the published precision, which is rare, so the check is against the exact
    value the engine holds rather than against a tolerance that would pass
    either way. The frozen payload agreed only after this was fixed.
    """
    _, signals = _replay(league)
    produced = legacy.match_predictions(load_league(league), list(signals.values()))
    exact = {s.fixture_id: s.raw_probabilities for s in signals.values()}
    checked = 0
    for row in produced["rows"]:
        probabilities = exact[row["fixture_id"]]
        for index, field in enumerate(
            ("home_win_fair_odds", "draw_fair_odds", "away_win_fair_odds")
        ):
            if not probabilities[index]:
                continue
            assert row[field] == round(1.0 / probabilities[index], 4)
            checked += 1
    assert checked, f"{league}: nothing was checked"
