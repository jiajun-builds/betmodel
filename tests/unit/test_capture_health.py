"""The capture dead-man's switch alerts on being down, not on being idle.

Captures fire every five minutes, so a per-failure alert is noise. A row-based
check would be worse: an idle tick with no fixture in its window correctly
appends nothing, so long quiet stretches are healthy and would look identical to
an outage. What is never healthy is the workflow not running at all, or running
and never once succeeding.
"""

from __future__ import annotations

import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "check_capture_health",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                 "check_capture_health.py"),
)
health = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(health)


def _runs(*conclusions):
    return [{"conclusion": c, "createdAt": "2026-08-30T10:00:00Z"} for c in conclusions]


def test_no_run_at_all_is_the_alarm_condition():
    ok, message = health.assess([], window_hours=24)
    assert not ok
    assert "not firing" in message


def test_every_run_failing_is_the_other_alarm_condition():
    ok, message = health.assess(_runs("failure", "failure", "failure"), window_hours=24)
    assert not ok
    assert "not one succeeded" in message


def test_one_success_among_failures_is_healthy():
    # Deliberately forgiving. A capture that succeeds at all is taking lines, and
    # a threshold tight enough to catch a partial degradation would fire on the
    # ordinary transient failures this pipeline sees every day.
    ok, _ = health.assess(_runs("failure", "success", "failure"), window_hours=24)
    assert ok


def test_cancelled_runs_count_as_neither():
    """Concurrency cancels an overlapping tick by design.

    Counting those as failures would make a busy matchday -- exactly when
    captures matter most -- look like an outage.
    """
    ok, message = health.assess(
        _runs("cancelled", "cancelled", "success"), window_hours=24)
    assert ok
    assert "0 failed" in message


def test_only_cancellations_and_no_success_still_alarms():
    ok, _ = health.assess(_runs("cancelled", "cancelled"), window_hours=24)
    assert not ok


def test_a_missing_conclusion_is_not_counted_as_a_failure():
    # An in-progress run has conclusion null; it has not failed yet.
    ok, _ = health.assess(_runs("success", None), window_hours=24)
    assert ok
