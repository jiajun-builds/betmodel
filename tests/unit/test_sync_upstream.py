"""The safety-net sync's dry run must be able to report a non-zero answer.

Before archiving the two upstream repositories the plan calls for one last
`sync_upstream_captures.py --check`, to see whether the pipelines that are still
running captured anything this one missed. That check was incapable of saying
yes: it measured the delta by reading the same file twice, once before an append
it had skipped and once after, so it printed +0 for any upstream whatsoever. A
green check meant nothing, which is worse than no check at all.
"""

from __future__ import annotations

import importlib.util
import os

import pandas as pd

from betmodel.odds import capture_store

_SPEC = importlib.util.spec_from_file_location(
    "sync_upstream_captures",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                 "sync_upstream_captures.py"),
)
sync = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sync)


def _row(event_id: str, book: str, last_update: str, snapshot_type: str = "close"):
    row = {column: "" for column in capture_store.HISTORY_COLUMNS}
    row.update(event_id=event_id, bookmaker=book, last_update=last_update,
               snapshot_type=snapshot_type)
    return row


def _history(tmp_path, rows):
    path = os.path.join(tmp_path, "history.csv")
    pd.DataFrame(rows, columns=capture_store.HISTORY_COLUMNS).to_csv(path, index=False)
    return path


def test_counts_the_rows_a_real_run_would_append(tmp_path):
    local = _history(str(tmp_path), [_row("a", "pinnacle", "T1")])
    upstream = pd.DataFrame(
        [_row("a", "pinnacle", "T1"), _row("a", "pinnacle", "T2"),
         _row("b", "pinnacle", "T1")],
        columns=capture_store.HISTORY_COLUMNS,
    )
    assert sync._missing_here(upstream, local) == 2


def test_a_snapshot_type_the_store_rejects_is_not_counted(tmp_path):
    local = _history(str(tmp_path), [_row("a", "pinnacle", "T1")])
    upstream = pd.DataFrame(
        [_row("b", "pinnacle", "T1", snapshot_type="now")],
        columns=capture_store.HISTORY_COLUMNS,
    )
    # Counting it would promise an append that `_append_preserving_type` skips.
    assert sync._missing_here(upstream, local) == 0


def test_an_upstream_with_nothing_new_still_reports_zero(tmp_path):
    local = _history(str(tmp_path), [_row("a", "pinnacle", "T1")])
    upstream = pd.DataFrame([_row("a", "pinnacle", "T1")],
                            columns=capture_store.HISTORY_COLUMNS)
    assert sync._missing_here(upstream, local) == 0
