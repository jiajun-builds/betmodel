"""Append-only history store for captured Pinnacle 1X2 (h2h) snapshots.

Phase 1 of the scheduled odds-capture pipeline (AGENTS.md roadmap #2; switched
from spreads to 1X2 in roadmap #10). The single-shot fetch in
``fetch_pinnacle_spreads`` overwrites one "current" snapshot; this module instead
*appends* timestamped rows to a history CSV so we retain the full price-movement
trail (opening prices, later snapshots, eventually close).

Schema = the base columns from ``fetch_pinnacle_spreads.OUTPUT_COLUMNS`` plus
three capture-metadata columns:

    snapshot_type   "open" | "close" | "ad_hoc"  — why this capture fired
    target_round    the round this capture targets (may be "")
    capture_reason  free-text audit label (e.g. the open-window fixture it anchored)

Dedup key ``(event_id, bookmaker, last_update, snapshot_type)``: ``last_update`` is the
book's own "line last moved" timestamp, so a repeated poll of an unmoved line is skipped
rather than appended. (``fetched_at`` — when *we* polled — is deliberately NOT part
of the key; it changes every poll and would defeat dedup.) ``bookmaker`` joined the key
in roadmap #8: the capture path now stores every book's price at the open window, and two
books can share a ``last_update`` second.

Usage:
    from csl.odds.snapshot_store import append_snapshots, HISTORY_CSV
    append_snapshots(rows, snapshot_type="open", target_round="18",
                     capture_reason="open-window: Shanghai Port vs Chengdu")
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from csl.odds.fetch_pinnacle_spreads import OUTPUT_COLUMNS
from csl.paths import data_raw_dir

log = logging.getLogger(__name__)

# Capture-metadata columns appended to each history row, in order.
SNAPSHOT_META_COLUMNS = ["snapshot_type", "target_round", "capture_reason"]

# Full history schema: base fetch columns first, then capture metadata.
HISTORY_COLUMNS = list(OUTPUT_COLUMNS) + SNAPSHOT_META_COLUMNS

# Columns whose combination uniquely identifies a captured line state.
DEDUP_KEY = ["event_id", "bookmaker", "last_update", "snapshot_type"]

VALID_SNAPSHOT_TYPES = frozenset({"open", "close", "ad_hoc"})

HISTORY_CSV = os.path.join(data_raw_dir(), "CHN_pinnacle_spreads_history.csv")


def load_history(path: str = HISTORY_CSV) -> pd.DataFrame:
    """Return the existing history frame, or an empty one with the right schema.

    Reading with ``dtype=str`` keeps the dedup key comparison exact (no float
    reformatting of ``last_update`` / ``event_id``) and avoids pandas guessing
    types differently across runs.
    """
    if not os.path.isfile(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Tolerate an older/newer file that is missing columns we expect.
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[HISTORY_COLUMNS]


def _prepare_new_frame(
    rows: pd.DataFrame,
    *,
    snapshot_type: str,
    target_round: str,
    capture_reason: str,
) -> pd.DataFrame:
    """Attach capture metadata to freshly-extracted rows and align columns."""
    frame = rows.copy()
    frame["snapshot_type"] = snapshot_type
    frame["target_round"] = target_round
    frame["capture_reason"] = capture_reason
    # Add any missing base columns as empty so column alignment never crashes.
    for col in HISTORY_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[HISTORY_COLUMNS].astype(str)


def append_snapshots(
    rows: pd.DataFrame,
    *,
    snapshot_type: str,
    target_round: str = "",
    capture_reason: str = "",
    path: str = HISTORY_CSV,
) -> tuple[pd.DataFrame, int]:
    """Append captured snapshot ``rows`` to the history CSV, skipping duplicates.

    ``rows`` is a frame shaped like ``fetch_pinnacle_spreads`` output (one row per
    event). Rows whose ``(event_id, last_update, snapshot_type)`` already exist in
    the history — or repeat within this batch — are dropped so an unmoved line is
    never stored twice.

    Returns ``(combined_history, appended_count)``. Writes the file only when at
    least one new row is appended.
    """
    if snapshot_type not in VALID_SNAPSHOT_TYPES:
        raise ValueError(
            f"snapshot_type must be one of {sorted(VALID_SNAPSHOT_TYPES)}, got {snapshot_type!r}"
        )

    new_frame = _prepare_new_frame(
        rows,
        snapshot_type=snapshot_type,
        target_round=str(target_round),
        capture_reason=capture_reason,
    )
    # Drop duplicates within the incoming batch first.
    new_frame = new_frame.drop_duplicates(subset=DEDUP_KEY, keep="last")

    existing = load_history(path)
    if not existing.empty:
        merged_keys = existing[DEDUP_KEY].apply(tuple, axis=1)
        seen = set(merged_keys)
        mask = new_frame[DEDUP_KEY].apply(lambda r: tuple(r) not in seen, axis=1)
        to_append = new_frame[mask]
    else:
        to_append = new_frame

    appended = len(to_append)
    if appended == 0:
        log.info("No new snapshot rows to append (all %d already in history)", len(new_frame))
        return existing, 0

    combined = pd.concat([existing, to_append], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info(
        "Appended %d/%d snapshot row(s) [type=%s] -> %s (%d total)",
        appended, len(new_frame), snapshot_type, path, len(combined),
    )
    return combined, appended
