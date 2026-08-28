"""Append-only history of deliberate odds captures. Tracked in git.

Two stores exist and they are not interchangeable:

``research/odds_snapshots.csv``
    Dense poll time-series, gitignored, rebuildable. Research fodder.

this one
    Sparse and provenance-stamped: one row per price we deliberately went and
    got.

It is tracked because the capture loop runs in CI, which has no durable disk, so
a tick that cannot commit its rows has done nothing. And unlike the poll store it
cannot be rebuilt: an opening line exists only while the book is showing it, and
neither odds provider sells opener history at any tier.

The 17-column schema is shared by both leagues, unchanged from before the merge,
which is why the two histories migrated with no conversion at all.

Dedup key ``(event_id, bookmaker, last_update, snapshot_type)``. ``last_update``
is the book's own "the line last moved" stamp, so re-polling an unmoved line is
skipped rather than appended. ``fetched_at`` is deliberately excluded: it changes
every poll and would defeat dedup entirely.

That key is necessary but **not sufficient** for openers, and the difference
matters. At least one book reports a single bulk feed-refresh timestamp shared by
every fixture, measured identical to the millisecond across nine priced matches,
so its ``last_update`` moves whether or not its price did. What actually
guarantees one open row per (fixture, book) is the pending-set gate in the open
capture, not this.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from betmodel import paths

log = logging.getLogger(__name__)

#: What the provider told us.
BASE_COLUMNS = [
    "event_id", "commence_time", "api_home_team", "api_away_team",
    "home_team", "away_team", "home_odds", "draw_odds", "away_odds",
    "bookmaker", "market", "regions", "last_update", "fetched_at",
]

#: Why the capture fired. Appended to every row, in this order.
CAPTURE_META_COLUMNS = ["snapshot_type", "target_round", "capture_reason"]

HISTORY_COLUMNS = BASE_COLUMNS + CAPTURE_META_COLUMNS

DEDUP_KEY = ["event_id", "bookmaker", "last_update", "snapshot_type"]

VALID_SNAPSHOT_TYPES = frozenset({"open", "close", "ad_hoc"})


def history_path(league: str) -> str:
    return paths.for_league(league).capture_history_csv


def load_history(path: str) -> pd.DataFrame:
    """The existing history, or an empty frame with the right schema.

    ``dtype=str`` with ``keep_default_na=False`` keeps the dedup comparison
    byte-exact. Without them pandas reformats event ids and timestamps, and an
    unmoved line reappears as new on the very next tick.
    """
    if not os.path.isfile(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in HISTORY_COLUMNS:  # tolerate a file written by an older schema
        if column not in frame.columns:
            frame[column] = ""
    return frame[HISTORY_COLUMNS]


def _prepare(
    rows: pd.DataFrame, *, snapshot_type: str, target_round: str, capture_reason: str
) -> pd.DataFrame:
    frame = rows.copy()
    frame["snapshot_type"] = snapshot_type
    frame["target_round"] = target_round
    frame["capture_reason"] = capture_reason
    for column in HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[HISTORY_COLUMNS].astype(str)


def append_snapshots(
    rows: pd.DataFrame,
    *,
    path: str,
    snapshot_type: str,
    target_round: str = "",
    capture_reason: str = "",
) -> tuple[pd.DataFrame, int]:
    """Append captured rows, dropping any that duplicate the key.

    Returns ``(combined_history, appended_count)``. The file is rewritten only
    when at least one row is genuinely new, so an idle tick leaves the working
    tree clean and the workflow's commit step honestly reports that nothing was
    appended. That honesty is load-bearing: the republish job is gated on it.
    """
    if snapshot_type not in VALID_SNAPSHOT_TYPES:
        raise ValueError(
            f"snapshot_type must be one of {sorted(VALID_SNAPSHOT_TYPES)}, "
            f"got {snapshot_type!r}"
        )

    fresh = _prepare(
        rows,
        snapshot_type=snapshot_type,
        target_round=str(target_round),
        capture_reason=capture_reason,
    ).drop_duplicates(subset=DEDUP_KEY, keep="last")

    existing = load_history(path)
    if not existing.empty:
        seen = set(existing[DEDUP_KEY].apply(tuple, axis=1))
        to_append = fresh[fresh[DEDUP_KEY].apply(lambda r: tuple(r) not in seen, axis=1)]
    else:
        to_append = fresh

    appended = len(to_append)
    if appended == 0:
        log.info("no new capture rows (all %d already in history)", len(fresh))
        return existing, 0

    combined = pd.concat([existing, to_append], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info(
        "appended %d/%d row(s) [%s] -> %s (%d total)",
        appended, len(fresh), snapshot_type, os.path.basename(path), len(combined),
    )
    return combined, appended
