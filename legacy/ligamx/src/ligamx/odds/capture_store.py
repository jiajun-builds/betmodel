"""Append-only history of scheduled odds captures (opening and closing lines).

Two stores exist and they are not interchangeable:

    odds_snapshots.csv      dense poll time-series, gitignored, research fodder.
                            32k rows, almost all Polymarket. Rebuilt by backfill.
    this one                sparse and provenance-stamped: one row per price we
                            deliberately went and got, tracked in git.

It is tracked because the capture loop runs in GitHub Actions, which has no
durable disk -- a tick that cannot commit its rows has done nothing. And unlike
the poll store it cannot be rebuilt: an opening line exists only while the book
is showing it, and odds-api.io sells no opener history at any tier.

Schema is the 17-column shape already used by MEX_betano_openers_history.csv, so
the hand-collected Betano openers and the captured ones speak the same language:

    event_id .. last_update   what the provider said
    fetched_at                when *we* polled
    snapshot_type             "open" | "close" | "ad_hoc" -- why the capture fired
    target_round              the round targeted (may be "")
    capture_reason            free-text audit label, incl. the lead time at capture

Dedup key (event_id, bookmaker, last_update, snapshot_type): last_update is the
book's own "line last moved" stamp, so re-polling an unmoved line is skipped
instead of appended. fetched_at is deliberately NOT in the key -- it changes every
poll and would defeat dedup entirely.

That key is necessary but NOT sufficient for openers, and the difference matters.
Duel reports a single bulk feed-refresh updatedAt shared by every fixture (measured
2026-08-12: identical to the millisecond across all 9 priced matches), so its
last_update moves whether or not its price did. What actually guarantees one open
row per (fixture, book) is the `missing` gate in fetch_oddsapiio_opens, not this.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ligamx import paths

log = logging.getLogger(__name__)

# What the provider told us. Mirrors MEX_betano_openers_history.csv's leading
# columns so the hand-collected file can be concatenated without a conversion.
BASE_COLUMNS = [
    "event_id", "commence_time", "api_home_team", "api_away_team",
    "home_team", "away_team", "home_odds", "draw_odds", "away_odds",
    "bookmaker", "market", "regions", "last_update", "fetched_at",
]

# Why the capture fired. Appended to every row, in this order.
CAPTURE_META_COLUMNS = ["snapshot_type", "target_round", "capture_reason"]

HISTORY_COLUMNS = BASE_COLUMNS + CAPTURE_META_COLUMNS

DEDUP_KEY = ["event_id", "bookmaker", "last_update", "snapshot_type"]

VALID_SNAPSHOT_TYPES = frozenset({"open", "close", "ad_hoc"})


def load_history(path: str | None = None) -> pd.DataFrame:
    """The existing history, or an empty frame with the right schema.

    dtype=str + keep_default_na=False keeps the dedup key comparison exact: without
    them pandas reformats event ids and timestamps, and an unmoved line reappears
    as new on the next tick.
    """
    path = path or paths.odds_capture_history_csv()
    if not os.path.isfile(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in HISTORY_COLUMNS:  # tolerate a file written by an older schema
        if col not in df.columns:
            df[col] = ""
    return df[HISTORY_COLUMNS]


def _prepare(rows: pd.DataFrame, *, snapshot_type: str, target_round: str,
             capture_reason: str) -> pd.DataFrame:
    """Stamp capture metadata onto fresh rows and align them to the schema."""
    frame = rows.copy()
    frame["snapshot_type"] = snapshot_type
    frame["target_round"] = target_round
    frame["capture_reason"] = capture_reason
    for col in HISTORY_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[HISTORY_COLUMNS].astype(str)


def append_snapshots(rows: pd.DataFrame, *, snapshot_type: str,
                     target_round: str = "", capture_reason: str = "",
                     path: str | None = None) -> tuple[pd.DataFrame, int]:
    """Append captured rows, dropping any that duplicate the key.

    Returns (combined_history, appended_count). The file is rewritten only when at
    least one row is genuinely new, so an idle tick leaves the working tree clean
    and the workflow's commit step correctly reports that nothing was appended.
    """
    if snapshot_type not in VALID_SNAPSHOT_TYPES:
        raise ValueError(
            f"snapshot_type must be one of {sorted(VALID_SNAPSHOT_TYPES)}, "
            f"got {snapshot_type!r}")

    path = path or paths.odds_capture_history_csv()
    new_frame = _prepare(rows, snapshot_type=snapshot_type,
                         target_round=str(target_round),
                         capture_reason=capture_reason)
    new_frame = new_frame.drop_duplicates(subset=DEDUP_KEY, keep="last")

    existing = load_history(path)
    if not existing.empty:
        seen = set(existing[DEDUP_KEY].apply(tuple, axis=1))
        to_append = new_frame[
            new_frame[DEDUP_KEY].apply(lambda r: tuple(r) not in seen, axis=1)]
    else:
        to_append = new_frame

    appended = len(to_append)
    if appended == 0:
        log.info("No new capture rows (all %d already in history)", len(new_frame))
        return existing, 0

    combined = pd.concat([existing, to_append], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info("Appended %d/%d row(s) [type=%s] -> %s (%d total)",
             appended, len(new_frame), snapshot_type, path, len(combined))
    return combined, appended
