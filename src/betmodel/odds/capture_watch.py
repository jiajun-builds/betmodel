"""When a (fixture, book) was first seen WITHOUT a price.

This file stores evidence, not odds, and it answers the one question the odds
history cannot: is a captured price actually an *opening* price?

The original test was ``(kickoff - lookahead) >= first capture we ever made``,
which asks whether the book could have priced the fixture before we started
watching. Sound, but conservative in a way that costs real data: it rejects a
fixture that entered the lookahead before capture began even when we can see,
right now, that no book has priced it yet. Measured on one round, that was all
nine fixtures, both books unpriced, every one destined to produce a genuine
opening line.

The fix is to record the observation instead of reasoning about what might have
happened. If we saw (fixture, book) unpriced at time T, any price captured after
T is by definition the first that book posted. That is a proof, not a proxy.

Both tests are kept and either suffices. The old one still covers fixtures that
entered the lookahead after capture began, where we may never happen to catch an
unpriced tick because the book was already quoting.

It has to be written as it happens. An unpriced fixture leaves no trace anywhere
else, by construction there is no odds row for it, and the window to observe it
shuts the moment the book posts. Nothing reconstructs this after the fact, which
is why it is recorded inline and tracked in git.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from betmodel import paths

log = logging.getLogger(__name__)

COLUMNS = ["home_team", "away_team", "bookmaker", "first_seen_unpriced_at"]

#: Proof strengths, strongest first.
OBSERVED = "observed"
WINDOW = "window"
NONE = ""


def watch_path(league: str) -> str:
    return paths.for_league(league).capture_watch_csv


def load_watch(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[COLUMNS]


def watched_before(path: str) -> dict[tuple[str, str, str], pd.Timestamp]:
    """``(home, away, book) -> earliest moment seen unpriced``."""
    frame = load_watch(path)
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["seen"] = pd.to_datetime(frame["first_seen_unpriced_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["seen"])
    grouped = frame.groupby(["home_team", "away_team", "bookmaker"])["seen"].min()
    return {key: value for key, value in grouped.items()}


def record_unpriced(
    observations, *, path: str, observed_at: str
) -> int:
    """Log each (fixture, book) seen without a price. Returns rows added.

    Only the **first** observation per triple is kept. That one bounds how early
    we were watching, which is the whole point; keeping every observation would
    grow the file by every unpriced fixture on every tick, forever, and prove
    nothing extra.
    """
    rows = [
        {
            "home_team": str(home),
            "away_team": str(away),
            "bookmaker": str(book),
            "first_seen_unpriced_at": observed_at,
        }
        for home, away, book in observations
    ]
    if not rows:
        return 0

    existing = load_watch(path)
    known = set(
        existing[["home_team", "away_team", "bookmaker"]].apply(tuple, axis=1)
    ) if not existing.empty else set()

    fresh = [
        r for r in rows
        if (r["home_team"], r["away_team"], r["bookmaker"]) not in known
    ]
    # Two ticks in one run can see the same pair; de-duplicate within the batch.
    seen_in_batch: set[tuple[str, str, str]] = set()
    deduped = []
    for row in fresh:
        key = (row["home_team"], row["away_team"], row["bookmaker"])
        if key in seen_in_batch:
            continue
        seen_in_batch.add(key)
        deduped.append(row)

    if not deduped:
        return 0

    combined = pd.concat([existing, pd.DataFrame(deduped)], ignore_index=True)[COLUMNS]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info("recorded %d unpriced observation(s) -> %s", len(deduped), os.path.basename(path))
    return len(deduped)


def watching_since(history: pd.DataFrame) -> pd.Timestamp | None:
    """The earliest moment we ever captured anything for this league.

    Taken over the WHOLE history, closes included, not just the opens. A close
    tick is also a moment we looked, and using a subset would date the start of
    watching later than it was and wrongly withhold the window proof.
    """
    if history.empty:
        return None
    stamps = pd.to_datetime(history["fetched_at"], utc=True, errors="coerce").dropna()
    return stamps.min() if len(stamps) else None


def opener_proof(
    *, home, away, bookmaker, captured_at, kickoff, watched, since, horizon
) -> str:
    """Why a captured price may be called this book's opener.

    ``observed`` is the strong proof and the only one that covers a fixture
    already inside the lookahead when capture began. ``window`` covers the
    reverse case. An empty string means the book may have been quoting before we
    ever looked, so the price is a first *sighting* and not an opener.

    One implementation because two callers need the same verdict, and a second
    copy would be free to drift.
    """
    seen = watched.get((home, away, str(bookmaker)))
    if seen is not None and captured_at is not None and seen < captured_at:
        return OBSERVED
    if kickoff is not None and since is not None and (kickoff - horizon) >= since:
        return WINDOW
    return NONE
