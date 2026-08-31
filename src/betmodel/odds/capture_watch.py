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

COLUMNS = [
    "home_team", "away_team", "bookmaker",
    "first_seen_unpriced_at",
    # Updated on every sighting, unlike the first. The two answer different
    # questions and only the second one proves an opener: the first bounds how
    # early we started watching, the last bounds how recently we confirmed the
    # book still had no price. A price captured long after the last confirmed
    # sighting may have been posted at any point in that gap.
    "last_seen_unpriced_at",
]

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


def _stamps(path: str, column: str, how: str) -> dict[tuple[str, str, str], pd.Timestamp]:
    frame = load_watch(path)
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["seen"] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame = frame.dropna(subset=["seen"])
    grouped = getattr(frame.groupby(["home_team", "away_team", "bookmaker"])["seen"], how)()
    return {key: value for key, value in grouped.items()}


def watched_before(path: str) -> dict[tuple[str, str, str], pd.Timestamp]:
    """``(home, away, book) -> earliest moment seen unpriced``."""
    return _stamps(path, "first_seen_unpriced_at", "min")


def watched_until(path: str) -> dict[tuple[str, str, str], pd.Timestamp]:
    """``(home, away, book) -> latest moment confirmed still unpriced``.

    This is the one that proves an opener. Rows written before the column
    existed have no value here, which is the safe direction: an absent last
    sighting withholds the proof rather than inventing one.
    """
    return _stamps(path, "last_seen_unpriced_at", "max")


def record_unpriced(
    observations, *, path: str, observed_at: str
) -> int:
    """Log each (fixture, book) seen without a price. Returns rows added.

    One row per triple, never more. The first sighting is written once and never
    moves; the last is overwritten on every tick. Keeping every observation as its
    own row would grow the file by every unpriced fixture on every tick forever,
    and the only thing the intermediate ones add is already carried by the last.

    The last sighting is not bookkeeping. It is what makes the opener proof
    survive a gap in capture: an unpriced sighting from two days ago proves
    nothing about a price captured today, because the book may have posted at any
    point in between and nobody was looking.
    """
    rows = [
        {
            "home_team": str(home),
            "away_team": str(away),
            "bookmaker": str(book),
            "first_seen_unpriced_at": observed_at,
            "last_seen_unpriced_at": observed_at,
        }
        for home, away, book in observations
    ]
    if not rows:
        return 0

    existing = load_watch(path)
    known = set(
        existing[["home_team", "away_team", "bookmaker"]].apply(tuple, axis=1)
    ) if not existing.empty else set()

    # Two ticks in one run can see the same pair; de-duplicate within the batch.
    batch: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        batch[(row["home_team"], row["away_team"], row["bookmaker"])] = row

    # Refresh the last sighting on pairs already on file, and append the rest.
    touched = 0
    if not existing.empty:
        keys = existing[["home_team", "away_team", "bookmaker"]].apply(tuple, axis=1)
        hit = keys.isin(batch.keys())
        touched = int(hit.sum())
        existing.loc[hit, "last_seen_unpriced_at"] = observed_at

    deduped = [row for key, row in batch.items() if key not in known]
    if not deduped and not touched:
        return 0

    combined = pd.concat([existing, pd.DataFrame(deduped)], ignore_index=True)[COLUMNS]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info("recorded %d new and refreshed %d unpriced observation(s) -> %s",
             len(deduped), touched, os.path.basename(path))
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
    *, home, away, bookmaker, captured_at, kickoff, watched, since, horizon,
    watched_last=None, max_gap=None,
) -> str:
    """Why a captured price may be called this book's opener.

    ``observed`` is the strong proof and the only one that covers a fixture
    already inside the lookahead when capture began. ``window`` covers the
    reverse case. An empty string means the book may have been quoting before we
    ever looked, so the price is a first *sighting* and not an opener.

    **Both proofs assume we were watching continuously, and one of them now
    checks.** An unpriced sighting only rules out an earlier price if nobody
    stopped looking in between; a gap lets the book post at any point inside it,
    unobserved. That is not hypothetical -- an exhausted API account refused every
    Pinnacle call for two days while `observed` would still have certified the
    price fetched afterwards, on the strength of a sighting 2d06h stale.

    So ``observed`` now requires the *last* confirmed unpriced sighting to be
    within ``max_gap`` of the capture. Callers that pass neither ``watched_last``
    nor ``max_gap`` get the old behaviour, which is why the anchor path passes
    both.

    One implementation because two callers need the same verdict, and a second
    copy would be free to drift.
    """
    key = (home, away, str(bookmaker))
    seen = watched.get(key)
    if seen is not None and captured_at is not None and seen < captured_at:
        if max_gap is None:
            return OBSERVED
        last = (watched_last or {}).get(key)
        if last is not None and (captured_at - last) <= max_gap:
            return OBSERVED
        # Watched once, then not recently enough for the price to be pinned to a
        # window we had eyes on. Falls through rather than returning: the window
        # proof may still cover it on its own terms.
    if kickoff is not None and since is not None and (kickoff - horizon) >= since:
        return WINDOW
    return NONE
