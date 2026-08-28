"""Record when a (fixture, book) was first seen WITHOUT a price.

This file stores evidence, not odds, and it exists to answer one question the
odds history cannot: is a captured price actually an *opening* price?

The problem
-----------
reduce_capture_history's original test was

    trustworthy  <=>  (kickoff - lookahead) >= first capture we ever made

which asks "could the book have priced this before we started watching?". That is
sound but conservative, and it is conservative in a way that costs real data: it
rejects a fixture that entered the lookahead before capture began even when we can
see, right now, that no book has priced it yet. Measured 2026-08-12, that was the
entire 8/22 round -- nine fixtures, both books unpriced, every one of which will
produce a genuine opening line, all of which the old rule would have discarded.

The fix
-------
Record the direct observation instead of reasoning about what might have happened.
If we saw (fixture, book) unpriced at time T, then any price captured after T is
by definition the first price that book posted. That is a proof, not a proxy, and
it is strictly stronger than the kickoff arithmetic.

Both tests are kept: either one being satisfied is sufficient. The old rule still
covers fixtures that entered the lookahead after capture began, where we may never
happen to observe an unpriced tick because the book was already quoting.

Why it has to be written as it happens
--------------------------------------
An unpriced fixture leaves no trace anywhere else -- by construction there is no
odds row for it -- and the window to observe it shuts the moment the book posts.
Nothing can reconstruct this after the fact, which is why the capture loop records
it inline and why the file is tracked in git.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ligamx import paths

log = logging.getLogger(__name__)

COLUMNS = ["home_team", "away_team", "bookmaker", "first_seen_unpriced_at"]


def load_watch(path: str | None = None) -> pd.DataFrame:
    """The watch log, or an empty frame with the right schema."""
    path = path or paths.capture_watch_csv()
    if not os.path.isfile(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def record_unpriced(observations, *, observed_at: str,
                    path: str | None = None) -> int:
    """Log (home, away, book) triples seen without a price. Returns rows added.

    Only the FIRST observation per triple is kept -- that is the one that bounds
    how early we were watching, and every later one is redundant. Keeping only the
    first also stops a file that would otherwise grow by every unpriced fixture on
    every tick, forever.
    """
    path = path or paths.capture_watch_csv()
    existing = load_watch(path)
    seen = set(zip(existing["home_team"], existing["away_team"], existing["bookmaker"]))

    fresh = []
    for home, away, book in observations:
        key = (str(home), str(away), str(book))
        if key in seen:
            continue
        seen.add(key)
        fresh.append({"home_team": key[0], "away_team": key[1], "bookmaker": key[2],
                      "first_seen_unpriced_at": observed_at})

    if not fresh:
        return 0
    combined = pd.concat([existing, pd.DataFrame(fresh)], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info("Watch: %d new (fixture, book) pair(s) confirmed unpriced at %s",
             len(fresh), observed_at)
    return len(fresh)


def watched_before(path: str | None = None) -> dict:
    """(home, away, bookmaker) -> the earliest moment we saw it unpriced."""
    watch = load_watch(path)
    if watch.empty:
        return {}
    watch = watch.copy()
    # No leading underscore: itertuples renames such columns to positional _N,
    # which then has to be read by index and silently breaks if a column moves.
    watch["seen_at"] = pd.to_datetime(watch["first_seen_unpriced_at"], utc=True,
                                      errors="coerce")
    watch = watch.dropna(subset=["seen_at"])
    out: dict = {}
    for row in watch.itertuples(index=False):
        key = (row.home_team, row.away_team, row.bookmaker)
        prior = out.get(key)
        if prior is None or row.seen_at < prior:
            out[key] = row.seen_at
    return out


def opener_proof(*, home, away, bookmaker, captured_at, kickoff,
                 watched: dict, watching_since, horizon) -> str:
    """Why a captured price can be called this book's OPENER: "observed", "window", "".

    The two proofs of the module docstring, in one place because two callers now
    need the same verdict and a second copy would be free to drift from this one.
    Either proof suffices; "" means the book may have been quoting before we ever
    looked, so the price is a first *sighting* and not an opener.

    OBSERVED is the strong one and the only one that covers a fixture already
    inside the lookahead when capture began. WINDOW covers the reverse case, a
    fixture that became observable only after capture began, where we may never
    happen to catch an unpriced tick because the book was already quoting.
    """
    seen = watched.get((home, away, str(bookmaker)))
    if seen is not None and captured_at is not None and seen < captured_at:
        return "observed"
    if kickoff is not None and watching_since is not None and (kickoff - horizon) >= watching_since:
        return "window"
    return ""
