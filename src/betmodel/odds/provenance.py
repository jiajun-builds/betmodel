"""How a stored price came to be, and what it is therefore allowed to prove.

The capture history is append-only and predates the merge, so it carries rows
collected by mechanisms that no longer exist. They are kept: the data is
irreproducible, and they were the prices the model actually used at the time.
But they are not interchangeable, and one class in particular is not an opening
price at all.

Classification reads ``capture_reason``, which is free text written by whichever
capture path produced the row. That is fragile if every consumer parses it, so
every consumer calls here instead and the exact historical spellings are pinned
by tests.

The classes, weakest evidence last:

``POLLED``
    A poller that was already watching the fixture saw this price appear. With a
    matching ``capture_watch`` observation this is a proof, not an inference:
    the book was seen unpriced beforehand, so this is the first price it posted.

``WINDOWED``
    Captured inside a predicted opening window. Genuinely early, but the window
    was a guess, so an earlier price may have existed and been missed. This
    mechanism has been retired in favour of polling.

``VERIFY``
    Captured by a one-off verification pass rather than the live loop.

``MANUAL``
    Hand-entered from a third-party odds archive. Plausible, unverifiable, and
    the only class whose accuracy depends on a source outside this system.

``BACKFILLED``
    **Not an opening price.** The predicted window elapsed unfilled and the
    current line was written into the open slot as a placeholder. Using these as
    opens biases any open-to-close measurement by however long the fallback ran.

``CLOSE``
    A closing-line capture, not an open at all.
"""

from __future__ import annotations

POLLED = "polled"
WINDOWED = "windowed"
VERIFY = "verify"
MANUAL = "manual"
BACKFILLED = "backfilled"
CLOSE = "close"
UNKNOWN = "unknown"

#: Substring -> class. Matched case-insensitively against ``capture_reason``.
#: Ordered most specific first; the first hit wins. These spellings are
#: historical facts about rows already committed, so they are additive only:
#: never edit one to tidy it up, or the row it describes silently reclassifies.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("open window missed", BACKFILLED),
    ("now-refresh fallback", BACKFILLED),
    ("manual-backfill", MANUAL),
    ("manually add", MANUAL),
    ("phase2-verify", VERIFY),
    ("open-window tick", WINDOWED),
    ("first-seen", POLLED),
)


def classify(snapshot_type: str | None, capture_reason: str | None) -> str:
    """Provenance class of one stored row.

    ``snapshot_type`` wins for closes: a close is a close regardless of which
    path recorded it.
    """
    stype = (snapshot_type or "").strip().lower()
    if stype == "close":
        return CLOSE
    reason = (capture_reason or "").strip().lower()
    for marker, cls in _MARKERS:
        if marker in reason:
            return cls
    return UNKNOWN


def is_opening_price(cls: str) -> bool:
    """Whether the row may be treated as an opening line at all.

    Excludes ``BACKFILLED``, which is a mid-market line wearing an open's label,
    and closes. Every open-to-close measurement must filter on this.
    """
    return cls in (POLLED, WINDOWED, VERIFY, MANUAL)


def is_provable_open(cls: str) -> bool:
    """Whether we can show the price was the first the book posted.

    Only polling can establish this, and only together with a ``capture_watch``
    observation from before the price appeared. A signal that requires price
    proof needs this, not merely :func:`is_opening_price`.
    """
    return cls == POLLED


def classify_frame(frame, *, column: str = "provenance"):
    """Add a provenance column to a capture-history DataFrame.

    Kept here rather than in the reducer so that the eval harnesses, which read
    the history directly, cannot disagree with production about what an open is.
    """
    frame = frame.copy()
    frame[column] = [
        classify(s, r)
        for s, r in zip(
            frame.get("snapshot_type", ""), frame.get("capture_reason", "")
        )
    ]
    return frame
