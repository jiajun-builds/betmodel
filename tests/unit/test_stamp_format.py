"""Stored timestamps are written in one format, to the second.

Every stamp went through a bare ``isoformat()``, which keeps whatever precision
its source happened to carry: a provider's value arrives to the second, this
pipeline's own clock arrives to the microsecond, and both were written to the
same column. The two pre-merge pipelines each published seconds, so the mixed
format is a regression the merge introduced, in a column a downstream board
reads. A strict parse of it raises, which is how it was found.
"""

from __future__ import annotations

import glob
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from betmodel.dates import stamp

SECONDS_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_stamp_drops_a_precision_nothing_here_measures():
    moment = datetime(2026, 8, 29, 10, 46, 35, 449586, tzinfo=timezone.utc)
    assert stamp(moment) == "2026-08-29T10:46:35Z"


def test_stamp_converts_rather_than_relabels():
    # A non-UTC input must be moved to UTC, not stamped with a Z it does not mean.
    moment = datetime(2026, 8, 29, 11, 46, 35,
                      tzinfo=timezone(timedelta(hours=1)))
    assert stamp(moment) == "2026-08-29T10:46:35Z"


@pytest.mark.parametrize("path", sorted(glob.glob("data/*/odds_capture_history.csv")))
def test_every_stored_fetched_at_uses_the_one_format(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    odd = [v for v in frame["fetched_at"] if v and not SECONDS_ONLY.match(v)]
    assert not odd, f"{path}: {odd[:3]}"


@pytest.mark.parametrize("path", sorted(glob.glob("data/*/odds_capture_history.csv")))
def test_the_column_parses_under_a_single_strict_format(path):
    # The point of one format is that a reader needs no fallbacks to read it.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    parsed = pd.to_datetime(frame["fetched_at"], utc=True,
                            format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
    assert parsed.notna().all()


def test_the_provider_stamp_that_identifies_a_row_is_left_alone(tmp_path):
    """`last_update` is the book's own clock and part of the dedup key.

    Normalising it would re-identify the row: two genuinely distinct ticks a
    fraction of a second apart would collapse into one, and every affected row
    would look new to the upstream sync and be appended a second time. It is
    stored exactly as the provider sent it, whatever precision that is.
    """
    from betmodel.odds import capture_store

    assert "last_update" in capture_store.DEDUP_KEY
    assert "fetched_at" not in capture_store.DEDUP_KEY
