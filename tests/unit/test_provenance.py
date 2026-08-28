"""Pin the historical capture_reason spellings.

These strings exist in committed, irreproducible rows. A classifier that stops
recognising one silently promotes a placeholder price to an opening line, which
is exactly the bias the open-to-close work is trying to measure.
"""

from __future__ import annotations

import pandas as pd
import pytest

from betmodel import paths
from betmodel.odds import provenance as pv


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("now-refresh fallback (open window missed)", pv.BACKFILLED),
        ("scheduler open-window tick @ 2026-08-22T12:53:35+00:00", pv.WINDOWED),
        ("odds-api.io first-seen open price @ 2026-08-12T10:00:00+00:00", pv.POLLED),
        ("odds-api.io first-seen 1xBet price @ 2026-07-04T09:00:00+00:00", pv.POLLED),
        ("manual-backfill oddspedia 1xbet open", pv.MANUAL),
        ("manual-backfill sportmarket betfair open", pv.MANUAL),
        ("manually add", pv.MANUAL),
        ("phase2-verify open", pv.VERIFY),
        ("something nobody has written yet", pv.UNKNOWN),
    ],
)
def test_every_historical_reason_classifies(reason, expected):
    assert pv.classify("open", reason) == expected


def test_a_close_is_a_close_whatever_the_reason_says():
    assert pv.classify("close", "pre-kickoff close tick @ x") == pv.CLOSE
    assert pv.classify("close", "open-window tick") == pv.CLOSE


def test_backfilled_rows_are_not_opening_prices():
    """The whole point: a Now line written into the open slot is not an open."""
    assert not pv.is_opening_price(pv.BACKFILLED)
    assert pv.is_opening_price(pv.WINDOWED)
    assert pv.is_opening_price(pv.POLLED)


def test_only_polling_can_prove_an_open():
    assert pv.is_provable_open(pv.POLLED)
    for cls in (pv.WINDOWED, pv.MANUAL, pv.VERIFY, pv.BACKFILLED, pv.UNKNOWN):
        assert not pv.is_provable_open(cls)


def test_no_row_in_either_committed_history_is_unclassified():
    """Guards the migrated data itself, not just the function."""
    for league in ("csl", "ligamx"):
        path = paths.for_league(league).capture_history_csv
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        tagged = pv.classify_frame(df)
        unknown = tagged[tagged.provenance == pv.UNKNOWN]
        assert unknown.empty, (
            f"{league}: {len(unknown)} rows with unrecognised capture_reason: "
            f"{sorted(unknown.capture_reason.unique())[:3]}"
        )
