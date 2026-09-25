"""The reducer's join from a captured fixture to its row in the match table.

``Date`` carries whichever timezone wrote the row (D12). From Apertura 2026 it is
Liga MX's local day, so a record labelled by its UTC day missed every kickoff
after 18:00 local, and 17 of 20 September closes never reached the table.
"""

from __future__ import annotations

import pandas as pd

from betmodel.odds import reduce as rd

TZ = "America/Mexico_City"


def _table():
    return pd.DataFrame([
        # 21:00 local on the 4th is 03:00 UTC on the 5th.
        {"Date": "2026-09-04", "Home": "FC Juarez", "Away": "Pachuca",
         "kickoff_utc": "2026-09-05T03:00:00Z"},
        # An older row with no kickoff, written in UTC days.
        {"Date": "2024/03/10", "Home": "FC Juarez", "Away": "Pachuca",
         "kickoff_utc": None},
    ])


def test_an_evening_kickoff_is_found_by_its_local_matchday():
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2026/09/05",
                        "2026-09-04", TZ) == 0


def test_without_the_matchday_it_is_missed_as_before():
    """What the reducer did until now, kept as the fallback."""
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2026/09/05") is None


def test_a_different_matchday_is_not_matched():
    """A rescheduled meeting must not be filed under the replacement's row."""
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2026/09/03",
                        "2026-09-02", TZ) is None


def test_a_row_without_a_kickoff_still_matches_on_its_date():
    assert rd._find_row(_table(), "FC Juarez", "Pachuca", "2024/03/10",
                        "2024-03-09", TZ) == 1
