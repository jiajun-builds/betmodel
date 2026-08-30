"""A stored event id follows the convention its provider's history already uses.

`event_id` is part of the dedup key, so the id form IS the row's identity. Both
pre-merge pipelines wrote odds-api.io ids as `oddsapiio:72055150` and The Odds
API ids bare. The merged capture wrote the raw provider id for everything, so
three Liga MX opening lines went in bare -- the same captures as upstream's, under
a different identity. Nothing was lost, because the pending gate keys on team
names, but the reconciliation due before the old repositories are archived would
have imported all three a second time.
"""

from __future__ import annotations

import glob

import pandas as pd

from betmodel.odds import capture_store


def test_odds_api_io_ids_carry_their_provider():
    assert capture_store.stored_event_id("oddsapiio", "72055150") == "oddsapiio:72055150"


def test_prefixing_is_idempotent():
    # Rows migrated from upstream already carry it; re-prefixing would create a
    # third identity for the same capture.
    assert capture_store.stored_event_id(
        "oddsapiio", "oddsapiio:72055150") == "oddsapiio:72055150"


def test_the_odds_api_ids_stay_bare():
    """Not an oversight: 677 rows of history are bare, and the id is the identity.

    Restyling them would make every one stop matching the same capture upstream.
    """
    assert capture_store.stored_event_id("theoddsapi", "860b8318") == "860b8318"


def test_an_empty_id_is_left_empty():
    assert capture_store.stored_event_id("oddsapiio", "") == ""


def test_event_id_is_part_of_the_identity():
    # If this stops being true the whole concern above evaporates -- and so does
    # the reason this module exists.
    assert "event_id" in capture_store.DEDUP_KEY


def test_stored_history_matches_its_own_convention():
    """Every row on disk is written the way its provider's rows are written."""
    for path in sorted(glob.glob("data/*/odds_capture_history.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for row in frame.itertuples(index=False):
            want = capture_store.stored_event_id(row.regions, row.event_id)
            assert row.event_id == want, f"{path}: {row.event_id!r} from {row.regions!r}"
