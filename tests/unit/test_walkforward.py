"""The walk-forward harness that trials a model setting (D34).

A research script, so what is pinned is only what would make its numbers wrong
without anyone noticing: that it prices each fixture from strictly earlier
matches, and that a trialled setting actually reaches the fit.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd

from betmodel.config import load_league

_SPEC = importlib.util.spec_from_file_location(
    "walkforward",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "walkforward.py"),
)
walkforward = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(walkforward)

START = "2026-08-01"


def test_a_short_window_is_priced_and_scored():
    config = load_league("csl")
    frame = walkforward.predict("csl", config, START)
    assert len(frame) > 0
    assert (pd.to_datetime(frame["_day"]) >= pd.Timestamp(START)).all()
    sums = frame[["raw_h", "raw_d", "raw_a"]].sum(axis=1)
    assert np.allclose(sums, 1.0)

    table = walkforward.report(walkforward.score("csl", config, frame))
    assert "all" in table.index
    assert table.loc["all", "fixtures"] == len(frame)


def test_a_fixture_is_priced_only_from_earlier_matches(tmp_path):
    """Pricing a match with its own result, or a later one, in the training set
    would flatter every score the harness reports. Deleting every match after
    the first priced day must leave that day's prices unchanged."""
    config = load_league("csl")
    full = walkforward.predict("csl", config, START)
    first_day = full["_day"].min()

    matches = pd.read_csv(walkforward.paths.for_league("csl").matches_csv)
    kept = matches[walkforward.parse_date_only_series(matches["Date"]) <= first_day]
    trimmed_path = tmp_path / "matches.csv"
    kept.to_csv(trimmed_path, index=False)
    trimmed = walkforward.predict("csl", config, START, matches_path=str(trimmed_path))

    columns = ["raw_h", "raw_d", "raw_a"]
    assert len(trimmed) > 0
    assert np.allclose(full[full["_day"] == first_day][columns].sort_index().to_numpy(),
                       trimmed[columns].sort_index().to_numpy())


def test_a_trialled_setting_reaches_the_fit():
    config = walkforward.configure(load_league("ligamx"), xi=0.004, blend=0.5, shrink=0)
    assert config.model.xi == 0.004
    assert (config.model.xg_blend.xg, config.model.xg_blend.goals) == (0.5, 0.5)
    assert config.model.shrinkage.enabled is False


def test_a_book_can_be_replayed_on_its_own():
    """How a paused league earns a book back (D35): its bets, not the best of all."""
    config = load_league("ligamx")
    frame = walkforward.predict("ligamx", config, "2026-07-01")
    every = walkforward.score("ligamx", config, frame)
    duel = walkforward.score("ligamx", config, frame, book="duel")
    none = walkforward.score("ligamx", config, frame, book="no-such-book")
    assert duel["clv"].notna().sum() <= every["clv"].notna().sum()
    assert none["clv"].notna().sum() == 0
