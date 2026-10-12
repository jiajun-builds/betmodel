"""The per-club market offset (D38).

What is pinned is what would make the correction wrong without anyone noticing:
that it reads only matches played before the day it is applied to, that it moves
the home/away split and nothing else, that a club with no history is left alone,
and that the 1X2 the engine reads carries it while the research-only handicap
columns do not.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from betmodel import paths
from betmodel.config import load_league
from betmodel.config.schema import TeamOffset
from betmodel.models import team_offset
from betmodel.models.dc import fit_from_csv
from betmodel.models.poisson import Outcome
from betmodel.models.run import simulations_frame

AS_OF = pd.Timestamp("2026-06-01")
SETTINGS = TeamOffset(enabled=True, xi=0.008, lam=0.01, window_months=18, min_history=30)
PLANTED = {"A": 0.30, "B": 0.10, "C": -0.15, "D": -0.25}
KAPPA = 0.05


def _residuals(n_rounds=12, start="2025-09-01", noise=0.0, seed=0):
    """A double round robin of four clubs whose residuals are exactly the planted
    offsets, one round a week."""
    rng = np.random.default_rng(seed)
    clubs = list(PLANTED)
    rows = []
    day = pd.Timestamp(start)
    for _ in range(n_rounds):
        for home in clubs:
            for away in clubs:
                if home == away:
                    continue
                rows.append({"day": day, "home": home, "away": away,
                             "residual": KAPPA + PLANTED[home] - PLANTED[away]
                             + noise * rng.standard_normal()})
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_the_planted_offsets_are_recovered():
    offsets = team_offset.estimate(_residuals(), AS_OF, SETTINGS)
    for home in PLANTED:
        for away in PLANTED:
            if home != away:
                assert offsets.shift(home, away) == pytest.approx(
                    KAPPA + PLANTED[home] - PLANTED[away], abs=1e-3)


def test_too_little_history_applies_no_correction():
    res = _residuals(n_rounds=1)            # 12 residuals, under min_history
    offsets = team_offset.estimate(res, AS_OF, SETTINGS)
    assert offsets.deltas == {} and offsets.shift("A", "D") == 0.0


def test_a_residual_on_or_after_the_day_cannot_reach_it():
    """The correction applied to a match must not have seen that match, or any
    later one: it would flatter every score the walk-forward reports."""
    res = _residuals(start="2025-12-01", n_rounds=30)   # runs past AS_OF
    assert (res["day"] >= AS_OF).any()
    before = res[res["day"] < AS_OF]
    poisoned = res.copy()
    poisoned.loc[poisoned["day"] >= AS_OF, "residual"] += 5.0
    a = team_offset.estimate(before, AS_OF, SETTINGS)
    b = team_offset.estimate(poisoned, AS_OF, SETTINGS)
    assert a.kappa == pytest.approx(b.kappa)
    assert a.deltas == pytest.approx(b.deltas)


def test_a_residual_outside_the_window_is_ignored():
    old = _residuals(start="2023-01-01")
    old["residual"] += 3.0
    recent = _residuals(start="2025-09-01")
    a = team_offset.estimate(recent, AS_OF, SETTINGS)
    b = team_offset.estimate(pd.concat([old, recent]), AS_OF, SETTINGS)
    assert a.deltas == pytest.approx(b.deltas)


def test_recent_residuals_weigh_more():
    """Two contradictory seasons: the fit follows the recent one."""
    old = _residuals(start="2024-12-01")
    recent = _residuals(start="2025-09-01")
    recent["residual"] = -recent["residual"] + 2 * KAPPA
    offsets = team_offset.estimate(pd.concat([old, recent]), AS_OF, SETTINGS)
    assert offsets.deltas["A"] < 0 < offsets.deltas["D"]


def test_the_penalty_pulls_a_thinly_seen_club_toward_zero():
    res = _residuals()
    thin = pd.DataFrame([{"day": AS_OF - pd.Timedelta(days=3), "home": "E", "away": "A",
                          "residual": 1.0 + KAPPA - PLANTED["A"]}])
    offsets = team_offset.estimate(pd.concat([res, thin]),
                                   AS_OF, TeamOffset(enabled=True, lam=4.0))
    assert 0.0 < offsets.deltas["E"] < 0.5


def test_the_correction_moves_the_split_and_leaves_the_draw():
    offsets = team_offset.Offsets(kappa=0.0, deltas={"A": 0.2, "B": -0.1})
    raw = Outcome(home=0.45, draw=0.27, away=0.28)
    out = offsets.adjust(raw, "A", "B")
    assert out.draw == raw.draw
    assert out.home + out.draw + out.away == pytest.approx(1.0)
    share = math.log(out.home / out.away) - math.log(raw.home / raw.away)
    assert share == pytest.approx(0.3)


def test_a_club_with_no_history_contributes_nothing_of_its_own():
    offsets = team_offset.Offsets(kappa=0.05, deltas={"A": 0.2})
    assert offsets.shift("A", "Promoted") == pytest.approx(0.25)
    assert offsets.shift("Promoted", "Other") == pytest.approx(0.05)


def test_the_identity_is_the_identity():
    raw = Outcome(home=0.45, draw=0.27, away=0.28)
    assert team_offset.Offsets().adjust(raw, "A", "B") == raw


def test_a_league_that_does_not_ask_for_it_is_not_corrected():
    config = load_league("ligamx")
    assert config.model.team_offset.enabled is False
    assert team_offset.from_history(pd.DataFrame(), config, AS_OF) == team_offset.Offsets()


def test_the_engine_reads_the_corrected_1x2_and_the_handicaps_stay_raw():
    config = load_league("csl")
    fit, _ = fit_from_csv(paths.for_league("csl").matches_csv, config)
    home, away = fit.teams[0], fit.teams[1]
    offsets = team_offset.Offsets(kappa=0.0, deltas={home: 0.3, away: -0.2})
    plain = simulations_frame(fit, config, "2026-10-01")
    corrected = simulations_frame(fit, config, "2026-10-01", offsets)
    pick = lambda f: f[(f["Home Team"] == home) & (f["Away Team"] == away)].iloc[0]  # noqa: E731
    a, b = pick(plain), pick(corrected)
    expected = offsets.adjust(
        Outcome(a["Home Win Probability"], a["Draw Probability"], a["Away Win Probability"]),
        home, away)
    assert b["Home Win Probability"] == pytest.approx(expected.home)
    assert b["Draw Probability"] == pytest.approx(a["Draw Probability"])
    assert b["Home Win Probability"] > a["Home Win Probability"]
    handicaps = [c for c in plain.columns if c.startswith(("Home -", "Away -"))]
    assert handicaps and (plain[handicaps].to_numpy() == corrected[handicaps].to_numpy()).all()
