"""Auxiliary evidence in a league's fit: China League One in the CSL's (D42).

What is pinned is the eight invariants of docs/LEAGUE_ONE.md, each stated
directly, because every one of them fails silently: a fit that moved when nothing
was configured, a CSL home advantage dragged toward League One's, a club priced
on another division's matches alone, a replay that saw a match it was pricing.
None of those raises; each just makes the published number a little wrong.

Every frame here is synthetic. The real League One file is never read.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from betmodel import paths, teams
from betmodel.config import load_league
from betmodel.config.schema import AuxiliarySource, SourceConfig
from betmodel.models import auxiliary, team_offset
from betmodel.models.auxiliary import AuxiliaryRows
from betmodel.models.dc import FitError, fit_from_csv, fit_production_model, prepare_training_frame
from betmodel.models.run import TEAM_STATS_COLUMNS, simulations_frame, team_stats_frame

PRIMARY = [f"P{i}" for i in range(10)]
LOWER = [f"L{i}" for i in range(8)]
#: Relegated clubs: the bridges that identify the gap between the divisions.
BRIDGES = ["P8", "P9"]
CONST, HOME = 0.30, 0.28
GAP, HOME_GAP = -0.15, -0.04      # League One scores less, and its home edge is smaller


def _strengths(seed=0):
    rng = np.random.default_rng(seed)
    clubs = PRIMARY + LOWER + ["Z"]
    return {c: (0.25 * rng.standard_normal(), 0.25 * rng.standard_normal()) for c in clubs}


STRENGTH = _strengths()


def _rate(home, away, level=0.0, home_shift=0.0):
    (atk_h, def_h), (atk_a, def_a) = STRENGTH[home], STRENGTH[away]
    lam_h = np.exp(CONST + level + atk_h + def_a + HOME + home_shift)
    lam_a = np.exp(CONST + level + atk_a + def_h)
    return lam_h, lam_a


def _matches(clubs, start, level=0.0, home_shift=0.0, every=1):
    """Each ordered pairing once, one match a day, with the expected rate as its
    target -- noise-free, so the generating parameters are the exact optimum."""
    rows = []
    day = pd.Timestamp(start)
    for home in clubs:
        for away in clubs:
            if home == away:
                continue
            lam_h, lam_a = _rate(home, away, level, home_shift)
            rows.append({"Date": day, "Home": home, "Away": away,
                         "HExpG+": lam_h, "AExpG+": lam_a, "Res": "H"})
            day += pd.Timedelta(days=every)
    return pd.DataFrame(rows)


def _config(**aux):
    config = load_league("csl")
    source = AuxiliarySource(id="league_one", name="China League One",
                             file="league_one_matches.csv", weight=aux.pop("weight", 0.5),
                             own_level=aux.pop("own_level", True))
    model = dataclasses.replace(
        config.model, min_train=30, auxiliary=(source,),
        team_offset=dataclasses.replace(config.model.team_offset, enabled=False))
    return dataclasses.replace(config, model=model)


def _aux(frame, config):
    return AuxiliaryRows(source=config.model.auxiliary[0], frame=frame.reset_index(drop=True))


def _train(config, frame=None):
    frame = _matches(PRIMARY, "2025-03-01") if frame is None else frame
    return prepare_training_frame(frame.assign(Date=frame["Date"].dt.strftime("%Y-%m-%d")),
                                  config)


def _lower(start="2025-01-01", clubs=LOWER + BRIDGES):
    return _matches(clubs, start, GAP, HOME_GAP)


# --------------------------------------------------------------------------- #
# 1. off means identical
# --------------------------------------------------------------------------- #

def _same_fit(a, b):
    assert np.array_equal(a.model.attack, b.model.attack)
    assert np.array_equal(a.model.defence, b.model.defence)
    assert a.model.const == b.model.const
    assert a.model.home_advantage == b.model.home_advantage
    assert np.array_equal(a.weights, b.weights)


def test_no_auxiliary_is_the_plain_fit_to_the_last_bit():
    """Off is not "close to" the old fit, it is the old fit. G1 holds the frozen
    pre-merge coefficients to 1e-12 and must not need an exemption for this."""
    config = _config()
    train = _train(config)
    plain = fit_production_model(train, config)
    empty = _aux(_lower().iloc[0:0], config)
    _same_fit(plain, fit_production_model(train, config, auxiliary=None))
    _same_fit(plain, fit_production_model(train, config, auxiliary=[]))
    _same_fit(plain, fit_production_model(train, config, auxiliary=[empty]))
    assert plain.model.n_groups == 0 and plain.auxiliary == ()


def test_fit_from_csv_stays_primary_only_unless_told(tmp_path):
    """G1 calls this with CSL's config, which now names League One. G1 asks about
    the pre-merge fitter, which never saw League One, so it must not pick it up."""
    config = _config()
    path = tmp_path / "matches.csv"
    frame = _matches(PRIMARY, "2025-03-01")
    frame.assign(Date=frame["Date"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False)
    assert config.model.enabled_auxiliary
    fit, train = fit_from_csv(str(path), config)
    _same_fit(fit, fit_production_model(train, config))
    pooled, _ = fit_from_csv(str(path), config, auxiliary=[_aux(_lower(), config)])
    assert pooled.auxiliary and pooled.auxiliary[0].rows > 0


def test_ligamx_pools_nothing():
    assert load_league("ligamx").model.auxiliary == ()


# --------------------------------------------------------------------------- #
# 2. the primary's weighting does not move
# --------------------------------------------------------------------------- #

def test_the_primary_rows_weights_window_and_evidence_do_not_move():
    config = _config()
    train = _train(config)
    plain = fit_production_model(train, config)
    pooled = fit_production_model(
        train, config, auxiliary=auxiliary.select_all([_aux(_lower(), config)], train, config))
    assert np.array_equal(plain.weights, pooled.weights)
    assert np.array_equal(pooled.model.weights[: len(train)], plain.weights)
    assert pooled.n_matches == plain.n_matches == len(train)
    assert list(pooled.teams) == list(plain.teams)
    for got, want in zip(pooled.primary_evidence, plain.primary_evidence):
        assert np.array_equal(got, want)


def test_auxiliary_rows_outside_the_primary_window_are_left_out():
    """After the primary's latest match they would move the Dixon-Coles reference
    date in spirit if not in code; before its window they are older than anything
    the primary itself is allowed to use."""
    config = _config()
    train = _train(config)
    latest = train["Date"].max()
    cutoff = latest - pd.DateOffset(months=config.model.lookback_months)
    lower = pd.concat([
        _lower("2023-01-01").head(3),                       # before the window
        _lower("2025-01-01").head(5),                       # inside
        _lower(str((latest + pd.Timedelta(days=1)).date())).head(4),   # after
    ], ignore_index=True)
    chosen = auxiliary.select(_aux(lower, config), train, config).frame
    assert len(chosen) == 5
    assert chosen["Date"].between(cutoff, latest).all()


def test_the_window_is_the_one_the_primary_used_even_with_a_blank_latest_row():
    """The primary's cutoff is taken before blank targets are dropped. A latest
    match still waiting on its xG moves the window, and the auxiliary must move
    with it rather than re-derive a different one from the rows that remain."""
    config = _config()
    frame = _matches(PRIMARY, "2025-03-01")
    late = frame.iloc[[0]].assign(Date=frame["Date"].max() + pd.Timedelta(days=40),
                                  **{"HExpG+": np.nan, "AExpG+": np.nan})
    train = _train(config, pd.concat([frame, late], ignore_index=True))
    expected = late["Date"].iloc[0] - pd.DateOffset(months=config.model.lookback_months)
    assert auxiliary.window_cutoff(train, config) == expected
    assert expected > train["Date"].max() - pd.DateOffset(months=config.model.lookback_months)


def test_min_train_counts_primary_matches_only():
    """However much League One there is, a CSL window too thin to fit is not fitted."""
    config = _config()
    history = team_offset.played_history(_matches(PRIMARY[:5], "2025-03-01"))   # 20 rows
    lower = _aux(_lower("2025-01-01"), config)
    day = history["_day"].max() + pd.Timedelta(days=1)
    probe = history.iloc[[0]].assign(_day=day)
    priced = team_offset.walk_forward(pd.concat([history, probe]), config, [day], [lower])
    assert priced.empty
    with pytest.raises(FitError, match="min_train"):
        prepare_training_frame(history.drop(columns="_day"), config)


# --------------------------------------------------------------------------- #
# 3. down-weighted, and on their own level
# --------------------------------------------------------------------------- #

def test_an_auxiliary_match_weighs_its_dixon_coles_weight_times_the_source_weight():
    config = _config(weight=0.5)
    train = _train(config)
    lower = auxiliary.select(_aux(_lower(), config), train, config)
    fit = fit_production_model(train, config, auxiliary=[lower])
    age = (train["Date"].max() - lower.frame["Date"]).dt.days.to_numpy(dtype=float)
    expected = np.exp(-config.model.xi * age) * 0.5
    assert np.allclose(fit.model.weights[len(train):], expected, rtol=0, atol=1e-15)


def test_its_own_level_is_recovered_and_the_primary_home_advantage_is_not_dragged():
    """Generated with League One scoring 0.15 lower and a home edge 0.04 smaller,
    and two relegated clubs playing in both. With its own level the fit reads both
    gaps back and leaves the CSL's home advantage where it is; pooled on the CSL's
    level, League One pulls it down. That second fit is what was rejected."""
    own = _config(own_level=True)
    train = _train(own)
    lower = auxiliary.select(_aux(_lower(), own), train, own)
    fit = fit_production_model(train, own, auxiliary=[lower])
    use = fit.auxiliary[0]
    assert use.bridges == tuple(BRIDGES)
    assert use.level_shift == pytest.approx(GAP, abs=1e-4)
    assert use.home_shift == pytest.approx(HOME_GAP, abs=1e-4)
    assert fit.model.home_advantage == pytest.approx(HOME, abs=1e-4)

    shared = _config(own_level=False)
    dragged = fit_production_model(train, shared, auxiliary=[_aux(lower.frame, shared)])
    assert dragged.auxiliary[0].level_shift is None
    assert dragged.model.home_advantage < HOME - 0.005


def test_no_bridge_club_leaves_the_primary_prices_alone_and_does_not_raise(caplog):
    """With no club in both divisions the gap between them is not identified: the
    fit has a flat direction. That must neither fail the stage nor move a CSL
    price, and it must say that the auxiliary informed nothing."""
    config = _config()
    train = _train(config)
    lower = auxiliary.select(_aux(_lower(clubs=LOWER), config), train, config)
    plain = fit_production_model(train, config)
    with caplog.at_level(logging.WARNING, logger="betmodel.models.dc"):
        isolated = fit_production_model(train, config, auxiliary=[lower])
    assert "unidentified" in caplog.text
    assert isolated.auxiliary[0].bridges == ()
    for home in PRIMARY[:4]:
        for away in PRIMARY[4:8]:
            a = plain.model.outcome_probabilities(home, away)
            b = isolated.model.outcome_probabilities(home, away)
            assert np.allclose(a, b, atol=1e-6)


def test_the_score_equation_is_checked_in_every_group():
    config = _config()
    train = _train(config)
    lower = auxiliary.select(_aux(_lower(), config), train, config)
    fit = fit_production_model(train, config, auxiliary=[lower])
    for group in (0, *fit.model.group_labels):
        assert np.allclose(fit.model.score_equation_ratio(group), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. only primary clubs are priced
# --------------------------------------------------------------------------- #

def test_a_club_seen_only_in_the_other_division_is_not_in_any_output():
    config = _config()
    train = _train(config)
    lower = auxiliary.select(_aux(_lower(), config), train, config)
    fit = fit_production_model(train, config, auxiliary=[lower])
    assert set(fit.teams) == set(PRIMARY)
    assert set(LOWER) <= set(fit.model.teams)

    stats = team_stats_frame(fit, "2025-06-01")
    assert list(stats.columns) == TEAM_STATS_COLUMNS
    assert TEAM_STATS_COLUMNS[:8] == ["Team", "Attack", "Defense", "Const", "HomeAdv",
                                      "Matches", "WeightedMatches", "Date"]
    assert set(stats["Team"]) == set(PRIMARY)
    bridge = stats.set_index("Team").loc["P8"]
    assert bridge["AuxMatches"] == 2 * (len(LOWER) + len(BRIDGES) - 1)
    assert stats.set_index("Team").loc["P0", "AuxMatches"] == 0
    # The ratings are the club's own, not a neighbour's in the larger model.
    index = list(fit.model.teams).index("P3")
    assert stats.set_index("Team").loc["P3", "Attack"] == fit.model.attack[index]

    with pytest.raises(KeyError, match="primary"):
        fit.predict("L0", "P0")

    sims = simulations_frame(fit, config, "2025-06-01")
    assert set(sims["Home Team"]) | set(sims["Away Team"]) == set(PRIMARY)


def _promotion_history():
    """A CSL season, then a promoted club Z's first CSL match and its second.

    P8 and P9 played League One too. Without them Z would be the only club in
    both divisions, the gap would be read off Z's single CSL match, and League
    One could say nothing about Z that that match did not already."""
    season = _matches(PRIMARY, "2025-03-01")
    debut_day = season["Date"].max() + pd.Timedelta(days=3)
    second_day = debut_day + pd.Timedelta(days=4)
    debut = _matches(["Z", "P0"], str(debut_day.date())).head(1)            # Z v P0
    # One match is a noisy read of a club; League One should pull against it.
    debut[["HExpG+", "AExpG+"]] *= (1.6, 0.6)
    second = _matches(["P1", "Z"], str(second_day.date())).head(1)          # P1 v Z
    frame = pd.concat([season, debut, second], ignore_index=True)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    return team_offset.played_history(frame), debut_day, second_day


def _lower_with_z():
    return _matches(LOWER + BRIDGES + ["Z"], "2024-12-01", GAP, HOME_GAP)


def test_a_promoted_club_is_priced_only_once_it_has_played_in_the_primary():
    config = _config()
    history, debut_day, second_day = _promotion_history()
    lower = _aux(_lower_with_z(), config)
    priced = team_offset.walk_forward(history, config, [debut_day, second_day], [lower])
    days = history.loc[priced.index, "_day"]
    assert (days == debut_day).sum() == 0, "Z was priced on League One alone"
    assert (days == second_day).sum() == 1

    plain = team_offset.walk_forward(history, config, [second_day])
    assert not np.allclose(priced[["raw_h", "raw_d", "raw_a"]].to_numpy(),
                           plain[["raw_h", "raw_d", "raw_a"]].to_numpy(), atol=1e-6), \
        "the League One evidence did not reach Z's rating"
    # ...and its evidence is its one CSL match, League One or not.
    assert priced["evidence"].to_numpy() == pytest.approx(plain["evidence"].to_numpy())


# --------------------------------------------------------------------------- #
# 5. no leak in the replay
# --------------------------------------------------------------------------- #

def test_a_replayed_day_sees_only_auxiliary_matches_before_it():
    config = _config()
    history, debut_day, second_day = _promotion_history()
    lower = _lower_with_z()
    lower = pd.concat([lower, _matches(["Z", "L0"], str(second_day.date()))],
                      ignore_index=True)          # Z plays in League One that day too
    on_the_day = lower.index[lower["Date"] >= second_day]
    before = lower.index[(lower["Home"] == "Z") & (lower["Date"] < debut_day)][-1]
    assert len(on_the_day) and before not in on_the_day

    def price(frame):
        return team_offset.walk_forward(history, config, [second_day],
                                        [_aux(frame, config)])[["raw_h", "raw_d", "raw_a"]]

    base = price(lower)
    future = lower.copy()
    future.loc[on_the_day, ["HExpG+", "AExpG+"]] = (9.0, 0.1)
    assert np.array_equal(price(future).to_numpy(), base.to_numpy())

    past = lower.copy()
    past.loc[before, ["HExpG+", "AExpG+"]] = (9.0, 0.1)
    assert not np.allclose(price(past).to_numpy(), base.to_numpy(), atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. a blank target stays blank
# --------------------------------------------------------------------------- #

def test_an_auxiliary_match_without_a_target_leaves_the_fit():
    config = _config()
    train = _train(config)
    lower = _lower()
    blanked = lower.copy()
    blanked.loc[blanked.index[:6], "HExpG+"] = np.nan
    chosen = auxiliary.select(_aux(blanked, config), train, config)
    assert chosen.frame[["HExpG+", "AExpG+"]].notna().all().all()
    without = auxiliary.select(_aux(lower.iloc[6:], config), train, config)
    a = fit_production_model(train, config, auxiliary=[chosen])
    b = fit_production_model(train, config, auxiliary=[without])
    assert np.array_equal(a.model.attack, b.model.attack)


# --------------------------------------------------------------------------- #
# 9. one club is one club
# --------------------------------------------------------------------------- #

def _mapping(tmp_path):
    path = tmp_path / "team_name_mapping.csv"
    path.write_text("standard_team,sofascore_team\n"
                    "Dalian Yingbo,Dalian Yingbo FC\n"
                    "Shanghai Port,Shanghai Port FC\n")
    return teams.load(str(path), league="csl")


def _file(tmp_path, rows):
    path = tmp_path / "league_one_matches.csv"
    pd.DataFrame(rows, columns=list(auxiliary.COLUMNS)).to_csv(path, index=False)
    return path


def _row(event_id, date, home, away, hxg="1.2", axg="0.8", hg=1, ag=0):
    return {"event_id": event_id, "Season": date[:4], "Round": "1", "Date": date,
            "Time": "11:00", "kickoff_utc": f"{date}T11:00:00+00:00",
            "home_sofascore": home, "away_sofascore": away,
            "home_team_id": 1, "away_team_id": 2, "HG": hg, "AG": ag,
            "HxG": hxg, "AxG": axg}


def test_a_sofascore_spelling_resolves_to_the_canonical_club(tmp_path, monkeypatch):
    """Dalian Yingbo in League One and Dalian Yingbo in the CSL are one club to the
    fitter, or the bridge the division gap is read from is lost. A League One club
    the mapping has never heard of is the normal case, and is not reported."""
    monkeypatch.setattr(teams, "report_unmapped",
                        lambda *a, **k: pytest.fail("an unmapped League One club was reported"))
    path = _file(tmp_path, [_row(1, "2025-04-01", "Dalian Yingbo FC", "Guangxi Pingguo")])
    config = _config()
    rows = auxiliary.read(str(path), config.model.auxiliary[0], config, _mapping(tmp_path))
    assert rows.frame.loc[0, "Home"] == "Dalian Yingbo"
    assert rows.frame.loc[0, "Away"] == "Guangxi Pingguo"


def test_the_target_is_the_leagues_own_blend_and_blank_without_xg(tmp_path):
    config = _config()
    path = _file(tmp_path, [_row(1, "2025-04-01", "A", "B", hxg="2.0", axg="1.0", hg=1, ag=0),
                            _row(2, "2025-04-02", "B", "A", hxg="", axg="")])
    rows = auxiliary.read(str(path), config.model.auxiliary[0], config, _mapping(tmp_path))
    blend = config.model.xg_blend
    assert rows.frame.loc[0, "HExpG+"] == pytest.approx(blend.xg * 2.0 + blend.goals * 1)
    assert rows.frame.loc[1, ["HExpG+", "AExpG+"]].isna().all()


def test_a_missing_file_fits_without_it(tmp_path, monkeypatch, caplog):
    """CI on a checkout without the file behaves as before."""
    monkeypatch.setattr(paths, "data_dir", lambda: str(tmp_path))
    (tmp_path / "csl").mkdir()
    config = _config()
    with caplog.at_level(logging.WARNING, logger="betmodel.models.auxiliary"):
        loaded = auxiliary.load_enabled("csl", config, mapping=_mapping(tmp_path))
    assert loaded == []
    assert "does not exist" in caplog.text


def test_a_disabled_auxiliary_is_not_read(tmp_path):
    config = _config()
    off = dataclasses.replace(config.model.auxiliary[0], enabled=False)
    config = dataclasses.replace(config, model=dataclasses.replace(
        config.model, auxiliary=(off,)))
    assert auxiliary.load_enabled("csl", config, mapping=_mapping(tmp_path)) == []


def test_the_csl_config_names_league_one_at_half_weight_on_its_own_level():
    (source,) = load_league("csl").model.auxiliary
    assert (source.id, source.file, source.weight, source.own_level, source.enabled) == (
        "league_one", "league_one_matches.csv", 0.5, True, True)
    assert source.source == SourceConfig(provider="sofascore", network="cloud",
                                         params={"unique_tournament_id": 782})
    assert paths.for_league("csl").auxiliary_csv(source.file).endswith(
        "data/csl/league_one_matches.csv")


def test_the_production_fit_logs_what_it_pooled(tmp_path, monkeypatch, caplog):
    """run_model reads the file through the same loader, so a fit that silently
    pooled nothing would look like one that pooled everything."""
    from betmodel.models import run

    monkeypatch.setattr(paths, "data_dir", lambda: str(tmp_path))
    league = tmp_path / "csl"
    league.mkdir()
    frame = _matches(PRIMARY, "2025-03-01")
    frame.assign(Date=frame["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        league / "matches.csv", index=False)
    lower = _lower()
    rows = [_row(i, str(r.Date.date()), r.Home, r.Away, hxg=r._4, axg=r._5, hg=1, ag=1)
            for i, r in enumerate(lower.itertuples())]
    _file(league, rows)
    (league / "team_name_mapping.csv").write_text(
        "standard_team,sofascore_team\n" + "".join(f"{c},\n" for c in PRIMARY))
    config = _config()
    monkeypatch.setattr(run, "load_league", lambda _: config)
    with caplog.at_level(logging.INFO, logger="betmodel.models.run"):
        fit = run.run_model("csl", write=False)
    assert "league_one pooled" in caplog.text and "bridge clubs" in caplog.text
    assert set(fit.teams) == set(PRIMARY)
    assert fit.auxiliary[0].rows == len(lower)


def test_the_walkforward_replays_production_with_and_without_league_one():
    """`--aux-weight 0` is the plain fit; any other weight reaches every source."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "walkforward", os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                                    "walkforward.py"))
    walkforward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walkforward)

    config = load_league("csl")
    off = walkforward.configure(config, aux_weight=0)
    assert off.model.auxiliary == () and walkforward.describe_auxiliary(off, []) == "off"
    quarter = walkforward.configure(config, aux_weight=0.25, aux_shared_level=True)
    assert [(a.weight, a.own_level) for a in quarter.model.auxiliary] == [(0.25, False)]
    assert walkforward.configure(config).model.auxiliary == config.model.auxiliary
    assert "missing" in walkforward.describe_auxiliary(config, [])

    # Configured but unreadable is the same replay as switched off.
    a = walkforward.predict("csl", config, "2026-09-01", auxiliary_rows=[])
    b = walkforward.predict("csl", off, "2026-09-01", auxiliary_rows=[])
    assert len(a) > 0
    assert np.array_equal(a[["raw_h", "raw_d", "raw_a"]].to_numpy(),
                          b[["raw_h", "raw_d", "raw_a"]].to_numpy())
