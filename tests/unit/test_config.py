"""Config layer guards.

Each test names the specific mistake it prevents. These are cheap because the
config is loaded unattended in CI and a wrong value publishes to a live board.
"""

from __future__ import annotations

import textwrap

import pytest

from betmodel import paths
from betmodel.config import ConfigError, available_leagues, load_all, load_league
from betmodel.config.loader import clear_cache


# --------------------------------------------------------------------------- #
# the real configs
# --------------------------------------------------------------------------- #

def test_every_shipped_league_loads():
    leagues = load_all()
    assert leagues, "no leagues found"
    assert set(leagues) == set(available_leagues())


def test_available_leagues_is_discovered_not_hardcoded():
    """The whole point of the merge: nothing may enumerate leagues, this test
    included. It asserts the mechanism, that what is on disk is what comes back,
    rather than a list that has to be edited whenever a league is added. The
    first version of this test named two leagues and broke the moment a third
    arrived, which is the failure it exists to prevent.
    """
    from pathlib import Path

    on_disk = {
        p.stem for p in Path(paths.leagues_dir()).glob("*.y*ml")
        if not p.stem.startswith("_")
    }
    assert set(available_leagues()) == on_disk
    assert len(on_disk) >= 2


@pytest.mark.parametrize(
    "league,xi,window,ev_min",
    [("csl", 0.001, 18, 0.20), ("ligamx", 0.0015, 24, 0.10)],
)
def test_model_and_signal_params_match_the_pre_merge_constants(
    league, xi, window, ev_min
):
    """These values are the ones the golden outputs were produced with.

    If one drifts, gates G1 and G3 fail for a reason that has nothing to do with
    the code being ported, so pin them here where the failure is legible.
    """
    c = load_league(league)
    assert c.model.xi == xi
    assert c.model.lookback_months == window
    assert c.signals.ev_min == ev_min


@pytest.mark.parametrize("league", ["csl", "ligamx"])
def test_the_scoreline_grid_is_the_same_for_every_league(league):
    """Deliberately NOT the pre-merge value. See docs/DECISIONS.md D2.

    The two pipelines used 10 and 15, and read the field differently on top of
    that: one built arange(max_goals), the other arange(max_goals + 1). It now
    means the highest scoreline modelled, and both leagues use 9. The cost to
    Liga MX is bounded by gate G1 rather than by this test.
    """
    assert load_league(league).model.max_goals == 9


def test_csl_anchors_the_draw_on_pinnacle_and_never_bets_it():
    c = load_league("csl")
    assert c.signals.debias.method == "market_anchor"
    assert c.odds.book(c.signals.debias.anchor_book).role == "anchor"
    assert "pinnacle" not in [b.key for b in c.odds.bet_books]
    assert c.signals.sides == ("home", "away")


def test_ligamx_anchors_the_draw_and_does_bet_it():
    """It bets the draw, which is why anchoring the draw matters most here.

    This league ran without de-bias until it had Pinnacle opening lines to anchor
    to -- the config said so in as many words, and the condition is now met. The
    combination is the point: the model prices the draw worst, this league is the
    one that will actually take a draw, and the anchor is the correction. Turning
    it on moved one fixture's draw EV by 17 points and dropped a firing signal.
    """
    c = load_league("ligamx")
    assert c.signals.debias.enabled is True
    assert c.signals.debias.anchor_book == "pinnacle"
    assert c.signals.sides == ("home", "draw", "away")


def test_no_league_anchors_on_a_book_it_bets():
    """Structural, not per league.

    Letting the price we intend to take also define the probability we price it
    against would make every quote look fair by construction. The schema refuses
    it at load; this asserts no shipped config is relying on that refusal never
    being tested.
    """
    for league in available_leagues():
        c = load_league(league)
        anchor = c.signals.debias.anchor_book
        if not anchor:
            continue
        assert anchor not in {b.key for b in c.odds.bet_books}, league


def test_onexbet_legacy_prefix_stays_frozen():
    """The board derives CSL's column names from this prefix.

    Letting it fall back to the derived "onexbet_open" would be correct by luck
    today; the guard is that any change has to be deliberate.
    """
    book = load_league("csl").odds.book("onexbet")
    assert book.effective_legacy_prefix == "onexbet_open"
    assert book.legacy_prefix_note, "a frozen prefix must record why"


def test_ligamx_never_takes_fixtures_from_thesportsdb():
    """A standing constraint, guarded because it is a tempting wrong turn.

    TheSportsDB is the only schedule provider a cloud runner can reach without a
    residential proxy, so switching Liga MX to it looks like a free way to give
    the league the automated refresh the other one already has. Its Liga MX match
    data is incomplete, so it is not available for that, and this league pays the
    residential-egress cost for fixtures as well as xG.
    """
    sources = load_league("ligamx").sources
    assert sources["fixtures"].provider == "sofascore"
    assert sources["xg"].provider == "sofascore"


def test_residential_stages_are_data_not_a_hardcoded_league_name():
    assert load_league("csl").residential_stages == ("xg",)
    assert load_league("ligamx").residential_stages == ("fixtures", "xg")


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #

MINIMAL = """
id: {id}
name: Test League
code: TST
season: "2026"
timezone: UTC
total_rounds: 20
sources:
  fixtures: {{provider: thesportsdb, league_id: "1"}}
  xg: {{provider: sofascore, network: residential}}
model:
  xi: 0.001
  lookback_months: 18
  max_goals: 10
  xg_blend: {{xg: {blend_xg}, goals: 0.3}}
odds:
  providers:
    oddsapiio: {{}}
    theoddsapi: {{}}
  books:
    - {{key: bookone, provider: oddsapiio, role: {role}{extra}}}
    - {{key: pinnacle, provider: theoddsapi, role: {anchor_role}}}
  close: {{books: [{close_books}]}}
signals:
  ev_min: {ev_min}
  debias: {{method: {debias}, anchor_book: pinnacle}}
publish:
  legacy_contract: per_book
"""


def _write(tmp_path, *, name="tst", id_="tst", blend_xg=0.7, role="bet",
           anchor_role="anchor", close_books="pinnacle", ev_min=0.2,
           debias="market_anchor", extra=""):
    (tmp_path / f"{name}.yml").write_text(
        textwrap.dedent(
            MINIMAL.format(id=id_, blend_xg=blend_xg, role=role, anchor_role=anchor_role,
                           close_books=close_books, ev_min=ev_min, debias=debias,
                           extra=extra)
        )
    )
    clear_cache()
    return str(tmp_path)


def test_baseline_fixture_is_actually_valid(tmp_path):
    load_league("tst", _write(tmp_path))


def test_percent_ev_threshold_is_rejected(tmp_path):
    """10 meaning "10 percent" is the single most likely config typo here.

    The two source repos disagreed on the unit and the disagreement reached the
    board, which now guesses the scale from the payload.
    """
    with pytest.raises(ConfigError, match="fraction"):
        load_league("tst", _write(tmp_path, ev_min=10))


def test_unknown_book_role_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="role"):
        load_league("tst", _write(tmp_path, role="favourite"))


def test_debias_anchor_may_not_be_a_bet_book(tmp_path):
    """Otherwise the price we intend to take also defines the probability."""
    with pytest.raises(ConfigError, match="anchor"):
        load_league("tst", _write(tmp_path, anchor_role="bet"))


def test_xg_blend_must_sum_to_one(tmp_path):
    with pytest.raises(ConfigError, match="sum to 1"):
        load_league("tst", _write(tmp_path, blend_xg=0.9))


def test_close_books_must_be_declared(tmp_path):
    with pytest.raises(ConfigError, match="undeclared"):
        load_league("tst", _write(tmp_path, close_books="matchbook"))


def test_book_needs_a_declared_provider(tmp_path):
    p = tmp_path / "tst.yml"
    _write(tmp_path)
    p.write_text(p.read_text().replace("provider: oddsapiio, role", "provider: nope, role"))
    clear_cache()
    with pytest.raises(ConfigError, match="not declared"):
        load_league("tst", str(tmp_path))


def test_nondefault_legacy_prefix_requires_a_reason(tmp_path):
    with pytest.raises(ConfigError, match="legacy_prefix_note"):
        load_league("tst", _write(tmp_path, extra=", legacy_prefix: weird_open"))


def test_filename_and_id_must_agree(tmp_path):
    """The id becomes a directory name and a published URL segment."""
    with pytest.raises(ConfigError, match="filename"):
        load_league("tst", _write(tmp_path, id_="other"))


def test_unknown_league_lists_what_exists(tmp_path):
    with pytest.raises(ConfigError, match="available"):
        load_league("nope", _write(tmp_path))


def test_league_id_may_not_escape_the_data_directory():
    with pytest.raises(ValueError):
        paths.for_league("../../etc")


@pytest.mark.parametrize("league", available_leagues())
def test_the_close_reserve_is_actually_reserved(league):
    """The open floor only reserves credit if the close floor sits below it.

    Both were 50. The open floor is there to stop spending the tail of a metered
    period on opening lines so that closing lines still have credit, but with the
    two equal the reserve protected nothing: at 48 remaining the opens stopped and
    so did the closes, leaving the last 50 credits of every period unspendable by
    anything at all. On 2026-08-31 that took out every CSL anchor slot and every
    CSL close on the same account, while every workflow run reported success.

    A close has a hard deadline at kickoff and cannot be re-bought at any price,
    so it must be the last thing to stop, not the first.
    """
    config = load_league(league)
    close = config.odds.close
    if close is None:
        pytest.skip(f"{league} captures no closing lines")
    assert close.min_remaining < config.odds.quota_floor, (
        f"{league}: closes floor at {close.min_remaining} and opens at "
        f"{config.odds.quota_floor}; the reserve frees nothing for the one thing "
        "that cannot be captured later"
    )
