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

def test_both_shipped_leagues_load():
    leagues = load_all()
    assert set(leagues) == {"csl", "ligamx"}


def test_available_leagues_is_discovered_not_hardcoded():
    # The whole point of the merge: the engine must never enumerate leagues.
    assert available_leagues() == ("csl", "ligamx")


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


def test_ligamx_does_not_debias_and_does_bet_the_draw():
    c = load_league("ligamx")
    assert c.signals.debias.enabled is False
    assert c.signals.sides == ("home", "draw", "away")


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
