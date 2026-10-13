"""A league configured for market_anchor must actually be publishing anchored rows.

One fixture falling back to raw is normal and deliberate -- no anchor book has
opened that match yet. A whole league falling back looks identical per row and
means something else entirely: the anchor is not being captured, and the league
has been publishing uncalibrated probabilities under a config that claims
otherwise.

That is not hypothetical. The Chinese Super League did it for three days, and the
only thing that surfaced it was comparing the config against the output by hand.
Measured on six fixtures, the correction moves EV by 2.3 to 6.4 percentage points,
always downward, against a 20-point threshold.
"""

from __future__ import annotations

import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "check_debias_applied",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                 "check_debias_applied.py"),
)
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


def _signals(*methods):
    return [{"model": {"method": m}} for m in methods]


def test_a_fully_anchored_league_is_healthy():
    ok, msg = check.assess("csl", "market_anchor", _signals("market_anchor") * 6)
    assert ok
    assert "100%" in msg


def test_a_league_publishing_only_raw_is_the_alarm():
    ok, msg = check.assess("csl", "market_anchor", _signals("raw") * 6)
    assert not ok
    assert "uncalibrated" in msg


def test_a_few_unanchored_fixtures_are_tolerated():
    # Early in a round most fixtures legitimately have no anchor yet. An alarm
    # that fires on a normal Monday gets muted, and a muted alarm is worse than
    # none.
    ok, _ = check.assess("csl", "market_anchor",
                         _signals("market_anchor", "market_anchor", "market_anchor", "raw"))
    assert ok


def test_a_league_that_does_not_use_the_anchor_is_not_judged():
    # Liga MX is deliberately debias: none. Applying the anchored-share rule there
    # would report a configuration choice as a fault.
    ok, msg = check.assess("ligamx", "none", _signals("raw") * 9)
    assert ok
    assert "nothing to check" in msg


def test_a_league_with_no_priced_fixture_is_not_judged():
    # Between rounds there is nothing to anchor, and dividing by zero would make
    # the check itself the failure.
    ok, msg = check.assess("csl", "market_anchor", [])
    assert ok
    assert "no priced fixture" in msg


def _named(fixture_id, method, home="A", away="B"):
    return {"fixture_id": fixture_id, "home_team": home, "away_team": away,
            "model": {"method": method}}


def test_the_alarm_tells_a_refused_anchor_from_a_missing_one():
    # Both publish raw and look identical per row. Liga MX round 11 was all the
    # first kind -- every Pinnacle opener captured, every one refused on `window`
    # -- under a message that said the openers were not being captured, which
    # sends the reader to the provider instead of to the proof.
    signals = [_named("f1", "raw", "Atlas", "Guadalajara"),
               _named("f2", "raw", "Pachuca", "Necaxa")]
    ok, msg = check.assess("ligamx", "market_anchor", signals,
                           {"f1": "window", "f2": check.MISSING})
    assert not ok
    assert "1 have an anchor opener on file that the anchor gate refused" in msg
    assert "proof 'window'" in msg and "Atlas v Guadalajara" in msg
    assert "1 have no anchor opening price on file" in msg
    assert "Pachuca v Necaxa" in msg


def test_without_the_anchor_view_the_alarm_does_not_guess_the_cause():
    ok, msg = check.assess("ligamx", "market_anchor", _signals("raw") * 3)
    assert not ok
    assert "not being captured or not proven" in msg
