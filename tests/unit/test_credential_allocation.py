"""Each league's Odds API calls go to that league's own account.

The two accounts were originally split by role -- one for opening polls, one for
closing captures -- on the reasoning that the two have different rhythms. The
measured outcome was 451 of 500 used on one and 102 of 500 on the other, because
role puts both leagues' continuous polling on a single monthly allowance while
the bursty half barely touches its own.

Per league they land near the same fraction each, and a league that exhausts its
quota can no longer starve the other one. It also matches how odds-api.io
credentials were already allocated.
"""

from __future__ import annotations

import pathlib
import re

from betmodel.config import available_leagues, load_all, load_league
from betmodel.providers.theoddsapi import key_env_for

WORKFLOW = pathlib.Path(__file__).parents[2] / ".github/workflows/capture.yml"


def _theoddsapi_credentials(config):
    """Every Odds API credential a league uses, opens and closes alike."""
    creds = {
        b.credential for b in config.odds.books
        if b.provider == "theoddsapi" and b.poll_interval_minutes
    }
    creds.add(config.odds.close.credential)
    return creds


def test_a_league_uses_exactly_one_account():
    for league, config in load_all().items():
        assert _theoddsapi_credentials(config) == {league}, league


def test_no_two_leagues_share_an_account():
    seen = {}
    for league, config in load_all().items():
        for cred in _theoddsapi_credentials(config):
            assert cred not in seen, f"{league} shares {cred!r} with {seen[cred]}"
            seen[cred] = league


def test_every_credential_the_config_names_is_injected_by_the_workflow():
    """A credential the workflow does not pass falls back to the shared key.

    The fallback is deliberate and it is also how this breaks quietly: the run
    keeps working, against the wrong account, until that account runs out.
    """
    text = WORKFLOW.read_text()
    for league in available_leagues():
        env = key_env_for(load_league(league).odds.close.credential)
        assert re.search(rf"^\s*{env}:", text, re.M), f"{env} is not passed by capture.yml"
