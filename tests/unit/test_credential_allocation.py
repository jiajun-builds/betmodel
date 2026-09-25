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

import datetime
import pathlib
import re

from betmodel.config import available_leagues, load_all, load_league
from betmodel.providers.theoddsapi import key_env_for

WORKFLOW = pathlib.Path(__file__).parents[2] / ".github/workflows/capture.yml"

#: ``league -> (the account it is borrowing, the day it must stop)``.
#:
#: Borrowing defeats the allocation this whole file exists to enforce, so it is
#: not allowed to be a quiet edit to a YAML line. It is declared here, and it
#: carries the day the lending account resets -- after which
#: :func:`test_a_borrowed_account_is_returned` fails.
#:
#: **The expiry is the mechanism, not documentation.** A comment saying "change
#: this back on the 24th" is read once, by the person who wrote it. A red suite
#: is read by whoever runs it next. Used once: Liga MX ran on CSL's account from
#: 2026-09-14, when its own was exhausted, until that one reset on 2026-09-24.
BORROWED: dict[str, tuple[str, datetime.date]] = {}


def _entitled(league: str) -> set[str]:
    """The accounts a league may legitimately name today."""
    borrowed = BORROWED.get(league)
    return {league} | ({borrowed[0]} if borrowed else set())


def _theoddsapi_credentials(config):
    """Every Odds API credential a league uses, opens and closes alike."""
    creds = {
        b.credential for b in config.odds.books
        if b.provider == "theoddsapi" and b.poll_interval_minutes
    }
    creds.add(config.odds.close.credential)
    return creds


def test_a_league_uses_exactly_one_account():
    """One account per league, opens and closes alike -- even while borrowing.

    Splitting a league across two accounts is the failure this prevents in
    either direction: it is what role-based allocation did, and it is also the
    tempting half-measure when an account runs dry ("move just the opens").
    """
    for league, config in load_all().items():
        creds = _theoddsapi_credentials(config)
        assert len(creds) == 1, f"{league} is split across {sorted(creds)}"
        assert creds <= _entitled(league), (
            f"{league} names {sorted(creds)}; it may use "
            f"{sorted(_entitled(league))}"
        )


def test_no_two_leagues_share_an_account():
    """Except where BORROWED says so, and then only one way round.

    The lender must not still be spending its own account through a second
    league, which would be the shared-allowance arrangement this replaced.
    """
    seen: dict[str, str] = {}
    for league, config in load_all().items():
        for cred in _theoddsapi_credentials(config):
            if cred in seen:
                lender = BORROWED.get(league, (None, None))[0]
                assert cred == lender and seen[cred] == cred, (
                    f"{league} shares {cred!r} with {seen[cred]}"
                )
            seen.setdefault(cred, league)


def test_a_borrowed_account_is_returned():
    """Fails the day the lending account resets. That is the point.

    When this goes red the fix is in the failure message: put the borrowing
    league's own credential back in its YAML and delete its BORROWED entry. The
    borrowed allowance is not free after the reset -- it is the lender's next
    month.
    """
    today = datetime.date.today()
    overdue = {
        league: (lender, until)
        for league, (lender, until) in BORROWED.items()
        if today >= until
    }
    assert not overdue, "; ".join(
        f"{league} has been on {lender}'s account since before {until}: set "
        f"`credential: {league}` in leagues/{league}.yml (the anchor book and "
        f"odds.close) and drop it from BORROWED"
        for league, (lender, until) in sorted(overdue.items())
    )


def test_every_credential_the_config_names_is_injected_by_the_workflow():
    """A credential the workflow does not pass falls back to the shared key.

    The fallback is deliberate and it is also how this breaks quietly: the run
    keeps working, against the wrong account, until that account runs out.
    """
    text = WORKFLOW.read_text()
    for league in available_leagues():
        env = key_env_for(load_league(league).odds.close.credential)
        assert re.search(rf"^\s*{env}:", text, re.M), f"{env} is not passed by capture.yml"
