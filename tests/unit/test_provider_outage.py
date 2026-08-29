"""An unreachable provider is not an empty result.

The first refresh run through CI reported success having fetched nothing. Every
SofaScore call had failed at the TCP level — the residential proxy was refusing
connections — and each stage read the empty answer as "the provider has nothing
new", logged a warning, and returned cleanly. A refresh that fetches nothing and
says so quietly is worse than one that fails, because the staleness it causes is
discovered days later by whoever reads a stale number.

The distinction is made where it is known: the request either got an answer or it
did not.
"""

from __future__ import annotations

import pytest

from betmodel.providers import http, sofascore


class _Boom:
    """An http layer where every request fails the way an outage fails."""

    proxy_url = "http://proxy.example:8080"

    def __init__(self, status=None):
        self.status = status

    def json(self, url, **kwargs):
        raise http.HttpError(f"GET {url} failed", status=self.status)


def _client(status=None):
    client = sofascore.SofascoreClient.__new__(sofascore.SofascoreClient)
    client._http = _Boom(status)
    client.pause = 0.0
    return client


def test_an_unreachable_provider_raises_rather_than_reporting_no_seasons():
    with pytest.raises(sofascore.SofascoreUnavailable):
        _client().seasons(649)


def test_the_stage_entry_point_raises_too():
    # current_season_id is what the xG stage calls; it must not turn the outage
    # into a None that reads as "no current season".
    with pytest.raises(sofascore.SofascoreUnavailable):
        _client().current_season_id(649)


def test_a_404_is_an_answer_and_stays_one():
    # A tournament that does not exist is a real reply, not an outage, and must
    # not be escalated: it would make a wrong id look like a proxy failure.
    assert _client(404).seasons(649) == []


def test_a_failing_league_makes_the_run_fail(monkeypatch):
    """The CLI runs every league and must still exit non-zero if one failed.

    Leagues are deliberately independent — one outage must not cost another
    league's capture — so the failure is recorded rather than raised, and it is
    the exit code that has to carry it.
    """
    from betmodel import cli

    calls = []

    def _boom(league, args):
        calls.append(league)
        raise sofascore.SofascoreUnavailable("proxy refused the connection")

    monkeypatch.setitem(cli.STAGES, "xg", _boom)
    code = cli.main(["all", "xg"])
    assert code == 1
    assert len(calls) > 1, "the other leagues must still have been attempted"
