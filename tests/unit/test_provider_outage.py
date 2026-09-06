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

from datetime import datetime, timedelta, timezone

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


# --------------------------------------------------------------------------- #
# a rate limit must not outlive the tick that met it
# --------------------------------------------------------------------------- #

class _Resp:
    """Minimal stand-in for a requests response."""

    def __init__(self, headers=None, status_code=200, payload=None):
        self.headers = headers or {}
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def _io_client(monkeypatch, responses, **kwargs):
    """A client whose transport replays `responses`, recording every sleep."""
    from betmodel.providers import oddsapiio

    client = oddsapiio.OddsApiIoClient(["a-league"], credential="default", **kwargs)
    calls = iter(responses)

    class _Transport:
        # Replaces the whole client: HttpClient is a frozen dataclass, so its
        # `get` cannot be patched in place.
        def get(self, url, params=None):
            outcome = next(calls)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    client._http = _Transport()
    slept: list[float] = []
    monkeypatch.setattr(oddsapiio.time, "sleep", slept.append)
    monkeypatch.setenv("ODDS_API_IO_KEY", "test")
    return client, slept


def _limited(reset_in_seconds=None):
    from betmodel.providers import http

    headers = {}
    if reset_in_seconds is not None:
        when = datetime.now(timezone.utc) + timedelta(seconds=reset_in_seconds)
        headers["x-ratelimit-reset"] = when.isoformat().replace("+00:00", "Z")
        headers["x-ratelimit-remaining"] = "0"
    return http.HttpError("429", status=429, body="", headers=headers)


def test_a_429_now_parks_for_the_window_rather_than_a_guess(monkeypatch):
    """The reset moment reaches the handler, so the wait is the real one.

    It could not before: `HttpError` discarded the headers, so the only 429 path
    fell back on an escalating 60/120/180/240/300 -- the blind backoff this
    module's own docstring says not to do.
    """
    from betmodel.providers import oddsapiio

    client, slept = _io_client(monkeypatch, [_limited(8), _Resp(payload=[{"id": "1"}])])
    assert client.get("events") == [{"id": "1"}]
    assert len(slept) == 1
    assert 8 <= slept[0] <= 11, slept  # the window, plus the two-second margin


def test_a_reset_beyond_the_budget_refuses_instead_of_waiting(monkeypatch):
    """The tick is five minutes long and shares its runner with the closes.

    Waiting out a long window does not merely arrive late, it holds the
    concurrency group and cancels whatever queued behind it. Refusing now and
    letting the next tick ask costs nothing, because nothing was going to be
    captured in either case.
    """
    from betmodel.providers import oddsapiio

    client, slept = _io_client(monkeypatch, [_limited(3415)], max_park_seconds=120.0)
    with pytest.raises(oddsapiio.RateLimited, match="park budget"):
        client.get("events")
    assert slept == [], "it must not have waited at all"


def test_an_unreadable_reset_refuses_rather_than_guessing(monkeypatch):
    from betmodel.providers import oddsapiio

    client, slept = _io_client(monkeypatch, [_limited(None)])
    with pytest.raises(oddsapiio.RateLimited, match="no readable reset"):
        client.get("events")
    assert slept == []


def test_the_park_budget_spans_every_call_the_client_makes(monkeypatch):
    """Five waits each under a per-attempt cap still outlast the tick.

    That arithmetic is what turned a rate limit into a fifteen-minute hang. The
    budget is therefore cumulative *and* per client rather than per call: a
    capture makes three requests, and three separate per-call budgets bound
    nothing that has to fit inside the five-minute tick.
    """
    from betmodel.providers import oddsapiio

    client, slept = _io_client(
        monkeypatch,
        [_limited(20), _Resp(payload=[{"id": "1"}]), _limited(20), _Resp()],
        max_park_seconds=30.0,
    )
    assert client.get("events") == [{"id": "1"}]      # first call parks 20-ish
    with pytest.raises(oddsapiio.RateLimited, match="park budget"):
        client.get("odds/multi")                       # second has nothing left
    assert len(slept) == 1 and sum(slept) <= 30.0, slept


def test_a_window_that_resets_inside_the_tick_is_waited_out(monkeypatch):
    """A 68-second reset is a burst, and riding it out costs a fifth of a tick.

    Measured: this provider answered one 429 with a 68s reset and another with
    3415s. Refusing both would throw away the requests the first would have won,
    and an earlier 60s budget did exactly that, by eight seconds.
    """
    from betmodel.providers import oddsapiio

    client, slept = _io_client(monkeypatch, [_limited(68), _Resp(payload=[1])])
    assert client.get("events") == [1]
    assert 68 <= sum(slept) <= 71, slept


def test_a_rate_limit_is_a_refusal_and_the_other_provider_still_runs(
    tmp_path, monkeypatch
):
    """Same handling as a quota refusal: counted, reported, survived.

    It used to escape as an unhandled error and fail the step.
    """
    from betmodel.config import load_league
    from betmodel.fixtures.upcoming import Fixture
    from betmodel.odds import capture_open as co
    from betmodel.providers import oddsapiio

    seen: list[str] = []

    def _limit(*_a, **_k):
        raise oddsapiio.RateLimited("window spent")

    def _anchor(*_a, **_k):
        seen.append("theoddsapi")
        return [], [], 0

    # Both providers must have something to do, or the second one is skipped for
    # an unrelated reason and the test passes without testing anything.
    def _pending(_league, _config, *, books, **_kwargs):
        return [co.Pending(
            Fixture(home="A", away="B", round="1",
                    kickoff=datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc)),
            tuple(books),
        )]

    monkeypatch.setattr(co, "pending_fixtures", _pending)
    monkeypatch.setattr(co, "_capture_oddsapiio", _limit)
    monkeypatch.setattr(co, "_capture_theoddsapi", _anchor)
    stats = co.capture_opens(
        "csl", load_league("csl"), providers=("oddsapiio", "theoddsapi"),
        ignore_schedule=True, now=datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc),
    )
    assert stats["refused"] == 1
    assert seen == ["theoddsapi"], "the anchor must still be polled"
