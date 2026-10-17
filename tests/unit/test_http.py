"""HTTP policy.

The one that matters: a 403 means opposite things depending on the provider, so
retrying it is correct in one place and wastes a paid allowance in another.
"""

from __future__ import annotations

import pytest

from betmodel.providers import http


class _Resp:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _stub(client, responses):
    """Replace the transport with a scripted sequence."""
    seq = list(responses)
    calls = {"n": 0}

    def fetch(url, params, headers, timeout):
        calls["n"] += 1
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    object.__setattr__(client, "_fetch", fetch)
    return calls


def test_quota_metered_providers_never_retry_a_refusal():
    """odds-api.io answers 403 when a bookmaker is not entitled.

    The whole request fails; a retry spends more of a daily allowance to be
    refused again in exactly the same way.
    """
    c = http.client("oddsapiio", retry=http.NO_RETRY)
    calls = _stub(c, [_Resp(403)])
    with pytest.raises(http.HttpError) as err:
        c.get("https://api.example/odds")
    assert err.value.status == 403
    assert calls["n"] == 1


def test_a_rotating_pool_retries_403_and_can_succeed():
    """Each retry opens a new connection and so draws a new residential exit."""
    c = http.client("sofascore", retry=http.ROTATING_PROXY).with_retry(
        http.RetryPolicy(attempts=4, retry_statuses=frozenset({403}), backoff_base=0.0)
    )
    calls = _stub(c, [_Resp(403), _Resp(403), _Resp(200, {"ok": True})])
    assert c.json("https://api.sofascore.com/x") == {"ok": True}
    assert calls["n"] == 3


def test_retries_are_bounded():
    c = http.client("sofascore", retry=http.RetryPolicy(
        attempts=3, retry_statuses=frozenset({403}), backoff_base=0.0))
    calls = _stub(c, [_Resp(403)] * 3)
    with pytest.raises(http.HttpError, match="after 3 attempt"):
        c.get("https://api.sofascore.com/x")
    assert calls["n"] == 3


def test_transient_policy_does_not_retry_403():
    c = http.client("theoddsapi", retry=http.RetryPolicy(
        attempts=3, retry_statuses=http.TRANSIENT_ONLY.retry_statuses, backoff_base=0.0))
    calls = _stub(c, [_Resp(403)])
    with pytest.raises(http.HttpError):
        c.get("https://api.example/x")
    assert calls["n"] == 1, "a 403 here is an auth answer, not noise"


@pytest.mark.parametrize("status", [520, 521, 522, 523, 524])
def test_thesportsdb_retries_a_cloudflare_origin_error(status):
    """TheSportsDB is behind Cloudflare; a 52x is its edge failing to reach the
    origin, and one of them on round one used to fail the whole fixtures walk."""
    from dataclasses import replace

    from betmodel.providers import thesportsdb

    c = http.client("thesportsdb",
                    retry=replace(thesportsdb.RETRY, backoff_base=0.0))
    calls = _stub(c, [_Resp(status), _Resp(200, {"events": []})])
    assert c.json("https://www.thesportsdb.com/x") == {"events": []}
    assert calls["n"] == 2


def test_retry_after_is_honoured_only_when_opted_in_and_capped():
    """Cloudflare's rate limit answers 429 with Retry-After: 119. Backing off five
    seconds lands inside the block every time, so the wait has to follow the header,
    but only for a provider that opted in and never past its cap."""
    plain = http.RetryPolicy(attempts=3, backoff_base=5.0)
    assert plain.delay(1, "119") == 5.0

    patient = http.RetryPolicy(attempts=3, backoff_base=5.0, backoff_max=60.0,
                              retry_after_max=150.0)
    assert patient.delay(1, "119") == 119.0
    assert patient.delay(1, "600") == 150.0
    assert patient.delay(1, None) == 5.0
    assert patient.delay(1, "Wed, 21 Oct 2026 07:28:00 GMT") == 5.0
    assert patient.delay(2, "1") == 10.0, "never shorter than the backoff"


def test_a_rate_limited_request_waits_out_the_block(monkeypatch):
    slept = []
    monkeypatch.setattr(http.time, "sleep", slept.append)
    c = http.client("thesportsdb", retry=http.RetryPolicy(
        attempts=2, retry_statuses=frozenset({429}), retry_after_max=150.0))
    calls = _stub(c, [_Resp(429, headers={"Retry-After": "119"}),
                      _Resp(200, {"events": []})])
    assert c.json("https://www.thesportsdb.com/x") == {"events": []}
    assert calls["n"] == 2
    assert slept == [119.0]


def test_a_200_carrying_html_is_an_error_not_data():
    """A challenge page is served with status 200."""
    c = http.client("sofascore")
    _stub(c, [_Resp(200, None, text="<html>are you a robot</html>")])
    with pytest.raises(http.HttpError, match="not JSON"):
        c.json("https://api.sofascore.com/x")


def test_api_keys_never_reach_a_log_line_or_an_exception():
    c = http.client("theoddsapi")
    _stub(c, [_Resp(500)])
    with pytest.raises(http.HttpError) as err:
        c.get("https://api.example/v4/odds?apiKey=SUPERSECRET&regions=us")
    assert "SUPERSECRET" not in str(err.value)
    assert "<redacted>" in str(err.value)


def test_proxy_is_opt_in_per_provider(monkeypatch):
    monkeypatch.setenv("SOFASCORE_PROXY_URL", "http://u:p@host:1")
    assert http.client("sofascore", proxy_env="SOFASCORE_PROXY_URL").proxy_url
    # A provider that does not declare a proxy never picks one up by accident.
    assert http.client("theoddsapi").proxy_url is None


def test_absent_proxy_config_means_direct(monkeypatch):
    """Same code path on a laptop with a residential IP and on a proxied runner."""
    monkeypatch.delenv("SOFASCORE_PROXY_URL", raising=False)
    assert http.client("sofascore", proxy_env="SOFASCORE_PROXY_URL").proxy_url is None
