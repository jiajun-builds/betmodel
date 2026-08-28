"""One HTTP layer for every provider, with the retry policy as an explicit choice.

Two constraints shape this, and both are easy to get wrong in a way that only
shows up in production.

**A 403 means opposite things depending on who said it.** From SofaScore behind a
rotating residential proxy it means "this exit IP is refused", and retrying is
correct because the next connection draws a different IP. From odds-api.io it
means "you asked for a bookmaker you are not entitled to", the whole request
fails, and retrying spends more of a daily allowance to be refused again. So
retry-on-403 is opt-in per provider, never a default.

**curl_cffi is imported lazily.** The odds capture jobs deliberately run on a
minimal ``pandas + requests`` install so a dispatch reaches the API in about
forty-five seconds instead of waiting on conda. Only SofaScore needs the browser
TLS impersonation, so importing it at module scope would tax every capture tick
for a dependency it never uses.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

log = logging.getLogger(__name__)

PLAIN = "plain"       # ordinary requests; fine for JSON APIs that do not fingerprint
BROWSER = "browser"   # curl_cffi with browser TLS impersonation


@dataclass(frozen=True)
class RetryPolicy:
    """When to try again, and how long to wait.

    ``attempts`` counts total tries, so 1 means no retry at all.
    """

    attempts: int = 1
    retry_statuses: frozenset[int] = field(default_factory=frozenset)
    backoff_base: float = 0.4
    backoff_max: float = 8.0
    retry_on_transport_error: bool = True

    def delay(self, attempt: int) -> float:
        return min(self.backoff_base * attempt, self.backoff_max)


#: Quota-metered providers. One shot at the request itself; a refusal is an
#: answer, not noise, and a retry would spend allowance to be refused again.
NO_RETRY = RetryPolicy(attempts=1)

#: Safe everywhere: these statuses do not consume a paid request, and a transport
#: error means the request never landed.
TRANSIENT_ONLY = RetryPolicy(
    attempts=3,
    retry_statuses=frozenset({408, 429, 500, 502, 503, 504}),
)

#: For a rotating residential pool. 403 is included because it describes the exit
#: IP rather than the request, and each retry opens a new connection and so draws
#: a new exit. Measured: individual exits in the pool are refused while others
#: from the same pool succeed on the very next attempt.
ROTATING_PROXY = RetryPolicy(
    attempts=4,
    retry_statuses=frozenset({403, 408, 429, 500, 502, 503, 504}),
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)


class HttpError(RuntimeError):
    """A request failed after exhausting its retry policy."""

    def __init__(self, message: str, *, status: int | None = None, attempts: int = 1):
        super().__init__(message)
        self.status = status
        self.attempts = attempts


def _redact(url: str) -> str:
    """Strip anything that looks like a key before a URL reaches a log line.

    Both odds providers pass the API key as a query parameter, so an unredacted
    URL in a CI log is a published credential.
    """
    out = url
    for marker in ("apiKey=", "api_key=", "apikey=", "key=", "token="):
        idx = out.lower().find(marker.lower())
        while idx != -1:
            end = out.find("&", idx)
            end = len(out) if end == -1 else end
            out = out[: idx + len(marker)] + "<redacted>" + out[end:]
            idx = out.lower().find(marker.lower(), idx + len(marker) + 10)
    return out


@dataclass(frozen=True)
class HttpClient:
    """A configured way of talking to one provider.

    Construct with :func:`client` rather than directly.
    """

    name: str
    transport: str = PLAIN
    retry: RetryPolicy = NO_RETRY
    timeout: float = 25.0
    proxy_env: str | None = None
    impersonate: str = "chrome"
    user_agent: str = DEFAULT_USER_AGENT

    # --- proxy ---------------------------------------------------------------
    @property
    def proxy_url(self) -> str | None:
        """The configured proxy, if this provider declares it needs one.

        Absent configuration means direct. That is deliberate: the same code then
        runs unchanged on a laptop with a residential IP and on a runner with a
        proxy secret, so there is no CI-only code path to rot.
        """
        if not self.proxy_env:
            return None
        return os.environ.get(self.proxy_env, "").strip() or None

    def _proxies(self) -> dict[str, str] | None:
        url = self.proxy_url
        return {"http": url, "https": url} if url else None

    # --- request -------------------------------------------------------------
    def _fetch(self, url: str, params, headers, timeout):
        proxies = self._proxies()
        merged = {"User-Agent": self.user_agent, **(headers or {})}
        if self.transport == BROWSER:
            # Imported here, not at module scope: the capture jobs run without it.
            try:
                from curl_cffi import requests as cffi
            except ImportError as exc:  # pragma: no cover - environment problem
                raise HttpError(
                    f"{self.name} needs curl_cffi for browser TLS impersonation. "
                    "A plain client is refused by this provider regardless of the "
                    "source IP.  pip install 'curl_cffi>=0.7'"
                ) from exc
            return cffi.get(
                url, params=params, headers=merged, proxies=proxies,
                timeout=timeout, impersonate=self.impersonate,
            )
        import requests

        return requests.get(
            url, params=params, headers=merged, proxies=proxies, timeout=timeout
        )

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ):
        """GET, honouring this provider's retry policy.

        Raises :class:`HttpError` once the policy is exhausted, so callers never
        have to remember to check a status code.
        """
        timeout = self.timeout if timeout is None else timeout
        attempts = max(1, self.retry.attempts)
        last_status: int | None = None
        last_detail = ""

        for attempt in range(1, attempts + 1):
            try:
                response = self._fetch(url, params, headers, timeout)
            except Exception as exc:  # noqa: BLE001 - any transport failure
                last_detail = f"{type(exc).__name__}: {exc}"[:200]
                if not self.retry.retry_on_transport_error or attempt == attempts:
                    raise HttpError(
                        f"{self.name}: {_redact(url)} failed after {attempt} "
                        f"attempt(s): {last_detail}",
                        attempts=attempt,
                    ) from exc
                time.sleep(self.retry.delay(attempt))
                continue

            status = response.status_code
            if 200 <= status < 300:
                if attempt > 1:
                    log.info(
                        "%s: %s succeeded on attempt %d", self.name, _redact(url), attempt
                    )
                return response

            last_status = status
            if status not in self.retry.retry_statuses or attempt == attempts:
                raise HttpError(
                    f"{self.name}: {_redact(url)} returned HTTP {status} "
                    f"after {attempt} attempt(s)",
                    status=status,
                    attempts=attempt,
                )
            log.debug(
                "%s: HTTP %d on attempt %d, retrying", self.name, status, attempt
            )
            time.sleep(self.retry.delay(attempt))

        raise HttpError(  # pragma: no cover - loop always returns or raises
            f"{self.name}: exhausted {attempts} attempts (last status {last_status})",
            status=last_status,
            attempts=attempts,
        )

    def json(self, url: str, **kwargs) -> Any:
        """GET and decode JSON.

        A 200 carrying HTML is a challenge page, not data, so it is an error here
        rather than a confusing failure three layers up.
        """
        response = self.get(url, **kwargs)
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise HttpError(
                f"{self.name}: {_redact(url)} returned HTTP "
                f"{response.status_code} but the body was not JSON "
                "(a challenge or error page?)",
                status=response.status_code,
            ) from exc

    def with_retry(self, retry: RetryPolicy) -> "HttpClient":
        """A copy with a different policy, for one unusual call."""
        return replace(self, retry=retry)


def client(
    name: str,
    *,
    transport: str = PLAIN,
    retry: RetryPolicy = NO_RETRY,
    timeout: float = 25.0,
    proxy_env: str | None = None,
) -> HttpClient:
    """Build a client for one provider.

    ``proxy_env`` names the environment variable holding a proxy URL, and is set
    only for providers that need a residential egress. Passing the variable name
    rather than the URL keeps credentials out of every call site and out of any
    object that might be logged or repr'd.
    """
    if transport not in (PLAIN, BROWSER):
        raise ValueError(f"unknown transport {transport!r}")
    return HttpClient(
        name=name, transport=transport, retry=retry, timeout=timeout, proxy_env=proxy_env
    )
