#!/usr/bin/env python3
"""Gate G0: can a CI runner reach SofaScore through a residential proxy?

SofaScore sits behind Cloudflare and refuses datacenter IPs with HTTP 403.
``curl_cffi`` impersonates a browser TLS handshake, which is necessary but not
sufficient: the source IP still has to look residential. Every automated
refresh for every league depends on the answer, so it is settled with a cheap
experiment before any design leans on it.

Run it in three places and compare:

    # 1. laptop, no proxy   -> expect 200. Proves the endpoints and the client work.
    python scripts/spike_sofascore_proxy.py

    # 2. laptop, via proxy  -> expect 200. Proves the proxy credentials work.
    SOFASCORE_PROXY_URL='http://user:pass@host:port' python scripts/spike_sofascore_proxy.py

    # 3. GitHub Actions, via proxy -> the actual question.
    #    Without the proxy the same job should 403, which is the control.

Exit code is 0 only if every probe returned usable data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

API = "https://api.sofascore.com/api/v1"

# One probe per capability the pipeline actually needs, for both leagues. A 200
# on the cheap endpoint but a block on the expensive one would be a false pass,
# so the event listing is included and its season id is resolved at runtime
# rather than pinned, which would rot every season.
TOURNAMENTS = {"ligamx": 11621, "csl": 649}


def _client(proxy: str | None):
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        sys.exit(
            "curl_cffi is not installed. It is what impersonates the browser TLS "
            "handshake; plain requests cannot pass Cloudflare here.\n"
            "  pip install 'curl_cffi>=0.7'"
        )
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return cffi, proxies


def build_probes(cffi, proxies, timeout: float) -> list[tuple[str, str]]:
    """Seasons endpoint first, then a real event listing for each league."""
    probes: list[tuple[str, str]] = []
    for league, tid in TOURNAMENTS.items():
        seasons_url = f"{API}/unique-tournament/{tid}/seasons"
        probes.append((f"{league}: seasons", seasons_url))
        try:
            r = cffi.get(seasons_url, impersonate="chrome", proxies=proxies, timeout=timeout)
            season_id = r.json()["seasons"][0]["id"]
        except Exception:
            # Leave a probe that will fail loudly rather than silently skipping
            # the expensive endpoint, which is the one that matters.
            probes.append((f"{league}: events (season id unresolved)", seasons_url + "/UNRESOLVED"))
            continue
        probes.append((
            f"{league}: events (season {season_id})",
            f"{API}/unique-tournament/{tid}/season/{season_id}/events/last/0",
        ))
    return probes


def probe(cffi, proxies, name: str, url: str, timeout: float) -> dict:
    started = time.monotonic()
    result = {"probe": name, "url": url}
    try:
        r = cffi.get(url, impersonate="chrome", proxies=proxies, timeout=timeout)
        result["status"] = r.status_code
        result["ms"] = round((time.monotonic() - started) * 1000)
        if r.status_code == 200:
            try:
                body = r.json()
                # A 200 carrying a Cloudflare interstitial is still a failure.
                result["ok"] = isinstance(body, dict) and bool(body)
                result["keys"] = sorted(body)[:4] if isinstance(body, dict) else None
            except Exception:
                result["ok"] = False
                result["note"] = "200 but body was not JSON (challenge page?)"
        else:
            result["ok"] = False
    except Exception as exc:  # noqa: BLE001 - the failure mode is the datapoint
        result["ok"] = False
        result["ms"] = round((time.monotonic() - started) * 1000)
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return result


def egress_identity(cffi, proxies, timeout: float) -> dict:
    """Which IP do we leave from, and who owns it?

    Half of gate G0 is not "does it work" but "is this really residential".
    A datacenter ASN behind a plan sold as residential is a refund conversation,
    and it is the difference between a fixable purchase and a fixable design.
    Known datacenter operators are named because the free tier that motivated
    this check exited via Leaseweb and was refused on every probe.
    """
    out: dict = {}
    try:
        r = cffi.get("https://ipinfo.io/json", impersonate="chrome",
                     proxies=proxies, timeout=timeout)
        d = r.json()
        out["ip"] = d.get("ip", "")
        out["org"] = d.get("org", "")
        out["city"] = d.get("city", "")
        out["country"] = d.get("country", "")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return out

    org = out.get("org", "").lower()
    hosting = (
        "leaseweb", "digitalocean", "linode", "vultr", "ovh", "hetzner", "aws",
        "amazon", "google", "microsoft", "azure", "oracle", "contabo", "scaleway",
        "choopa", "quadranet", "hostwinds", "m247", "datacamp", "cloudflare",
    )
    hit = next((h for h in hosting if h in org), None)
    out["looks_like_datacenter"] = bool(hit)
    out["matched"] = hit or ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    proxy = os.environ.get("SOFASCORE_PROXY_URL", "").strip() or None
    cffi, proxies = _client(proxy)

    # Never print the proxy URL: it carries credentials.
    where = "via proxy" if proxy else "direct (no proxy configured)"
    ident = egress_identity(cffi, proxies, args.timeout)
    results = [
        probe(cffi, proxies, n, u, args.timeout)
        for n, u in build_probes(cffi, proxies, args.timeout)
    ]
    passed = sum(1 for r in results if r.get("ok"))

    if args.json:
        print(json.dumps(
            {"proxy_configured": bool(proxy), "egress": ident, "results": results},
            indent=2,
        ))
    else:
        print(f"SofaScore reachability {where}")
        if ident.get("error"):
            print(f"  egress: could not identify ({ident['error']})")
        else:
            print(f"  egress: {ident.get('ip','?')}  {ident.get('org','?')}  "
                  f"{ident.get('city','')} {ident.get('country','')}")
            if ident.get("looks_like_datacenter"):
                print(f"  WARNING: {ident['matched']} is a hosting provider. This is a "
                      "datacenter IP,\n           not residential, and SofaScore is "
                      "expected to refuse it.")
        for r in results:
            mark = "PASS" if r.get("ok") else "FAIL"
            detail = r.get("error") or r.get("note") or f"HTTP {r.get('status')}"
            print(f"  [{mark}] {r['probe']:32s} {r.get('ms','?')}ms  {detail}")
        print(f"  {passed}/{len(results)} probes returned usable data")

    if passed != len(results):
        print(
            "\nA 403 here means the egress IP was rejected, not that the client is "
            "wrong.\nIf this ran on a CI runner without a proxy, that is the expected "
            "control result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
