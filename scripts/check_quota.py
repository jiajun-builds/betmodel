#!/usr/bin/env python3
"""Report the remaining allowance on every The Odds API account, spending nothing.

The quota lives in the response headers of ``/sports``, which is free, so this
answers "how much is left on which account" without touching the metered
endpoints. It exists because that question could not be answered at all: the
capture logs the number only when it is already below the floor, which is the
one moment it is too late to act on.

Keys are never printed. Accounts are identified by a short digest of the key so
that two secrets holding the *same* account are visible as such -- the by-role
secrets predate the by-league ones and may be the same two accounts renamed.
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from betmodel.providers import theoddsapi  # noqa: E402

#: Any sport key works; the headers do not depend on it.
PROBE_SPORT = "soccer_epl"


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def main() -> int:
    names = sorted(
        n for n in os.environ
        if n.startswith("THE_ODDS_API_KEY") and os.environ[n].strip()
    )
    if not names:
        print("no THE_ODDS_API_KEY* in the environment", file=sys.stderr)
        return 1

    print(f"{'secret':28} {'account':9} {'remaining':>9} {'used':>6}")
    for name in names:
        key = os.environ[name].strip()
        credential = name[len("THE_ODDS_API_KEY"):].lstrip("_").lower() or "default"
        try:
            quota = theoddsapi.TheOddsApiClient(
                PROBE_SPORT, credential=credential
            ).quota()
            remaining, used = quota.remaining, quota.used
        except Exception as exc:  # noqa: BLE001
            remaining, used = f"error: {exc}", ""
        print(f"{name:28} {digest(key):9} {str(remaining):>9} {str(used):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
