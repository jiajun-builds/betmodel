#!/usr/bin/env python3
"""Alert when a league's configured de-bias is not reaching its published signals.

Falling back to raw probabilities for one fixture is normal and deliberate: no
anchor book has opened that match yet, and `signals.debias.apply` says so in its
own docstring. The label on each signal records which path ran, so nothing is
hidden.

A whole league falling back is a different condition wearing the same clothes. It
means the anchor is not being captured at all, and the league has been publishing
uncalibrated probabilities under a config that says otherwise. That happened here:
the Chinese Super League ran raw for three days because its anchor poll was being
refused by its own quota floor, and the only thing that surfaced it was reading
the config and the output side by side by hand.

The correction is not cosmetic. Measured on six fixtures, de-biasing moved EV by
2.3 to 6.4 percentage points, always downward, against a signal threshold of 20 --
enough to turn a fixture that should not fire into one that does.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

#: Below this share of anchored signals, a league configured for market_anchor is
#: not merely waiting on a few unopened fixtures. Deliberately forgiving: early in
#: a round most fixtures legitimately have no anchor yet, and an alarm that fires
#: on a normal Monday gets muted.
MIN_ANCHORED_SHARE = 0.5

ANCHORED = "market_anchor"
RAW = "raw"


def assess(league: str, method: str, signals: list[dict]) -> tuple[bool, str]:
    """Decide for one league. Pure, so the threshold can be tested."""
    if method != ANCHORED:
        return True, f"{league}: de-bias is {method!r}; nothing to check"
    if not signals:
        return True, f"{league}: no priced fixture to judge"

    anchored = sum(1 for s in signals if s.get("model", {}).get("method") == ANCHORED)
    share = anchored / len(signals)
    if share >= MIN_ANCHORED_SHARE:
        return True, (f"{league}: {anchored}/{len(signals)} signals anchored "
                      f"({share:.0%})")
    return False, (
        f"{league} is configured for market_anchor but only {anchored} of "
        f"{len(signals)} published signals are anchored ({share:.0%}). The anchor "
        f"book's opening prices are not being captured, so the league is "
        f"publishing uncalibrated probabilities."
    )


def alert(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat):
        print("no telegram credentials; alerting skipped", file=sys.stderr)
        return
    try:
        subprocess.run(
            ["curl", "-sS", "--max-time", "20", "-o", "/dev/null",
             f"https://api.telegram.org/bot{token}/sendMessage",
             "--data-urlencode", f"chat_id={chat}",
             "--data-urlencode", f"text={text}"],
            check=True, capture_output=True)
    except Exception as exc:  # noqa: BLE001
        print(f"alert could not be sent: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from betmodel import paths
    from betmodel.config import available_leagues, load_league

    problems = []
    for league in available_leagues():
        config = load_league(league)
        if not config.publish.published:
            continue
        path = paths.for_league(league).public_json("signals")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            signals = json.load(fh).get("signals", [])
        ok, message = assess(league, config.signals.debias.method, signals)
        print(message)
        if not ok:
            problems.append(message)

    if problems and not args.dry_run:
        alert("betmodel: " + " | ".join(problems))
    # Reported, not fatal: the refresh has its own work and its own failure path.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
