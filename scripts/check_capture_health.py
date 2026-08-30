#!/usr/bin/env python3
"""Alert when the capture ticks stop working, without alerting on every failure.

The capture workflow fires every five minutes. A per-failure alert would be
noise, and a data-based check ("has a row been appended lately?") would be wrong:
long quiet stretches are normal, because an idle tick with no fixture in its
window correctly appends nothing.

What is never normal is the workflow not running, or running and never once
succeeding. Both mean the pipeline is down rather than idle, and a missed capture
is unrecoverable -- an opening line not taken is gone, and no provider sells it
back. So this is a dead-man's switch on the runs, not on the rows.

Run from the daily refresh, which already holds the Telegram credentials:

    python scripts/check_capture_health.py --window-hours 24
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

WORKFLOW = "capture.yml"

#: Conclusions that mean the tick did its job.
GOOD = {"success"}
#: Concurrency cancels an overlapping tick by design; that is the mechanism
#: working, not a fault, so it counts as neither good nor bad.
IGNORED = {"cancelled", "skipped", None, ""}


def assess(runs: list[dict], *, window_hours: int) -> tuple[bool, str]:
    """Decide from a list of runs. Pure, so the thresholds can be tested.

    ``runs`` are dicts with ``conclusion`` and ``createdAt``, newest first, as
    the GitHub API returns them.
    """
    if not runs:
        return False, (
            f"no capture run started in {window_hours}h. The timer that dispatches "
            "them is not firing, so no opening or closing line is being taken."
        )

    good = sum(1 for r in runs if r.get("conclusion") in GOOD)
    bad = sum(
        1 for r in runs
        if r.get("conclusion") not in GOOD and r.get("conclusion") not in IGNORED
    )
    if good == 0:
        return False, (
            f"{len(runs)} capture runs in {window_hours}h and not one succeeded "
            f"({bad} failed). Captures are being attempted and every one is "
            "failing, so no line is being taken."
        )
    return True, f"capture healthy: {good} succeeded, {bad} failed in {window_hours}h"


def recent_runs(repo: str, window_hours: int, limit: int = 100) -> list[dict]:
    """Runs of the capture workflow inside the window, newest first."""
    out = subprocess.run(
        ["gh", "run", "list", "--repo", repo, "--workflow", WORKFLOW,
         "--limit", str(limit), "--json", "conclusion,createdAt"],
        capture_output=True, text=True, check=True,
    ).stdout
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    runs = []
    for run in json.loads(out or "[]"):
        created = run.get("createdAt", "")
        try:
            when = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when >= cutoff:
            runs.append(run)
    return runs


def alert(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat):
        print("no telegram credentials; alerting skipped", file=sys.stderr)
        return
    # Best-effort, like every other alert here: a send that fails must not
    # replace the condition it was reporting.
    try:
        subprocess.run(
            ["curl", "-sS", "--max-time", "20", "-o", "/dev/null",
             f"https://api.telegram.org/bot{token}/sendMessage",
             "--data-urlencode", f"chat_id={chat}",
             "--data-urlencode", f"text={text}"],
            check=True, capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"alert could not be sent: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.repo:
        print("no repository; pass --repo or set GITHUB_REPOSITORY", file=sys.stderr)
        return 2

    runs = recent_runs(args.repo, args.window_hours)
    ok, message = assess(runs, window_hours=args.window_hours)
    print(message)
    if ok:
        return 0
    if not args.dry_run:
        alert(f"betmodel: {message}")
    # Reported, not fatal. The refresh that hosts this check has its own work,
    # and failing it would confuse a capture outage with a refresh outage.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
