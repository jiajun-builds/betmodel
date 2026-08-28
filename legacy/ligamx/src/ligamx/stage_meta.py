"""When each pipeline stage last actually ran.

The dashboard's `updated_at` answers "when were these JSON files written", which
CI re-answers every few minutes: `capture-odds.yml` re-runs the offline exports
after every odds capture. It says nothing about when the *inputs* were refreshed,
and two of them can only be refreshed by hand -- SofaScore bans datacenter IPs, so
`fixtures.mex_fixture` (and therefore the model fed by it) never runs in Actions.

So a fixture list three weeks stale is republished with a timestamp of "now"
several times an hour. These sidecars exist to keep that distinguishable: each
stage stamps one when it runs, the exporter reads them, and the dashboard can show
how old the data actually is rather than how recently it was copied.

Tracked in git, because the reader (export_dashboard) runs in CI while the writers
run locally -- an untracked stamp would never reach the machine that reads it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def stamp(path: str, key: str) -> str:
    """Record `key` = now (UTC, seconds) in the JSON sidecar at `path`.

    Merges into whatever is already there so two keys can share a file, and
    rewrites the file wholesale rather than appending, so a corrupt sidecar
    repairs itself on the next run instead of staying broken forever.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {}
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            payload = loaded
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {}

    payload[key] = now
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return now


def read(path: str, key: str) -> str | None:
    """Read `key` from the sidecar at `path`, or None if it isn't there.

    None, not "now": a missing stamp means the stage's last run is *unknown*, and
    defaulting to the current time is precisely the claim this module exists to
    stop making.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            val = json.load(fh).get(key)
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
        return None
    return str(val) if val else None
