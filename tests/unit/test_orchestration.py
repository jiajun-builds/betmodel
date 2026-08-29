"""The parts of the pipeline that live in YAML.

Configuration and schedule are two halves of one decision here, and they are
written in different files by different tools. A test is the only thing that can
hold them together.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from betmodel.config import available_leagues, load_league

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools/capture-timer"


def _crons() -> list[str]:
    text = (WORKER / "wrangler.toml").read_text()
    match = re.search(r"crons\s*=\s*\[(.*?)\]", text, re.S)
    assert match, "the Worker declares no cron"
    return re.findall(r'"([^"]+)"', match.group(1))


def _cron_minutes(expression: str) -> int:
    match = re.match(r"\*/(\d+) ", expression)
    assert match, f"cannot read a cadence from {expression!r}"
    return int(match.group(1))


def test_the_trigger_is_fast_enough_for_the_close_window_every_league_configured():
    """The coupling that must not be broken by editing one file.

    Narrowing the close window without speeding the trigger leaves fixtures with
    no close at all, and a close missed is unrecoverable at any price. Measured
    before the change: a 60-minute window on a 10-minute trigger reached its
    target band on 27% of fixtures, worst case 33.6 minutes out. Three ticks per
    fixture is the floor.
    """
    cadence = min(_cron_minutes(c) for c in _crons())
    for league in available_leagues():
        close = load_league(league).odds.close
        ticks = close.window_minutes / cadence
        assert ticks >= 3, (
            f"{league}: a {close.window_minutes:.0f}-minute window on a "
            f"{cadence}-minute trigger is only {ticks:.1f} ticks per fixture"
        )


def test_the_close_window_is_the_same_for_every_league():
    """It differed only because one trigger was unreliable. That reason is gone."""
    windows = {load_league(l).odds.close.window_minutes for l in available_leagues()}
    assert len(windows) == 1, f"windows have diverged again: {windows}"


def test_the_worker_lead_is_wider_than_the_capture_window():
    """Firing early costs nothing, since the capture re-reads the window and
    declines. Firing late costs a close."""
    lead = int(re.search(r"CLOSE_LEAD_MINUTES\s*=\s*(\d+)",
                         (WORKER / "src/worker.js").read_text()).group(1))
    for league in available_leagues():
        assert lead >= load_league(league).odds.close.window_minutes


def test_the_worker_discovers_leagues_rather_than_listing_them():
    """Adding a league must not need a change here, which is only true if the
    manifest is read."""
    source = (WORKER / "src/worker.js").read_text()
    assert "public/index.json" in source
    for league in available_leagues():
        assert f'"{league}"' not in source, f"{league} is hardcoded in the Worker"


@pytest.mark.parametrize("workflow", ["capture", "refresh", "tests"])
def test_every_workflow_parses(workflow):
    yaml.safe_load((ROOT / f".github/workflows/{workflow}.yml").read_text())


def test_the_capture_workflow_republishes_only_when_something_was_appended():
    """Otherwise the published tree is rewritten several times an hour with an
    identical payload and a fresh timestamp."""
    text = (ROOT / ".github/workflows/capture.yml").read_text()
    assert "steps.opens.outputs.appended == 'true'" in text
    assert "steps.closes.outputs.appended == 'true'" in text


def test_the_published_commit_does_not_skip_ci():
    """The alert reads the previously committed signals file as its baseline, so
    a commit that never lands leaves the next run comparing against a stale
    snapshot."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/capture.yml").read_text())
    steps = workflow["jobs"]["capture"]["steps"]
    publish = [s for s in steps if s.get("with", {}).get("paths") == "public"]
    assert publish, "no step commits the published tree"
    assert all(s["with"]["skip_ci"] == "false" for s in publish)


def test_the_capture_job_installs_only_what_capture_needs():
    """A conda solve here is the difference between reaching the API in forty-five
    seconds and four minutes, for a price that moves."""
    text = (ROOT / ".github/workflows/capture.yml").read_text()
    assert "--no-deps" in text
    assert "scipy" not in text and "penaltyblog" not in text


def test_the_residential_stages_are_the_ones_given_the_proxy():
    """Which stages need it is data, in the league config."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/refresh.yml").read_text())
    steps = workflow["jobs"]["refresh"]["steps"]
    proxied = {
        s["name"].lower() for s in steps
        if "SOFASCORE_PROXY_URL" in (s.get("env") or {})
    }
    needed = {
        stage for league in available_leagues()
        for stage in load_league(league).residential_stages
    }
    for stage in needed:
        assert any(stage in name for name in proxied), f"{stage} runs without the proxy"


def test_the_capture_install_does_not_strip_dependencies_from_pandas():
    """--no-deps must apply to this package alone.

    Listing it beside the runtime dependencies applied it to all of them, so
    pandas installed without numpy and every capture failed on import. The
    workflow reported it because the step does not swallow its exit code.
    """
    text = (ROOT / ".github/workflows/capture.yml").read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("pip install")]
    assert lines, "no install line found"
    for line in lines:
        if "--no-deps" in line:
            assert "pandas" not in line, (
                "--no-deps is on the same command as pandas, which strips numpy"
            )
