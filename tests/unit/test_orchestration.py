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


def test_the_staleness_check_runs_whatever_happened_before_it():
    """A monitor that only runs when the thing it monitors succeeded is silent
    exactly when it is needed. If the xG step fails or never runs, that IS the
    condition to alert on."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/refresh.yml").read_text())
    steps = workflow["jobs"]["refresh"]["steps"]
    check = [s for s in steps if "staleness" in s.get("name", "").lower()]
    assert check, "no staleness step in the refresh workflow"
    assert check[0].get("if") == "always()"
    # And it must sit after the stage it watches, or it measures the old state.
    names = [s.get("name", "") for s in steps]
    assert names.index(check[0]["name"]) > names.index("xG")


def test_every_data_file_has_something_that_writes_it():
    """A data file nobody writes is a snapshot pretending to be state.

    Six of them survived the migration: the old CSL schedule and xG tables, a
    market comparison, and three Liga MX research files. They sat in `data/`
    with plausible names and timestamps frozen at the migration, which is exactly
    what a live file looks like the moment its pipeline stops. One was asked about
    directly -- "does the workflow update this?" -- and the honest answer was no,
    and had been no since the day it was copied in.

    The check is against the path helpers rather than a list, so a file added
    later is covered without editing this test.
    """
    import glob
    import os
    import re

    from betmodel import paths
    from betmodel.config import available_leagues

    source = pathlib.Path(paths.__file__).read_text()
    known = set(re.findall(r'os\.path\.join\(self\.root, "([^"]+)"\)', source))
    # Directories the helpers name, whose contents are managed as a unit.
    dirs = {n for n in known if not n.endswith(".csv")}

    orphans = []
    for league in available_leagues():
        root = paths.for_league(league).root
        for path in glob.glob(os.path.join(root, "*.csv")):
            name = os.path.basename(path)
            if name not in known and not any(d in path for d in dirs):
                orphans.append(f"{league}/{name}")
    assert not orphans, (
        "no path helper names these, so nothing in the pipeline writes them: "
        + ", ".join(sorted(orphans))
    )


# --------------------------------------------------------------------------- #
# the anchor rescue's position in the pipeline
# --------------------------------------------------------------------------- #

def _step_names(workflow: str, job: str) -> list[str]:
    spec = yaml.safe_load((ROOT / f".github/workflows/{workflow}.yml").read_text())
    return [s.get("name") or "" for s in spec["jobs"][job]["steps"]]


@pytest.mark.parametrize("workflow,job", [("capture", "capture"), ("refresh", "refresh")])
def test_the_anchor_is_fetched_before_anything_is_published(workflow, job):
    """Order is the whole design, not a detail.

    The engine refuses to fire an edge it could not calibrate, and the notifier
    warns about one. If publish ran before the rescue, every newly listed fixture
    would produce a warning first and a bet second -- two messages for one
    decision, the first of them wrong. Running the rescue first is what makes the
    warning mean "the fetch was tried and did not help" rather than "the fetch has
    not happened yet".
    """
    names = _step_names(workflow, job)
    anchor = next(i for i, n in enumerate(names) if "Anchor for a stranded edge" in n)
    publish = next(i for i, n in enumerate(names) if n in {"Publish", "Republish signals"})
    assert anchor < publish, f"{workflow}: the rescue must precede publish"


def test_the_rescue_is_bought_only_by_a_capture_that_appended():
    """Spend is bounded by the trigger. On an idle tick no price appeared, so
    nothing can have become stranded, and a metered provider must not be touched
    to discover that."""
    spec = yaml.safe_load((ROOT / ".github/workflows/capture.yml").read_text())
    step = next(s for s in spec["jobs"]["capture"]["steps"]
                if "Anchor for a stranded edge" in (s.get("name") or ""))
    assert "steps.opens.outputs.appended == 'true'" in step["if"]


def test_a_tick_where_only_the_anchor_landed_still_republishes():
    """That tick is precisely the one that turns `unanchored` into a bet.

    Without the anchor step in the republish gate the rescue would write the
    anchor to disk and never publish the signal it unlocked, which is the whole
    point of fetching it.
    """
    text = (ROOT / ".github/workflows/capture.yml").read_text()
    republish = text.split("Republish signals", 1)[1].split("- name:", 1)[0]
    assert "steps.anchor.outputs.appended == 'true'" in republish
