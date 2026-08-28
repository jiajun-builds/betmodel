"""Tests for the stale-xG alert.

Two failure modes matter, and they pull in opposite directions:

* **Must fire** on a dead fetcher. Pinned by ``test_dead_fetcher_alerts``, which
  replays the real July 2026 outage — xG frozen at round 19 (2026-07-18) while the
  results feed had already moved to round 20 (2026-07-26).
* **Must stay quiet** when a single played match legitimately has no xG. SofaScore
  does not cover every CSL fixture (match 16484884, a round-18 makeup, has full stats
  and no Expected-goals item), and that one hole alone pushed the day gap to exactly 3
  for three days. A monitor that cries wolf forever gets muted, so
  ``test_isolated_uncovered_match_stays_quiet`` pins it.

The second is why the check needs the match count as well as the day gap.

Runnable either way::

    pytest tests/test_xg_freshness.py          # if pytest is installed
    python tests/test_xg_freshness.py          # no pytest needed
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csl.xg.check_freshness import check, run  # noqa: E402

LEAGUE_HEADER = "Country,League,Season,Round,Date,Time,Home,Away,HxG,AxG,HG,AG"
XG_HEADER = "match_id,round,date,home_team,away_team,home_score,away_score,home_xg,away_xg,status"


def _league_csv(path: str, matches: list[tuple[str, str]]) -> str:
    """Write a results CSV from ``(date, round)`` pairs; every row carries a score."""
    lines = [LEAGUE_HEADER]
    for i, (day, rnd) in enumerate(matches):
        lines.append(f"China,Super League,2026,{rnd},{day},11:00,Home{i},Away{i},,,1,0")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _xg_csv(path: str, matches: list[tuple[str, str, bool]]) -> str:
    """Write an xG CSV from ``(date, round, has_xg)`` triples."""
    lines = [XG_HEADER]
    for i, (day, rnd, has_xg) in enumerate(matches):
        xg = "1.50,0.80" if has_xg else ","
        lines.append(f"{i},{rnd},{day},Home{i},Away{i},1,0,{xg},Ended")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _fixture(tmp: str, league: list, xg: list) -> tuple[str, str]:
    return (
        _league_csv(os.path.join(tmp, "league.csv"), league),
        _xg_csv(os.path.join(tmp, "xg.csv"), xg),
    )


def test_dead_fetcher_alerts() -> None:
    """The real July 2026 outage: a whole round played, xG frozen 8 days back."""
    with tempfile.TemporaryDirectory() as tmp:
        league, xg = _fixture(
            tmp,
            [("2026-07-18", "19")] * 8 + [("2026-07-26", "20")] * 8,
            [("2026-07-18", "19", True)] * 8 + [("2026-07-26", "20", False)] * 8,
        )
        gap, missing, _, _ = check(league_csv=league, xg_csv=xg)
        assert gap == 8, gap
        assert missing == 8, missing
        assert run(league_csv=league, xg_csv=xg, dry_run=True) == 0  # dry run sends nothing
        # Without a token nothing goes out, but the verdict above is what gates the send.


def test_isolated_uncovered_match_stays_quiet() -> None:
    """One played match SofaScore has no xG for must not alert, however old it gets."""
    with tempfile.TemporaryDirectory() as tmp:
        league, xg = _fixture(
            tmp,
            [("2026-07-11", "18")] * 8 + [("2026-07-14", "18")],
            [("2026-07-11", "18", True)] * 8 + [("2026-07-14", "18", False)],
        )
        gap, missing, _, _ = check(league_csv=league, xg_csv=xg)
        assert gap == 3, gap
        assert missing == 1, missing  # below MIN_MISSING_MATCHES -> no alert


def test_healthy_feed_is_quiet() -> None:
    """xG level with results: zero gap, nothing past the frontier."""
    with tempfile.TemporaryDirectory() as tmp:
        league, xg = _fixture(
            tmp,
            [("2026-07-26", "20")] * 8,
            [("2026-07-26", "20", True)] * 8,
        )
        gap, missing, _, _ = check(league_csv=league, xg_csv=xg)
        assert gap == 0, gap
        assert missing == 0, missing


def test_offseason_break_is_quiet() -> None:
    """A month-long break freezes both feeds together, so the gap stays 0."""
    with tempfile.TemporaryDirectory() as tmp:
        league, xg = _fixture(
            tmp,
            [("2026-05-31", "15")] * 8,
            [("2026-05-31", "15", True)] * 8,
        )
        gap, missing, _, _ = check(league_csv=league, xg_csv=xg)
        assert gap == 0, gap
        assert missing == 0, missing


def test_unreadable_input_returns_none() -> None:
    """A monitor that cannot see must stay quiet rather than raise."""
    with tempfile.TemporaryDirectory() as tmp:
        league, _ = _fixture(tmp, [("2026-07-26", "20")], [("2026-07-26", "20", True)])
        assert check(league_csv=league, xg_csv=os.path.join(tmp, "missing.csv")) is None
        assert run(league_csv=league, xg_csv=os.path.join(tmp, "missing.csv")) == 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - test runner reports everything
            failures.append((test.__name__, exc))
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
