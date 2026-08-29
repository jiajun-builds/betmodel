"""Alert when the xG feed falls behind the results feed.

The failure this exists for is silent by construction. The xG merge never
erases, so a run that fetches nothing keeps the previous values and every
downstream stage rebuilds green on stale data. One fetcher sat wedged for ten
days and nothing anywhere reported a problem.

**The comparison is against results, not against the xG feed's own state.** When
a fetcher dies the xG side freezes whole, so any self-consistency check on it
alone sees nothing wrong. Results come from a different provider that keeps
running, and the gap between the two is what makes the failure visible.

**Two conditions, both required.** The feed must be behind by more than
``stale_days``, *and* at least ``min_missing`` played matches must sit past the
xG frontier. The second is what makes it usable: a provider does not cover every
fixture, so an isolated match with a result and no xG is normal and alerting on
it trains the reader to ignore the channel. A dead feed strands a whole round.

Fail-open. A missing token or an unreachable Telegram logs and returns, because a
monitor that can fail the run it monitors is worse than no monitor.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from betmodel import paths
from betmodel.config.schema import LeagueConfig
from betmodel.dates import parse_date_only_series
from betmodel.notify.telegram import CHAT_ENV, TOKEN_ENV, send

log = logging.getLogger(__name__)

DEFAULT_STALE_DAYS = 3
DEFAULT_MIN_MISSING = 3

STALE_DAYS_ENV = "XG_STALE_DAYS"


def measure(league: str, *, matches_path: str | None = None) -> dict:
    """How far the xG feed is behind, and how much is stranded past it."""
    path = matches_path or paths.for_league(league).matches_csv
    frame = pd.read_csv(path)
    frame["MatchDate"] = parse_date_only_series(frame["Date"])
    played = frame[pd.to_numeric(frame["HG"], errors="coerce").notna()]
    with_xg = played[pd.to_numeric(played["HxG"], errors="coerce").notna()]

    if played.empty:
        return {"gap_days": 0, "stranded": 0, "results_to": None, "xg_to": None}

    results_to = played["MatchDate"].max()
    xg_to = with_xg["MatchDate"].max() if not with_xg.empty else None
    if xg_to is None:
        return {
            "gap_days": int((results_to - played["MatchDate"].min()).days),
            "stranded": int(len(played)),
            "results_to": results_to.strftime("%Y-%m-%d"), "xg_to": None,
        }
    return {
        "gap_days": int((results_to - xg_to).days),
        # Played matches past the xG frontier, which is what a dead feed strands.
        "stranded": int((played["MatchDate"] > xg_to).sum()),
        "results_to": results_to.strftime("%Y-%m-%d"),
        "xg_to": xg_to.strftime("%Y-%m-%d"),
    }


def is_stale(reading: dict, *, stale_days: int, min_missing: int) -> bool:
    return reading["gap_days"] > stale_days and reading["stranded"] >= min_missing


def check(
    league: str,
    config: LeagueConfig,
    *,
    stale_days: int | None = None,
    min_missing: int = DEFAULT_MIN_MISSING,
    dry_run: bool = False,
    matches_path: str | None = None,
) -> dict:
    """Measure, and alert if both conditions hold."""
    if stale_days is None:
        try:
            stale_days = int(os.environ.get(STALE_DAYS_ENV, "") or DEFAULT_STALE_DAYS)
        except ValueError:
            stale_days = DEFAULT_STALE_DAYS

    reading = measure(league, matches_path=matches_path)
    reading["stale"] = is_stale(reading, stale_days=stale_days, min_missing=min_missing)
    log.info(
        "%s: results to %s, xG to %s, gap %d day(s), %d stranded -> %s",
        league, reading["results_to"], reading["xg_to"], reading["gap_days"],
        reading["stranded"], "STALE" if reading["stale"] else "ok",
    )
    if not reading["stale"] or dry_run:
        return reading

    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if not (token and chat_id):
        log.warning("%s: xG is stale but %s or %s is unset", league, TOKEN_ENV, CHAT_ENV)
        return reading

    send(token, chat_id, "\n".join([
        "⚠️ <b>xG 数据陈旧</b>",
        f"<b>{config.name}</b>",
        f"比分更新至: {reading['results_to']}",
        f"xG 更新至: {reading['xg_to'] or '无'}",
        f"落后: <b>{reading['gap_days']} 天</b>，{reading['stranded']} 场已踢比赛没有 xG",
        "模型仍会在旧数据上重新拟合并输出绿色状态。",
    ]))
    return reading
