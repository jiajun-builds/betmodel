"""After each round, tell a person which of its games have no closing line.

The close is the yardstick for CLV, and it has a hard deadline: the fixture leaves
the pre-match feed at kickoff and no provider sells it back. Whether a close was
taken is settled the moment the match starts -- and until now nothing told anyone.

**It was missed for four days because a refused close looks like a quiet one.**
On 2026-09-26..28 the Liga MX Odds API account sat at 4 requests against a close
floor of 5. Every close tick for seven of matchday 10's nine fixtures found its
fixture, declined to spend, logged a WARNING inside a green step, and the capture
dead-man's switch -- which watches whether runs succeed -- had nothing to say.
This reads the outcome instead of the runs: once the round is over, which of its
games have a close?

**One message per round, once every game in it has been played**, listing each
with what was captured. Played means kicked off more than :data:`MATCH_LENGTH`
ago; the close was settled at kickoff, so waiting changes nothing in the report,
only when it arrives. A round with no label is grouped by its local matchday.

**A postponement does not hold the round open.** A game of the round still more
than :data:`ROUND_HOLD` away is treated as moved: the report counts it as still
to play, and it gets a report of its own once it has been played.

**Captured means what CLV means.** The same rule :mod:`betmodel.odds.reduce`
applies to fill the master table's closing columns: the latest close from the
finalisation book, if it was taken between kickoff and six hours before. One
taken earlier than the close window's target still yields a CLV and is shown as
early, not missing.

**Not gated on ``notify.telegram``.** That switch is about bet signals, and Liga
MX turned it off when it paused. A paused league's closes are exactly the record
that decides whether it unpauses (D35), so they are reported regardless.

**Recorded only after Telegram accepted it**, in ``close_report.csv``, which the
capture workflow commits. A send that fails is retried on the next tick; a run
whose commit never lands sends again, the same trade the signal alerts make.

Env: ``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``, and the league's Odds API key
for a free balance check when a close is missing.
"""

from __future__ import annotations

import csv
import html
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from betmodel import paths
from betmodel.config.schema import LeagueConfig
from betmodel.dates import local_matchday, stamp
from betmodel.notify import telegram
from betmodel.odds import capture_store
from betmodel.odds.capture_close import finalisation_book
from betmodel.odds.reduce import MAX_CLOSE_LEAD_HOURS
from betmodel.providers import theoddsapi

log = logging.getLogger(__name__)

#: How long after kickoff a game counts as played: ninety minutes, half-time and
#: stoppage time, rounded up.
MATCH_LENGTH = timedelta(hours=2)

#: A game of the round kicking off further out than this is a postponement, not
#: the rest of the round, and does not hold the report back. Four days, so that a
#: Liguilla tie's two legs -- one round label, about three days apart -- are one
#: report rather than a report and a 补赛.
ROUND_HOLD = timedelta(days=4)

#: How long after its last game a round may still be reported, and how far before
#: that last game the round reaches back. Short enough that the first deploy, or a
#: run after a long outage, does not replay weeks of rounds nobody can act on.
#: Measured from the round's end and not from each game, because per game it cut
#: a round in half: four days on, a Liga MX round's Friday games had aged out and
#: its Sunday ones had not. Longer than any round is long, so reaching back by it
#: takes the whole round and nothing before it -- which for a postponed game
#: played weeks later is the game alone.
LOOKBACK = timedelta(days=7)

CAPTURED = "captured"
EARLY = "early"
MISSING = "missing"

REPORT_COLUMNS = [
    "home_team", "away_team", "matchday", "season", "round",
    "kickoff_utc", "status", "close_lead_min", "reported_at",
]

ICON = {CAPTURED: "✅", EARLY: "⚠️", MISSING: "❌"}

Key = tuple[str, str, str]


@dataclass(frozen=True)
class Game:
    """One fixture, identified the way every other layer identifies it."""

    season: str
    round: str
    home: str
    away: str
    kickoff: datetime
    matchday: str

    @property
    def key(self) -> Key:
        return self.home, self.away, self.matchday

    @property
    def group(self) -> tuple[str, str, str]:
        """The round it is reported with: its round, or failing that its day."""
        if self.round:
            return self.season, self.round, ""
        return self.season, "", self.matchday


@dataclass(frozen=True)
class Outcome:
    """A played game and what its close turned out to be."""

    game: Game
    #: Minutes before kickoff of the close CLV will use, or None when there is none.
    lead_minutes: float | None
    status: str


@dataclass(frozen=True)
class RoundReport:
    season: str
    round: str
    matchday: str
    outcomes: tuple[Outcome, ...]
    #: Games of this round not yet played: postponed, or rescheduled.
    still_to_play: int
    #: Games of this round covered by an earlier report.
    reported_before: int

    @property
    def missing(self) -> int:
        return sum(1 for o in self.outcomes if o.status == MISSING)


# --------------------------------------------------------------------------- #
# what happened
# --------------------------------------------------------------------------- #

def load_games(
    league: str, config: LeagueConfig, *,
    matches_path: str | None = None, fixtures_path: str | None = None,
) -> list[Game]:
    """Every fixture with a kickoff, from the match table and the schedule.

    Both, because a fixture moves from one to the other at the daily refresh:
    until then it is only in the schedule, and after it only in the match table.
    A round straddles that moment every time it is played across a night.
    """
    lp = paths.for_league(league)
    games: dict[Key, Game] = {}
    for path in (matches_path or lp.matches_csv, fixtures_path or lp.upcoming_fixtures_csv):
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                home = (row.get("Home") or "").strip()
                away = (row.get("Away") or "").strip()
                raw = (row.get("kickoff_utc") or "").strip()
                if not (home and away and raw):
                    continue
                try:
                    kickoff = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
                game = Game(
                    season=(row.get("Season") or "").strip(),
                    round=(row.get("Round") or "").strip(),
                    home=home, away=away, kickoff=kickoff,
                    matchday=local_matchday(kickoff, config.timezone),
                )
                games[game.key] = game  # the schedule, read last, is the newer word
    return list(games.values())


def close_leads(
    league: str, config: LeagueConfig, *, history_path: str | None = None,
) -> dict[Key, float]:
    """Minutes before kickoff of the close CLV will be computed from, per fixture.

    The latest close from the finalisation book, as :func:`reduce.build_records`
    chooses it. Keyed on the matchday, so a rescheduled fixture is not credited
    with the close of the date it was moved off.
    """
    history = capture_store.load_history(
        history_path or paths.for_league(league).capture_history_csv
    )
    closes = history[
        (history["snapshot_type"] == "close")
        & (history["bookmaker"] == finalisation_book(config))
    ]
    if closes.empty:
        return {}
    kickoffs = pd.to_datetime(closes["commence_time"], utc=True, format="ISO8601",
                              errors="coerce")
    fetched = pd.to_datetime(closes["fetched_at"], utc=True, format="ISO8601",
                             errors="coerce")
    priced = closes[["home_odds", "draw_odds", "away_odds"]].apply(
        pd.to_numeric, errors="coerce").notna().all(axis=1)

    latest: dict[Key, tuple[pd.Timestamp, float]] = {}
    for home, away, kickoff, at, ok in zip(
        closes["home_team"], closes["away_team"], kickoffs, fetched, priced
    ):
        if not ok or pd.isna(kickoff) or pd.isna(at):
            continue
        key = (home, away, local_matchday(kickoff.to_pydatetime(), config.timezone))
        if key not in latest or at > latest[key][0]:
            latest[key] = (at, (kickoff - at).total_seconds() / 60.0)
    return {key: lead for key, (_, lead) in latest.items()}


def status(lead_minutes: float | None, config: LeagueConfig) -> str:
    if lead_minutes is None or not 0 <= lead_minutes <= MAX_CLOSE_LEAD_HOURS * 60:
        return MISSING
    if lead_minutes <= config.odds.close.target_minutes:
        return CAPTURED
    return EARLY


# --------------------------------------------------------------------------- #
# what is due
# --------------------------------------------------------------------------- #

def load_reported(path: str) -> set[Key]:
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row.get("home_team", ""), row.get("away_team", ""), row.get("matchday", ""))
            for row in csv.DictReader(handle)
        }


def due_rounds(
    games: list[Game], leads: dict[Key, float], reported: set[Key],
    config: LeagueConfig, *, now: datetime,
) -> list[RoundReport]:
    """Rounds whose every game has now been played. Pure, so the timing is testable."""
    by_round: dict[tuple[str, str, str], list[Game]] = defaultdict(list)
    for game in games:
        by_round[game.group].append(game)

    reports = []
    for (season, label, day), members in by_round.items():
        # Kicked off, but not yet over -- or kicking off soon enough to be the rest
        # of this round rather than a postponement.
        if any(now - MATCH_LENGTH < g.kickoff <= now + ROUND_HOLD for g in members):
            continue
        played = [g for g in members if g.kickoff <= now - MATCH_LENGTH]
        if not played:
            continue
        end = max(g.kickoff for g in played)
        if end < now - LOOKBACK:
            continue  # long finished
        fresh = sorted(
            (g for g in played if g.kickoff >= end - LOOKBACK and g.key not in reported),
            key=lambda g: g.kickoff,
        )
        if not fresh:
            continue
        reports.append(RoundReport(
            season=season, round=label, matchday=day,
            outcomes=tuple(
                Outcome(g, leads.get(g.key), status(leads.get(g.key), config))
                for g in fresh
            ),
            still_to_play=sum(1 for g in members if g.kickoff > now),
            reported_before=sum(1 for g in members if g.key in reported),
        ))
    return sorted(reports, key=lambda r: r.outcomes[0].game.kickoff)


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #

def _escape(text) -> str:
    return html.escape(str(text), quote=False)


def round_title(report: RoundReport) -> str:
    if not report.round:
        return report.matchday
    return f"第{report.round}轮" if report.round.isdigit() else report.round


def _lead_text(outcome: Outcome, config: LeagueConfig) -> str:
    if outcome.status == MISSING:
        return "无收盘价"
    text = f"开赛前 {outcome.lead_minutes:.0f} 分钟"
    if outcome.status == EARLY:
        text += f"（早于 {config.odds.close.target_minutes:g} 分钟目标）"
    return text


def balance(config: LeagueConfig) -> str | None:
    """The close account's balance, from the free ``/sports`` probe.

    Read now, not at kickoff, so it is evidence rather than proof: but the balance
    only falls between resets, so an account under its floor now was almost
    certainly under it then. Fail-open: a report must never fail on its footnote.
    """
    close = config.odds.close
    provider = config.odds.providers.get("theoddsapi")
    env = theoddsapi.key_env_for(close.credential)
    if provider is None:
        return None
    try:
        quota = theoddsapi.TheOddsApiClient(
            provider.require("sport_key"),
            credential=close.credential,
            base_url=provider.get("base_url", theoddsapi.BASE_URL),
        ).quota()
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: could not read the balance of %s (%s)", config.id, env, exc)
        return None
    if quota.remaining is None:
        return None
    if quota.below(close.min_remaining):
        return (f"{env} 剩余 {quota.remaining} 次，低于收盘下限 {close.min_remaining}："
                "收盘请求会一直被拒绝，直到额度重置或充值")
    return f"{env} 剩余 {quota.remaining} 次"


def format_report(config: LeagueConfig, report: RoundReport,
                  balance_line: str | None) -> str:
    title = f"{_escape(config.name)} {_escape(round_title(report))}"
    if report.reported_before:
        title += " · 补赛"
    counted = len(report.outcomes) - report.missing
    summary = f"可算 CLV {counted}/{len(report.outcomes)} 场"
    if report.missing:
        summary += f"，缺失 {report.missing} 场"
    icon = "❌" if report.missing else "✅"
    lines = [f"{icon} <b>收盘价捕获 · {title}</b>", summary]
    for outcome in report.outcomes:
        game = outcome.game
        lines.append(
            f"{ICON[outcome.status]} {_escape(game.home)} vs {_escape(game.away)}"
            f" · {_lead_text(outcome, config)}"
        )
    if report.missing:
        lines.append("❌ 的比赛没有 Pinnacle 收盘价，无法计算 CLV。")
    if report.still_to_play:
        lines.append(f"另有 {report.still_to_play} 场尚未开赛，赛后单独报告")
    if report.missing and balance_line:
        lines.append(f"Odds API: {_escape(balance_line)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def _rows(report: RoundReport, reported_at: str) -> list[dict]:
    return [{
        "home_team": o.game.home, "away_team": o.game.away, "matchday": o.game.matchday,
        "season": o.game.season, "round": o.game.round,
        "kickoff_utc": stamp(o.game.kickoff),
        "status": o.status,
        "close_lead_min": "" if o.lead_minutes is None else f"{o.lead_minutes:.1f}",
        "reported_at": reported_at,
    } for o in report.outcomes]


def _record(path: str, rows: list[dict]) -> None:
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def report(
    league: str,
    config: LeagueConfig,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    matches_path: str | None = None,
    fixtures_path: str | None = None,
    history_path: str | None = None,
    report_path: str | None = None,
    balance_of=balance,
) -> int:
    """Send the report for every round that has finished. Returns how many."""
    now = now or datetime.now(timezone.utc)
    report_path = report_path or paths.for_league(league).close_report_csv

    due = due_rounds(
        load_games(league, config, matches_path=matches_path, fixtures_path=fixtures_path),
        close_leads(league, config, history_path=history_path),
        load_reported(report_path),
        config, now=now,
    )
    if not due:
        log.info("%s: no round has finished since the last closing-line report", league)
        return 0

    token = os.environ.get(telegram.TOKEN_ENV, "").strip()
    chat_id = os.environ.get(telegram.CHAT_ENV, "").strip()
    if not dry_run and not (token and chat_id):
        log.warning("%s: %s or %s unset; %d closing-line report(s) not sent",
                    league, telegram.TOKEN_ENV, telegram.CHAT_ENV, len(due))
        return 0

    balance_line = balance_of(config) if any(r.missing for r in due) else None
    reported_at = stamp(now)

    sent = 0
    for round_report in due:
        text = format_report(config, round_report, balance_line)
        if dry_run:
            log.info("%s: would send\n%s", league, text)
            sent += 1
            continue
        if telegram.send(token, chat_id, text):
            _record(report_path, _rows(round_report, reported_at))
            sent += 1
    log.info("%s: %d closing-line report(s) %s", league, sent,
             "prepared" if dry_run else "sent")
    return sent
