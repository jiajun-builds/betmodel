"""The closing-line report sent once a round has been played.

It exists because seven of Liga MX matchday 10's nine closes were refused by an
exhausted account on 2026-09-26..28 and nobody was told for four days. Its two
failure modes are the same two every alert here has: telling too early or too
often trains the reader to ignore it, and not telling is the incident again.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from betmodel.config import load_league
from betmodel.notify import close_report as cr
from betmodel.odds import capture_store

LEAGUE = "ligamx"
CONFIG = load_league(LEAGUE)

#: A round of three: Friday, Saturday, and a Sunday game that is the last.
FRI = datetime(2026, 9, 26, 1, 0, tzinfo=timezone.utc)
SAT = datetime(2026, 9, 26, 23, 7, tzinfo=timezone.utc)
SUN = datetime(2026, 9, 28, 1, 0, tzinfo=timezone.utc)
ROUND = [("Atlante", "Monterrey", FRI), ("Guadalajara", "Queretaro", SAT),
         ("Leon", "FC Juarez", SUN)]

AFTER = SUN + cr.MATCH_LENGTH + timedelta(minutes=5)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _table(path, rows, round_="10"):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Season", "Round", "Home", "Away",
                                                    "kickoff_utc"])
        writer.writeheader()
        for row in rows:
            home, away, kickoff = row[:3]
            writer.writerow({"Season": "Apertura 2026",
                             "Round": row[3] if len(row) > 3 else round_,
                             "Home": home, "Away": away, "kickoff_utc": _iso(kickoff)})
    return str(path)


def _close(home, away, kickoff, minutes_before, book="pinnacle"):
    fetched = kickoff - timedelta(minutes=minutes_before)
    return {
        "event_id": f"{home}{away}{minutes_before}", "commence_time": _iso(kickoff),
        "api_home_team": home, "api_away_team": away, "home_team": home, "away_team": away,
        "home_odds": "2.1", "draw_odds": "3.3", "away_odds": "3.4",
        "bookmaker": book, "market": "h2h", "regions": "theoddsapi",
        "last_update": _iso(fetched), "fetched_at": _iso(fetched),
    }


class Setup:
    def __init__(self, tmp_path, *, played=(), upcoming=(), closes=()):
        self.matches = _table(tmp_path / "matches.csv", played)
        self.fixtures = _table(tmp_path / "upcoming.csv", upcoming)
        self.history = str(tmp_path / "history.csv")
        if closes:
            capture_store.append_snapshots(pd.DataFrame(list(closes)),
                                           path=self.history, snapshot_type="close")
        self.reports = str(tmp_path / "close_report.csv")
        self.balance_reads = 0

    def _balance(self, config):
        self.balance_reads += 1
        return "THE_ODDS_API_KEY_LIGAMX 剩余 4 次"

    def run(self, now, **kwargs):
        return cr.report(
            LEAGUE, CONFIG, now=now, matches_path=self.matches,
            fixtures_path=self.fixtures, history_path=self.history,
            report_path=self.reports, balance_of=self._balance, **kwargs,
        )

    def due(self, now):
        return cr.due_rounds(
            cr.load_games(LEAGUE, CONFIG, matches_path=self.matches,
                          fixtures_path=self.fixtures),
            cr.close_leads(LEAGUE, CONFIG, history_path=self.history),
            cr.load_reported(self.reports), CONFIG, now=now,
        )


@pytest.fixture
def sent(monkeypatch):
    """Messages Telegram accepted."""
    out: list[str] = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    monkeypatch.setattr(cr.telegram, "send", lambda token, chat, text: out.append(text) or True)
    return out


# --------------------------------------------------------------------------- #
# when
# --------------------------------------------------------------------------- #

def test_nothing_is_reported_while_the_round_still_has_games_to_play(tmp_path):
    """Matchday 10's Saturday games are over long before Sunday's kick off. The
    round is not, and a report then would be the first of several."""
    setup = Setup(tmp_path, played=ROUND[:2], upcoming=ROUND[2:])
    assert setup.due(SAT + cr.MATCH_LENGTH + timedelta(hours=1)) == []


def test_nothing_is_reported_while_the_last_game_is_in_progress(tmp_path):
    setup = Setup(tmp_path, played=ROUND)
    assert setup.due(SUN + timedelta(minutes=30)) == []


def test_the_round_is_reported_once_its_last_game_has_been_played(tmp_path):
    setup = Setup(tmp_path, played=ROUND)
    [report] = setup.due(AFTER)
    assert [o.game.home for o in report.outcomes] == ["Atlante", "Guadalajara", "Leon"]
    assert report.round == "10"


def test_a_reported_round_is_not_reported_again(tmp_path, sent):
    setup = Setup(tmp_path, played=ROUND)
    assert setup.run(AFTER) == 1
    assert setup.run(AFTER + timedelta(minutes=5)) == 0
    assert len(sent) == 1


def test_a_send_telegram_refused_is_retried_on_the_next_tick(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    setup = Setup(tmp_path, played=ROUND)

    monkeypatch.setattr(cr.telegram, "send", lambda *a: False)
    assert setup.run(AFTER) == 0
    monkeypatch.setattr(cr.telegram, "send", lambda *a: True)
    assert setup.run(AFTER + timedelta(minutes=5)) == 1


def test_a_dry_run_sends_and_records_nothing(tmp_path, sent):
    setup = Setup(tmp_path, played=ROUND)
    assert setup.run(AFTER, dry_run=True) == 1
    assert sent == []
    assert setup.run(AFTER) == 1


def test_a_postponed_game_does_not_hold_the_round_open(tmp_path, sent):
    """It is reported on its own once played, marked as the round's 补赛, rather
    than keeping the whole round's report back for weeks."""
    later = SUN + timedelta(days=12)
    setup = Setup(tmp_path, played=ROUND, upcoming=[("Necaxa", "Club America", later)])

    [report] = setup.due(AFTER)
    assert report.still_to_play == 1
    assert setup.run(AFTER) == 1
    assert "另有 1 场尚未开赛" in sent[0]

    [makeup] = setup.due(later + cr.MATCH_LENGTH + timedelta(minutes=5))
    assert [o.game.home for o in makeup.outcomes] == ["Necaxa"]
    assert makeup.reported_before == 3
    assert setup.run(later + cr.MATCH_LENGTH + timedelta(minutes=5)) == 1
    assert "补赛" in sent[1]


def test_a_two_legged_tie_is_one_report(tmp_path):
    """A Liguilla round is two legs about three days apart under one label. Split,
    the second leg would arrive as a 补赛 of a round that was never postponed."""
    first = datetime(2026, 11, 27, 3, 0, tzinfo=timezone.utc)
    second = first + timedelta(days=3)
    setup = Setup(tmp_path, played=[("Toluca", "Cruz Azul", first, "Quarterfinals"),
                                    ("Cruz Azul", "Toluca", second, "Quarterfinals")])
    assert setup.due(first + cr.MATCH_LENGTH + timedelta(minutes=5)) == []
    [report] = setup.due(second + cr.MATCH_LENGTH + timedelta(minutes=5))
    assert len(report.outcomes) == 2 and report.reported_before == 0


def test_a_round_long_finished_is_not_replayed(tmp_path):
    """The first deploy, or a run after an outage, must not send old rounds."""
    setup = Setup(tmp_path, played=ROUND)
    assert setup.due(SUN + cr.LOOKBACK + timedelta(hours=1)) == []


def test_a_round_is_reported_whole_however_long_ago_its_first_game_was(tmp_path):
    """The lookback once applied per game: four days after matchday 10 began, its
    Friday games had aged out and the report named only the last two of nine."""
    setup = Setup(tmp_path, played=ROUND)
    [report] = setup.due(FRI + cr.LOOKBACK + timedelta(hours=1))
    assert len(report.outcomes) == 3


def test_a_round_is_read_from_the_match_table_and_the_schedule_together(tmp_path):
    """A round played across a night is split between the two by the refresh."""
    setup = Setup(tmp_path, played=ROUND[:2], upcoming=ROUND[2:])
    [report] = setup.due(AFTER)
    assert len(report.outcomes) == 3


def test_a_game_with_no_round_label_is_reported_with_its_day(tmp_path):
    setup = Setup(tmp_path, played=[("A", "B", SAT, ""), ("C", "D", SAT, "")])
    [report] = setup.due(SAT + cr.MATCH_LENGTH + timedelta(minutes=5))
    assert report.round == "" and report.matchday == "2026-09-26"
    assert len(report.outcomes) == 2


# --------------------------------------------------------------------------- #
# what counts as captured
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("minutes, expected", [
    (9, cr.CAPTURED), (10, cr.CAPTURED),
    (14, cr.EARLY),                          # Tijuana v Atlas, 2026-09-26
    (5 * 60, cr.EARLY),
    (7 * 60, cr.MISSING),                    # reduce refuses a close this early
    (None, cr.MISSING),
])
def test_status_follows_the_rule_clv_is_computed_with(minutes, expected):
    assert cr.status(minutes, CONFIG) == expected


def test_each_game_gets_its_own_status(tmp_path):
    setup = Setup(tmp_path, played=ROUND, closes=[
        _close("Atlante", "Monterrey", FRI, 9),
        _close("Guadalajara", "Queretaro", SAT, 14),
    ])
    [report] = setup.due(AFTER)
    assert [o.status for o in report.outcomes] == [cr.CAPTURED, cr.EARLY, cr.MISSING]
    assert report.missing == 1


def test_only_the_finalisation_book_counts(tmp_path):
    """CLV is measured against Pinnacle. An exchange close beside it is context."""
    setup = Setup(tmp_path, played=ROUND[:1],
                  closes=[_close("Atlante", "Monterrey", FRI, 9, book="betfair_ex_eu")])
    [report] = setup.due(FRI + cr.MATCH_LENGTH + timedelta(minutes=5))
    assert report.outcomes[0].status == cr.MISSING


def test_a_rescheduled_game_is_not_credited_with_the_close_of_its_old_date(tmp_path):
    moved = FRI + timedelta(days=5)
    setup = Setup(tmp_path, played=[("Atlante", "Monterrey", moved)],
                  closes=[_close("Atlante", "Monterrey", FRI, 9)])
    [report] = setup.due(moved + cr.MATCH_LENGTH + timedelta(minutes=5))
    assert report.outcomes[0].status == cr.MISSING


# --------------------------------------------------------------------------- #
# the message
# --------------------------------------------------------------------------- #

def test_the_message_names_every_game_and_which_have_no_clv(tmp_path, sent):
    setup = Setup(tmp_path, played=ROUND, closes=[
        _close("Atlante", "Monterrey", FRI, 9),
        _close("Guadalajara", "Queretaro", SAT, 14),
    ])
    setup.run(AFTER)
    [text] = sent
    assert "Liga MX 第10轮" in text
    assert "可算 CLV 2/3 场，缺失 1 场" in text
    assert "✅ Atlante vs Monterrey · 开赛前 9 分钟" in text
    assert "⚠️ Guadalajara vs Queretaro · 开赛前 14 分钟" in text
    assert "❌ Leon vs FC Juarez · 无收盘价" in text
    assert "无法计算 CLV" in text
    assert "THE_ODDS_API_KEY_LIGAMX 剩余 4 次" in text


def test_the_balance_is_read_only_when_a_close_is_missing(tmp_path, sent):
    """It is a network call, if a free one, and a full round needs no footnote."""
    setup = Setup(tmp_path, played=ROUND, closes=[
        _close(home, away, kickoff, 9) for home, away, kickoff in ROUND
    ])
    setup.run(AFTER)
    assert setup.balance_reads == 0
    assert "Odds API" not in sent[0]
    assert "可算 CLV 3/3 场" in sent[0]


def test_the_report_is_not_switched_off_with_the_signal_alerts(tmp_path, sent):
    """Liga MX's signal alerts are off while it is paused. Its closes are the
    record that decides the unpause, so they are reported regardless."""
    assert CONFIG.notify.telegram is False
    Setup(tmp_path, played=ROUND).run(AFTER)
    assert len(sent) == 1


def test_missing_credentials_send_nothing_and_record_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    setup = Setup(tmp_path, played=ROUND)
    assert setup.run(AFTER) == 0
    assert cr.load_reported(setup.reports) == set()
