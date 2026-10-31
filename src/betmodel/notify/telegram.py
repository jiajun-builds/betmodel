"""Push an alert the moment a new bet signal appears.

Without this a signal reaches the user only when they happen to open the board, so
"signal exists" to "human sees it" has no upper bound. This runs right after the
signals are published, on every path that regenerates them.

**The baseline is the previously committed signals file**, read with
``git show HEAD:``, not an accumulated log. Each run commits the published tree,
so the last commit is the last published signal set, and a fixture already
alerted there is not alerted again. One snapshot deep is enough because an
opening line is immutable once banked: the per-book capture gate plus
earliest-capture selection mean the bettable book only ever moves once, when a
better one opens.

**The book is part of the dedup key**, because the price is the best across books
and the answer is not always the same one. When a second book opens later at a
better price, that is genuinely new information and deserves a second message.

**A baseline missing the book is treated as a wildcard**: any book counts as
already alerted for that fixture and side. Without it, the first run after a key
change compares nothing against something and re-alerts every currently firing
signal. It self-heals after one committed run.

**The fixture is compared without its round** (:func:`fixture_key`). The round
is part of ``fixture_id`` but not of what makes a fixture one fixture -- that is
the pairing on a local matchday -- and until 2026-10-31 every signal id carried
round 0 because the upcoming-fixtures reader looked for the wrong column. Fixing
it renamed every live signal, and comparing ids exactly would have re-alerted
each one under its new name and retracted each old name in the same run.

**An alert Telegram did not accept is owed, not lost.** The baseline advances
whether or not the send landed -- the workflow commits the published tree either
way -- so a failed send used to be gone for good: the next run found the signal
in its baseline and took it as already alerted. What failed is now kept in
``data/<league>/alert_outbox.json``, committed with the signals file, and the
next run sends it again if it still stands. Recorded only on failure and cleared
only on success, so a delivered alert is never repeated from it.

The exit code stays 0 on purpose. A failing ``notify`` would not stop the
capture workflow's republish commit, which runs ``always()``, so the baseline
would advance anyway; and in the daily refresh it *would* skip the one commit
that carries the run's data, including an anchor bought minutes earlier.

**Fail-open throughout.** A missing token, an unreachable Telegram, or an
unreadable baseline logs and returns. The notifier must never fail a publish. And
when there is no baseline at all, the first-ever run sends nothing rather than
blasting every signal that happens to be live.

Env: ``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import subprocess
from datetime import datetime

import requests

from betmodel import display, paths
from betmodel.config.schema import LeagueConfig
from betmodel.signals.engine import slugify

log = logging.getLogger(__name__)

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"

API = "https://api.telegram.org/bot{token}/sendMessage"

PICK_LABEL = {"home": "主胜", "draw": "平局", "away": "客胜"}

#: A signal is the same alert as before when all three match -- the fixture
#: compared by :func:`fixture_key`, i.e. without its round.
KEY_FIELDS = ("fixture_id", "side", "book")

#: The state the engine uses for an edge it found on uncalibrated probabilities.
#: It is never alerted on. An uncalibrated signal is not a weaker signal, it is a
#: fake one -- the backtest that validates this strategy was run on true opening
#: prices, so a row calibrated on anything else is an untested variant. Pushing it
#: with a caveat attached would add noise to a decision rather than information.
#: It stays in the published payload for diagnosis; nothing acts on it.
STATE_UNANCHORED = "unanchored"
#: A club in the fixture has too little history for its rating to be bettable.
STATE_THIN_EVIDENCE = "thin_evidence"


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

def previous_signals(path: str) -> list[dict] | None:
    """The last committed version of the signals file, or ``None``.

    ``None`` means no baseline, which the caller treats as "send nothing": the
    first run that introduces the file would otherwise alert every live signal
    at once.
    """
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        relative = os.path.relpath(os.path.abspath(path), root)
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        log.warning("no committed baseline for %s (%s); sending nothing", path, exc)
        return None
    try:
        return json.loads(blob).get("signals", [])
    except (ValueError, AttributeError) as exc:
        log.warning("committed baseline for %s is unreadable (%s); sending nothing",
                    path, exc)
        return None


#: The local matchday inside a ``fixture_id``: the segment the round sits before.
_MATCHDAY = re.compile(r"-\d{4}-\d{2}-\d{2}-")


def fixture_key(signal: dict) -> str:
    """The row's ``fixture_id`` with the round taken out: what alerts compare.

    League, season, local matchday, home and away -- the pairing on a matchday,
    which is what CLAUDE.md says a fixture is. The round is cut using the row's
    own ``round`` field, the same value ``make_fixture_id`` was given, so a
    hyphenated season or stage name cannot be mistaken for it.

    Not a transition shim to delete later. It exists because the round-0 fix of
    2026-10-31 renamed every live signal (``CSL-2026-0-2026-11-08-...`` became
    ``CSL-2026-30-2026-11-08-...``) and the baseline still held the old names:
    compared exactly, the first run after the fix would have re-sent every live
    bet and retracted every one of them in the same breath. A provider relabelling
    a round would do the same thing again, and it changes nothing about the bet.

    An id that does not have the expected shape is returned whole, which falls
    back to the exact comparison this replaced.
    """
    fixture = str(signal.get("fixture_id") or "")
    day = _MATCHDAY.search(fixture)
    if not day:
        return fixture
    head = fixture[: day.start()]
    round_segment = "-" + (slugify(str(signal.get("round") or "")) or "0")
    if not head.endswith(round_segment):
        return fixture
    return head[: -len(round_segment)] + fixture[day.start():]


def _bet_key(signal: dict) -> tuple[str, str, str]:
    """``(fixture, side, book)`` of a row's bet: the dedup key of one alert."""
    bet = signal.get("bet") or {}
    return fixture_key(signal), bet.get("side", ""), (bet.get("book") or "").strip()


def _alerted(signals: list[dict]) -> dict[tuple[str, str], set[str] | None]:
    """``(fixture, side) -> books already alerted``, or ``None`` for a wildcard."""
    out: dict[tuple[str, str], set[str] | None] = {}
    for signal in signals:
        bet = signal.get("bet")
        if not bet:
            continue
        key = (fixture_key(signal), bet.get("side", ""))
        book = (bet.get("book") or "").strip()
        if not book:
            out[key] = None  # baseline predates the book: swallow the comparison
            continue
        if out.get(key, set()) is None:
            continue
        out.setdefault(key, set()).add(book)  # type: ignore[union-attr]
    return out


def new_signals(current: list[dict], previous: list[dict]) -> list[dict]:
    """Firing signals not already alerted.

    New means the fixture and side are new outright, or the same side is now
    best-priced at a different book.
    """
    seen = _alerted(previous)
    fresh = []
    for signal in current:
        bet = signal.get("bet")
        if not bet:
            continue
        key = (fixture_key(signal), bet.get("side", ""))
        if key not in seen:
            fresh.append(signal)
            continue
        books = seen[key]
        if books is None:
            continue
        if (bet.get("book") or "").strip() not in books:
            fresh.append(signal)
    return fresh


def withdrawn_signals(current: list[dict], previous: list[dict]) -> list[tuple[dict, dict]]:
    """Signals alerted before that are no longer bets, paired with their new row.

    The counterpart of the bet alert, and it exists for the same reason: a signal
    reaches you when it fires, so it has to reach you when it stops. The model is
    refitted daily on new results, and a fixture that cleared the bar on Tuesday's
    probabilities can fail it on Wednesday's without any price moving. Told about
    the first and not the second, you would be holding a recommendation the tool
    itself no longer stands behind.

    A fixture that has left the file entirely is not a withdrawal: it kicked off,
    which is not news. Self-deduplicating, because the baseline is the previous
    published file -- once the withdrawal is published, the next run's baseline
    has no bet to withdraw.
    """
    latest = {fixture_key(s): s for s in current}
    out = []
    for before in previous:
        bet = before.get("bet")
        if not bet:
            continue
        after = latest.get(fixture_key(before))
        if after is None:
            continue
        still_on = after.get("bet")
        if still_on and still_on.get("side") == bet.get("side"):
            continue
        out.append((before, after))
    return out


def withdrawal_reason(config: LeagueConfig, before: dict, after: dict) -> str:
    """Why it stopped, told apart by what actually changed.

    The price and the probability are the only two inputs, so comparing the side's
    price across the two publishes says which of them moved. That distinction is
    the whole value of the message: a model that changed its mind is a different
    thing to react to than a line that moved.
    """
    side = before["bet"]["side"]
    if after.get("state") == STATE_UNANCHORED:
        return "锚定盘开盘价不可用，信号无法校准"
    if after.get("state") == STATE_THIN_EVIDENCE:
        return (
            f"该场有球队的历史样本不足（少于 {config.signals.min_team_evidence:g} 场加权），"
            "模型对其实力的估计不可靠，不作为可下注信号"
        )
    if after.get("state") == "odds_cap":
        return "赔率超出该联赛上限"
    if after.get("state") == "paused":
        return "该联赛已暂停下注，信号仅作记录"

    best = (after.get("best") or {}).get(side) or {}
    odds, ev = best.get("odds"), best.get("ev")
    if odds is None:
        return "该方向已无可用报价"

    was = before["bet"].get("odds")
    moved = was is not None and abs(odds - was) > 1e-9
    cause = f"赔率从 {was:.2f} 变为 {odds:.2f}" if moved else "模型更新后概率变化，赔率未动"
    if ev is None:
        return cause
    # Only call it a threshold miss when it actually is one. A signal can be
    # withdrawn for a reason that leaves the edge intact, and saying "EV +0.213 is
    # below +0.100" in that case is not merely unhelpful, it is false.
    if ev <= config.signals.ev_min:
        return f"{cause}；EV 现为 {ev:+.3f}，低于阈值 {config.signals.ev_min:+.3f}"
    return f"{cause}；EV 现为 {ev:+.3f}"


# --------------------------------------------------------------------------- #
# the outbox: what Telegram did not accept
# --------------------------------------------------------------------------- #

def load_outbox(path: str) -> list[dict]:
    """Alerts an earlier run owed and could not deliver. Empty when there are none."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            pending = json.load(handle).get("pending", [])
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("alert outbox %s is unreadable (%s); treating it as empty", path, exc)
        return []
    return [entry for entry in pending if isinstance(entry, dict)]


def save_outbox(path: str, pending: list[dict]) -> None:
    """Write what is still owed. Never deletes the file, only empties it.

    The commit action stages a path only if it exists, so a deleted outbox would
    never reach main: the next run would check out the old one and send its
    alerts a second time.
    """
    if not pending and not os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"pending": pending}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")


def retry_pending(
    current: list[dict], pending: list[dict],
    fresh: list[dict], withdrawn: list[tuple[dict, dict]],
) -> tuple[list[dict], list[tuple[dict, dict]]]:
    """Fold the alerts an earlier run failed to send back into this run's.

    Each is sent again only if it still stands, judged against the current file
    rather than replayed from the outbox: a bet goes out only while the fixture
    still fires on that side at that book, so the reader gets today's price and
    never one the engine has stopped standing behind; a withdrawal only while the
    fixture is still on the board and not firing that side. Anything else is
    dropped -- a stale alert is worse than none.

    And a bet the reader never received is not retracted. Withdrawal is computed
    against the baseline, which holds every bet that was *published*, delivered or
    not; retracting an unseen bet is a message about nothing. Only when it was
    the first alert on that side, though -- if an earlier book's alert did land,
    the reader is holding it and the withdrawal is still owed.
    """
    latest = {fixture_key(s): s for s in current}

    fresh = list(fresh)
    owed = {_bet_key(s) for s in fresh}
    for entry in pending:
        if entry.get("kind") != "bet":
            continue
        row = latest.get(fixture_key(entry))
        if row is None or not row.get("bet") or _bet_key(row) != _bet_key(entry):
            continue
        if _bet_key(row) not in owed:
            fresh.append(row)
            owed.add(_bet_key(row))

    unseen = {_bet_key(e) for e in pending if e.get("kind") == "bet" and e.get("first")}
    withdrawn = [(b, a) for b, a in withdrawn if _bet_key(b) not in unseen]
    retracting = {_bet_key(b)[:2] for b, _ in withdrawn}
    for entry in pending:
        if entry.get("kind") != "withdrawal":
            continue
        before = entry.get("signal") or {}
        if not before.get("bet"):
            continue
        after = latest.get(fixture_key(before))
        if after is None:
            continue  # kicked off: not news
        side = before["bet"].get("side")
        if (after.get("bet") or {}).get("side") == side:
            continue  # firing on that side again; that is a bet, not a retraction
        if _bet_key(before)[:2] not in retracting:
            withdrawn.append((before, after))
            retracting.add(_bet_key(before)[:2])
    return fresh, withdrawn


def _owed_bet(signal: dict, *, first: bool) -> dict:
    bet = signal["bet"]
    return {
        "kind": "bet",
        "fixture_id": signal.get("fixture_id", ""),
        "round": signal.get("round") or "",
        "home_team": signal.get("home_team", ""),
        "away_team": signal.get("away_team", ""),
        "bet": {"side": bet.get("side", ""), "book": bet.get("book", "")},
        "first": first,
    }


# --------------------------------------------------------------------------- #
# message
# --------------------------------------------------------------------------- #

def _escape(text) -> str:
    return html.escape(str(text), quote=False)


def format_message(config: LeagueConfig, signal: dict) -> str:
    """A bet instruction readable at a glance.

    The quoted price is the best across books and the named book is where to get
    it. Those two must never be separated, or the reader takes the right side at
    the wrong price. Any other book clearing both bars is listed as an alternate
    with its own price, so a limited or unavailable primary has a fallback.
    """
    bet = signal["bet"]
    side = bet["side"]
    probability = signal["model"][side]
    # The model's no-vig fair price: the odds at which EV is exactly zero. Shown
    # as a reference, NOT a betting floor. The signal required EV above a
    # materially higher bar, and calling this a floor would imply the band
    # between them is bettable when it is a slice no backtest validated.
    fair = (1.0 / probability) if probability else None

    kickoff = datetime.fromisoformat(signal["kickoff_utc"].replace("Z", "+00:00"))

    book = next((b for b in config.odds.bet_books if b.key == bet["book"]), None)
    label = book.label if book else bet["book"]

    lines = [
        "🟢 <b>BET 信号</b>",
        f"<b>{_escape(config.name)}</b>",
        f"<b>{_escape(signal['home_team'])} vs {_escape(signal['away_team'])}</b>",
        f"方向: <b>{_escape(PICK_LABEL.get(side, side))}</b>",
        f"{_escape(label)} 开盘价: <b>{bet['odds']:.2f}</b>",
        f"EV: <b>{bet['ev']:+.3f}</b>",
        f"Fair odds (模型): <b>{fair:.2f}</b>" if fair else "Fair odds (模型): --",
        # One timezone for every league, and it is the reader's. Showing each
        # league in its own zone puts two timezones in one inbox and answers a
        # question nobody asked: what matters is when to be at a screen.
        f"开赛: {display.moment(kickoff)} {display.label()}",
    ]

    alternates = []
    for other in config.odds.bet_books:
        if other.key == bet["book"] or other.key not in bet.get("books", []):
            continue
        quote = next(
            (q for q in signal["quotes"] if q["book"] == other.key and q["side"] == side),
            None,
        )
        alternates.append(
            f"{_escape(other.label)} {quote['odds']:.2f}" if quote else _escape(other.label)
        )
    if alternates:
        lines.append(f"备选: {' · '.join(alternates)}")

    if book and book.url:
        lines.append(f'下注: <a href="{book.url}">{_escape(book.label)}</a>')
    return "\n".join(lines)


def format_withdrawal_message(config: LeagueConfig, before: dict, after: dict) -> str:
    """A retraction, shaped so the reader knows immediately it is not a new bet."""
    bet = before["bet"]
    side = bet["side"]
    kickoff = datetime.fromisoformat(before["kickoff_utc"].replace("Z", "+00:00"))
    book = next((b for b in config.odds.bet_books if b.key == bet.get("book")), None)
    label = book.label if book else bet.get("book", "--")

    return "\n".join([
        "🔴 <b>信号撤回</b>",
        f"<b>{_escape(config.name)}</b>",
        f"<b>{_escape(before['home_team'])} vs {_escape(before['away_team'])}</b>",
        f"原方向: <b>{_escape(PICK_LABEL.get(side, side))}</b>"
        f" @ {bet['odds']:.2f} ({_escape(label)})",
        f"原 EV: <b>{bet['ev']:+.3f}</b>",
        f"原因: {_escape(withdrawal_reason(config, before, after))}",
        f"开赛: {display.moment(kickoff)} {display.label()}",
    ])


def send(token: str, chat_id: str, text: str, *, timeout: int = 15) -> bool:
    """Post one message. True only when Telegram says it delivered it.

    **The receipt is logged**: the message number Telegram assigned and the chat it
    went to. Without it a run can say only that it sent something, and "it never
    arrived" cannot be settled from this side. That is where CSL round 27's
    closing-line report left things on 2026-10-11: the run logged "sent" at
    13:36Z, the reader had no message, and nothing on record said which chat
    Telegram had put it in.
    """
    try:
        response = requests.post(
            API.format(token=token),
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log.warning("telegram send failed: %s", exc)
        return False
    if response.status_code != 200:
        log.warning("telegram returned HTTP %s: %s", response.status_code,
                    response.text[:200])
        return False
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict) or body.get("ok") is False:
        log.warning("telegram answered 200 without delivering: %s", response.text[:200])
        return False
    result = body.get("result") or {}
    chat = result.get("chat") or {}
    log.info("telegram: delivered as message %s to %s chat %r",
             result.get("message_id", "?"), chat.get("type", "?"),
             chat.get("title") or chat.get("username") or chat.get("first_name") or "?")
    return True


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def notify(
    league: str, config: LeagueConfig, *, dry_run: bool = False,
    signals_path: str | None = None, outbox_path: str | None = None,
) -> int:
    """Alert on newly firing signals. Returns how many were sent."""
    if not config.notify.telegram:
        log.debug("%s: telegram notifications are off", league)
        return 0

    path = signals_path or paths.for_league(league).public_json("signals")
    if not os.path.exists(path):
        log.warning("%s: no signals file at %s", league, path)
        return 0
    with open(path, encoding="utf-8") as handle:
        current = json.load(handle).get("signals", [])

    previous = previous_signals(path)
    if previous is None:
        return 0

    outbox_path = outbox_path or paths.for_league(league).alert_outbox_json
    pending = load_outbox(outbox_path)
    fresh, withdrawn = retry_pending(
        current, pending,
        new_signals(current, previous), withdrawn_signals(current, previous),
    )
    if not fresh and not withdrawn:
        log.info("%s: no new signals to alert", league)
        if pending and not dry_run:
            log.info("%s: %d unsent alert(s) no longer stand; dropped", league, len(pending))
            save_outbox(outbox_path, [])
        return 0

    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if not (token and chat_id):
        log.warning("%s: %s or %s unset; %d alert(s) not sent",
                    league, TOKEN_ENV, CHAT_ENV, len(fresh) + len(withdrawn))
        return 0

    # A bet alert is the reader's first on its side unless the baseline already
    # had one there that was actually delivered. Decides whether a later
    # withdrawal of it is owed if this send fails too (see retry_pending).
    alerted = _alerted(previous)
    undelivered = {_bet_key(e)[:2] for e in pending if e.get("kind") == "bet" and e.get("first")}

    sent = 0
    still_owed: list[dict] = []
    # Withdrawals first. A retraction that arrives under a fresh bet reads as part
    # of it; on its own above, it is unambiguous.
    outgoing = [
        (format_withdrawal_message(config, b, a), {"kind": "withdrawal", "signal": b})
        for b, a in withdrawn
    ]
    for signal in fresh:
        side_key = _bet_key(signal)[:2]
        first = side_key not in alerted or side_key in undelivered
        outgoing.append((format_message(config, signal), _owed_bet(signal, first=first)))
    for message, owed in outgoing:
        if dry_run:
            log.info("%s: would send\n%s", league, message)
            sent += 1
            continue
        if send(token, chat_id, message):
            sent += 1
        else:
            still_owed.append(owed)
    if not dry_run:
        save_outbox(outbox_path, still_owed)
    if still_owed:
        log.warning("%s: %d alert(s) Telegram did not accept; kept in %s for the next run",
                    league, len(still_owed), outbox_path)
    log.info("%s: %d alert(s) %s (%d new, %d withdrawn)",
             league, sent, "prepared" if dry_run else "sent", len(fresh), len(withdrawn))
    return sent
