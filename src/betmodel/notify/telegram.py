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
import subprocess
from datetime import datetime

import requests

from betmodel import display, paths
from betmodel.config.schema import LeagueConfig

log = logging.getLogger(__name__)

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"

API = "https://api.telegram.org/bot{token}/sendMessage"

PICK_LABEL = {"home": "主胜", "draw": "平局", "away": "客胜"}

#: A signal is the same alert as before when all three match.
KEY_FIELDS = ("fixture_id", "side", "book")

#: The state the engine uses for an edge it found on uncalibrated probabilities.
#: It is never alerted on. An uncalibrated signal is not a weaker signal, it is a
#: fake one -- the backtest that validates this strategy was run on true opening
#: prices, so a row calibrated on anything else is an untested variant. Pushing it
#: with a caveat attached would add noise to a decision rather than information.
#: It stays in the published payload for diagnosis; nothing acts on it.
STATE_UNANCHORED = "unanchored"


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


def _alerted(signals: list[dict]) -> dict[tuple[str, str], set[str] | None]:
    """``(fixture_id, side) -> books already alerted``, or ``None`` for a wildcard."""
    out: dict[tuple[str, str], set[str] | None] = {}
    for signal in signals:
        bet = signal.get("bet")
        if not bet:
            continue
        key = (signal.get("fixture_id", ""), bet.get("side", ""))
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
        key = (signal.get("fixture_id", ""), bet.get("side", ""))
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
    latest = {s.get("fixture_id", ""): s for s in current}
    out = []
    for before in previous:
        bet = before.get("bet")
        if not bet:
            continue
        after = latest.get(before.get("fixture_id", ""))
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
    if after.get("state") == "odds_cap":
        return "赔率超出该联赛上限"

    best = (after.get("best") or {}).get(side) or {}
    odds, ev = best.get("odds"), best.get("ev")
    if odds is None:
        return "该方向已无可用报价"

    was = before["bet"].get("odds")
    moved = was is not None and abs(odds - was) > 1e-9
    cause = f"赔率从 {was:.2f} 变为 {odds:.2f}" if moved else "模型更新后概率变化，赔率未动"
    if ev is not None:
        return f"{cause}；EV 现为 {ev:+.3f}，低于阈值 {config.signals.ev_min:+.3f}"
    return cause


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
    return True


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def notify(
    league: str, config: LeagueConfig, *, dry_run: bool = False,
    signals_path: str | None = None,
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

    fresh = new_signals(current, previous)
    withdrawn = withdrawn_signals(current, previous)
    if not fresh and not withdrawn:
        log.info("%s: no new signals to alert", league)
        return 0

    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if not (token and chat_id):
        log.warning("%s: %s or %s unset; %d alert(s) not sent",
                    league, TOKEN_ENV, CHAT_ENV, len(fresh) + len(withdrawn))
        return 0

    sent = 0
    # Withdrawals first. A retraction that arrives under a fresh bet reads as part
    # of it; on its own above, it is unambiguous.
    outgoing = [format_withdrawal_message(config, b, a) for b, a in withdrawn]
    outgoing += [format_message(config, s) for s in fresh]
    for message in outgoing:
        if dry_run:
            log.info("%s: would send\n%s", league, message)
            sent += 1
            continue
        if send(token, chat_id, message):
            sent += 1
    log.info("%s: %d alert(s) %s (%d new, %d withdrawn)",
             league, sent, "prepared" if dry_run else "sent", len(fresh), len(withdrawn))
    return sent
