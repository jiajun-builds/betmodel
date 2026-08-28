"""Push a Telegram alert the moment a NEW opening-line bet signal appears (P0-2).

Closes the terminal-side gap in the execution chain: a signal used to reach the user
only when they happened to open the dashboard, so "signal on site -> human sees it"
had no upper bound. This module runs right after the market-comparison export (inside
``scripts/csl.sh`` publish / republish / all — every path that regenerates signals)
and pushes each newly-fired ``signal_state == "bet"`` fixture to a Telegram chat, so
the full bet instruction lands on the user's lock screen without a page load.

Two books (2026-08-08). The message must name **which book to bet**, because the price
is now the best across ``books.BET_BOOKS`` and the answer is no longer always 1xBet.
``signal_book`` is therefore part of the dedup key: when a second book opens later at a
better price, that is genuinely new information and deserves a second alert.

Dedup baseline = the *previously committed* comparison CSV. Each workflow commits
``CHN_upcoming_market_comparison.csv`` every run, so ``git show HEAD:<csv>`` is the last
published signal set; a fixture+pick+book that was already a "bet" there is NOT
re-notified. A price that merely moved on an already-notified pick is likewise not
re-sent (dedup ignores odds) — the quoted price and the fair odds in the message are what
let the user judge a moved line at execution time.

⚠️ Baselines written before ``signal_book`` existed are treated as a **wildcard**: any
book counts as already-alerted for that fixture+pick. Without this, the first run after
the two-book migration would compare "" against "onexbet" and re-alert every currently
firing signal. It self-heals after one committed run, and the same guard protects any
future key change. See ``_bet_books_by_key``.

Note the baseline is one git snapshot deep, not an accumulated log, so an A->B->A
sequence would re-alert on the return. That cannot happen from price movement: opening
lines are immutable once banked (per-book capture gate + earliest-``fetched_at``
selection), so ``signal_book`` only ever moves once, when a better book opens.

Fail-open by design: a missing token, an unreachable Telegram, or an unreadable
baseline logs and returns without raising, so the notifier can never fail a publish.
When the baseline CSV is unavailable (first-ever run) nothing is sent, to avoid a
one-off blast of every currently-firing signal.

Env:
    TELEGRAM_BOT_TOKEN   bot token from @BotFather
    TELEGRAM_CHAT_ID     numeric chat id (see the setup notes)

Usage (repo root, PYTHONPATH=src):
    python -m csl.notify.signal_alert
    python -m csl.notify.signal_alert --dry-run   # print what would be sent, send nothing
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from csl.odds.books import BET_BOOKS, BOOK_BY_KEY
from csl.paths import data_output_dir

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_COMPARISON_CSV = os.path.join(data_output_dir(), "CHN_upcoming_market_comparison.csv")
DISPLAY_TZ = ZoneInfo("Europe/London")

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"

# Dedup identity of a fired signal. `signal_book` is included so a better price opening
# at the OTHER book re-alerts; `signal_books` deliberately is NOT, or the key would
# churn whenever the second book merely crosses the threshold without becoming best.
SIGNAL_KEY = ("fixture_id", "signal_pick", "signal_book")


def _bet_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Map (fixture_id, signal_pick) -> row for every ``signal_state == "bet"`` row."""
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        if (row.get("signal_state") or "").strip() != "bet":
            continue
        pick = (row.get("signal_pick") or "").strip()
        if pick not in ("home", "draw", "away"):
            continue
        out[(str(row.get("fixture_id", "")), pick)] = row
    return out


def _bet_books_by_key(rows: list[dict]) -> dict[tuple[str, str], set[str] | None]:
    """Map (fixture_id, pick) -> the set of books already alerted, or ``None`` for "any".

    ``None`` is the migration wildcard: it means these rows predate ``signal_book``, so
    we cannot tell which book was alerted and must assume all of them were. Detected per
    row via the header keys ``csv.DictReader`` supplies, so a baseline CSV written by the
    old single-book exporter suppresses the re-alert blast instead of causing one.
    """
    out: dict[tuple[str, str], set[str] | None] = {}
    for key, row in _bet_rows(rows).items():
        if "signal_book" not in row:
            out[key] = None
        else:
            out[key] = {(row.get("signal_book") or "").strip()}
    return out


def _read_csv_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _previous_committed_rows(path: str) -> list[dict] | None:
    """Rows of the last-committed version of ``path`` via ``git show HEAD:<relpath>``.

    Returns None when git or the committed blob is unavailable (e.g. the first run
    that introduces the file), which the caller treats as "no baseline -> send nothing".
    """
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        rel = os.path.relpath(os.path.abspath(path), root)
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        log.warning("No committed baseline for %s (%s); sending nothing this run.", path, exc)
        return None
    return list(csv.DictReader(io.StringIO(blob)))


def _fmt_kickoff(kickoff_at: str, match_time: str) -> str:
    if kickoff_at:
        try:
            dt = datetime.fromisoformat(kickoff_at.replace("Z", "+00:00"))
            return dt.astimezone(DISPLAY_TZ).strftime("%a %d %b %H:%M") + " (London)"
        except ValueError:
            pass
    return match_time or "TBD"


def _pick_cn(row: dict) -> str:
    pick = (row.get("signal_pick") or "").strip()
    if pick == "home":
        return f"主胜 {row.get('home_team', '')}".strip()
    if pick == "away":
        return f"客胜 {row.get('away_team', '')}".strip()
    return "平局 Draw"


def _f(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _esc(text) -> str:
    s = "" if text is None else str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_message(row: dict) -> str:
    """One-glance bet instruction: match, side, book, price, EV, fair odds, kickoff.

    The price quoted is the BEST across books and the book named is where to get it —
    those two must never be separated, or the user takes the right side at the wrong
    price. Any other book that also clears both bars is listed as an alternate with its
    own price, so a limited or unavailable primary has an immediate fallback.
    """
    pick = (row.get("signal_pick") or "").strip()
    book = BOOK_BY_KEY.get((row.get("signal_book") or "").strip())
    odds = _f(row.get(f"best_open_{pick}_odds"))
    evv = _f(row.get(f"best_open_{pick}_ev"))
    if odds is None:
        # Pre-migration CSV (no best_* columns). Fall back so an old file still alerts.
        odds = _f(row.get(f"onexbet_open_{pick}_odds"))
        evv = _f(row.get(f"onexbet_open_{pick}_ev"))
    prob = _f(row.get({"home": "home_win_prob", "draw": "draw_prob", "away": "away_win_prob"}[pick]))
    # 1/p — the model's no-vig fair price for this outcome, i.e. the odds at which
    # EV is exactly zero. Shown as a reference point against the quoted price, NOT as a
    # betting floor: the signal itself required EV > SIGNAL_EV_MIN, which is a
    # materially higher bar (at p=0.2175 that is 5.52, against fair odds of 4.60).
    # Labelling it a floor implied the band between the two was bettable; it is a slice
    # the backtest never validated, and §11.7's vig wall says EV > 0 alone loses.
    fair_odds = (1.0 / prob) if prob and prob > 0 else None

    match = f"{row.get('home_team', '')} vs {row.get('away_team', '')}".strip()
    kickoff = _fmt_kickoff(row.get("kickoff_at", ""), row.get("match_time", ""))
    label = book.label if book else "开盘"

    lines = [
        "🟢 <b>BET 信号</b>",
        f"<b>{_esc(match)}</b>",
        f"方向: <b>{_esc(_pick_cn(row))}</b>",
        f"{_esc(label)} 开盘价: <b>{odds:.2f}</b>" if odds is not None else f"{_esc(label)} 开盘价: --",
        f"EV: <b>{evv:+.3f}</b>" if evv is not None else "EV: --",
        f"Fair odds (模型): <b>{fair_odds:.2f}</b>" if fair_odds is not None else "Fair odds (模型): --",
        f"开赛: {_esc(kickoff)}",
    ]

    alts = [
        b for b in BET_BOOKS
        if b.key in _signal_book_keys(row) and (book is None or b.key != book.key)
    ]
    if alts:
        parts = []
        for b in alts:
            alt_odds = _f(row.get(b.odds_col(pick)))
            parts.append(f"{_esc(b.label)} {alt_odds:.2f}" if alt_odds is not None else _esc(b.label))
        lines.append(f"备选: {' · '.join(parts)}")

    if book:
        lines.append(f'下注: <a href="{book.url}">{_esc(book.label)}</a>')
    return "\n".join(lines)


def _signal_book_keys(row: dict) -> list[str]:
    """Book keys from the "|"-joined ``signal_books`` column ("" -> [])."""
    return [k for k in (row.get("signal_books") or "").strip().split("|") if k]


def send_telegram(token: str, chat_id: str, text: str, *, timeout: int = 15) -> bool:
    """POST one message; return True on success, False (logged) on any failure."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Telegram send failed: %s", exc)
        return False
    return True


def new_signals(current_rows: list[dict], previous_rows: list[dict]) -> list[dict]:
    """Bet rows in ``current`` not already alerted in ``previous``.

    New means either the (fixture, pick) is new outright, or it is the same pick now
    best-priced at a **different book** — a better price is real new information worth
    a second message. A pre-``signal_book`` baseline maps to ``None`` and swallows the
    book comparison entirely, so the migration itself never triggers a blast.
    """
    prev = _bet_books_by_key(previous_rows)
    fresh = []
    for key, row in _bet_rows(current_rows).items():
        if key not in prev:
            fresh.append(row)
            continue
        seen = prev[key]
        if seen is None:  # baseline predates books: treat any book as already sent
            continue
        if (row.get("signal_book") or "").strip() not in seen:
            fresh.append(row)
    return fresh


def run(*, comparison_csv: str = DEFAULT_COMPARISON_CSV, dry_run: bool = False) -> int:
    """Send Telegram alerts for newly-fired bet signals. Returns the number sent."""
    current = _read_csv_rows(comparison_csv)
    if not current:
        log.info("No comparison rows at %s; nothing to alert.", comparison_csv)
        return 0

    previous = _previous_committed_rows(comparison_csv)
    if previous is None:
        return 0  # no baseline -> stay silent rather than blast every open signal

    fresh = new_signals(current, previous)
    if not fresh:
        log.info("No new bet signals since the last published comparison.")
        return 0

    log.info("New bet signal(s): %d", len(fresh))
    if dry_run:
        for row in fresh:
            log.info("Would send:\n%s", format_message(row))
        return 0

    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if not token or not chat_id:
        log.warning("%s / %s not set; %d new signal(s) not sent.", TOKEN_ENV, CHAT_ENV, len(fresh))
        return 0

    sent = 0
    for row in fresh:
        if send_telegram(token, chat_id, format_message(row)):
            sent += 1
    log.info("Sent %d/%d Telegram signal alert(s).", sent, len(fresh))
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram alerts for newly-fired opening-line bet signals (1xBet / Duel)."
    )
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON_CSV,
                        help="Full market-comparison CSV (with signal_state/signal_pick/signal_book).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be sent; send nothing.")
    args = parser.parse_args()

    # Fail-open: a notifier must never fail a publish. Log and exit 0 on any error.
    try:
        run(comparison_csv=args.comparison, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - top-level guard, must not break the pipeline
        log.error("signal_alert failed (ignored): %s", exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
