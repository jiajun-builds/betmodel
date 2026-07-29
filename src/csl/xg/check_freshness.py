"""Alert when the xG feed goes stale (P0-3).

xG is not fetched by CI: SofaScore's Cloudflare 403s GitHub Actions' datacenter IPs,
so a home Mac on a residential IP fetches ``xg_data.csv`` and pushes it (see
``scripts/LOCAL_XG_SETUP.md``). That machine is a single point of failure *and* the
failure is silent — the xG merge is deliberately no-erase, so a run with no fresh xG
keeps the old file, and every downstream step (model, dashboard, site) rebuilds green
on stale data. In July 2026 the fetcher sat wedged mid-rebase for 10 days and nothing
anywhere reported a problem.

This closes that hole: once a day the full refresh compares how far the xG feed has
fallen behind the results feed and pushes a Telegram alert when the gap exceeds
``XG_STALE_DAYS``.

**The comparison must be against the results CSV, not against ``xg_data.csv``'s own
status column.** When the fetcher dies, ``xg_data.csv`` freezes whole — played matches
still read "Not started" in it — so any self-consistency check on that file alone sees
nothing wrong. ``CHN_Super League.csv`` comes from a different source that CI *does*
keep current, which is what makes the gap visible.

Two conditions must hold before it alerts: the feed is behind by more than
``XG_STALE_DAYS``, *and* at least ``MIN_MISSING_MATCHES`` played matches sit past the
xG frontier. The second is what makes it robust: SofaScore does not cover every CSL
fixture, so an isolated match with a result and no xG is normal (match 16484884, a
round-18 makeup, has full stats but no Expected-goals item — it alone pushed the day
gap to exactly 3 for three days in July 2026). A dead fetcher, by contrast, strands a
whole round at once. Requiring half a round separates the two, so an uncovered fixture
can never become a permanent false alarm.

Fail-open by design, like ``csl.notify.signal_alert``: a missing token, an unreachable
Telegram, or an unreadable CSV logs and returns without raising, so a monitor can never
fail the refresh it monitors.

Env:
    TELEGRAM_BOT_TOKEN   bot token from @BotFather
    TELEGRAM_CHAT_ID     numeric chat id
    XG_STALE_DAYS        gap in days before alerting (default 3)

Usage (repo root, PYTHONPATH=src):
    python -m csl.xg.check_freshness
    python -m csl.xg.check_freshness --dry-run   # print the verdict, send nothing
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import date, datetime

from csl.notify.signal_alert import send_telegram
from csl.paths import data_raw_dir

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_LEAGUE_CSV = os.path.join(data_raw_dir(), "CHN_Super League.csv")
DEFAULT_XG_CSV = os.path.join(data_raw_dir(), "xg_data.csv")

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"
STALE_DAYS_ENV = "XG_STALE_DAYS"

# SofaScore publishes xG a little after the final whistle, and the fetcher runs once a
# day, so a 1-2 day gap is normal operation. 3 leaves margin for a single missed run
# without crying wolf, while still catching a real outage within days rather than weeks.
DEFAULT_STALE_DAYS = 3

# Half a CSL round (8 matches). Below this, the gap is explained by fixtures SofaScore
# simply does not cover, not by a dead fetcher — see the module docstring.
MIN_MISSING_MATCHES = 4

LOG_PATH = "~/Library/Logs/cslmonitor-fetch-xg.log"


def _parse_date(value: str) -> date | None:
    """Parse an ISO date; return None for blanks and anything unparseable."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _latest_dated_row(
    path: str, date_field: str, required_fields: tuple[str, ...], round_field: str
) -> tuple[date, str] | None:
    """Latest (date, round) among rows where every ``required_fields`` cell is filled."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return None

    best: tuple[date, str] | None = None
    for row in rows:
        if any(not (row.get(field) or "").strip() for field in required_fields):
            continue
        parsed = _parse_date(row.get(date_field, ""))
        if parsed is None:
            continue
        if best is None or parsed > best[0]:
            best = (parsed, (row.get(round_field) or "?").strip())
    return best


def latest_result(league_csv: str = DEFAULT_LEAGUE_CSV) -> tuple[date, str] | None:
    """Latest match with a recorded score in the (CI-maintained) results CSV."""
    return _latest_dated_row(league_csv, "Date", ("HG", "AG"), "Round")


def latest_xg(xg_csv: str = DEFAULT_XG_CSV) -> tuple[date, str] | None:
    """Latest match carrying xG in the (home-Mac-maintained) xG CSV."""
    return _latest_dated_row(xg_csv, "date", ("home_xg", "away_xg"), "round")


def played_since(cutoff: date, league_csv: str = DEFAULT_LEAGUE_CSV) -> int:
    """How many matches have a recorded score dated after ``cutoff``."""
    try:
        with open(league_csv, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        log.warning("Cannot read %s: %s", league_csv, exc)
        return 0

    count = 0
    for row in rows:
        if not (row.get("HG") or "").strip() or not (row.get("AG") or "").strip():
            continue
        parsed = _parse_date(row.get("Date", ""))
        if parsed is not None and parsed > cutoff:
            count += 1
    return count


def format_message(
    result: tuple[date, str], xg: tuple[date, str], gap: int, missing: int, threshold: int
) -> str:
    """Telegram HTML body naming the gap and where to look."""
    return "\n".join(
        [
            "⚠️ <b>xG feed stale</b>",
            "",
            f"Latest result:  <b>{result[0]}</b>  (round {result[1]})",
            f"Latest xG:      <b>{xg[0]}</b>  (round {xg[1]})",
            f"Gap:            <b>{gap} days</b>  (alerts above {threshold})",
            f"Played, no xG:  <b>{missing} match(es)</b>",
            "",
            "The home Mac has not pushed xG. The model trains on xG, so",
            "those matches are excluded from the fit.",
            "",
            "<code>launchctl list | grep cslmonitor</code>",
            f"<code>tail -50 {LOG_PATH}</code>",
        ]
    )


def check(
    *, league_csv: str = DEFAULT_LEAGUE_CSV, xg_csv: str = DEFAULT_XG_CSV
) -> tuple[int, int, tuple[date, str], tuple[date, str]] | None:
    """Return ``(gap_days, missing_matches, latest_result, latest_xg)``.

    None when either CSV is unreadable — a monitor that cannot see stays quiet.
    """
    result = latest_result(league_csv)
    xg = latest_xg(xg_csv)
    if result is None or xg is None:
        log.warning("Cannot compare freshness (results=%s, xg=%s).", result, xg)
        return None
    return (result[0] - xg[0]).days, played_since(xg[0], league_csv), result, xg


def run(
    *,
    league_csv: str = DEFAULT_LEAGUE_CSV,
    xg_csv: str = DEFAULT_XG_CSV,
    stale_days: int | None = None,
    dry_run: bool = False,
) -> int:
    """Alert if xG trails results by more than the threshold. Returns 1 if sent, else 0."""
    if stale_days is None:
        raw = os.environ.get(STALE_DAYS_ENV, "").strip()
        try:
            stale_days = int(raw) if raw else DEFAULT_STALE_DAYS
        except ValueError:
            log.warning("%s=%r is not an integer; using %d.", STALE_DAYS_ENV, raw, DEFAULT_STALE_DAYS)
            stale_days = DEFAULT_STALE_DAYS

    verdict = check(league_csv=league_csv, xg_csv=xg_csv)
    if verdict is None:
        return 0

    gap, missing, result, xg = verdict
    if gap <= stale_days or missing < MIN_MISSING_MATCHES:
        log.info(
            "xG is fresh: results %s, xG %s, gap %d day(s), %d match(es) past the frontier.",
            result[0], xg[0], gap, missing,
        )
        return 0

    message = format_message(result, xg, gap, missing, stale_days)
    log.warning(
        "xG is STALE: results %s, xG %s, gap %d day(s), %d played match(es) with no xG.",
        result[0], xg[0], gap, missing,
    )
    if dry_run:
        log.info("Would send:\n%s", message)
        return 0

    token = os.environ.get(TOKEN_ENV, "").strip()
    chat_id = os.environ.get(CHAT_ENV, "").strip()
    if not token or not chat_id:
        log.warning("%s / %s not set; stale-xG alert not sent.", TOKEN_ENV, CHAT_ENV)
        return 0

    return 1 if send_telegram(token, chat_id, message) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram alert when the xG feed goes stale.")
    parser.add_argument("--league", default=DEFAULT_LEAGUE_CSV, help="Results CSV (CI-maintained).")
    parser.add_argument("--xg", default=DEFAULT_XG_CSV, help="xG CSV (home-Mac-maintained).")
    parser.add_argument("--stale-days", type=int, default=None,
                        help=f"Gap in days before alerting (default {DEFAULT_STALE_DAYS}).")
    parser.add_argument("--dry-run", action="store_true", help="Print the verdict; send nothing.")
    args = parser.parse_args()

    # Fail-open: a monitor must never fail the refresh it monitors.
    try:
        run(league_csv=args.league, xg_csv=args.xg, stale_days=args.stale_days, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - top-level guard, must not break the pipeline
        log.error("check_freshness failed (ignored): %s", exc)


if __name__ == "__main__":
    main()
