"""Polymarket CSL open/close odds capture -> merge into the raw CSV.

Polymarket lists every Chinese Super League fixture as a *Full-time result* event
holding three independent binary markets — Home win (Yes/No), Draw (Yes/No), Away win
(Yes/No). Each market's "Yes" price is a tradeable implied probability, so the decimal
odds for an outcome are ``1 / price``.

Two public, key-free endpoints drive this:

  * **Gamma** (``gamma-api.polymarket.com/events?tag_id=102764``) — enumerates the CSL
    slate. Tag ``102764`` is ``chinese-super-league``. Coverage runs from 2025-10 (tail
    of the 2025 season) through the live 2026 season.
  * **CLOB** (``clob.polymarket.com/prices-history``) — the price time series per market
    token. ``open`` is the first point (market seeding / first trade), ``close`` is the
    last point at/before kickoff. Resolved markets need a bounded ``startTs``/``endTs``
    window (the ``interval=max`` form returns nothing once a market has resolved, and the
    window may not exceed ~31 days), so we clamp to ``[createdAt, kickoff]``.

The three "Yes" prices carry an overround (each market is priced independently), so the
decimal odds are *not* vig-free — they sit alongside ``pinnacle_*`` / ``onexbet_*`` as
another book, filling six columns::

    polymarket_open_h  polymarket_open_d  polymarket_open_a
    polymarket_close_h polymarket_close_d polymarket_close_a

Fixtures are matched to CSV rows by ``(Date, Home, Away)`` after normalising Polymarket's
team names (strip the `` FC`` suffix; a short override map handles the transliterated
names, e.g. *Shanghai Haigang* -> *Shanghai Port*, *Qingdao Xihaian* -> *Qingdao West
Coast*). Polymarket's title order is ``Home vs. Away`` (verified against the CSV), and its
``endDate`` kickoff is UTC, matching the CSV's UTC date convention — so calendar dates line
up without timezone juggling.

Usage (repo root, ``PYTHONPATH=src``):
    python -m csl.odds.fetch_polymarket                # fetch + write CSV
    python -m csl.odds.fetch_polymarket --dry-run      # fetch + report, no write
    python -m csl.odds.fetch_polymarket --since 2026-03-01
    python -m csl.odds.fetch_polymarket --overwrite    # refresh already-filled cells
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from csl.paths import data_raw_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
CSL_TAG_ID = 102764  # chinese-super-league
USER_AGENT = "Mozilla/5.0 (cslmonitor polymarket capture)"

CSV_PATH = os.path.join(data_raw_dir(), "CHN_Super League.csv")

NEW_COLUMNS = [
    "polymarket_open_h",
    "polymarket_open_d",
    "polymarket_open_a",
    "polymarket_close_h",
    "polymarket_close_d",
    "polymarket_close_a",
]

# Polymarket team name (after stripping " FC") -> CSV standard name. Only the names that
# do not already match the CSV need an entry; everything else passes through unchanged.
# NOTE: the canonical alias registry is data/output_data/CHN_team_name_mapping.csv
# (consumed via fetch_pinnacle_spreads.load_team_mapping). If a second Polymarket-specific
# rename ever appears, promote these into a ``polymarket_team`` column there and normalize
# through the shared loader rather than growing this dict — it keeps one source of truth.
TEAM_OVERRIDES = {
    "Shanghai Haigang": "Shanghai Port",
    "Qingdao Xihaian": "Qingdao West Coast",
    "Tianjin Jinmen Hu": "Tianjin Jinmen Tiger",
    "Wuhan San Zhen": "Wuhan Three Towns",
    "Zhejiang Zhiye": "Zhejiang Professional",
    "Henan": "Henan Songshan Longmen",
}

# The CLOB rejects any startTs/endTs window longer than 15 days ("interval is too
# long"), independent of fidelity. Mid-season markets are bulk-created ~27 days before
# kickoff, so a single window cannot span both the opening seed and the pre-match close.
# We therefore read the open from a short window anchored at creation and the close from
# a short window anchored at kickoff. 14d leaves a safety margin under the 15d ceiling.
MAX_HISTORY_WINDOW = timedelta(days=14)

# When no CSV row shares a fixture's exact kickoff date, accept the nearest same-teams
# row within this many days (postponements shift the date but not the pairing).
DATE_TOLERANCE_DAYS = 3


def normalize_team(poly_name: str) -> str:
    """Map a Polymarket team name to the CSV's standard spelling."""
    base = poly_name.strip()
    if base.endswith(" FC"):
        base = base[:-3].strip()
    return TEAM_OVERRIDES.get(base, base)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _parse_ts(iso: str) -> datetime:
    """Parse a Polymarket ISO timestamp (trailing ``Z``) to an aware UTC datetime."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def is_main_ft_event(event: dict) -> bool:
    """True for the top-level 1X2 event only (skip Halftime / Exact Score / ... variants).

    The main event's title is exactly ``"Home vs. Away"`` with no ``-`` suffix, and it
    carries exactly three markets (home win / draw / away win).
    """
    title = event.get("title", "")
    parts = title.split(" vs. ")
    if len(parts) != 2:
        return False
    if "-" in parts[1]:  # "... - Halftime Result", "... - Exact Score", etc.
        return False
    return len(event.get("markets", [])) == 3


def enumerate_events(session: requests.Session, since: str | None) -> list[dict]:
    """Page the CSL tag (both resolved and live) and return the main 1X2 events.

    ``since`` (``YYYY-MM-DD``) drops events whose kickoff date is earlier, cutting the
    number of price-history calls when only recent rounds are wanted.
    """
    events: dict[str, dict] = {}
    for closed in ("true", "false"):
        offset = 0
        while True:
            resp = session.get(
                f"{GAMMA_BASE}/events",
                params={
                    "tag_id": CSL_TAG_ID,
                    "limit": 100,
                    "offset": offset,
                    "closed": closed,
                },
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for event in batch:
                if not is_main_ft_event(event):
                    continue
                if since and (event.get("endDate", "")[:10] < since):
                    continue
                events[str(event["id"])] = event
            if len(batch) < 100:
                break
            offset += 100
    ordered = sorted(events.values(), key=lambda e: e.get("endDate", ""))
    log.info("Found %d Polymarket CSL main events%s", len(ordered),
             f" since {since}" if since else "")
    return ordered


def classify_markets(event: dict) -> dict[str, dict] | None:
    """Map each of the event's three markets to 'h' / 'd' / 'a'.

    Home/away come from the title (``Home vs. Away``); the draw market's question
    contains ``"end in a draw"``; the two win markets name the winning team, which we
    match back to home vs away.
    """
    parts = event["title"].split(" vs. ")
    home_raw, away_raw = parts[0].strip(), parts[1].strip()
    slots: dict[str, dict] = {}
    for market in event.get("markets", []):
        question = market.get("question", "")
        if "end in a draw" in question:
            slots["d"] = market
        elif home_raw in question:
            slots["h"] = market
        elif away_raw in question:
            slots["a"] = market
    if set(slots) != {"h", "d", "a"}:
        log.warning("Could not classify all 3 markets for %r (got %s)",
                    event["title"], sorted(slots))
        return None
    return slots


def yes_token(market: dict) -> str | None:
    """The clobTokenId of the market's 'Yes' outcome (index of 'Yes' in outcomes)."""
    try:
        outcomes = ast.literal_eval(market["outcomes"])
        tokens = ast.literal_eval(market["clobTokenIds"])
    except (KeyError, ValueError, SyntaxError):
        return None
    if "Yes" not in outcomes:
        return None
    return tokens[outcomes.index("Yes")]


def _history(
    session: requests.Session, token: str, start: datetime, end: datetime, fidelity: int,
) -> list[dict]:
    """Raw price-history points for ``token`` in ``[start, end]`` (window must be <=15d)."""
    resp = session.get(
        f"{CLOB_BASE}/prices-history",
        params={
            "market": token,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "fidelity": fidelity,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("history", [])


def open_close_prices(
    session: requests.Session, token: str, created: datetime, kickoff: datetime,
    fidelity: int,
) -> tuple[float | None, float | None]:
    """Return (open, close) prices using two <=14d windows either side of the 15d cap.

    ``open`` is the first point in ``[created, created+14d]`` (the seeding / first trade);
    ``close`` is the last point in ``[kickoff-14d, kickoff]`` (the pre-match line). When the
    market was created within 14 days of kickoff a single ``[created, kickoff]`` window
    already spans both anchors, so we fetch once and read both ends off it — only the
    genuinely long-lived (bulk-created ~27d out) markets need the second call.
    """
    open_end = min(kickoff, created + MAX_HISTORY_WINDOW)
    open_hist = _history(session, token, created, open_end, fidelity)
    open_p = open_hist[0]["p"] if open_hist else None

    if open_end >= kickoff:  # one window covers open→close; no second round-trip needed
        close_hist = open_hist
    else:
        close_start = max(created, kickoff - MAX_HISTORY_WINDOW)
        close_hist = _history(session, token, close_start, kickoff, fidelity)
    close_p = close_hist[-1]["p"] if close_hist else None

    return open_p, close_p


def price_to_odds(price: float | None) -> str:
    """Decimal odds = 1/price, 2dp, as a string. Empty for missing/degenerate prices."""
    if not price or price <= 0:
        return ""
    return f"{round(1.0 / price, 2):.2f}"


def collect_odds(
    session: requests.Session, events: list[dict], fidelity: int, pause: float,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Fetch open/close odds for every event, keyed by (date, home_std, away_std)."""
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for i, event in enumerate(events, 1):
        slots = classify_markets(event)
        if slots is None:
            continue
        parts = event["title"].split(" vs. ")
        home = normalize_team(parts[0])
        away = normalize_team(parts[1])
        date = event.get("endDate", "")[:10]
        kickoff = _parse_ts(event["endDate"])
        row: dict[str, str] = {}
        for slot, market in slots.items():
            token = yes_token(market)
            if token is None:
                continue
            created = _parse_ts(market.get("createdAt") or event.get("createdAt")
                                or event["startDate"])
            try:
                open_p, close_p = open_close_prices(
                    session, token, created, kickoff, fidelity)
            except requests.RequestException as exc:
                log.warning("price history failed for %s (%s): %s",
                            event["title"], slot, exc)
                open_p = close_p = None
            row[f"polymarket_open_{slot}"] = price_to_odds(open_p)
            row[f"polymarket_close_{slot}"] = price_to_odds(close_p)
            if pause:
                time.sleep(pause)
        out[(date, home, away)] = row
        log.info("[%3d/%d] %s  %s vs %s  open H/D/A=%s/%s/%s",
                 i, len(events), date, home, away,
                 row.get("polymarket_open_h", ""), row.get("polymarket_open_d", ""),
                 row.get("polymarket_open_a", ""))
    return out


def merge_into_csv(
    csv_path: str, odds: dict[tuple[str, str, str], dict[str, str]],
    overwrite: bool, dry_run: bool,
) -> tuple[int, int]:
    """Write the Polymarket columns into matching CSV rows. Returns (rows_matched, cells_written)."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for col in NEW_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    # Index CSV rows by (home, away) so a Polymarket fixture whose kickoff date differs
    # from the CSV's (postponements/reschedules shift the date by a day or three while the
    # teams stay put) can still be matched to the nearest same-teams row within tolerance.
    by_teams: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_teams.setdefault((row.get("Home", ""), row.get("Away", "")), []).append(row)

    def resolve_row(date: str, home: str, away: str) -> dict | None:
        candidates = by_teams.get((home, away))
        if not candidates:
            return None
        exact = [r for r in candidates if r.get("Date") == date]
        if exact:
            return exact[0]
        try:
            target = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return None
        best, best_gap = None, timedelta(days=DATE_TOLERANCE_DAYS)
        for r in candidates:
            try:
                gap = abs(datetime.strptime(r.get("Date", ""), "%Y-%m-%d") - target)
            except ValueError:
                continue
            if gap <= best_gap:
                best, best_gap = r, gap
        return best

    rows_matched = 0
    cells_written = 0
    unmatched = set(odds)
    for date, home, away in odds:
        key = (date, home, away)
        values = odds[key]
        row = resolve_row(date, home, away)
        if row is None:
            continue
        unmatched.discard(key)
        if row.get("Date") != date:
            log.info("date-tolerant match: Polymarket %s %s vs %s -> CSV row dated %s",
                     date, home, away, row.get("Date"))
        touched = False
        for col in NEW_COLUMNS:
            new_val = values.get(col, "")
            if not new_val:
                continue
            if row.get(col) and not overwrite:
                continue
            if row.get(col, "") != new_val:
                row[col] = new_val
                cells_written += 1
                touched = True
        if touched:
            rows_matched += 1

    for key in sorted(unmatched):
        log.debug("Polymarket fixture not in CSV: %s", key)
    if unmatched:
        log.info("%d Polymarket fixtures had no matching CSV row (future rounds / "
                 "name gaps)", len(unmatched))

    if dry_run:
        log.info("[dry-run] would update %d rows / %d cells (no write)",
                 rows_matched, cells_written)
        return rows_matched, cells_written

    # Ensure every row carries the new keys so DictWriter emits aligned columns.
    for row in rows:
        for col in NEW_COLUMNS:
            row.setdefault(col, "")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows / %d cells to %s", rows_matched, cells_written, csv_path)
    return rows_matched, cells_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket CSL open/close odds and merge into the raw CSV.")
    parser.add_argument("--csv", default=CSV_PATH, help="target CSV (default: raw CSL CSV)")
    parser.add_argument("--since", default=None,
                        help="only fixtures kicking off on/after YYYY-MM-DD")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="price-history resolution in minutes (default 60)")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="seconds to sleep between CLOB calls (politeness)")
    parser.add_argument("--overwrite", action="store_true",
                        help="refresh cells that already hold a value")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and report, but do not write the CSV")
    args = parser.parse_args()

    session = _session()
    events = enumerate_events(session, args.since)
    if not events:
        log.error("No Polymarket CSL events found; aborting.")
        sys.exit(1)
    odds = collect_odds(session, events, args.fidelity, args.pause)
    merge_into_csv(args.csv, odds, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
