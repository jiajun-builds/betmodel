"""Predict when Pinnacle opens each upcoming CSL match, from the observed pattern:

    A future match's line opens within ~1 hour AFTER the later of the two teams'
    most-recent matches has kicked off.

So for an upcoming fixture (home i, away j) we take each team's previous match
kickoff, use the later of the two as the anchor, and predict the opening window
[anchor, anchor + 1h].

This is a timing test: run it with the current round as target (default) to check
predicted windows against the round's already-opened Pinnacle lines. Once the timing
is confirmed, point --target at the next round to schedule live captures.

Timezone: the source CSVs store kickoff times in UTC (GMT, i.e. UK time WITHOUT
daylight saving). We parse them as UTC and convert to Europe/London so the output is
the real UK local kickoff — in summer this correctly adds the +1h BST offset, matching
the observed open times. (An earlier version treated the raw values as already-local
wall-clock and so came out 1h early in summer.)

Deliberately stdlib-only (csv/datetime/zoneinfo), so it runs anywhere without pandas.

Run (repo root):
    python -m csl.odds.opening_calendar
    python -m csl.odds.opening_calendar --target data/raw_data/chn_upcoming_fixtures.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from csl.paths import data_output_dir, data_raw_dir

# Source CSVs store kickoff times in UTC; we convert to real UK local time.
UK_TZ = ZoneInfo("Europe/London")

DEFAULT_SCHEDULE_CSV = os.path.join(data_raw_dir(), "chinese_super_league_data.csv")
DEFAULT_TARGET_CSV = os.path.join(data_raw_dir(), "chn_upcoming_fixtures.csv")
DEFAULT_OUT_CSV = os.path.join(data_output_dir(), "CHN_opening_time_calendar.csv")

DEFAULT_WINDOW_HOURS = 1.0

OUTPUT_COLUMNS = [
    "round",
    "kickoff_at",
    "home",
    "away",
    "home_prev_opp",
    "home_prev_kickoff",
    "away_prev_opp",
    "away_prev_kickoff",
    "anchor_kickoff",
    "predicted_open_from",
    "predicted_open_to",
    "note",
]

DT_FMT = "%Y-%m-%d %H:%M"


def _parse_kickoff(date_str: str, time_str: str) -> datetime | None:
    """Combine a canonical YYYY-MM-DD date and an HH:MM time into a UTC-aware datetime.

    The source stores kickoff in UTC, so we attach UTC here and keep all internal
    comparison/arithmetic in UTC; conversion to UK local happens only at format time.
    Returns None when either field is missing or unparseable, so rows without a
    scheduled time are skipped rather than crashing the run.
    """
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()
    if not date_str or not time_str:
        return None
    # Times occasionally arrive as HH:MM:SS; keep the first five chars (HH:MM).
    time_str = time_str[:5]
    try:
        naive = datetime.strptime(f"{date_str} {time_str}", DT_FMT)
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)


def _fmt(dt: datetime | None) -> str:
    """Render a UTC-aware datetime as real UK local (Europe/London, DST-aware)."""
    return dt.astimezone(UK_TZ).strftime(DT_FMT) if dt is not None else ""


def load_schedule(path: str) -> dict[str, list[tuple[datetime, str, str]]]:
    """Build a per-team schedule: team -> list of (kickoff, opponent, date_str).

    Reads the full-season file (results blank for unplayed rounds). Every match
    contributes an entry for both the home and away team. utf-8-sig strips the BOM
    on the leading header column.
    """
    per_team: dict[str, list[tuple[datetime, str, str]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            kickoff = _parse_kickoff(row.get("Date", ""), row.get("Time", ""))
            if kickoff is None:
                continue
            home = (row.get("Home") or "").strip()
            away = (row.get("Away") or "").strip()
            date_str = (row.get("Date") or "").strip()
            if not home or not away:
                continue
            per_team.setdefault(home, []).append((kickoff, away, date_str))
            per_team.setdefault(away, []).append((kickoff, home, date_str))
    for entries in per_team.values():
        entries.sort(key=lambda e: e[0])
    return per_team


def previous_match(
    per_team: dict[str, list[tuple[datetime, str, str]]],
    team: str,
    target_kickoff: datetime,
    other_team: str,
    target_date: str,
) -> tuple[datetime, str] | None:
    """The team's most-recent match strictly before target_kickoff.

    Excludes the target fixture itself (same date and opponent) so that a schedule
    entry for this very match — even if its stored time differs slightly from the
    target file — can never be mistaken for the "previous" match.
    Returns (kickoff, opponent) or None if the team has no prior match.
    """
    best: tuple[datetime, str] | None = None
    for kickoff, opponent, date_str in per_team.get(team, []):
        if kickoff >= target_kickoff:
            continue
        if date_str == target_date and opponent == other_team:
            continue  # the target fixture itself
        if best is None or kickoff > best[0]:
            best = (kickoff, opponent)
    return best


@dataclass(frozen=True)
class OpenWindow:
    """A fixture's predicted Pinnacle opening window, with UTC-aware datetimes.

    build_calendar formats everything to UK-local strings for display/CSV; a
    scheduler needs real tz-aware datetimes to compare against "now", so this is
    the structured form the timing logic is derived from. All datetimes are UTC.
    """

    round: str
    home: str
    away: str
    kickoff: datetime
    home_prev: tuple[datetime, str] | None
    away_prev: tuple[datetime, str] | None
    anchor: datetime | None
    open_from: datetime | None
    open_to: datetime | None
    note: str


def build_open_windows(
    schedule_path: str = DEFAULT_SCHEDULE_CSV,
    target_path: str = DEFAULT_TARGET_CSV,
    window_hours: float = DEFAULT_WINDOW_HOURS,
) -> list[OpenWindow]:
    """Compute each target fixture's predicted open window as tz-aware datetimes."""
    per_team = load_schedule(schedule_path)
    window = timedelta(hours=window_hours)
    windows: list[OpenWindow] = []

    with open(target_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            home = (row.get("Home") or "").strip()
            away = (row.get("Away") or "").strip()
            date_str = (row.get("Date") or "").strip()
            kickoff = _parse_kickoff(date_str, row.get("Time", ""))
            round_str = (row.get("Wk") or row.get("Round") or "").strip()
            if not home or not away or kickoff is None:
                continue

            home_prev = previous_match(per_team, home, kickoff, away, date_str)
            away_prev = previous_match(per_team, away, kickoff, home, date_str)

            anchor: datetime | None = None
            note = ""
            if home_prev is None or away_prev is None:
                missing = [t for t, p in ((home, home_prev), (away, away_prev)) if p is None]
                note = "no prior match for: " + ", ".join(missing)
            else:
                anchor = max(home_prev[0], away_prev[0])

            # Cap the window at kickoff: an "open" captured after kickoff is not an
            # opening line (the fixture also drops out of the pre-match feed then), and
            # leaving open_to past kickoff would make a never-captured fixture stay
            # "pending" after kickoff and burn a credit on every tick for nothing.
            open_to = min(anchor + window, kickoff) if anchor else None

            windows.append(
                OpenWindow(
                    round=round_str,
                    home=home,
                    away=away,
                    kickoff=kickoff,
                    home_prev=home_prev,
                    away_prev=away_prev,
                    anchor=anchor,
                    open_from=anchor,
                    open_to=open_to,
                    note=note,
                )
            )
    return windows


def build_calendar(
    schedule_path: str,
    target_path: str,
    window_hours: float,
) -> list[dict[str, str]]:
    """UK-local string rows for display/CSV, derived from build_open_windows."""
    rows: list[dict[str, str]] = []
    for w in build_open_windows(schedule_path, target_path, window_hours):
        rows.append(
            {
                "round": w.round,
                "kickoff_at": _fmt(w.kickoff),
                "home": w.home,
                "away": w.away,
                "home_prev_opp": w.home_prev[1] if w.home_prev else "",
                "home_prev_kickoff": _fmt(w.home_prev[0]) if w.home_prev else "",
                "away_prev_opp": w.away_prev[1] if w.away_prev else "",
                "away_prev_kickoff": _fmt(w.away_prev[0]) if w.away_prev else "",
                "anchor_kickoff": _fmt(w.anchor),
                "predicted_open_from": _fmt(w.anchor),
                "predicted_open_to": _fmt(w.open_to) if w.anchor else "",
                "note": w.note,
            }
        )

    rows.sort(key=lambda r: (r["predicted_open_from"] or "9999", r["kickoff_at"]))
    return rows


def write_csv(rows: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, str]]) -> None:
    header = f"{'kickoff':<16} {'home':<24} {'away':<24} {'anchor':<16} {'predicted open (UK)':<33}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["predicted_open_from"]:
            window = f"{r['predicted_open_from']} - {r['predicted_open_to'][11:]}"
        else:
            window = f"(!) {r['note']}"
        print(f"{r['kickoff_at']:<16} {r['home']:<24} {r['away']:<24} {r['anchor_kickoff']:<16} {window:<33}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Pinnacle opening-time windows for upcoming CSL fixtures."
    )
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE_CSV,
                        help="Full-season schedule CSV (kickoff times for all rounds).")
    parser.add_argument("--target", default=DEFAULT_TARGET_CSV,
                        help="Upcoming fixtures CSV to predict opening times for.")
    parser.add_argument("--out", default=DEFAULT_OUT_CSV, help="Output calendar CSV path.")
    parser.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
                        help="Hours after the anchor kickoff that the line is expected to open.")
    args = parser.parse_args()

    rows = build_calendar(args.schedule, args.target, args.window_hours)
    print(f"Fixtures: {len(rows)} | anchor pattern: later team's prev kickoff "
          f"+ 0..{args.window_hours:g}h | timezone: UK (Europe/London)\n")
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
