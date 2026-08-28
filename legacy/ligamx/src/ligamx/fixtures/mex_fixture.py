"""
Fetch Liga MX fixtures / results / upcoming from the SofaScore API (single source)
and:
  - upsert newly-played matches (with xG) into MEX_ligamx.csv,
  - write the unplayed fixtures to MEX_upcoming_fixtures.csv.

Times are stored in America/Mexico_City. To stay duplicate-safe against the
historical cache (whose kickoff timezone is inconsistent), only matches strictly
newer than the latest cached match date are upserted; the rolling history already
covers everything before that.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ligamx import config, paths, stage_meta
from ligamx.sofascore_client import SofascoreClient, event_goals, round_label
from ligamx.xg.data_updater import DataUpdater
from ligamx.date_utils import parse_date_only_series

# Tournaments that make up a Liga MX year (each has its own SofaScore season ids).
TOURNAMENTS = [
    (config.SOFASCORE_APERTURA, "Apertura"),
    (config.SOFASCORE_CLAUSURA, "Clausura"),
]
MX_TZ = ZoneInfo("America/Mexico_City")

UPCOMING_COLUMNS = ["Season", "round", "Date", "Time", "Home", "Away", "kickoff_utc", "sofa_event_id"]


def _std(name: str) -> str:
    return config.SOFASCORE_TO_STANDARD.get(name, name)


def _parts(ts: int):
    """(date_mx 'YYYY-MM-DD', time_mx 'HH:MM', kickoff_utc ISO) from a unix ts."""
    utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    local = utc.astimezone(MX_TZ)
    return local.strftime("%Y-%m-%d"), local.strftime("%H:%M"), utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _max_existing_date():
    """Latest cached match date (a date), or None if unavailable."""
    try:
        df = pd.read_csv(paths.ligamx_data_csv())
        parsed = parse_date_only_series(df["Date"])
        m = parsed.max()
        return None if pd.isna(m) else m.date()
    except Exception as e:
        print(f"Warning: could not read latest cached date: {e}")
        return None


def _existing_upcoming_count(path: str) -> int:
    """Fixture rows already on disk, or 0 if the file is absent/unreadable.

    Guards the overwrite below: a SofaScore outage yields an empty fetch that is
    indistinguishable from a finished season, and the unconditional write turned
    that into a silently blanked schedule (107 fixtures lost, 2026-08-24).
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)  # minus the header
    except OSError:
        return 0


def run(allow_empty_upcoming: bool = False):
    client = SofascoreClient()
    updater = DataUpdater()

    max_existing = _max_existing_date()
    now_ts = datetime.now(timezone.utc).timestamp()
    print(f"Latest cached match date: {max_existing}")

    played_updates = []
    upcoming_rows = []
    unmapped = set()
    missing_xg = []

    for ut, tour_name in TOURNAMENTS:
        seasons = client.seasons(ut)
        if not seasons:
            print(f"No seasons for tournament {ut}")
            continue
        sid = seasons[0]["id"]
        sname = seasons[0].get("name") or tour_name
        season_label = sname.split(",")[-1].strip() or tour_name
        print(f"\n{season_label}: tournament={ut} season={sid}")

        # Enumerate the season through the paginated events endpoints rather than
        # walking rounds 1..N. The old walk stopped at the first empty round, so it
        # only ever saw the 17-round regular season and silently dropped every
        # liguilla tie (SofaScore files those under non-sequential round numbers).
        events = client.season_events(ut, sid, "last") + client.season_events(ut, sid, "next")
        seen_ids = set()
        for e in events:
            if e["id"] in seen_ids:
                continue  # an in-progress match can surface in both pages
            seen_ids.add(e["id"])

            home_sofa = e["homeTeam"]["name"]
            away_sofa = e["awayTeam"]["name"]
            if home_sofa not in config.SOFASCORE_TO_STANDARD:
                unmapped.add(home_sofa)
            if away_sofa not in config.SOFASCORE_TO_STANDARD:
                unmapped.add(away_sofa)

            ts = e.get("startTimestamp")
            if not ts:
                continue
            date_mx, time_mx, kickoff_utc = _parts(ts)
            status = e.get("status", {}).get("type")
            rnd = round_label(e)

            if status == "finished":
                # Skip anything already covered by the rolling history cache.
                if max_existing is not None and date_mx <= str(max_existing):
                    continue
                hxg, axg = client.event_xg(e["id"])
                if hxg is None and axg is None:
                    # xG not published yet; a 0/0 row would be rejected anyway.
                    # This skip is permanent: once a later match advances the
                    # max-date watermark above, this one is never reconsidered.
                    # verify_xg is what catches it (MISSING / NO_XG).
                    missing_xg.append(f"{date_mx} {_std(home_sofa)} vs {_std(away_sofa)}")
                    continue
                hg, ag = event_goals(e)
                played_updates.append({
                    "commence_time": f"{date_mx}T{time_mx}:00",
                    "home_team": home_sofa,
                    "away_team": away_sofa,
                    "home_goals": 0 if hg is None else hg,
                    "away_goals": 0 if ag is None else ag,
                    "HxG": hxg or 0.0,
                    "AxG": axg or 0.0,
                    "season": season_label,
                    "round": rnd,
                })
            else:
                # Only genuinely-future fixtures (skip postponed/abandoned past
                # matches that never received a 'finished' status). 6h grace keeps
                # in-progress games.
                if ts < now_ts - 21600:
                    continue
                upcoming_rows.append({
                    "Season": season_label,
                    "round": rnd,
                    "Date": date_mx,
                    "Time": time_mx,
                    "Home": _std(home_sofa),
                    "Away": _std(away_sofa),
                    "kickoff_utc": kickoff_utc,
                    "sofa_event_id": e["id"],
                })

    stats = updater.upsert_matches(played_updates)
    print(f"\nUpsert into MEX_ligamx.csv: {stats}")

    upcoming_rows.sort(key=lambda r: r["kickoff_utc"])
    out = paths.upcoming_fixtures_csv()
    if not upcoming_rows and not allow_empty_upcoming and _existing_upcoming_count(out):
        raise RuntimeError(
            f"Refusing to blank {out}: the fetch returned no upcoming fixtures but "
            f"the file currently holds {_existing_upcoming_count(out)}. A partial "
            "SofaScore outage looks exactly like this. Re-run once SofaScore answers; "
            "if the season really is over, pass --allow-empty-upcoming."
        )
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=UPCOMING_COLUMNS)
        w.writeheader()
        w.writerows(upcoming_rows)
    print(f"Wrote {len(upcoming_rows)} upcoming fixtures -> {out}")

    # Only a real SofaScore round-trip may move this: it is what the dashboard
    # shows as the age of the fixture list, and CI (which cannot reach SofaScore)
    # re-exports that dashboard several times an hour. See ligamx.stage_meta.
    stamped = stage_meta.stamp(paths.fixtures_meta_json(), "fixtures_updated_at")
    print(f"Fixture-fetch timestamp recorded: {stamped}")

    if unmapped:
        print("\n[WARN] Unmapped SofaScore team names (add rows to ligamx_team_name_mapping.csv):")
        for t in sorted(unmapped):
            print(f"   - {t}")

    if missing_xg:
        print(
            f"\n[WARN] {len(missing_xg)} finished match(es) skipped -- SofaScore has "
            "not published xG for them yet:"
        )
        for m in sorted(missing_xg):
            print(f"   - {m}")
        print(
            "   These will NOT be picked up by a later 'update' (the max-date\n"
            "   watermark has moved past them). Re-run once SofaScore publishes,\n"
            "   then reconcile with: ./scripts/ligamx.sh verify-xg"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-empty-upcoming",
        action="store_true",
        help="Permit blanking MEX_upcoming_fixtures.csv. Only for a genuinely "
             "finished season -- otherwise an empty fetch is a SofaScore outage.",
    )
    run(allow_empty_upcoming=ap.parse_args().allow_empty_upcoming)
