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

import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ligamx import config, paths
from ligamx.sofascore_client import SofascoreClient
from ligamx.xg.data_updater import DataUpdater

# Tournaments that make up a Liga MX year (each has its own SofaScore season ids).
TOURNAMENTS = [
    (config.SOFASCORE_APERTURA, "Apertura"),
    (config.SOFASCORE_CLAUSURA, "Clausura"),
]
MAX_ROUNDS = 40
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
        parsed = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
        m = parsed.max()
        return None if pd.isna(m) else m.date()
    except Exception as e:
        print(f"Warning: could not read latest cached date: {e}")
        return None


def run():
    client = SofascoreClient()
    updater = DataUpdater()

    max_existing = _max_existing_date()
    now_ts = datetime.now(timezone.utc).timestamp()
    print(f"Latest cached match date: {max_existing}")

    played_updates = []
    upcoming_rows = []
    unmapped = set()

    for ut, tour_name in TOURNAMENTS:
        seasons = client.seasons(ut)
        if not seasons:
            print(f"No seasons for tournament {ut}")
            continue
        sid = seasons[0]["id"]
        sname = seasons[0].get("name") or tour_name
        season_label = sname.split(",")[-1].strip() or tour_name
        print(f"\n{season_label}: tournament={ut} season={sid}")

        for rnd in range(1, MAX_ROUNDS + 1):
            events = client.round_events(ut, sid, rnd)
            if not events:
                break
            for e in events:
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

                if status == "finished":
                    # Skip anything already covered by the rolling history cache.
                    if max_existing is not None and date_mx <= str(max_existing):
                        continue
                    hxg, axg = client.event_xg(e["id"])
                    if hxg is None and axg is None:
                        continue  # xG not published yet; a 0/0 row would be rejected anyway
                    played_updates.append({
                        "commence_time": f"{date_mx}T{time_mx}:00",
                        "home_team": home_sofa,
                        "away_team": away_sofa,
                        "home_goals": e.get("homeScore", {}).get("current", 0),
                        "away_goals": e.get("awayScore", {}).get("current", 0),
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
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=UPCOMING_COLUMNS)
        w.writeheader()
        w.writerows(upcoming_rows)
    print(f"Wrote {len(upcoming_rows)} upcoming fixtures -> {out}")

    if unmapped:
        print("\n[WARN] Unmapped SofaScore team names (add rows to ligamx_team_name_mapping.csv):")
        for t in sorted(unmapped):
            print(f"   - {t}")


if __name__ == "__main__":
    run()
