import sys
import subprocess

for _pkg in ['pandas', 'requests']:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg, '--quiet'])

import requests
import pandas as pd

import shutil
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict

from csl.date_utils import format_date_only_series, parse_date_only_series
from csl.paths import data_output_dir, data_raw_dir

# ═══════════════════════════════════════════════════════════════════
#  Chinese Super League — Fixture Fetcher + Auto-Updater
#
#  Step 1: Fetch 2026 CSL fixtures/results from TheSportsDB
#  Step 2: Save fresh data to chinese_super_league_data.csv
#  Step 3: Merge NEW played matches into CHN_Super League.csv
#  Step 4: Write next 14-day fixtures into "Fixtures" sheet
#          in the Excel tracker (other sheets untouched)
# ═══════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────
_raw = data_raw_dir()
FRESH_DATA_PATH = os.path.join(_raw, "chinese_super_league_data.csv")
RAW_DATA_PATH = os.path.join(_raw, "CHN_Super League.csv")
BACKUP_DIR = os.path.join(_raw, "backups")
FIXTURE_CSV_PATH = os.path.join(_raw, "chn_upcoming_fixtures.csv")
# ── TheSportsDB config ────────────────────────────────────────────
BASE_URL   = "https://www.thesportsdb.com/api/v1/json/123"
CSL_ID     = "4359"
MAX_ROUNDS = 35

SEASONS_TO_FETCH = [
    ("2026", ["2026", "2026-2027"]),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
}

# ── Team name mapping（项目内 data/output_data）────────────────────
TEAM_MAPPING_PATH = os.path.join(data_output_dir(), "CHN_team_name_mapping.csv")
team_mapping_dict = {}
if os.path.exists(TEAM_MAPPING_PATH):
    df_map = pd.read_csv(TEAM_MAPPING_PATH)
    # match_team (TheSportsDB name) → standard_team (your standardised name)
    team_mapping_dict = dict(zip(df_map['match_team'], df_map['standard_team']))
    print(f"  ✓ Team name mapping loaded: {len(team_mapping_dict)} teams from {TEAM_MAPPING_PATH}")
else:
    print(f"⚠️  Team name mapping not found at {TEAM_MAPPING_PATH} — using TheSportsDB names as-is")


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — FETCH FROM API
# ══════════════════════════════════════════════════════════════════

def safe_get(url, params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code} on attempt {attempt+1}/{max_retries}")
                time.sleep(5)
        except Exception as e:
            print(f"  Error on attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(5)
    return {}


def resolve_season_string(format_strings):
    url = f"{BASE_URL}/eventsround.php"
    for fmt in format_strings:
        data = safe_get(url, params={"id": CSL_ID, "r": 1, "s": fmt})
        if data.get("events"):
            print(f"  Season string resolved: '{fmt}'")
            return fmt
        time.sleep(1)
    return None


def fetch_all_rounds(season_str):
    url, all_events, empty_streak = f"{BASE_URL}/eventsround.php", [], 0
    for rnd in range(1, MAX_ROUNDS + 1):
        data   = safe_get(url, params={"id": CSL_ID, "r": rnd, "s": season_str})
        events = data.get("events") or []
        if events:
            empty_streak = 0
            played = sum(1 for e in events if e.get("intHomeScore") not in (None, ""))
            print(f"  Round {rnd:>2}: {len(events):>2} matches ({played} played)")
            all_events.extend(events)
        else:
            empty_streak += 1
            print(f"  Round {rnd:>2}: no data")
            if empty_streak >= 3:
                print(f"  3 consecutive empty rounds — stopping")
                break
        time.sleep(2)
    return all_events


def parse_event(event, season_label):
    hs, as_ = event.get("intHomeScore"), event.get("intAwayScore")
    try:
        hg = int(hs) if hs not in (None, "") else None
        ag = int(as_) if as_ not in (None, "") else None
    except (ValueError, TypeError):
        hg, ag = None, None
    result = None
    if hg is not None and ag is not None:
        result = "H" if hg > ag else ("A" if hg < ag else "D")
    mt = event.get("strTime") or ""
    mt = mt[:5] if len(mt) >= 5 else None
    rnd_raw = event.get("intRound")
    try:
        round_num = int(rnd_raw) if rnd_raw not in (None, "") else None
    except (TypeError, ValueError):
        round_num = None
    return {
        "Country": "China", "League": "Super League", "Season": season_label,
        "Round": round_num, "Date": event.get("dateEvent"), "Time": mt,
        "Home": event.get("strHomeTeam"), "Away": event.get("strAwayTeam"),
        "HG": hg, "AG": ag, "HxG": None, "AxG": None,
        "HExpG+": None, "AExpG+": None, "Res": result,
        "PSCH": None, "PSCD": None, "PSCA": None,
    }


def normalize_match_columns(df):
    df["Season"] = df["Season"].astype(str).str.strip()
    df["Home"] = df["Home"].astype(str).str.strip()
    df["Away"] = df["Away"].astype(str).str.strip()
    df["Date"] = parse_date_only_series(df["Date"])
    return df


def warn_on_duplicate_matches(df, label):
    dup_mask = df.duplicated(subset=["Season", "Home", "Away", "Date"], keep=False)
    dup_count = int(dup_mask.sum())
    if not dup_count:
        return 0

    sample = df.loc[dup_mask, ["Season", "Date", "Home", "Away"]].copy()
    sample["Date"] = format_date_only_series(sample["Date"])
    sample = sample.drop_duplicates().head(10)
    print(f"  ⚠️  {label} contains {dup_count} duplicate row(s) on exact match key; sample:")
    print(sample.to_string(index=False))
    return dup_count


def deduplicate_matches(df):
    work = df.copy()
    work["_row_order"] = range(len(work))
    work["_info_score"] = work.notna().sum(axis=1)
    work["_round_penalty"] = pd.to_numeric(work["Round"], errors="coerce").isna().astype(int)

    work = work.sort_values(
        ["Season", "Home", "Away", "Date", "_info_score", "_round_penalty", "_row_order"],
        ascending=[True, True, True, True, False, True, False],
    )
    work = work.drop_duplicates(subset=["Season", "Home", "Away", "Date"], keep="first")
    work = work.sort_values("_row_order").drop(columns=["_row_order", "_info_score", "_round_penalty"])
    return work.reset_index(drop=True)


print("\n" + "=" * 60)
print("  STEP 1: Fetching CSL fixtures from TheSportsDB")
print("=" * 60)

all_rows = []
for season_label, format_strings in SEASONS_TO_FETCH:
    print(f"\nSeason {season_label} — resolving season string...")
    season_str = resolve_season_string(format_strings)
    if not season_str:
        print(f"  ⚠️  No data found for {season_label} — may not be available yet")
        continue
    events = fetch_all_rounds(season_str)
    for event in events:
        all_rows.append(parse_event(event, season_label))
    total  = sum(1 for r in all_rows if r["Season"] == season_label)
    played = sum(1 for r in all_rows if r["Season"] == season_label and r["Res"] is not None)
    print(f"  → {total} matches total | {played} played | {total - played} upcoming")

if not all_rows:
    print("\n❌ No data retrieved from API. Exiting.")
    sys.exit(1)

col_order = [
    "Country", "League", "Season", "Round", "Date", "Time",
    "Home", "Away", "HG", "AG", "HxG", "AxG",
    "HExpG+", "AExpG+", "Res", "PSCH", "PSCD", "PSCA"
]
df_fresh = pd.DataFrame(all_rows)[col_order]
df_fresh = df_fresh.sort_values(["Season", "Date", "Time"]).reset_index(drop=True)
if team_mapping_dict:
    for col in ("Home", "Away"):
        df_fresh[col] = df_fresh[col].map(team_mapping_dict).fillna(df_fresh[col])

os.makedirs(os.path.dirname(FRESH_DATA_PATH), exist_ok=True)
df_fresh.to_csv(FRESH_DATA_PATH, index=False, encoding="utf-8-sig")
print(f"\n✓ Fresh data saved to:\n  {FRESH_DATA_PATH}")


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — MERGE INTO CHN_Super League.csv
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 2: Merging new results into CHN_Super League.csv")
print("=" * 60)

if not os.path.exists(RAW_DATA_PATH):
    print(f"  ❌ Raw data file not found:\n  {RAW_DATA_PATH}")
    sys.exit(1)

df_raw = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
print(f"\n  Loaded {len(df_raw)} existing rows from CHN_Super League.csv")

os.makedirs(BACKUP_DIR, exist_ok=True)
ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(BACKUP_DIR, f"CHN_Super League_backup_{ts}.csv")
shutil.copy2(RAW_DATA_PATH, backup_path)
print(f"  Backup saved: {backup_path}")

df_fresh = normalize_match_columns(df_fresh)
df_raw = normalize_match_columns(df_raw)

df_played = df_fresh[df_fresh["Res"].notna() & (df_fresh["Res"] != "")].copy()
print(f"\n  Played matches in fresh data : {len(df_played)}")
print(f"  Upcoming (skipped)           : {len(df_fresh) - len(df_played)}")

fresh_dup_count = warn_on_duplicate_matches(df_played, "Fresh played data")
if fresh_dup_count:
    before = len(df_played)
    df_played = deduplicate_matches(df_played)
    print(f"  Deduplicated fresh played rows: {before} -> {len(df_played)}")

raw_dup_count = warn_on_duplicate_matches(df_raw, "Existing raw data")
if raw_dup_count:
    before = len(df_raw)
    df_raw = deduplicate_matches(df_raw)
    print(f"  Deduplicated existing raw rows: {before} -> {len(df_raw)}")

# Exact key first, then keep the old ±1 day fallback only as a compatibility
# guard for feeds that slightly shift kickoff dates.
exact_lookup = set()
tolerant_lookup = defaultdict(set)   # (season, home, away) → set of dates

for _, row in df_raw.iterrows():
    season = str(row["Season"]).strip()
    home = str(row["Home"]).strip()
    away = str(row["Away"]).strip()
    if pd.notna(row["Date"]):
        match_date = pd.Timestamp(row["Date"])
        exact_lookup.add((season, home, away, match_date))
        tolerant_lookup[(season, home, away)].add(match_date)

def is_already_in_raw(row):
    season = str(row["Season"]).strip()
    home = str(row["Home"]).strip()
    away = str(row["Away"]).strip()
    match_date = row["Date"]
    if pd.isna(match_date):
        return False
    exact_key = (season, home, away, pd.Timestamp(match_date))
    if exact_key in exact_lookup:
        return True

    team_key = (season, home, away)
    if team_key not in tolerant_lookup:
        return False
    return any(abs((pd.Timestamp(match_date) - d).days) <= 1 for d in tolerant_lookup[team_key])

new_rows, skipped = [], 0
for _, row in df_played.iterrows():
    if is_already_in_raw(row):
        skipped += 1
    else:
        new_rows.append(row)

print(f"  Already in raw (skipped)     : {skipped}")
print(f"  New matches to append        : {len(new_rows)}")

if new_rows:
    df_new = pd.DataFrame(new_rows)
    final_dup_count = warn_on_duplicate_matches(df_new, "New rows before append")
    if final_dup_count:
        before = len(df_new)
        df_new = deduplicate_matches(df_new)
        print(f"  Deduplicated pending append rows: {before} -> {len(df_new)}")
    for col in df_raw.columns:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[df_raw.columns]
    df_raw = pd.concat([df_raw, df_new], ignore_index=True)
    combined_dup_count = warn_on_duplicate_matches(df_raw, "Combined raw data before save")
    if combined_dup_count:
        before = len(df_raw)
        df_raw = deduplicate_matches(df_raw)
        print(f"  Deduplicated combined raw rows: {before} -> {len(df_raw)}")
    df_raw = df_raw.sort_values(["Season", "Date", "Time"]).reset_index(drop=True)
    latest_date = parse_date_only_series(df_new["Date"]).max().strftime("%Y-%m-%d")
    print(f"\n  ✓ {len(new_rows)} new played match(es) added")
    print(f"  ✓ Latest match date in new results: {latest_date}")
    print(f"\n  Newly appended matches:")
    df_new["Date"] = format_date_only_series(df_new["Date"])
    print(df_new[["Season", "Date", "Home", "Away", "HG", "AG", "Res"]].to_string(index=False))

df_raw["Date"] = format_date_only_series(df_raw["Date"])

# Invariant: a played row (Res populated) must have a Date. If this fires it
# almost always means a parse regression silently dropped legacy date formats —
# refuse to overwrite the CSV so the bad state cannot be auto-committed.
played_mask = df_raw["Res"].notna() & (df_raw["Res"].astype(str).str.strip() != "")
date_str = df_raw["Date"].astype("string").str.strip()
missing_date_mask = played_mask & (date_str.isna() | (date_str == ""))
if missing_date_mask.any():
    sample = df_raw.loc[missing_date_mask, ["Season", "Round", "Home", "Away", "Res"]].head(10)
    raise RuntimeError(
        f"Refusing to write CHN_Super League.csv: "
        f"{int(missing_date_mask.sum())} played row(s) have empty Date after normalization. "
        f"Likely a date-parse regression in csl.date_utils.parse_date_only_series. "
        f"Sample of affected rows:\n{sample.to_string(index=False)}"
    )

df_raw.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
print(f"\n  ✓ CHN_Super League.csv saved ({len(df_raw)} total rows)")


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — SAVE 14-DAY FIXTURES TO CSV
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  STEP 3: Saving 14-day fixture CSV")
print("=" * 60)

today    = pd.Timestamp(datetime.now().date())
deadline = today + timedelta(days=14)

df_fresh["Date"] = parse_date_only_series(df_fresh["Date"])
df_upcoming = df_fresh[
    df_fresh["Res"].isna() &
    df_fresh["Date"].notna() &
    (df_fresh["Date"] >= today) &
    (df_fresh["Date"] <= deadline)
].copy().sort_values(["Date", "Time"]).reset_index(drop=True)

# Keep only the columns useful for the tracker
df_upcoming_out = df_upcoming.copy()
df_upcoming_out["Wk"] = pd.to_numeric(df_upcoming_out["Round"], errors="coerce").astype("Int64")
df_upcoming_out["Date"] = format_date_only_series(df_upcoming_out["Date"])
df_upcoming_out = df_upcoming_out[["Wk", "Date", "Time", "Home", "Away"]]

df_upcoming_out.to_csv(FIXTURE_CSV_PATH, index=False, encoding="utf-8-sig")

print(f"\n  Fixtures in next 14 days : {len(df_upcoming_out)}")
if len(df_upcoming_out):
    print(f"  From : {df_upcoming_out['Date'].min()}")
    print(f"  To   : {df_upcoming_out['Date'].max()}")
print(f"\n  ✓ Saved to:\n  {FIXTURE_CSV_PATH}")

# ── Final summary ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  DONE")
print(f"{'=' * 60}")
print(f"  New results added          : {len(new_rows)}")
if new_rows:
    print(f"  Latest played match date   : {latest_date}")
print(f"  14-day fixtures saved      : {len(df_upcoming_out)} → chn_upcoming_fixtures.csv")
print(f"  CSV backup                 : {backup_path}")
