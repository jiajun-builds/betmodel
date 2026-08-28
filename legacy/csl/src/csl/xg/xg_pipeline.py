"""
xG Scraping Pipeline — official SofaScore API
==============================================
By default, re-scrapes only the most recently completed round plus the
previous round, then merges those updates back into a full-season cache.
Use --full-season to fetch every round from round 1 until the first empty one.

Data comes from the official SofaScore API (api.sofascore.com) via curl_cffi
browser impersonation — the same xG the SofaScore app shows, no API key needed.
The merge lets fresh values win (so xG tracks SofaScore's latest, including
post-match revisions) but a blank scrape never erases an xG already in the
cache. See merge_with_existing_cache.

Team names: load data/output_data/CHN_team_name_mapping.csv and treat the
known alias columns (sofa_team / match_team / odds_team / standard_team) as
valid inputs that all normalize to standard_team.

Usage (仓库根目录，PYTHONPATH=src):
    python -m csl.xg.xg_pipeline
    python -m csl.xg.xg_pipeline --full-season

Output:
    data/raw_data/xg_data.csv  — one row per match with xG values
"""

import argparse
import os
import sys
import time
import logging
import pandas as pd
from curl_cffi import requests
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

from csl.date_utils import format_date_only_series
from csl.paths import data_output_dir, data_raw_dir

_TEAM_MAPPING_CANDIDATES = (
    os.path.join(data_output_dir(), "CHN_team_name_mapping.csv"),
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Official SofaScore API. Cloudflare fingerprints the TLS handshake, so requests
# must be made with curl_cffi's browser impersonation (see _get). This returns
# the same xG the SofaScore app shows — and fresher than the RapidAPI reseller.
BASE_URL            = "https://api.sofascore.com/api/v1"
IMPERSONATE         = "chrome"   # curl_cffi browser TLS profile
BROWSER_HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

UNIQUE_TOURNAMENT_ID = 649       # Chinese Super League
SEASON_ID            = 90049   # 2025/26 season
MAX_ROUND            = 40      # safety cap; loop normally ends at first empty round
OUTPUT_DIR = data_raw_dir()
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "xg_data.csv")
SCHEDULE_FILE = os.path.join(OUTPUT_DIR, "chinese_super_league_data.csv")

REQUEST_DELAY        = 1.2     # seconds between API calls (be polite)
MAX_RETRIES          = 3
RETRY_BACKOFF        = 2.0     # seconds; doubles on each retry
SKIP_STATUSES        = {"not started", "postponed", "canceled"}
OUTPUT_COLUMNS = [
    "match_id",
    "round",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_xg",
    "away_xg",
    "status",
]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def resolve_team_mapping_path() -> str:
    """Return the first existing path to CHN_team_name_mapping.csv."""
    for p in _TEAM_MAPPING_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "CHN_team_name_mapping.csv not found. Tried:\n  "
        + "\n  ".join(_TEAM_MAPPING_CANDIDATES)
    )


def load_sofa_standard_mapping(csv_path: str) -> dict[str, str]:
    """
    Build alias -> standard_team using known team-name columns.
    Duplicate aliases keep the last row and log a warning.
    """
    df = pd.read_csv(csv_path)
    for col in ("standard_team",):
        if col not in df.columns:
            log.error("Mapping file missing column %r: %s", col, csv_path)
            sys.exit(1)

    alias_columns = [col for col in ("sofa_team", "match_team", "odds_team", "standard_team") if col in df.columns]
    alias_rows: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        standard_raw = row.get("standard_team", "")
        if pd.isna(standard_raw):
            continue
        standard = str(standard_raw).strip()
        if not standard:
            continue
        seen_aliases: set[str] = set()
        for col in alias_columns:
            alias_raw = row.get(col, "")
            if pd.isna(alias_raw):
                continue
            alias = str(alias_raw).strip()
            if alias and alias not in seen_aliases:
                alias_rows.append((alias, standard))
                seen_aliases.add(alias)

    sub = pd.DataFrame(alias_rows, columns=["alias", "standard_team"])
    dup_mask = sub["alias"].duplicated(keep=False)
    if dup_mask.any():
        dupes = sorted(sub.loc[dup_mask, "alias"].unique().tolist())
        log.warning("Duplicate team aliases; using last row: %s", dupes)

    sub = sub.drop_duplicates(subset=["alias"], keep="last")
    mapping = dict(zip(sub["alias"], sub["standard_team"]))
    log.info("Loaded %d team aliases from %s", len(mapping), csv_path)
    return mapping


def normalize_team_name(sofa_name: str, mapping: dict[str, str]) -> str:
    """Return standard_team if mapped, else the original name unchanged."""
    cleaned = str(sofa_name or "").strip()
    return mapping.get(cleaned, cleaned)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MatchXG:
    match_id:       int
    round:          int
    date:           str
    home_team:      str
    away_team:      str
    home_score:     Optional[int]
    away_score:     Optional[int]
    home_xg:        Optional[float]
    away_xg:        Optional[float]
    status:         str
    errors:         str = field(default="", repr=False)


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> Optional[Union[dict, list]]:
    """GET with retry/back-off. Returns parsed JSON (dict or list) or None on failure.

    Uses curl_cffi browser impersonation so SofaScore's Cloudflare accepts the
    request; a plain requests/urllib call is rejected with HTTP 403.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers=BROWSER_HEADERS,
                params=params,
                timeout=15,
                impersonate=IMPERSONATE,
            )
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                log.warning("Rate-limited. Waiting %.1fs before retry %d…", wait, attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestsError as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    log.error("All retries exhausted for %s", url)
    return None


# ── Core pipeline ─────────────────────────────────────────────────────────────

def fetch_round_matches(round_num: int) -> list[dict]:
    """Return raw match objects for a single round."""
    url = (
        f"{BASE_URL}/unique-tournament/{UNIQUE_TOURNAMENT_ID}"
        f"/season/{SEASON_ID}/events/round/{round_num}"
    )
    data = _get(url)
    if not data:
        return []

    # API may return a bare list, or a dict wrapping the list
    if isinstance(data, list):
        matches = data
    elif isinstance(data, dict):
        matches = (
            data.get("events")
            or data.get("matches")
            or data.get("data", {}).get("events")
            or data.get("data", {}).get("matches")
            or []
        )
        # If still empty, surface the top-level keys so it's easy to patch
        if not matches:
            log.warning("  Round %2d — unknown response shape, top-level keys: %s",
                        round_num, list(data.keys()))
    else:
        log.warning("  Round %2d — unexpected response type: %s", round_num, type(data))
        matches = []

    log.info("  Round %2d → %d match(es) found", round_num, len(matches))
    return matches


def fetch_match_statistics(match_id: int) -> Optional[Union[dict, list]]:
    """Return the raw statistics payload for a single match."""
    url = f"{BASE_URL}/event/{match_id}/statistics"
    return _get(url)


def extract_xg(stats_payload: Optional[Union[dict, list]]) -> tuple[Optional[float], Optional[float]]:
    """
    Walk the statistics payload and return (home_xg, away_xg).

    SofaScore's statistics are grouped into periods.  We look for
    'ALL' period first, then fall back to 'REGULAR_TIME'.
    The xG stat key is typically 'Expected goals' or 'xG'.
    """
    if not stats_payload:
        return None, None

    # Navigate to the statistics list; API may return a bare list at top level
    if isinstance(stats_payload, list):
        stats_root = stats_payload
    else:
        data = stats_payload.get("data")
        stats_root = (
            stats_payload.get("statistics")
            or (data.get("statistics") if isinstance(data, dict) else None)
            or []
        )
        if not stats_root and isinstance(data, list):
            stats_root = data

    # Prefer the ALL-period aggregation
    period_order = ["ALL", "REGULAR_TIME", "1ST", "2ND"]
    period_map: dict[str, list] = {}
    for period_block in stats_root:
        if not isinstance(period_block, dict):
            continue
        period = period_block.get("period", "")
        period_map[period] = period_block.get("groups", [])

    chosen_groups: list = []
    for period_key in period_order:
        if period_key in period_map:
            chosen_groups = period_map[period_key]
            break

    xg_keys = {"Expected goals", "xG", "xg", "expected_goals", "expectedGoals"}

    for group in chosen_groups:
        if not isinstance(group, dict):
            continue
        for stat_item in group.get("statisticsItems", []):
            if stat_item.get("name") in xg_keys or stat_item.get("key") in xg_keys:
                try:
                    home_xg = float(stat_item.get("home", 0) or 0)
                    away_xg = float(stat_item.get("away", 0) or 0)
                    return home_xg, away_xg
                except (ValueError, TypeError):
                    pass

    return None, None  # xG not available for this match


def _current_score(score_block: object) -> Optional[int]:
    """Extract current score; API may return a dict or a list of period snapshots."""
    if score_block is None:
        return None
    if isinstance(score_block, dict):
        cur = score_block.get("current")
        if cur is None:
            return None
        try:
            return int(cur)
        except (TypeError, ValueError):
            return None
    if isinstance(score_block, list):
        for el in reversed(score_block):
            if isinstance(el, dict):
                cur = el.get("current")
                if cur is not None:
                    try:
                        return int(cur)
                    except (TypeError, ValueError):
                        continue
        return None
    return None


def _status_description(status_block: object) -> str:
    if isinstance(status_block, dict):
        return str(status_block.get("description", "") or "")
    return ""


def _team_name(team_block: object) -> str:
    if isinstance(team_block, dict):
        return str(team_block.get("name", "Unknown") or "Unknown")
    return "Unknown"


def _merge_event_wrapper(raw: dict) -> dict:
    """If API nests the match under 'event', merge so top-level fields resolve."""
    ev = raw.get("event")
    if isinstance(ev, dict):
        return {**raw, **ev}
    return raw


def _kickoff_unix_seconds(raw: dict) -> Optional[int]:
    """
    Resolve kickoff to Unix seconds (UTC). Handles multiple field names and ms timestamps.
    """
    candidates: list[object] = []
    for key in ("startTimestamp", "startTime", "scheduledStartTimestamp", "timestamp"):
        candidates.append(raw.get(key))
    nested = raw.get("event")
    if isinstance(nested, dict):
        for key in ("startTimestamp", "startTime", "scheduledStartTimestamp"):
            candidates.append(nested.get(key))

    for v in candidates:
        if v is None:
            continue
        try:
            ts = int(float(v))
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        # Heuristic: values > 1e12 are typically milliseconds (epoch ms)
        if ts > 10_000_000_000:
            ts = ts // 1000
        return ts
    return None


def parse_match(raw: dict, round_num: int) -> MatchXG:
    """Extract the fields we care about from a raw match object."""
    raw = _merge_event_wrapper(raw)

    match_id   = raw.get("id", 0)
    home_team  = _team_name(raw.get("homeTeam"))
    away_team  = _team_name(raw.get("awayTeam"))
    home_score = _current_score(raw.get("homeScore"))
    away_score = _current_score(raw.get("awayScore"))
    status     = _status_description(raw.get("status"))

    # Unix timestamp → date only (YYYY-MM-DD, UTC)
    kickoff = _kickoff_unix_seconds(raw)
    date_str = (
        pd.Timestamp(kickoff, unit="s", tz="UTC").strftime("%Y-%m-%d")
        if kickoff is not None
        else ""
    )

    return MatchXG(
        match_id=match_id,
        round=round_num,
        date=date_str,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        home_xg=None,
        away_xg=None,
        status=status,
    )


def _extract_round_number(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+)\s*$", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def determine_incremental_rounds(schedule_path: str) -> list[int]:
    if not os.path.isfile(schedule_path):
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")

    schedule = pd.read_csv(schedule_path)
    required = {"Round", "Res"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule file missing required columns: {sorted(missing)}")

    schedule = schedule.copy()
    schedule["round_num"] = _extract_round_number(schedule["Round"])
    schedule["Res"] = schedule["Res"].astype(str).str.strip()
    played = schedule[schedule["Res"].isin({"H", "D", "A"}) & schedule["round_num"].notna()].copy()
    if played.empty:
        raise ValueError("Schedule file has no completed matches with parseable rounds")

    latest_round = int(played["round_num"].max())
    target_rounds = sorted({round_num for round_num in (latest_round - 1, latest_round) if round_num >= 1})
    if not target_rounds:
        raise ValueError("Could not derive incremental target rounds")
    return target_rounds


def _is_match_finished(status: str) -> bool:
    return status.strip().lower() not in SKIP_STATUSES


def _records_to_frame(records: list[MatchXG]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame([asdict(r) for r in records])
    df.drop(columns=["errors"], inplace=True, errors="ignore")
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].copy()


def _normalize_xg_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[OUTPUT_COLUMNS].copy()
    out["match_id"] = pd.to_numeric(out["match_id"], errors="coerce").astype("Int64")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").round().astype("Int64")
    out["date"] = format_date_only_series(out["date"])
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce").astype("Int64")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce").astype("Int64")
    out["home_xg"] = pd.to_numeric(out["home_xg"], errors="coerce")
    out["away_xg"] = pd.to_numeric(out["away_xg"], errors="coerce")
    out["status"] = out["status"].fillna("").astype(str).str.strip()
    return out


def _standardize_team_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return df

    out = df.copy()
    for col in ("home_team", "away_team"):
        if col in out.columns:
            cleaned = out[col].astype(str).str.strip()
            out[col] = cleaned.map(mapping).fillna(cleaned)
    return out


def _add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_match_key"] = (
        out["date"].fillna("").astype(str)
        + "|"
        + out["home_team"].fillna("").astype(str)
        + "|"
        + out["away_team"].fillna("").astype(str)
    )
    return out


def _warn_duplicate_keys(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = _add_match_key(df)
    dup_mask = out.duplicated(subset=["_match_key"], keep=False)
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        sample = out.loc[dup_mask, ["date", "round", "home_team", "away_team"]].drop_duplicates().head(10)
        log.warning("%s contains %d duplicate match row(s); using last row per match key", label, dup_count)
        log.warning("\n%s", sample.to_string(index=False))
        out = out.drop_duplicates(subset=["_match_key"], keep="last")
    return out


# Columns whose existing value is carried forward when a fresh scrape returns a
# blank — a fresh gap never erases data already in the cache (see below).
COALESCE_COLUMNS = ("home_xg", "away_xg", "home_score", "away_score")


def _add_protection_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable per-match key (`_key`) used to align existing vs fresh rows.

    Prefers match_id (immutable across re-scrapes); falls back to the
    date|home|away match key for the rare row missing a match_id.
    """
    out = df if "_match_key" in df.columns else _add_match_key(df)
    out = out.copy()
    mid = pd.to_numeric(out["match_id"], errors="coerce").astype("Int64").astype("string")
    out["_key"] = mid.where(mid.notna(), "mk:" + out["_match_key"].astype(str))
    return out


def _merge_fresh_over_existing(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge a fresh scrape over the cache — fresh wins, but never erases.

    Since the official SofaScore API is the source of truth, a fresh non-blank
    value always replaces the cached one (so xG tracks SofaScore's latest,
    including post-match revisions). The one guard: a *blank* in the fresh row
    (e.g. the API hasn't computed xG yet just after full-time) is back-filled
    from the existing row, so a missing/late scrape can't wipe good data.
    Existing matches absent from the fresh scrape are retained (history).
    """
    existing = _add_protection_key(existing).drop_duplicates(subset=["_key"], keep="last")
    fresh = _add_protection_key(fresh).drop_duplicates(subset=["_key"], keep="last")
    existing = existing.set_index("_key", drop=False)
    fresh = fresh.set_index("_key", drop=False)

    # Fresh rows win, back-filling any blank xG/score from the existing row.
    fresh_keep = fresh.copy()
    for col in COALESCE_COLUMNS:
        exist_col = existing[col].reindex(fresh_keep.index)
        fresh_keep[col] = fresh_keep[col].where(fresh_keep[col].notna(), exist_col)

    # Existing matches the scrape didn't touch — retain as history.
    remaining = existing[~existing.index.isin(fresh.index)].copy()

    new_matches = int((~fresh.index.isin(existing.index)).sum())
    log.info(
        "xG cache merge: %d updated, %d retained (not re-scraped), %d new",
        len(fresh_keep) - new_matches,
        len(remaining),
        new_matches,
    )

    return pd.concat([fresh_keep, remaining], ignore_index=True)


def merge_with_existing_cache(
    fresh_df: pd.DataFrame,
    output_path: str,
    *,
    full_season: bool,
    team_aliases: dict[str, str],
) -> pd.DataFrame:
    fresh = _warn_duplicate_keys(
        _standardize_team_columns(_normalize_xg_frame(fresh_df), team_aliases),
        "Fresh xG payload",
    )

    if os.path.isfile(output_path):
        existing = pd.read_csv(output_path)
        existing = _warn_duplicate_keys(
            _standardize_team_columns(_normalize_xg_frame(existing), team_aliases),
            "Existing xG cache",
        )
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS + ["_match_key"])

    log.info("Merging fresh xG into cache (mode=%s)", "full-season" if full_season else "incremental")
    merged = _merge_fresh_over_existing(existing, fresh)
    merged = merged.drop(columns=["_match_key", "_key"], errors="ignore")
    return merged[OUTPUT_COLUMNS].sort_values(
        ["round", "date", "home_team", "away_team"], na_position="last"
    ).reset_index(drop=True)


def scrape_round_records(
    round_num: int,
    raw_matches: list[dict],
    sofa_to_standard: dict[str, str],
) -> tuple[list[MatchXG], int]:
    records: list[MatchXG] = []
    stats_requests = 0

    for raw in raw_matches:
        record = parse_match(raw, round_num)
        record.home_team = normalize_team_name(record.home_team, sofa_to_standard)
        record.away_team = normalize_team_name(record.away_team, sofa_to_standard)

        if not _is_match_finished(record.status):
            log.info("    ↳ %s vs %s — skipped (%s)", record.home_team, record.away_team, record.status)
            records.append(record)
            continue

        stats_requests += 1
        stats_payload = fetch_match_statistics(record.match_id)
        time.sleep(REQUEST_DELAY)

        home_xg, away_xg = extract_xg(stats_payload)
        record.home_xg = home_xg
        record.away_xg = away_xg

        xg_str = f"xG {home_xg:.2f}–{away_xg:.2f}" if home_xg is not None else "xG N/A"
        log.info(
            "    ✓ %s %s–%s %s  [%s]",
            record.home_team,
            record.home_score,
            record.away_score,
            record.away_team,
            xg_str,
        )
        records.append(record)

    return records, stats_requests


def fetch_specific_rounds(target_rounds: list[int], sofa_to_standard: dict[str, str]) -> tuple[pd.DataFrame, int]:
    records: list[MatchXG] = []
    stats_requests = 0

    for round_num in target_rounds:
        raw_matches = fetch_round_matches(round_num)
        time.sleep(REQUEST_DELAY)
        if not raw_matches:
            log.warning("Round %d returned no matches during incremental scrape; skipping", round_num)
            continue

        round_records, round_stats = scrape_round_records(round_num, raw_matches, sofa_to_standard)
        records.extend(round_records)
        stats_requests += round_stats

    return _records_to_frame(records), stats_requests


def fetch_full_season(sofa_to_standard: dict[str, str]) -> tuple[pd.DataFrame, int, list[int]]:
    records: list[MatchXG] = []
    stats_requests = 0
    fetched_rounds: list[int] = []
    last_round_with_matches = 0

    round_num = 1
    while round_num <= MAX_ROUND:
        raw_matches = fetch_round_matches(round_num)
        time.sleep(REQUEST_DELAY)

        if not raw_matches:
            if round_num == 1:
                log.error("Round 1 returned no matches (no API data or wrong season/tournament ID). Exiting.")
                sys.exit(1)
            log.info(
                "Round %d has no matches; scraped through round %d (stop at first empty round).",
                round_num,
                last_round_with_matches,
            )
            break

        last_round_with_matches = round_num
        fetched_rounds.append(round_num)
        round_records, round_stats = scrape_round_records(round_num, raw_matches, sofa_to_standard)
        records.extend(round_records)
        stats_requests += round_stats
        round_num += 1

    if last_round_with_matches == MAX_ROUND:
        log.warning(
            "Processed through round %d (MAX_ROUND) with no empty round; increase MAX_ROUND if the season is longer.",
            MAX_ROUND,
        )

    return _records_to_frame(records), stats_requests, fetched_rounds


def run_pipeline(full_season: bool = False) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        mapping_path = resolve_team_mapping_path()
    except FileNotFoundError as e:
        log.error("%s", e)
        sys.exit(1)

    sofa_to_standard = load_sofa_standard_mapping(mapping_path)

    mode = "full-season" if full_season else "incremental"
    target_rounds: list[int] = []

    if not full_season:
        try:
            target_rounds = determine_incremental_rounds(SCHEDULE_FILE)
        except Exception as exc:
            log.warning(
                "Could not derive incremental rounds from %s: %s. Falling back to full-season mode.",
                SCHEDULE_FILE,
                exc,
            )
            full_season = True
            mode = "full-season"

    if full_season:
        log.info(
            "Starting xG pipeline in %s mode — Tournament %d | Season %d | rounds from 1 until first empty (max %d)",
            mode,
            UNIQUE_TOURNAMENT_ID,
            SEASON_ID,
            MAX_ROUND,
        )
        fresh_df, stats_requests, fetched_rounds = fetch_full_season(sofa_to_standard)
        target_rounds = fetched_rounds
    else:
        log.info(
            "Starting xG pipeline in %s mode — Tournament %d | Season %d | target rounds: %s",
            mode,
            UNIQUE_TOURNAMENT_ID,
            SEASON_ID,
            target_rounds,
        )
        fresh_df, stats_requests = fetch_specific_rounds(target_rounds, sofa_to_standard)

    final_df = merge_with_existing_cache(
        fresh_df,
        OUTPUT_FILE,
        full_season=full_season,
        team_aliases=sofa_to_standard,
    )
    metadata = {
        "mode": mode,
        "target_rounds": target_rounds,
        "stats_requests": stats_requests,
        "fresh_rows": len(fresh_df),
    }
    log.info(
        "xG pipeline summary — mode=%s | target_rounds=%s | stats_requests=%d | fresh_rows=%d | final_rows=%d",
        metadata["mode"],
        metadata["target_rounds"],
        metadata["stats_requests"],
        metadata["fresh_rows"],
        len(final_df),
    )
    return final_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape CSL xG data from the official SofaScore API")
    parser.add_argument(
        "--full-season",
        action="store_true",
        help="Fetch every round from round 1 until the first empty round instead of the default two-round incremental refresh",
    )
    args = parser.parse_args()

    df, metadata = run_pipeline(full_season=args.full_season)

    if df.empty:
        log.warning("No data collected — check your API key and tournament/season IDs.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        log.info("\n✅  Done! %d matches saved to '%s'", len(df), OUTPUT_FILE)
        log.info(
            "   Mode: %s | Target rounds: %s | Stats requests: %d",
            metadata["mode"],
            metadata["target_rounds"],
            metadata["stats_requests"],
        )

        finished = df.dropna(subset=["home_xg", "away_xg"])
        log.info("   xG available for %d / %d matches", len(finished), len(df))
        if not finished.empty:
            log.info("\n%s", finished[
                ["date", "round", "home_team", "home_xg", "away_xg", "away_team"]
            ].to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
