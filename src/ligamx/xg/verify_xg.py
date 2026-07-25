"""Audit MEX_ligamx.csv xG against the SofaScore API (source of truth).

The production fetcher (fixtures.mex_fixture) walks rounds 1..N and stops at the
first empty round, so it only ever captures the 17-round regular season -- the
liguilla (Quarterfinals/Semifinals/Final, exposed under non-sequential round
numbers) is never scraped. This audit uses the paginated season events endpoint
instead, which returns EVERY finished event including the playoff bracket, and
reconciles it against the CSV:

  MISSING   finished SofaScore match (in the CSV date range) absent from the CSV
  XG_DIFF   matched match whose stored HxG/AxG differ from SofaScore (> tol)
  SCORE     matched match whose stored HG/AG differ from SofaScore
  NO_XG     stored xG is 0/0 but SofaScore publishes real xG (a missed scrape)
  UNVERIFIED CSV row with no SofaScore match (can't be checked)

The fetched events are cached to JSON so the (slow) network pass runs once.

    python -m ligamx.xg.verify_xg [--min-date 2023-07-01] [--refresh] [--tol 0.05]
    python -m ligamx.xg.verify_xg --report-json <path>   # machine-readable dump
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ligamx import config, paths
from ligamx.sofascore_client import SofascoreClient

MX_TZ = ZoneInfo("America/Mexico_City")
CACHE_PATH = os.path.join(paths.data_output_dir(), "sofascore_events_cache.json")
MIN_DATE_DEFAULT = "2023-07-01"


def _std(name: str) -> str:
    return config.SOFASCORE_TO_STANDARD.get(name, name)


def _date_mx(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(MX_TZ).strftime("%Y-%m-%d")


def _relevant_seasons(client: SofascoreClient) -> list:
    """(unique_tournament_id, season_id, label) for seasons overlapping the CSV.

    Apertura runs Jul-Dec, Clausura Jan-May. The CSV starts 2023-07-01, so we take
    Apertura >= 2023 and Clausura >= 2024 (Clausura 2023 ended before the window).
    """
    out = []
    for ut, floor in ((config.SOFASCORE_APERTURA, ("Apertura", 2023)),
                      (config.SOFASCORE_CLAUSURA, ("Clausura", 2024))):
        kind, min_year = floor
        for s in client.seasons(ut):
            m = re.search(r"(Apertura|Clausura)\s+(\d{4})", s.get("name", ""))
            if not m:
                continue
            year = int(m.group(2))
            if m.group(1) == kind and year >= min_year:
                out.append((ut, s["id"], f"{kind} {year}"))
    return out


def fetch_all(client: SofascoreClient, min_date: str, log=print) -> list:
    """Every finished SofaScore Liga MX event >= min_date, with xG. Slow (per-event stat call)."""
    seasons = _relevant_seasons(client)
    log(f"Seasons in scope: {', '.join(l for _, _, l in seasons)}")
    records, unmapped = [], set()
    for ut, sid, label in seasons:
        page, n_season = 0, 0
        while True:
            data = client._get(f"/unique-tournament/{ut}/season/{sid}/events/last/{page}")
            if not data:
                break
            events = data.get("events", [])
            for e in events:
                ts = e.get("startTimestamp")
                if not ts or e.get("status", {}).get("type") != "finished":
                    continue
                d = _date_mx(ts)
                if d < min_date:
                    continue
                hs, as_ = e["homeTeam"]["name"], e["awayTeam"]["name"]
                if hs not in config.SOFASCORE_TO_STANDARD:
                    unmapped.add(hs)
                if as_ not in config.SOFASCORE_TO_STANDARD:
                    unmapped.add(as_)
                hxg, axg = client.event_xg(e["id"])
                ri = e.get("roundInfo", {}) or {}
                records.append({
                    "event_id": e["id"], "date": d, "season": label,
                    "round": ri.get("name") or ri.get("round"),
                    "home": _std(hs), "away": _std(as_),
                    "hg": e.get("homeScore", {}).get("current"),
                    "ag": e.get("awayScore", {}).get("current"),
                    "hxg": hxg, "axg": axg,
                })
                n_season += 1
            if not data.get("hasNextPage"):
                break
            page += 1
        log(f"  {label}: {n_season} finished events >= {min_date}")
    if unmapped:
        log("[WARN] unmapped SofaScore names: " + ", ".join(sorted(unmapped)))
    return records


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return None


def save_cache(records):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(records, f)


def _key(date_str, home, away):
    return (date_str, home, away)


def reconcile(records: list, tol: float = 0.05) -> dict:
    """Compare SofaScore records against the CSV; return categorized discrepancies."""
    df = pd.read_csv(paths.ligamx_data_csv(), dtype=str, keep_default_na=False)
    df["_d"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    # index CSV rows by (home, away) -> list of (row_index, date, hxg, axg, hg, ag)
    csv_by_pair = {}
    for i, r in df.iterrows():
        csv_by_pair.setdefault((r["Home"], r["Away"]), []).append(i)

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    matched_rows = set()
    missing, xg_diff, score_diff, no_xg = [], [], [], []
    for rec in records:
        cands = csv_by_pair.get((rec["home"], rec["away"]), [])
        # match on same pairing within +/-2 days (kickoff tz drift in the cache)
        hit = None
        for i in cands:
            if i in matched_rows:
                continue
            d_csv = df.at[i, "_d"]
            if d_csv and abs((pd.Timestamp(d_csv) - pd.Timestamp(rec["date"])).days) <= 2:
                hit = i
                break
        if hit is None:
            missing.append(rec)
            continue
        matched_rows.add(hit)
        row = df.loc[hit]
        s_hxg, s_axg = rec["hxg"], rec["axg"]
        c_hxg, c_axg = _f(row["HxG"]), _f(row["AxG"])
        if s_hxg is not None and s_axg is not None:
            if (c_hxg in (None, 0.0) and c_axg in (None, 0.0)) and (s_hxg or s_axg):
                no_xg.append((hit, rec, c_hxg, c_axg))
            elif c_hxg is not None and c_axg is not None and (
                abs(c_hxg - s_hxg) > tol or abs(c_axg - s_axg) > tol):
                xg_diff.append((hit, rec, c_hxg, c_axg))
        c_hg, c_ag = _f(row["HG"]), _f(row["AG"])
        if rec["hg"] is not None and c_hg is not None and (
            int(c_hg) != int(rec["hg"]) or int(c_ag) != int(rec["ag"])):
            score_diff.append((hit, rec, c_hg, c_ag))

    unverified = [i for i in range(len(df)) if i not in matched_rows]
    return {"df": df, "missing": missing, "xg_diff": xg_diff,
            "score_diff": score_diff, "no_xg": no_xg, "unverified": unverified}


def _res_from_score(hg, ag) -> str:
    if hg is None or ag is None:
        return ""
    return "H" if hg > ag else ("A" if hg < ag else "D")


def apply_fix(res: dict, add_missing: bool = True, correct_xg: bool = False) -> dict:
    """Backfill MISSING matches (and optionally correct XG_DIFF/NO_XG) drift-free.

    Writes only the touched cells (and appends new rows); every untouched column
    keeps its exact string form. HExpG+/AExpG+ are left for the downstream
    recompute (compute_expg) to fill/normalize. Returns a small stats dict.

    correct_xg is OFF by default: overwriting existing xG is gated because the CSV's
    historical xG uses a different SofaScore aggregation (sum-of-halves) than
    event_xg's ALL-period read, so a bulk overwrite is a deliberate, not incidental,
    choice. See the module audit for the breakdown.
    """
    df = res["df"]  # already dtype=str, keep_default_na=False
    cols = list(df.columns)
    stats = {"added": 0, "xg_corrected": 0}

    # Correct xG in place for mis-scraped / empty-xG rows (only when explicitly asked).
    if correct_xg:
        for hit, rec, _c1, _c2 in res["xg_diff"] + res["no_xg"]:
            df.at[hit, "HxG"] = str(rec["hxg"])
            df.at[hit, "AxG"] = str(rec["axg"])
            stats["xg_corrected"] += 1

    # Append missing matches (xG present only -- skip if SofaScore has no xG yet).
    if add_missing:
        new_rows = []
        for rec in res["missing"]:
            if rec["hxg"] is None or rec["axg"] is None:
                continue
            row = {c: "" for c in cols}
            row.update({
                "Country": "Mexico", "League": "Liga MX", "Season": rec["season"],
                "Round": str(rec["round"]) if rec["round"] is not None else "",
                "Date": rec["date"].replace("-", "/"), "Time": "",
                "Home": rec["home"], "Away": rec["away"],
                "HxG": str(rec["hxg"]), "AxG": str(rec["axg"]),
                "HG": str(rec["hg"]), "AG": str(rec["ag"]),
                "Res": _res_from_score(rec["hg"], rec["ag"]),
            })
            new_rows.append(row)
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
            stats["added"] = len(new_rows)

    # Sort by date to keep the history ordered, then write drift-free.
    df["_sort"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.sort_values("_sort", kind="stable").drop(columns=["_sort"]).reset_index(drop=True)
    df.to_csv(paths.ligamx_data_csv(), index=False)
    return stats


def print_report(res: dict):
    df = res["df"]
    print(f"\n{'='*80}\nSofaScore xG audit  |  CSV rows: {len(df)}")
    print(f"{'='*80}")
    print(f"MISSING (in SofaScore, not in CSV):     {len(res['missing'])}")
    print(f"XG_DIFF (stored != SofaScore > tol):    {len(res['xg_diff'])}")
    print(f"SCORE   (stored score != SofaScore):    {len(res['score_diff'])}")
    print(f"NO_XG   (stored 0/0, SofaScore has xG): {len(res['no_xg'])}")
    print(f"UNVERIFIED (CSV row, no SofaScore match):{len(res['unverified'])}")

    if res["missing"]:
        print("\n-- MISSING matches (would be added) --")
        for r in sorted(res["missing"], key=lambda x: x["date"]):
            print(f"  {r['date']}  {r['season']:<14} {r['round'] or '':<13} "
                  f"{r['home']:<18} {r['hg']}-{r['ag']} {r['away']:<18} "
                  f"xG {r['hxg']}/{r['axg']}")
    for tag, rows in (("XG_DIFF", res["xg_diff"]), ("NO_XG", res["no_xg"])):
        if rows:
            print(f"\n-- {tag} --")
            for hit, rec, chx, cax in rows:
                print(f"  {rec['date']} {rec['home']:<16} v {rec['away']:<16} "
                      f"CSV xG {chx}/{cax}  ->  SofaScore {rec['hxg']}/{rec['axg']}")
    if res["score_diff"]:
        print("\n-- SCORE mismatches --")
        for hit, rec, chg, cag in res["score_diff"]:
            print(f"  {rec['date']} {rec['home']} v {rec['away']}  "
                  f"CSV {int(chg)}-{int(cag)}  ->  SofaScore {rec['hg']}-{rec['ag']}")
    if res["unverified"]:
        print(f"\n-- UNVERIFIED CSV rows ({len(res['unverified'])}) --")
        for i in res["unverified"][:40]:
            print(f"  {df.at[i,'Date']}  {df.at[i,'Home']:<18} v {df.at[i,'Away']:<18}")
        if len(res["unverified"]) > 40:
            print(f"  ... and {len(res['unverified']) - 40} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", default=MIN_DATE_DEFAULT)
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-fetch from SofaScore")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--fix", action="store_true",
                    help="backfill MISSING matches only (safe; then run compute_expg)")
    ap.add_argument("--fix-xg-diffs", action="store_true",
                    help="ALSO overwrite existing xG for XG_DIFF/NO_XG rows to the "
                         "SofaScore ALL-period value (deliberate bulk change -- see audit)")
    args = ap.parse_args()

    records = None if args.refresh else load_cache()
    if records is None:
        records = fetch_all(SofascoreClient(), args.min_date)
        save_cache(records)
        print(f"Cached {len(records)} SofaScore events -> {CACHE_PATH}")
    else:
        print(f"Loaded {len(records)} cached SofaScore events (use --refresh to re-fetch)")

    res = reconcile(records, tol=args.tol)
    print_report(res)

    if args.fix or args.fix_xg_diffs:
        stats = apply_fix(res, add_missing=True, correct_xg=args.fix_xg_diffs)
        print(f"\n[FIX] added {stats['added']} missing matches, "
              f"corrected {stats['xg_corrected']} xG rows.")
        print("[FIX] run `python -m ligamx.xg.compute_expg` to fill HExpG+/AExpG+ and normalize.")


if __name__ == "__main__":
    main()
