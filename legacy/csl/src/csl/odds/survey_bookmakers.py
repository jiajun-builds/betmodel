"""Which books quote CSL 1X2, and at what overround? (roadmap #8 reconnaissance)

`backtest/backtest.md` §11.7 is the reason this exists: EV > 0 ⟺ CLV > p × R. The
model's replicable excess CLV is ~1.2–1.4pp (§12), while Pinnacle's 7.55% opening
overround alone costs 2.61pp — so the edge only becomes real at a book with a
**≤5% overround** (bar ≈ 1.75pp) and comfortably at 4% (≈1.4pp). This module answers
the first screening question — *which books are cheap enough to be worth chasing* —
before any effort goes into sourcing their historical opening lines.

Quota: The Odds API bills `markets × regions` per /odds call and the `bookmakers`
filter is FREE, so dropping it returns every book for the same price.
`--regions us` = 1 credit; `us,eu,uk` = 3 credits (1xbet is typically `eu`,
bet365 `uk`/`eu`, so the wider sweep is the meaningful one). A pre-spend guard reads
the free `/sports` endpoint first.

What it CANNOT answer: nobody's *historical* opening prices — the free plan has no
historical endpoint. This is a snapshot of who is on the board now and how cheap they
are. Per-book *opening* prices accrue going forward via `capture_scheduler`, which
since roadmap #8 stores every book present at each open window (a book already
showing a price when Pinnacle opens is a book that opened earlier).

Usage (repo root, PYTHONPATH=src, THE_ODDS_API_KEY set):
    python -m csl.odds.survey_bookmakers --regions us,eu,uk
    python -m csl.odds.survey_bookmakers --dry-run     # spend nothing
    python -m csl.odds.survey_bookmakers --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from csl.odds.capture_snapshot import DEFAULT_MIN_REMAINING, read_quota
from csl.odds.fetch_pinnacle_spreads import (
    BOOKMAKER,
    _book_prices,
    _clean_name,
    _event_bookmakers,
    _find_market,
    fetch_odds_response,
    get_api_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_SURVEY_REGIONS = "us,eu,uk"

# §11.7 bars: a book must undercut these for the model's excess CLV to clear p×R.
OVERROUND_TARGET = 0.05   # bar ≈ 1.75pp — the model's ~1.2–1.4pp is marginal here
OVERROUND_GOOD = 0.04     # bar ≈ 1.40pp — clears comfortably


def survey_rows(events: list[dict], *, fetched_at: str) -> list[dict]:
    """One row per (event, book) with an h2h market, incl. overround.

    Team names are the API's own — no mapping/normalization — so an unmapped book or
    a fixture the mapping doesn't know still shows up in the survey.
    """
    rows: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        home = _clean_name(event.get("home_team"))
        away = _clean_name(event.get("away_team"))
        if not home or not away:
            continue
        for book in _event_bookmakers(event):
            market = _find_market(book)
            prices = _book_prices(book, home, away)
            if prices is None:
                # No usable 3-outcome h2h (e.g. a book quoting no draw): still record
                # the book's presence — coverage without a draw is itself a finding.
                rows.append(
                    {
                        "bookmaker": book.get("key"),
                        "bookmaker_title": book.get("title"),
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "home_team": home,
                        "away_team": away,
                        "has_h2h_draw": False,
                        "home_odds": None,
                        "draw_odds": None,
                        "away_odds": None,
                        "overround": None,
                        "last_update": (market or {}).get("last_update") or book.get("last_update"),
                        "fetched_at": fetched_at,
                    }
                )
                continue
            home_odds, draw_odds, away_odds = prices
            overround = sum(1.0 / o for o in prices) - 1.0
            rows.append(
                {
                    "bookmaker": book.get("key"),
                    "bookmaker_title": book.get("title"),
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": home,
                    "away_team": away,
                    "has_h2h_draw": True,
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "overround": overround,
                    "last_update": (market or {}).get("last_update") or book.get("last_update"),
                    "fetched_at": fetched_at,
                }
            )
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-book summary, cheapest first — the screening table."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["bookmaker", "bookmaker_title"], dropna=False)
    summary = grouped.agg(
        events=("event_id", "nunique"),
        with_draw=("has_h2h_draw", "sum"),
        mean_overround=("overround", "mean"),
        min_overround=("overround", "min"),
        max_overround=("overround", "max"),
        oldest_update=("last_update", "min"),
        newest_update=("last_update", "max"),
    ).reset_index()
    return summary.sort_values("mean_overround", na_position="last").reset_index(drop=True)


def _verdict(overround: float | None) -> str:
    if overround is None or pd.isna(overround):
        return "no 3-way price"
    if overround <= OVERROUND_GOOD:
        return "CLEARS the bar (<=4%)"
    if overround <= OVERROUND_TARGET:
        return "marginal (<=5%)"
    return "too expensive"


def print_report(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    if summary.empty:
        log.warning("No bookmaker rows returned — nothing to report.")
        return

    print("\n=== CSL 1X2 bookmaker survey ===")
    print(f"{frame['event_id'].nunique()} fixtures · {len(summary)} books · "
          f"fetched {frame['fetched_at'].iloc[0]}\n")
    header = f"{'book':<22} {'events':>6} {'draw':>5} {'mean OR':>9} {'range':>15}  verdict"
    print(header)
    print("-" * len(header))
    for row in summary.itertuples(index=False):
        mean_or = "--" if pd.isna(row.mean_overround) else f"{row.mean_overround * 100:.2f}%"
        if pd.isna(row.min_overround):
            rng = "--"
        else:
            rng = f"{row.min_overround * 100:.2f}-{row.max_overround * 100:.2f}%"
        marker = " <-- reference" if row.bookmaker == BOOKMAKER else ""
        print(
            f"{str(row.bookmaker):<22} {row.events:>6} {int(row.with_draw):>5} "
            f"{mean_or:>9} {rng:>15}  {_verdict(row.mean_overround)}{marker}"
        )

    ref = summary[summary["bookmaker"] == BOOKMAKER]
    if not ref.empty and pd.notna(ref.iloc[0]["mean_overround"]):
        ref_or = float(ref.iloc[0]["mean_overround"])
        print(f"\nReference ({BOOKMAKER}) mean overround: {ref_or * 100:.2f}% "
              f"(bar = 0.35 x OR ~ {0.35 * ref_or * 100:.2f}pp)")
    cheaper = summary[summary["mean_overround"] <= OVERROUND_TARGET]
    cheaper = cheaper[cheaper["bookmaker"] != BOOKMAKER]
    if cheaper.empty:
        print(f"\nNo book at or below the {OVERROUND_TARGET * 100:.0f}% bar. On this evidence the "
              "model's ~1.2-1.4pp excess CLV (backtest.md §12) does not clear p x R anywhere\n"
              "in this slate — sourcing historical opening lines is premature.")
    else:
        print(f"\nCandidates at or below {OVERROUND_TARGET * 100:.0f}% overround "
              f"(worth sourcing historical opens for):")
        for row in cheaper.itertuples(index=False):
            print(f"  - {row.bookmaker} ({row.bookmaker_title}): "
                  f"{row.mean_overround * 100:.2f}% over {row.events} fixtures")
    print()


def run(*, regions: str, min_remaining: int, dry_run: bool, json_out: str | None) -> pd.DataFrame:
    api_key = get_api_key()
    n_regions = len([r for r in regions.split(",") if r.strip()])

    remaining, used, csl_available = read_quota(api_key)
    log.info("Quota: remaining=%s used=%s | CSL in slate: %s", remaining, used, csl_available)
    log.info("This survey costs %d credit(s) (1 market x %d region(s))", n_regions, n_regions)
    if remaining is not None and remaining < max(min_remaining, n_regions):
        log.warning("Aborting: quota remaining=%d below threshold=%d.", remaining, min_remaining)
        return pd.DataFrame()

    if dry_run:
        log.info("Dry run: would spend %d credit(s) on regions=%s; nothing fetched.", n_regions, regions)
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    response = fetch_odds_response(api_key, regions, bookmakers=None)  # n_regions paid credits
    log.info("Quota after call: remaining=%s", response.headers.get("x-requests-remaining"))
    events = response.json()
    if not isinstance(events, list):
        raise ValueError(f"Expected list response from The Odds API, got: {type(events)}")
    log.info("Fetched %d events", len(events))

    frame = pd.DataFrame(survey_rows(events, fetched_at=fetched_at))
    if frame.empty:
        log.warning("No bookmaker/h2h rows in the response.")
        return frame

    summary = summarize(frame)
    print_report(frame, summary)

    if json_out:
        payload = {
            "fetched_at": fetched_at,
            "regions": regions,
            "credits_spent": n_regions,
            "books": json.loads(summary.to_json(orient="records")),
            "rows": json.loads(frame.to_json(orient="records")),
        }
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        log.info("Wrote %s", json_out)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey every bookmaker quoting CSL 1X2 (roadmap #8 recon): coverage + overround"
    )
    parser.add_argument("--regions", default=DEFAULT_SURVEY_REGIONS,
                        help="Odds API regions, comma separated. COSTS 1 credit per region "
                             "(default: us,eu,uk = 3 credits; 1xbet is usually eu, bet365 uk)")
    parser.add_argument("--min-remaining", type=int, default=DEFAULT_MIN_REMAINING,
                        help="Abort before spending if quota remaining is below this")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Optional path to write the full survey (summary + per-fixture rows)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the quota/cost only; spend nothing, fetch nothing")
    args = parser.parse_args()

    try:
        run(regions=args.regions, min_remaining=args.min_remaining,
            dry_run=args.dry_run, json_out=args.json_out)
    except requests.HTTPError as exc:
        log.error("The Odds API HTTP error: %s", exc)
        sys.exit(1)
    except requests.RequestException as exc:
        log.error("The Odds API request failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
