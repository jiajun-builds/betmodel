"""Collapse the capture history into the master table's open/close columns.

    open   earliest capture per (fixture, book), if it can be shown to be an opener
    close  latest capture per (fixture, book), if it landed close enough to kickoff

Every row here was taken at a deliberate moment, so unlike a dense price curve
there is no "snapshot nearest kickoff minus delta" search to do and no in-play
guard to apply: the capture window already did that work.

**Which opens can be trusted is the load-bearing question in this file.** The
opening-line series is the only positive-expectation result the project has, and
quietly mixing a mid-market price into it would corrupt the one thing worth
measuring. Three independent tests have to pass.

First, the row has to *be* an opening price. A capture labelled ``open`` is not
necessarily one: before the predicted-window mechanism was retired, a fixture
whose window elapsed unfilled had the current line written into its open slot as
a placeholder. Twenty such rows exist, eight of them on a book the model bets.
:mod:`betmodel.odds.provenance` classifies them and they are excluded here.

Then it has to be provably *first*, by either of the two proofs in
:mod:`betmodel.odds.capture_watch`. Either suffices, and both self-calibrate: on
day one, with no evidence of either kind, everything is correctly rejected.

Rows that fail stay in the history. They are perfectly good price observations,
they just never reach the master table.

**Nothing here overwrites.** Existing values were assembled from several sources
and repaired by hand; a reducer that clobbered them would undo work that cannot
be redone. Only blanks are filled.
"""

from __future__ import annotations

import logging

import pandas as pd

from betmodel import paths
from betmodel.config.schema import LeagueConfig
from betmodel.dates import parse_date_only_series
from betmodel.odds import capture_store, capture_watch, provenance

log = logging.getLogger(__name__)

#: A capture later than this before kickoff is not a closing line.
MAX_CLOSE_LEAD_HOURS = 6.0


def schema_prefixes(config: LeagueConfig) -> dict[str, str]:
    """``bookmaker key -> master-table column prefix``.

    A book with no prefix is store-only: captured for research, with no column to
    reduce into. That is deliberate rather than an oversight, so it is expressed
    by omission here rather than by a special case downstream.
    """
    return {
        book.key: book.schema_prefix
        for book in config.odds.books
        if book.schema_prefix
    }


def _prepared(history: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    """Typed view of the history, with unusable rows dropped."""
    if history.empty:
        return history
    frame = history.copy()
    frame["_at"] = pd.to_datetime(frame["fetched_at"], utc=True, errors="coerce")
    frame["_ko"] = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
    for side in ("home_odds", "draw_odds", "away_odds"):
        frame[side] = pd.to_numeric(frame[side], errors="coerce")
    frame["_prefix"] = frame["bookmaker"].map(schema_prefixes(config))
    frame["_provenance"] = [
        provenance.classify(s, r)
        for s, r in zip(frame["snapshot_type"], frame["capture_reason"])
    ]
    usable = (
        frame["_at"].notna()
        & frame["_ko"].notna()
        & frame["_prefix"].notna()
        & frame[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
    )
    return frame[usable]


#: How far the last confirmed unpriced sighting may be from the capture before it
#: stops proving anything. Two and a half poll intervals: one tick may be missed
#: to a runner delay or a concurrency cancel without voiding the proof, while a
#: sustained outage -- the case this exists for -- is well outside it.
PROOF_GAP_INTERVALS = 2.5


def _proof_gap(config: LeagueConfig, book: str) -> pd.Timedelta | None:
    """The tolerated silence for one book, or None when it is not polled.

    None means the old, unchecked behaviour, which is correct for a book whose
    opens nobody polls for: there is no cadence to measure a gap against.
    """
    for candidate in config.odds.books:
        if candidate.key == book and candidate.poll_interval_minutes:
            return pd.Timedelta(
                minutes=candidate.poll_interval_minutes * PROOF_GAP_INTERVALS
            )
    return None


def collapse_opens(
    league: str,
    config: LeagueConfig,
    *,
    lookahead_days: int | None = None,
    history_path: str | None = None,
    watch_path: str | None = None,
) -> dict[tuple[str, str, str], dict]:
    """Earliest usable open per ``(home, away, book)``, with its proof.

    Shared by the reducer and the signal engine so the two cannot disagree about
    what an opening price is. They used to: the reducer applied a proof test that
    the signal path did not, so a price the match table refused could still fire
    a bet.
    """
    lp = paths.for_league(league)
    history = capture_store.load_history(history_path or lp.capture_history_csv)
    frame = _prepared(history, config)
    if frame.empty:
        return {}

    watch_file = watch_path or lp.capture_watch_csv
    watched = capture_watch.watched_before(watch_file)
    watched_last = capture_watch.watched_until(watch_file)
    since = capture_watch.watching_since(history)
    horizon = pd.Timedelta(
        days=config.odds.open.lookahead_days if lookahead_days is None else lookahead_days
    )

    opens = frame[
        (frame["snapshot_type"] == "open")
        & frame["_provenance"].map(provenance.is_opening_price)
    ]
    out: dict[tuple[str, str, str], dict] = {}
    for (home, away, book), group in opens.groupby(
        ["home_team", "away_team", "bookmaker"], dropna=False
    ):
        first = group.loc[group["_at"].idxmin()]
        kickoff = group["_ko"].max()
        proof = capture_watch.opener_proof(
            home=home, away=away, bookmaker=book,
            captured_at=first["_at"], kickoff=kickoff,
            watched=watched, since=since, horizon=horizon,
            watched_last=watched_last, max_gap=_proof_gap(config, str(book)),
        )
        out[(str(home), str(away), str(book))] = {
            "home_odds": float(first.home_odds),
            "draw_odds": float(first.draw_odds),
            "away_odds": float(first.away_odds),
            "captured_at": first["_at"],
            "last_update": str(first.last_update),
            "proof": proof,
            "provenance": first["_provenance"],
            "lead_h": (kickoff - first["_at"]).total_seconds() / 3600.0,
        }
    return out


def build_records(
    league: str,
    config: LeagueConfig,
    *,
    lookahead_days: int | None = None,
    max_close_lead: float = MAX_CLOSE_LEAD_HOURS,
) -> list[dict]:
    """One record per (fixture, book), carrying whichever of open/close is usable."""
    lp = paths.for_league(league)
    history = capture_store.load_history(lp.capture_history_csv)
    frame = _prepared(history, config)
    if frame.empty:
        return []

    watched = capture_watch.watched_before(lp.capture_watch_csv)
    since = capture_watch.watching_since(history)
    horizon = pd.Timedelta(
        days=config.odds.open.lookahead_days if lookahead_days is None else lookahead_days
    )

    records: list[dict] = []
    for (home, away, prefix), group in frame.groupby(
        ["home_team", "away_team", "_prefix"], dropna=False
    ):
        kickoff = group["_ko"].max()
        record = {
            "home": home, "away": away, "prefix": prefix,
            "date": kickoff.strftime("%Y/%m/%d"),
            "open_h": None, "open_d": None, "open_a": None, "open_lead_h": None,
            "close_h": None, "close_d": None, "close_a": None, "close_lead_h": None,
            "open_trusted": False, "open_proof": "", "open_provenance": "",
        }

        # Only rows that are actually opening prices. A backfilled placeholder is
        # labelled 'open' but is a mid-market line.
        opens = group[
            (group["snapshot_type"] == "open")
            & group["_provenance"].map(provenance.is_opening_price)
        ]
        if not opens.empty:
            first = opens.loc[opens["_at"].idxmin()]
            record["open_lead_h"] = (kickoff - first["_at"]).total_seconds() / 3600.0
            record["open_provenance"] = first["_provenance"]
            proof = capture_watch.opener_proof(
                home=home, away=away, bookmaker=first.bookmaker,
                captured_at=first["_at"], kickoff=kickoff,
                watched=watched, since=since, horizon=horizon,
            )
            record["open_proof"] = proof
            record["open_trusted"] = bool(proof)
            if record["open_trusted"]:
                record.update(
                    open_h=first.home_odds, open_d=first.draw_odds, open_a=first.away_odds
                )

        closes = group[group["snapshot_type"] == "close"]
        if not closes.empty:
            last = closes.loc[closes["_at"].idxmax()]
            lead = (kickoff - last["_at"]).total_seconds() / 3600.0
            record["close_lead_h"] = lead
            # A negative lead is an in-play print; too large is not a close at all.
            if 0 <= lead <= max_close_lead:
                record.update(
                    close_h=last.home_odds, close_d=last.draw_odds, close_a=last.away_odds
                )

        records.append(record)
    return records


def _find_row(matches: pd.DataFrame, home: str, away: str, date: str) -> int | None:
    """Locate one fixture in the master table by parsed date, not by string.

    Matching on the raw string silently misses when two sources format the same
    day differently, and matching on teams alone can hit the reverse fixture from
    another round.
    """
    want = pd.Timestamp(date.replace("/", "-"))
    same = matches[(matches["Home"] == home) & (matches["Away"] == away)]
    if same.empty:
        return None
    dates = parse_date_only_series(same["Date"])
    hit = same[dates == want]
    return int(hit.index[0]) if not hit.empty else None


def missing_columns(league: str, records: list[dict]) -> list[str]:
    """Columns a trusted value would need, that the master table does not have.

    Captured data with nowhere to go is silent waste, so it is surfaced rather
    than counted. One league reached the merge with no reducer at all, and so
    with no column for two of the books it had been capturing.
    """
    matches = pd.read_csv(paths.for_league(league).matches_csv, nrows=1)
    wanted: set[str] = set()
    for record in records:
        for phase in ("open", "close"):
            if any(record[f"{phase}_{side}"] is not None for side in ("h", "d", "a")):
                for side in ("h", "d", "a"):
                    wanted.add(f"{record['prefix']}_{phase}_{side}")
    return sorted(w for w in wanted if w not in matches.columns)


def merge(
    league: str, records: list[dict], *, dry_run: bool = False,
    create_columns: bool = False, matches_path: str | None = None,
) -> dict[str, int]:
    """Fill blanks in the master table. Never overwrites.

    ``no_row`` is a normal outcome, not an error: a fixture has no row in the
    match history until it has been played.

    ``create_columns`` adds a column a trusted value needs but the table lacks.
    Off by default: growing a tracked file's schema should be a deliberate step,
    not a side effect of a routine tick.
    """
    target = matches_path or paths.for_league(league).matches_csv
    matches = pd.read_csv(target)
    stats = {
        "records": len(records), "no_row": 0, "written": 0, "kept": 0,
        "no_column": 0, "created_columns": 0,
    }

    for record in records:
        index = _find_row(matches, record["home"], record["away"], record["date"])
        if index is None:
            stats["no_row"] += 1
            continue
        for phase in ("open", "close"):
            for side in ("h", "d", "a"):
                value = record[f"{phase}_{side}"]
                if value is None or pd.isna(value):
                    continue
                column = f"{record['prefix']}_{phase}_{side}"
                if column not in matches.columns:
                    if not create_columns:
                        stats["no_column"] += 1
                        continue
                    matches[column] = pd.NA
                    stats["created_columns"] += 1
                if pd.notna(matches.at[index, column]):
                    stats["kept"] += 1
                    continue
                if not dry_run:
                    matches.at[index, column] = value
                stats["written"] += 1

    if not dry_run and (stats["written"] or stats["created_columns"]):
        matches.to_csv(target, index=False)
    log.info("%s reduce: %s", league, stats)
    return stats
