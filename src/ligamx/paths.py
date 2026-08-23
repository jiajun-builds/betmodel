"""Repo-root-relative path helpers (resolved from this file, not the CWD).

Every stage reads/writes files under the repo, so paths must not depend on the
current working directory. Import these helpers instead of hardcoding relative or
absolute paths.
"""

from __future__ import annotations

import os


def project_root() -> str:
    """Repo root (the directory containing data/, models/, scripts/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def data_dir() -> str:
    return os.path.join(project_root(), "data")


def models_dir() -> str:
    return os.path.join(project_root(), "models")


def output_dir() -> str:
    return os.path.join(project_root(), "output")


def data_output_dir() -> str:
    """Derived model/odds outputs (comparison datasets)."""
    return os.path.join(data_dir(), "output_data")


def pinnacle_h2h_csv() -> str:
    """Current ('Now') Pinnacle 1X2 (h2h) line snapshot."""
    return os.path.join(data_dir(), "MEX_pinnacle_h2h.csv")


def odds_snapshots_csv() -> str:
    """Append-only long-format store of every captured odds snapshot (all venues)."""
    return os.path.join(data_dir(), "odds_snapshots.csv")


def odds_capture_history_csv() -> str:
    """Append-only history of scheduled captures (opening lines, closing lines).

    Tracked in git, unlike odds_snapshots_csv(): the capture loop runs in GitHub
    Actions and has to commit what it collects, and an opening line missed is an
    opening line lost forever. Kept separate from the snapshot store because the
    two answer different questions -- the store is a dense poll time-series for
    research (Polymarket curves), this is a sparse, provenance-stamped record of
    prices captured at a deliberate moment. See odds/capture_store.
    """
    return os.path.join(data_dir(), "MEX_odds_capture_history.csv")


def capture_watch_csv() -> str:
    """When each (fixture, book) was first *observed without a price*.

    Evidence, not odds. A captured opening line is only provably an opening line
    if we were watching before the book posted it, and the strongest proof of
    that is having seen the fixture unpriced at a known moment. The capture loop
    is the only thing that ever sees this -- an unpriced fixture leaves no trace
    in the odds history by construction -- and the window to observe it closes
    the instant the book posts. So it is recorded as it happens, and tracked in
    git for the same reason the capture history is. See odds/capture_watch.
    """
    return os.path.join(data_dir(), "MEX_capture_watch.csv")


def market_comparison_csv() -> str:
    """Model-vs-market comparison (drives the dashboard EV view)."""
    return os.path.join(data_output_dir(), "MEX_upcoming_market_comparison.csv")


def polymarket_price_quality_csv() -> str:
    """Per-snapshot tradedness flags for Polymarket (companion to the snapshot store).

    Derived, regenerable, and deliberately NOT part of the append-only store's
    schema: it exists to tell the CLV layer which stored prices were real traded
    prices rather than pre-trading placeholders. See odds/price_quality.
    """
    return os.path.join(data_dir(), "polymarket_price_quality.csv")


def oddsapi_io_closes_csv() -> str:
    """Cached odds-api.io historical closing lines (long format, one row per line).

    Regenerable by re-pulling, and quota-limited rather than free, so it is cached
    beside the snapshot store instead of inside it: the store is a poll time-series
    while this endpoint only ever returns a single close per market.
    See odds/oddsapi_io.
    """
    return os.path.join(data_dir(), "oddsapi_io_closes.csv")


def polymarket_trades_csv() -> str:
    """Tick-level Polymarket trade tape, normalized to Yes-space aggressor terms.

    Derived and regenerable (re-pull with odds/polymarket_trades), so it is cached
    beside the snapshot store rather than inside it: the store is one row per poll
    per match, while this is one row per print. See odds/polymarket_trades.
    """
    return os.path.join(data_dir(), "polymarket_trades.csv")


# --- raw / input data -------------------------------------------------------

def ligamx_data_csv() -> str:
    """Full match history / model input table."""
    return os.path.join(data_dir(), "MEX_ligamx.csv")


def upcoming_fixtures_csv() -> str:
    """Unplayed fixtures derived from the schedule provider."""
    return os.path.join(data_dir(), "MEX_upcoming_fixtures.csv")


def football_data_csv() -> str:
    """Cached football-data.co.uk Liga MX CSV (free market/exchange closing odds).

    Backfills closing 1X2 benchmarks (AvgC market-average, BFEC Betfair exchange)
    after football-data stopped publishing Pinnacle closes in Oct-2025. Used by the
    eval/CLV layer only; not part of the production schema.
    """
    return os.path.join(data_dir(), "football_data_mex.csv")


def team_mapping_csv() -> str:
    """Team-name mapping across the provider namespaces + canonical name."""
    return os.path.join(data_dir(), "ligamx_team_name_mapping.csv")


# --- model outputs ----------------------------------------------------------

def team_stats_csv() -> str:
    return os.path.join(models_dir(), "MEX_team_stats.csv")


def match_simulations_csv() -> str:
    return os.path.join(models_dir(), "MEX_team_stats_match_simulations.csv")


def model_meta_json() -> str:
    """Sidecar recording when the model was last (re)fit. See ligamx.stage_meta."""
    return os.path.join(models_dir(), "MEX_model_meta.json")


def fixtures_meta_json() -> str:
    """Sidecar recording when fixtures/results were last fetched from SofaScore.

    Separate from the model's because the two stages go stale independently: CI
    can never refresh either (SofaScore blocks datacenter IPs), but a local
    `recompute -> model` run advances the model without touching the fixture list.
    See ligamx.stage_meta.
    """
    return os.path.join(data_dir(), "MEX_fixtures_meta.json")


# --- dashboard datasets (site serves the json/ tree) ------------------------

def data_dashboard_dir() -> str:
    return os.path.join(data_dir(), "dashboard")


def data_dashboard_csv_dir() -> str:
    return os.path.join(data_dashboard_dir(), "csv")


def data_dashboard_json_dir() -> str:
    return os.path.join(data_dashboard_dir(), "json")
