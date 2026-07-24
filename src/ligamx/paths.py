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


def market_comparison_csv() -> str:
    """Model-vs-market comparison (drives the dashboard EV view)."""
    return os.path.join(data_output_dir(), "MEX_upcoming_market_comparison.csv")


# --- raw / input data -------------------------------------------------------

def ligamx_data_csv() -> str:
    """Full match history / model input table."""
    return os.path.join(data_dir(), "MEX_ligamx.csv")


def upcoming_fixtures_csv() -> str:
    """Unplayed fixtures derived from the schedule provider."""
    return os.path.join(data_dir(), "MEX_upcoming_fixtures.csv")


def team_mapping_csv() -> str:
    """Team-name mapping across the provider namespaces + canonical name."""
    return os.path.join(data_dir(), "ligamx_team_name_mapping.csv")


# --- model outputs ----------------------------------------------------------

def team_stats_csv() -> str:
    return os.path.join(models_dir(), "MEX_team_stats.csv")


def match_simulations_csv() -> str:
    return os.path.join(models_dir(), "MEX_team_stats_match_simulations.csv")


def model_meta_json() -> str:
    """Sidecar recording when the model was last (re)fit."""
    return os.path.join(models_dir(), "MEX_model_meta.json")


# --- dashboard datasets (site serves the json/ tree) ------------------------

def data_dashboard_dir() -> str:
    return os.path.join(data_dir(), "dashboard")


def data_dashboard_csv_dir() -> str:
    return os.path.join(data_dashboard_dir(), "csv")


def data_dashboard_json_dir() -> str:
    return os.path.join(data_dashboard_dir(), "json")
