"""Every file location in one place, scoped by league.

Paths resolve from this file, never from the current working directory, because
stages run from GitHub Actions, from a shell script and from a laptop cron, and
all three have different CWDs.

Layout, per league::

    data/<league>/
        matches.csv                 master match table; model input
        upcoming_fixtures.csv       unplayed fixtures from the schedule provider
        xg.csv                      per-match xG from the xG provider
        odds_capture_history.csv    append-only capture store  (TRACKED)
        capture_watch.csv           unpriced-observation evidence (TRACKED)
        team_name_mapping.csv       provider namespaces -> canonical name
        market_comparison.csv       model-vs-market, the signal engine's output
        now_line.csv                current reference line   (ignored, transient)
        model/                      team_stats, simulations, meta sidecars
        research/                   large regenerable stores (ignored)
        cache/                      provider response caches (ignored)

Two of these are irreplaceable and that is why they are committed:
``odds_capture_history.csv`` and ``capture_watch.csv``. An opening line exists
only while a book shows it, and neither odds provider sells opener history at any
tier. Everything else can be refetched or recomputed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def project_root() -> str:
    """Repo root: the directory holding data/, leagues/, public/."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def leagues_dir() -> str:
    return os.path.join(project_root(), "leagues")


def data_dir() -> str:
    return os.path.join(project_root(), "data")


def public_dir() -> str:
    """The published contract. This tree is the repo's only public surface."""
    return os.path.join(project_root(), "public")


def public_index_json() -> str:
    """League manifest. The one address a downstream consumer hardcodes."""
    return os.path.join(public_dir(), "index.json")


@dataclass(frozen=True)
class LeaguePaths:
    """Every path a single league's pipeline touches.

    Construct via :func:`for_league` rather than directly, so the league id is
    validated against the filesystem-safe charset before it reaches a path join.
    """

    league: str

    # --- roots ---------------------------------------------------------------
    @property
    def root(self) -> str:
        return os.path.join(data_dir(), self.league)

    @property
    def model_dir(self) -> str:
        return os.path.join(self.root, "model")

    @property
    def research_dir(self) -> str:
        """Large, regenerable research stores. Gitignored by design.

        Kept apart from the tracked stores because the two answer different
        questions: these are dense poll time-series for research, the tracked
        capture history is a sparse, provenance-stamped record of prices taken
        at a deliberate moment.
        """
        return os.path.join(self.root, "research")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.root, "cache")

    # --- tracked inputs ------------------------------------------------------
    @property
    def matches_csv(self) -> str:
        """Master match table: results, xG, and the reduced odds columns."""
        return os.path.join(self.root, "matches.csv")

    @property
    def upcoming_fixtures_csv(self) -> str:
        return os.path.join(self.root, "upcoming_fixtures.csv")

    @property
    def xg_csv(self) -> str:
        return os.path.join(self.root, "xg.csv")

    @property
    def team_mapping_csv(self) -> str:
        return os.path.join(self.root, "team_name_mapping.csv")

    # --- tracked, irreplaceable ---------------------------------------------
    @property
    def capture_history_csv(self) -> str:
        """Append-only history of deliberate captures (opening + closing lines).

        Tracked in git: the capture loop runs in CI and has to commit what it
        collects, and an opening line missed is an opening line lost forever.
        """
        return os.path.join(self.root, "odds_capture_history.csv")

    @property
    def capture_watch_csv(self) -> str:
        """When each (fixture, book) was first observed WITHOUT a price.

        Evidence, not odds. A captured opening line is provably an opening line
        only if we were watching before the book posted it, and the strongest
        proof is having seen the fixture unpriced at a known moment. The window
        to observe that closes the instant the book posts, so it is recorded as
        it happens.
        """
        return os.path.join(self.root, "capture_watch.csv")

    # --- derived -------------------------------------------------------------
    @property
    def market_comparison_csv(self) -> str:
        return os.path.join(self.root, "market_comparison.csv")

    @property
    def now_line_csv(self) -> str:
        """Current reference line. Overwritten every fetch, so not tracked."""
        return os.path.join(self.root, "now_line.csv")

    # --- model outputs -------------------------------------------------------
    @property
    def team_stats_csv(self) -> str:
        return os.path.join(self.model_dir, "team_stats.csv")

    @property
    def simulations_csv(self) -> str:
        return os.path.join(self.model_dir, "simulations.csv")

    @property
    def model_meta_json(self) -> str:
        """When the model was last refit.

        Deliberately not touched by odds-only refreshes, so the published
        ``model_updated_at`` stays pinned to the last real fit even while the
        capture loop republishes the dashboard several times an hour.
        """
        return os.path.join(self.model_dir, "meta.json")

    @property
    def fixtures_meta_json(self) -> str:
        """When fixtures/results were last fetched.

        Separate from the model's sidecar because the two stages go stale
        independently: a local refit advances the model without touching the
        fixture list, and the fixture provider may be unreachable from CI.
        """
        return os.path.join(self.model_dir, "fixtures_meta.json")

    # --- research (gitignored) ----------------------------------------------
    @property
    def odds_snapshots_csv(self) -> str:
        return os.path.join(self.research_dir, "odds_snapshots.csv")

    @property
    def polymarket_trades_csv(self) -> str:
        return os.path.join(self.research_dir, "polymarket_trades.csv")

    @property
    def price_quality_csv(self) -> str:
        return os.path.join(self.research_dir, "price_quality.csv")

    @property
    def oddsapiio_closes_csv(self) -> str:
        return os.path.join(self.research_dir, "oddsapi_io_closes.csv")

    @property
    def football_data_csv(self) -> str:
        return os.path.join(self.research_dir, "football_data.csv")

    # --- published contract --------------------------------------------------
    @property
    def public_dir(self) -> str:
        return os.path.join(public_dir(), self.league)

    @property
    def public_legacy_dir(self) -> str:
        """Pre-merge JSON shapes. Delete once the tracker reads index.json."""
        return os.path.join(public_dir(), "legacy", self.league)

    def public_json(self, name: str) -> str:
        return os.path.join(self.public_dir, f"{name}.json")

    def public_legacy_json(self, name: str) -> str:
        return os.path.join(self.public_legacy_dir, f"{name}.json")

    # --- helpers -------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create every directory this league writes into."""
        for d in (self.root, self.model_dir, self.research_dir, self.cache_dir,
                  self.public_dir, self.public_legacy_dir):
            os.makedirs(d, exist_ok=True)


_SAFE = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def for_league(league: str) -> LeaguePaths:
    """Paths for one league.

    The id is restricted to lowercase alphanumerics, ``_`` and ``-`` because it
    is interpolated into filesystem paths and into published URLs.
    """
    if not league or not set(league) <= _SAFE:
        raise ValueError(
            f"invalid league id {league!r}: use lowercase letters, digits, '_' or '-'"
        )
    return LeaguePaths(league)
