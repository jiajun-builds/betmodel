"""Load and validate ``leagues/*.yml``.

Every failure here is fatal. The pipeline runs unattended and publishes to a
live board, so a config that cannot be understood must stop the run rather than
fall back to a default. The pre-merge code loaded its team mapping inside a bare
``except Exception`` that returned empty dictionaries, which turned a missing
file into silently unmapped team names much later in the run.
"""

from __future__ import annotations

import os
from functools import lru_cache

import yaml

from betmodel import paths
from betmodel.config.schema import ConfigError, LeagueConfig

__all__ = ["ConfigError", "available_leagues", "load_league", "load_all", "clear_cache"]

_SUFFIXES = (".yml", ".yaml")


def _league_files(directory: str | None = None) -> dict[str, str]:
    directory = directory or paths.leagues_dir()
    if not os.path.isdir(directory):
        raise ConfigError(f"league directory not found: {directory}")
    found: dict[str, str] = {}
    for entry in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(entry)
        if ext not in _SUFFIXES or stem.startswith("_"):
            continue
        if stem in found:
            raise ConfigError(
                f"two config files define league {stem!r} in {directory}"
            )
        found[stem] = os.path.join(directory, entry)
    return found


def available_leagues(directory: str | None = None) -> tuple[str, ...]:
    """League ids discoverable on disk, sorted.

    Callers iterate this instead of hardcoding a league list, which is what
    makes adding a league a matter of dropping in one file.
    """
    return tuple(_league_files(directory))


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return raw


@lru_cache(maxsize=None)
def _load_cached(path: str, stem: str) -> LeagueConfig:
    raw = _read(path)
    cfg = LeagueConfig.parse(raw, os.path.basename(path))
    # The filename is the league id used in paths and published URLs, so a
    # mismatch would mean data written under one name and published under another.
    if cfg.id != stem:
        raise ConfigError(
            f"{path}: id is {cfg.id!r} but the filename says {stem!r}; "
            "they must match because the id becomes a directory and a URL segment"
        )
    paths.for_league(cfg.id)  # rejects ids unsafe in a path or URL
    return cfg


def load_league(league: str, directory: str | None = None) -> LeagueConfig:
    """One league's validated config."""
    files = _league_files(directory)
    if league not in files:
        raise ConfigError(
            f"unknown league {league!r}; available: {sorted(files) or 'none'}"
        )
    return _load_cached(files[league], league)


def load_all(directory: str | None = None) -> dict[str, LeagueConfig]:
    """Every league, validated. Raises on the first bad file."""
    return {name: _load_cached(path, name) for name, path in _league_files(directory).items()}


def clear_cache() -> None:
    """Drop memoised configs. For tests and for long-lived processes."""
    _load_cached.cache_clear()
