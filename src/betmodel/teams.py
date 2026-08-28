"""One team-name mapping per league, and one way to resolve a name against it.

Four upstreams spell the same club four ways, and none of them is authoritative.
The mapping file is: one canonical name per row, plus one column per upstream
namespace holding that upstream's spelling.

Two rules make it hold together.

**A lookup folds accents rather than enumerating them.** The pre-merge code
carried a dictionary listing "Club América" and "Club America", "León" and
"Leon", and so on, which only works until an upstream adds a club nobody
remembered to double-enter. Folding at lookup time covers every such pair at
once, including the ones not written down yet.

**A collision is fatal.** If two canonical clubs both claim the same spelling,
every price for one of them silently lands on the other. That is a data
corruption that produces no error and no empty column, so it is refused at load.
"""

from __future__ import annotations

import csv
import os
import unicodedata
from dataclasses import dataclass, field

#: The canonical column. Every other column is an alias namespace.
CANONICAL_COLUMN = "standard_team"

#: Namespaces the engine knows how to ask for. A file may carry others, which are
#: still indexed for lookup but cannot be requested by name.
KNOWN_NAMESPACES = (
    "match_team",        # the fixtures/schedule provider's spelling
    "sofascore_team",    # SofaScore
    "theoddsapi_team",   # The Odds API
    "oddsapiio_team",    # odds-api.io
)


class TeamMappingError(ValueError):
    """The mapping file is unusable. Never degraded into an empty mapping."""


def normalize(name: str) -> str:
    """Fold a name to its comparison form.

    Accents are stripped, case is dropped and whitespace is collapsed, so
    "Club América", "club america" and "Club  America" all meet.
    """
    if name is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


@dataclass(frozen=True)
class TeamMapping:
    """Aliases to canonical names, for one league."""

    league: str
    path: str
    canonical: tuple[str, ...]
    namespaces: tuple[str, ...]
    _to_canonical: dict[str, str] = field(repr=False, default_factory=dict)
    _by_namespace: dict[str, dict[str, str]] = field(repr=False, default_factory=dict)

    # --- lookups ------------------------------------------------------------
    def to_standard(self, name: str) -> str | None:
        """Canonical name for any known spelling, or ``None``."""
        return self._to_canonical.get(normalize(name))

    def require_standard(self, name: str, *, source: str = "") -> str:
        """Canonical name, or raise.

        Callers that write a price into a match row use this: an unmapped club
        must stop the run, because dropping the row silently loses a capture that
        cannot be taken again, and guessing writes it onto the wrong match.
        """
        found = self.to_standard(name)
        if found is None:
            where = f" (from {source})" if source else ""
            raise TeamMappingError(
                f"{self.league}: unmapped team name {name!r}{where}. "
                f"Add it to {os.path.basename(self.path)}."
            )
        return found

    def spelling(self, standard: str, namespace: str) -> str | None:
        """How one upstream spells a canonical club."""
        if namespace not in self._by_namespace:
            raise TeamMappingError(
                f"{self.league}: no namespace {namespace!r} in "
                f"{os.path.basename(self.path)}; have {sorted(self._by_namespace)}"
            )
        return self._by_namespace[namespace].get(normalize(standard))

    def __contains__(self, name: str) -> bool:
        return normalize(name) in self._to_canonical

    def __len__(self) -> int:
        return len(self.canonical)


def load(path: str, *, league: str = "") -> TeamMapping:
    """Read and validate a mapping file.

    Every failure raises. An empty or missing mapping used to be tolerated and
    the result was team names flowing through unmapped, which looks like working
    software right up until two upstreams disagree.
    """
    league = league or os.path.basename(os.path.dirname(path))
    if not os.path.exists(path):
        raise TeamMappingError(f"{league}: no team mapping at {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise TeamMappingError(f"{league}: team mapping {path} is empty")

    columns = [c for c in (rows[0].keys()) if c]
    if CANONICAL_COLUMN not in columns:
        raise TeamMappingError(
            f"{league}: {path} has no {CANONICAL_COLUMN!r} column; found {columns}"
        )
    namespaces = tuple(c for c in columns if c != CANONICAL_COLUMN)

    canonical: list[str] = []
    to_canonical: dict[str, str] = {}
    by_namespace: dict[str, dict[str, str]] = {ns: {} for ns in namespaces}

    for line, row in enumerate(rows, start=2):
        standard = (row.get(CANONICAL_COLUMN) or "").strip()
        if not standard:
            raise TeamMappingError(f"{league}: {path} line {line} has no canonical name")
        if normalize(standard) in {normalize(c) for c in canonical}:
            raise TeamMappingError(
                f"{league}: {path} line {line} repeats canonical name {standard!r}"
            )
        canonical.append(standard)

        aliases = [standard] + [
            (row.get(ns) or "").strip() for ns in namespaces
        ]
        for alias in aliases:
            if not alias:
                continue
            key = normalize(alias)
            clash = to_canonical.get(key)
            if clash is not None and clash != standard:
                raise TeamMappingError(
                    f"{league}: {path} line {line}: spelling {alias!r} is claimed by "
                    f"both {clash!r} and {standard!r}. One club's prices would "
                    "silently be filed under the other."
                )
            to_canonical[key] = standard

        for ns in namespaces:
            value = (row.get(ns) or "").strip()
            if value:
                by_namespace[ns][normalize(standard)] = value

    return TeamMapping(
        league=league,
        path=path,
        canonical=tuple(canonical),
        namespaces=namespaces,
        _to_canonical=to_canonical,
        _by_namespace=by_namespace,
    )


def for_league(league: str) -> TeamMapping:
    """The mapping for one league, from its configured data directory."""
    from betmodel import paths

    return load(paths.for_league(league).team_mapping_csv, league=league)
