#!/usr/bin/env python3
"""Collapse every team-name mapping surface into one file per league.

Before: two CSVs with different column names for the same namespace, plus a
hardcoded dictionary of odds-api.io spellings, plus an accent-repair dictionary.
Four places to update when a club is renamed, and three of them silent when they
are missed.

After: one CSV per league, columns fixed by betmodel.teams.KNOWN_NAMESPACES, and
accents handled by folding at lookup rather than by listing both forms.

Verifies that every spelling the old surfaces could resolve still resolves.
"""

from __future__ import annotations

import ast
import csv
import os
import pathlib
import sys

from betmodel import paths, teams

RENAMES = {
    "sofa_team": "sofascore_team",     # CSL's name for SofaScore's namespace
    "odds_team": "theoddsapi_team",    # both leagues: The Odds API's namespace
}

#: Where the odds-api.io spellings currently live: a hardcoded dictionary in the
#: pre-merge provider, keyed the other way round (their spelling -> ours). Parsed
#: from source rather than transcribed, because transcribing nineteen near-identical
#: club names by hand is exactly the error this whole exercise is removing.
LIGAMX_ODDSAPIIO_SOURCE = (
    "legacy/ligamx/src/ligamx/odds/oddsapi_io.py",
    "API_TO_STANDARD",
)


def _dict_from_source(rel_path: str, name: str) -> dict[str, str]:
    """Read one module-level dict literal out of a file without importing it."""
    text = (pathlib.Path(paths.project_root()) / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return {str(k): str(v) for k, v in ast.literal_eval(node.value).items()}
    raise KeyError(f"{name} not found in {rel_path}")


def _read(path: str) -> tuple[list[str], list[dict]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return [c for c in reader.fieldnames or [] if c], list(reader)


def unify(league: str, *, extra: dict[str, str] | None = None, check: bool) -> list[str]:
    path = paths.for_league(league).team_mapping_csv
    before = teams.load(path, league=league)
    columns, rows = _read(path)

    new_columns = [RENAMES.get(c, c) for c in columns]
    out_rows = [{RENAMES.get(k, k): v for k, v in row.items() if k} for row in rows]

    if extra:
        # extra is keyed the upstream way round; invert it once, here.
        standard_to_spelling = {v: k for k, v in extra.items()}
        if len(standard_to_spelling) != len(extra):
            raise SystemExit("two upstream spellings map to one canonical name")
        if "oddsapiio_team" not in new_columns:
            new_columns.append("oddsapiio_team")
        for row in out_rows:
            row.setdefault("oddsapiio_team", "")
            standard = (row.get(teams.CANONICAL_COLUMN) or "").strip()
            spelling = standard_to_spelling.get(standard)
            if spelling:
                row["oddsapiio_team"] = spelling
        missing = set(standard_to_spelling) - {
            (r.get(teams.CANONICAL_COLUMN) or "").strip() for r in out_rows
        }
        if missing:
            raise SystemExit(f"{league}: no CSV row for {sorted(missing)}")

    # canonical column last, namespaces in a stable order
    ordered = [c for c in teams.KNOWN_NAMESPACES if c in new_columns]
    ordered += [c for c in new_columns
                if c not in ordered and c != teams.CANONICAL_COLUMN]
    ordered.append(teams.CANONICAL_COLUMN)

    if not check:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            for row in out_rows:
                writer.writerow({c: (row.get(c) or "").strip() for c in ordered})

    after = teams.load(path, league=league) if not check else None
    problems: list[str] = []
    if after is not None:
        # Nothing resolvable before may become unresolvable.
        for alias in before._to_canonical:
            if alias not in after._to_canonical:
                problems.append(f"{league}: lost alias {alias!r}")
        if set(before.canonical) != set(after.canonical):
            problems.append(f"{league}: canonical set changed")
        # Both directions. Resolving the spelling is not enough: a wrong value in
        # the column can still resolve through the canonical name while making the
        # outgoing request ask for a club that does not exist.
        for spelling, standard in (extra or {}).items():
            if after.to_standard(spelling) != standard:
                problems.append(
                    f"{league}: {spelling!r} resolves to "
                    f"{after.to_standard(spelling)!r}, expected {standard!r}"
                )
            written = after.spelling(standard, "oddsapiio_team")
            if written != spelling:
                problems.append(
                    f"{league}: oddsapiio_team for {standard!r} is {written!r}, "
                    f"expected {spelling!r}"
                )
    print(f"  {league}: {len(out_rows)} clubs, columns -> {ordered}")
    return problems


def main() -> int:
    check = "--check" in sys.argv
    ligamx_oddsapiio = _dict_from_source(*LIGAMX_ODDSAPIIO_SOURCE)
    print(f"  lifted {len(ligamx_oddsapiio)} odds-api.io spellings from source")
    problems = unify("csl", check=check)
    problems += unify("ligamx", extra=ligamx_oddsapiio, check=check)
    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("  every spelling that resolved before still resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
