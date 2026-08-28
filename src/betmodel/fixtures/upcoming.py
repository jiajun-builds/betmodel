"""Upcoming fixtures, with one unambiguous kickoff.

Every stage that decides *when* to do something reads this: the open poller
decides what is still pending, the close poller decides what is inside its
window, and the signal engine decides what is still upcoming. All three are wrong
in the same way if the kickoff is wrong.

So there is exactly one kickoff column and it is UTC by construction. One of the
merged pipelines learned this the hard way: its match table's ``Time`` column
mixes local and UTC, and the ambiguity fabricated a result once. It responded by
writing an explicit ``kickoff_utc`` and never reading ``Date``/``Time`` again.
The other pipeline was still parsing the pair and calling the result UTC, which
is correct for that league today and is exactly the assumption that failed
elsewhere.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)

KICKOFF_COLUMN = "kickoff_utc"


@dataclass(frozen=True)
class Fixture:
    """One upcoming match, with names already in canonical form."""

    home: str
    away: str
    kickoff: datetime
    round: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.home, self.away

    @property
    def label(self) -> str:
        return f"{self.home} v {self.away}"


def _parse_kickoff(value: str) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def load_upcoming(path: str, *, mapping=None) -> list[Fixture]:
    """Read the upcoming-fixtures file.

    ``kickoff_utc`` is required. A file without it is refused rather than parsed
    from ``Date`` and ``Time``, because that pair is only unambiguous by
    convention and the convention has already been broken once.

    ``mapping`` optionally normalises team names to canonical form. Without it
    the names are taken as written, which is right for a file this repo produced
    and wrong for anything else.
    """
    fixtures: list[Fixture] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if KICKOFF_COLUMN not in columns:
            raise ValueError(
                f"{path} has no {KICKOFF_COLUMN!r} column. Every timing decision "
                "reads it, and deriving one from Date and Time assumes a "
                "convention that has been wrong before. Regenerate the file with "
                "the fixtures stage."
            )
        for row in reader:
            home = (row.get("Home") or "").strip()
            away = (row.get("Away") or "").strip()
            kickoff = _parse_kickoff((row.get(KICKOFF_COLUMN) or "").strip())
            if not home or not away or kickoff is None:
                continue
            if mapping is not None:
                home = mapping.require_standard(home, source=path)
                away = mapping.require_standard(away, source=path)
            fixtures.append(
                Fixture(home=home, away=away, kickoff=kickoff,
                        round=str(row.get("round") or row.get("Wk") or "").strip())
            )
    return fixtures
