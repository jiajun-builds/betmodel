"""The one timezone anything human-facing is shown in.

Three layers use time differently and only this one is about a reader.

**Storage and computation** are UTC everywhere, without exception. Every stored
timestamp ends in ``Z`` and every comparison happens in UTC.

**Identity** uses the league's own local matchday. That is not display: a match
belongs to the day it was played in its own country, and naming a 19:00 kickoff
in a UTC-6 league by the next UTC date names a day nobody played on. It is also
frozen, because published identifiers have to keep matching the ones already
issued.

**Display** is this module, and it is a property of the reader rather than of the
league. Showing each league in its own zone puts two timezones in one inbox and
tells the reader nothing actionable: "19:00" does not say whether that is now, or
one in the morning where they are. One zone, theirs, answers the only question a
kickoff time is asked.

Override with ``BETMODEL_DISPLAY_TZ`` for a reader somewhere else.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Europe/London"
ENV = "BETMODEL_DISPLAY_TZ"


def timezone_name() -> str:
    return os.environ.get(ENV, "").strip() or DEFAULT_TIMEZONE


def zone() -> ZoneInfo:
    return ZoneInfo(timezone_name())


def moment(when: datetime, *, fmt: str = "%m-%d %H:%M") -> str:
    """A timestamp as the reader should see it, in the one display zone."""
    return when.astimezone(zone()).strftime(fmt)


def label() -> str:
    """Short zone label to put beside a shown time.

    Shown once per message rather than per line: the point is that there is only
    ever one, and repeating it invites the reader to look for a second.
    """
    return datetime.now(zone()).strftime("%Z")
