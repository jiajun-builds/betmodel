from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

DATE_ONLY_FORMAT = "%Y-%m-%d"


def stamp(moment: datetime | None = None) -> str:
    """The one format a timestamp is written to storage in: UTC, to the second.

    Everything that records when something happened went through a bare
    ``isoformat()``, which keeps whatever precision the clock handed over. Values
    read from a provider arrive to the second, values taken from ``now()`` arrive
    to the microsecond, and both land in the same column — so ``fetched_at`` held
    two formats at once and a plain parse of it raised. The precision was never
    meaningful: nothing here measures anything in fractions of a second.

    The published contract already truncates the same way, in
    ``publish.contract.utc``. This is the storage-side counterpart.
    """
    moment = moment or datetime.now(timezone.utc)
    return (
        moment.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def local_matchday(moment: datetime, timezone_name: str) -> str:
    """The league-local day a kickoff belongs to, ``YYYY-MM-DD``.

    The same notion :func:`betmodel.signals.engine.fixture_id` builds an
    identifier from, factored out because three more places now need to tell two
    meetings of the same pair apart and a second definition would be free to
    drift from the first.

    **A day, not a timestamp, and that is the whole design.** Keyed on the exact
    kickoff, a schedule provider nudging a match by ten minutes would mint a new
    fixture -- and a fixture with no opening price on file is one the capture
    goes and opens, writing whatever the book shows now into an open slot. That
    is precisely the mid-market-price-as-opener failure the capture gate exists
    to prevent, so the key has to absorb drift. Measured on the current stores,
    every repeated pairing but one differs by 5 to 120 minutes on the same local
    day; the exception is a genuine reschedule, which a day-level key separates
    and a timestamp-level key would separate for the wrong reason.

    Local, because a 19:00 kickoff in a UTC-6 league falls on the next UTC day
    and calling that its matchday names it by a day nobody played on.
    """
    return moment.astimezone(ZoneInfo(timezone_name)).strftime(DATE_ONLY_FORMAT)


def parse_date_only_series(series: pd.Series) -> pd.Series:
    """
    Parse date-only values while accepting:
    - YYYY-MM-DD / YYYY/MM/DD (canonical)
    - DD/MM/YYYY / DD-MM-YYYY (legacy manual edits / locale exports)

    Both shapes are matched by an explicit anchored regex and parsed with an
    explicit ``format=`` so behaviour does not depend on pandas' dayfirst
    inference, which changes between minor versions.

    The return value is normalized to midnight timestamps so callers can safely
    compare and sort on calendar dates without time components.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.normalize()

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin(["", "nan", "none", "<na>"]))

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    # Canonical ISO: YYYY-MM-DD or YYYY/MM/DD
    iso_mask = cleaned.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", na=False)
    if iso_mask.any():
        iso_text = cleaned.loc[iso_mask].str.replace("/", "-", regex=False)
        result.loc[iso_mask] = pd.to_datetime(iso_text, errors="coerce", format=DATE_ONLY_FORMAT)

    # Legacy DMY: DD/MM/YYYY or DD-MM-YYYY (older manually-curated rows)
    dmy_mask = cleaned.str.match(r"^\d{2}[/-]\d{2}[/-]\d{4}$", na=False) & result.isna()
    if dmy_mask.any():
        dmy_text = cleaned.loc[dmy_mask].str.replace("-", "/", regex=False)
        result.loc[dmy_mask] = pd.to_datetime(dmy_text, errors="coerce", format="%d/%m/%Y")

    return result.dt.normalize()


def format_date_only_series(series: pd.Series, *, missing_value: str = "") -> pd.Series:
    """
    Format date-only values to the canonical YYYY-MM-DD string representation.
    """
    parsed = parse_date_only_series(series)
    formatted = parsed.dt.strftime(DATE_ONLY_FORMAT)
    if missing_value is not None:
        formatted = formatted.fillna(missing_value)
    return formatted
