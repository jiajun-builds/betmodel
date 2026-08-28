from __future__ import annotations

import pandas as pd

DATE_ONLY_FORMAT = "%Y-%m-%d"


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
