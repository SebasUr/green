"""Shared protocol helpers for the daily-refit forecasting stack.

Slimmed to what the production pipeline actually consumes: UTC coercion,
hourly regularization, and the dispatchable-gas regime labels used to
supervise the physical mixture-of-experts. The historical multi-protocol
evaluation machinery lives in the ``snapshot2007`` branch.
"""

from __future__ import annotations

import pandas as pd


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Coerce a timestamp-like value to a tz-aware UTC Timestamp."""
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def regularize_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert missing UTC hours so positional lags remain genuine hour lags."""
    frame = frame.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    complete = pd.date_range(frame.index.min(), frame.index.max(), freq="1h")
    return frame.reindex(complete)


#: Gas subtypes the operator actually dispatches (cogeneration excluded).
DISPATCHABLE_GAS_COLUMNS = ("gas_ccg_mw", "gas_turbine_mw", "gas_other_mw")


def dispatchable_gas(frame: pd.DataFrame) -> pd.Series:
    """Observed CCG/TAC/other gas MW used only to construct supervised labels."""
    missing = [column for column in DISPATCHABLE_GAS_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"fossil-regime frame is missing columns: {missing}")
    return frame.loc[:, DISPATCHABLE_GAS_COLUMNS].sum(axis=1, min_count=1)


def fossil_regime_labels(
    frame: pd.DataFrame, *, ccg_threshold_mw: float, peak_threshold_mw: float
) -> pd.Series:
    """0=baseload, 1=CCG, 2=peak labels from dispatchable gas thresholds."""
    if not 0.0 <= ccg_threshold_mw < peak_threshold_mw:
        raise ValueError("regime thresholds must satisfy 0 <= ccg < peak")
    gas = dispatchable_gas(frame)
    labels = pd.Series(0, index=frame.index, dtype="int8", name="fossil_regime")
    labels.loc[gas >= ccg_threshold_mw] = 1
    labels.loc[gas >= peak_threshold_mw] = 2
    return labels.where(gas.notna())


__all__ = [
    "DISPATCHABLE_GAS_COLUMNS",
    "_utc",
    "dispatchable_gas",
    "fossil_regime_labels",
    "regularize_hourly",
]
