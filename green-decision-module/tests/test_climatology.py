"""Climatology + persistence tests: DST-correct grouping, fallback, no leakage."""

import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import (
    ClimatologyModel,
    PersistenceForecaster,
    local_calendar,
)
from green_observatory.providers.carbon_base import CARBON


def _hourly_carbon_by_local_hour(start, end, tz="Europe/Paris"):
    """Synthetic frame where carbon intensity equals the Paris-local hour."""
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    local_hour = idx.tz_convert(tz).hour
    return pd.DataFrame({CARBON: np.asarray(local_hour, dtype=float)}, index=idx)


def test_local_calendar_handles_dst():
    idx = pd.DatetimeIndex(
        ["2026-01-15T11:00:00Z", "2026-07-15T11:00:00Z"]
    ).tz_convert("UTC")
    cal = local_calendar(idx, "Europe/Paris")
    assert cal["hour_of_day"].tolist() == [12, 13]  # CET(+1) in winter, CEST(+2) in summer


def test_climatology_recovers_bucket_center():
    # month x dow x hour buckets need a few years to reach the default
    # min_samples=8; with 2 synthetic months we lower it so the 'full' path runs
    # (a (Jan, Monday, 15h) bucket has ~4 samples here).
    df = _hourly_carbon_by_local_hour("2025-01-01", "2025-03-01")
    model = ClimatologyModel(min_samples=3).fit(df)
    target = pd.DatetimeIndex(["2025-01-20T14:00:00Z"])  # 15:00 Paris (CET), a Monday
    out = model.predict(target)
    assert out["source"].iloc[0] == "full"
    assert abs(out["center"].iloc[0] - 15.0) < 1e-6


def test_climatology_falls_back_for_unseen_month():
    df = _hourly_carbon_by_local_hour("2025-01-01", "2025-03-01")  # Jan-Feb only
    model = ClimatologyModel().fit(df)
    # July timestamp: (month=7,...) bucket unseen -> fallback on (dow, hour).
    target = pd.DatetimeIndex(["2025-07-20T13:00:00Z"])  # 15:00 Paris (CEST, +2)
    out = model.predict(target)
    assert out["source"].iloc[0] == "fallback"
    assert abs(out["center"].iloc[0] - 15.0) < 1e-6
    assert np.isfinite(out["center"].iloc[0])


def test_persistence_uses_only_history_up_to_origin():
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    hist = pd.DataFrame({CARBON: [10, 20, 30, 999, 999]}, index=idx)  # future values poisoned
    origin = idx[2]  # value 30
    out = PersistenceForecaster().predict(hist, origin, [1, 24])
    assert (out["prediction"] == 30).all()  # ignores the 999 future rows
