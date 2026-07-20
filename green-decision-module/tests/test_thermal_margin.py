"""Vintage causality and unit-safety of the implied thermal-margin features."""

import numpy as np
import pandas as pd
import pytest

from green_observatory.carbon.thermal_margin import (
    NUCLEAR_FUELS,
    day_ahead_thermal_margin_features,
    unavailable_capacity_hourly,
)

# Two full Paris days in winter: midnights are 23:00 UTC.
DAY1 = pd.Timestamp("2025-01-14 23:00:00+00:00")
DAY2 = pd.Timestamp("2025-01-15 23:00:00+00:00")
HOURS = pd.date_range(DAY1, DAY2 + pd.Timedelta("23h"), freq="1h")


def _message(identifier, publication, status, mw, start, end, version=1):
    return {
        "identifier": identifier,
        "message_id": f"{identifier}_{version:03d}",
        "version": version,
        "publication_date": pd.Timestamp(publication),
        "event_status": status,
        "fuel_type": "NUCLEAR",
        "interval_start": pd.Timestamp(start),
        "interval_end": pd.Timestamp(end),
        "unavailable_capacity_mw": mw,
    }


def test_latest_version_before_paris_midnight_wins():
    intervals = pd.DataFrame(
        [
            # v1 published before day1 midnight: 1000 MW across both days
            _message("A", "2025-01-14 12:00:00+00:00", "ACTIVE", 1000.0, DAY1, DAY2 + pd.Timedelta("24h")),
            # v2 published during day1 (before day2 midnight): reduced to 400 MW
            _message("A", "2025-01-15 10:00:00+00:00", "ACTIVE", 400.0, DAY1, DAY2 + pd.Timedelta("24h"), version=2),
        ]
    )
    out = unavailable_capacity_hourly(intervals, HOURS, {"nuclear": NUCLEAR_FUELS})
    assert (out.loc[DAY1 : DAY2 - pd.Timedelta("1h"), "nuclear"] == 1000.0).all()
    assert (out.loc[DAY2:, "nuclear"] == 400.0).all()


def test_non_active_latest_version_cancels_message():
    intervals = pd.DataFrame(
        [
            _message("B", "2025-01-14 12:00:00+00:00", "ACTIVE", 900.0, DAY1, DAY2 + pd.Timedelta("24h")),
            _message("B", "2025-01-15 09:00:00+00:00", "DISMISSED", 900.0, DAY1, DAY2 + pd.Timedelta("24h"), version=2),
        ]
    )
    out = unavailable_capacity_hourly(intervals, HOURS, {"nuclear": NUCLEAR_FUELS})
    assert (out.loc[DAY1 : DAY2 - pd.Timedelta("1h"), "nuclear"] == 900.0).all()
    assert (out.loc[DAY2:, "nuclear"] == 0.0).all()


def test_multi_interval_version_keeps_all_rows():
    publication = "2025-01-14 12:00:00+00:00"
    intervals = pd.DataFrame(
        [
            _message("C", publication, "ACTIVE", 500.0, DAY2, DAY2 + pd.Timedelta("6h")),
            _message("C", publication, "ACTIVE", 500.0, DAY2 + pd.Timedelta("12h"), DAY2 + pd.Timedelta("18h")),
        ]
    )
    out = unavailable_capacity_hourly(intervals, HOURS, {"nuclear": NUCLEAR_FUELS})
    day2 = out.loc[DAY2:, "nuclear"]
    assert (day2.iloc[0:6] == 500.0).all()
    assert (day2.iloc[6:12] == 0.0).all()
    assert (day2.iloc[12:18] == 500.0).all()


def test_millisecond_resolution_inputs_are_safe():
    # Regression: parquet round-trips yield ms-resolution datetimes; mixing
    # them with ns must not silently break searchsorted comparisons.
    intervals = pd.DataFrame(
        [_message("D", "2025-01-14 12:00:00+00:00", "ACTIVE", 750.0, DAY1, DAY2)]
    )
    for column in ("publication_date", "interval_start", "interval_end"):
        intervals[column] = intervals[column].astype("datetime64[ms, UTC]")
    hours_ms = HOURS.as_unit("ms")
    out = unavailable_capacity_hourly(intervals, hours_ms, {"nuclear": NUCLEAR_FUELS})
    assert (out.loc[DAY1 : DAY2 - pd.Timedelta("1h"), "nuclear"] == 750.0).all()


def test_tightness_combines_residual_and_outages():
    forecast = pd.DataFrame(
        {
            "load_day_ahead_forecast_mw": 50000.0,
            "wind_onshore_day_ahead_forecast_mw": 5000.0,
            "wind_offshore_day_ahead_forecast_mw": 1000.0,
            "solar_day_ahead_forecast_mw": 4000.0,
        },
        index=HOURS,
    )
    intervals = pd.DataFrame(
        [_message("E", "2025-01-14 12:00:00+00:00", "ACTIVE", 2000.0, DAY1, DAY2 + pd.Timedelta("24h"))]
    )
    out = day_ahead_thermal_margin_features(forecast, intervals)
    assert (out["residual_demand_day_ahead_mw"] == 40000.0).all()
    assert (out["thermal_tightness_day_ahead_mw"] == 42000.0).all()
    # 24h delta defined from the second day on, zero for constant inputs
    assert np.isnan(out["thermal_tightness_delta_day_ahead_mw"].iloc[0])
    assert (out["thermal_tightness_delta_day_ahead_mw"].iloc[24:] == 0.0).all()


def test_missing_residual_columns_raise():
    with pytest.raises(ValueError, match="residual-demand"):
        day_ahead_thermal_margin_features(pd.DataFrame(index=HOURS), pd.DataFrame())
