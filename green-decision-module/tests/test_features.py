"""Feature tests: the no-look-ahead invariant, label alignment, holidays."""

import numpy as np
import pandas as pd

from green_observatory.carbon.features import FeatureBuilder
from green_observatory.providers.carbon_base import CARBON


def _frame(periods=300, start="2025-06-01"):
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    hour = idx.tz_convert("Europe/Paris").hour
    carbon = 20 + 10 * np.sin(2 * np.pi * np.asarray(hour) / 24.0)
    return pd.DataFrame(
        {
            CARBON: carbon,
            "consumption_mw": 60000 + 100 * np.arange(periods),
            "wind_mw": 5000.0,
            "solar_mw": 1000.0,
            "hydro_mw": 3000.0,
            "nuclear_mw": 45000.0,
            "physical_exchange_mw": -8000.0,
        },
        index=idx,
    )


def test_origin_features_do_not_look_ahead():
    """Origin features at t0 must be identical whether computed on the full
    series or only on history up to t0 - the invariant the fast backtest uses."""
    frame = _frame()
    fb = FeatureBuilder(climatology=None)
    full = fb.origin_features(frame)
    t0 = frame.index[200]
    truncated = fb.origin_features(frame.loc[:t0])
    pd.testing.assert_series_equal(full.loc[t0], truncated.loc[t0], check_names=False)


def test_supervised_label_is_future_value_and_calendar_is_target():
    frame = _frame()
    fb = FeatureBuilder(climatology=None)
    h = 6
    x, y = fb.build_supervised(frame, h)
    t0 = x.index[100]
    # label at origin t0 equals actual carbon at t0 + h
    assert abs(y.loc[t0] - frame[CARBON].loc[t0 + pd.Timedelta(hours=h)]) < 1e-9
    # target-calendar hour matches the *target* local hour, not the origin's
    target_local_hour = (t0 + pd.Timedelta(hours=h)).tz_convert("Europe/Paris").hour
    assert int(x.loc[t0, "tgt_hour_of_day"]) == target_local_hour


def test_holiday_flag():
    fb = FeatureBuilder(climatology=None, holidays_country="FR")
    idx = pd.DatetimeIndex(["2026-01-01T08:00:00Z", "2026-01-02T08:00:00Z"])
    cal = fb.calendar_features(idx)
    assert cal["is_holiday"].tolist() == [1, 0]  # New Year's Day is a FR holiday
