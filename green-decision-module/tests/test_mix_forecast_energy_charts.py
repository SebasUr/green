"""Energy-Charts mix-forecast provider tests without network I/O."""

import pandas as pd

from green_observatory.providers.mix_forecast_energy_charts import (
    EnergyChartsMixForecastProvider,
)


def test_parse_day_ahead_forecast_to_hourly_mean():
    start = pd.Timestamp("2026-02-01T00:00:00Z")
    timestamps = [int((start + pd.Timedelta(minutes=15 * i)).timestamp()) for i in range(8)]
    payload = {
        "unix_seconds": timestamps,
        "forecast_values": [100, 200, 300, 400, 500, 600, 700, 800],
        "production_type": "wind_onshore",
        "forecast_type": "day-ahead",
    }
    out = EnergyChartsMixForecastProvider.parse(payload)
    assert list(out.columns) == ["wind_onshore_day_ahead_forecast_mw"]
    assert out.iloc[:, 0].tolist() == [250.0, 650.0]
    assert str(out.index.tz) == "UTC"


def test_parse_rejects_mismatched_payload_lengths():
    payload = {
        "unix_seconds": [1, 2],
        "forecast_values": [100],
        "production_type": "solar",
    }
    try:
        EnergyChartsMixForecastProvider.parse(payload)
    except ValueError as exc:
        assert "different lengths" in str(exc)
    else:
        raise AssertionError("expected malformed payload to fail")
