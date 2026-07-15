"""Open-Meteo weather aggregation tests (no network)."""

from green_observatory.providers.weather_openmeteo import WeatherProvider

_PAYLOAD = [
    {"hourly": {"time": ["2026-01-01T00:00"], "wind_speed_100m": [10.0], "shortwave_radiation": [100.0]}},
    {"hourly": {"time": ["2026-01-01T00:00"], "wind_speed_100m": [30.0], "shortwave_radiation": [200.0]}},
]


def test_capacity_weighted_wind_simple_solar():
    out = WeatherProvider.parse(_PAYLOAD, weights=[3.0, 1.0])
    assert abs(out["wind_speed_100m"].iloc[0] - 15.0) < 1e-9   # (10*3 + 30*1) / 4
    assert abs(out["solar_radiation"].iloc[0] - 150.0) < 1e-9  # solar stays a simple mean


def test_unweighted_falls_back_to_mean():
    out = WeatherProvider.parse(_PAYLOAD)
    assert abs(out["wind_speed_100m"].iloc[0] - 20.0) < 1e-9
