"""Contract tests for the typed models (UTC enforcement, bounds, serialization)."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from green_observatory.models import (
    CarbonForecast,
    CarbonSignal,
    GreenWindow,
    ModelName,
    WindowType,
)

PARIS = timezone(timedelta(hours=2))


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        CarbonSignal(timestamp=datetime(2026, 7, 8, 12), carbon_intensity_gco2_kwh=42.0)


def test_aware_datetime_normalized_to_utc():
    s = CarbonSignal(
        timestamp=datetime(2026, 7, 8, 12, tzinfo=PARIS), carbon_intensity_gco2_kwh=42.0
    )
    assert s.timestamp.utcoffset() == timedelta(0)
    assert s.timestamp.hour == 10  # 12:00+02:00 -> 10:00Z


def test_green_score_bounds_enforced():
    with pytest.raises(ValidationError):
        CarbonSignal(
            timestamp=datetime(2026, 7, 8, tzinfo=timezone.utc),
            carbon_intensity_gco2_kwh=10.0,
            green_score=1.5,
        )


def test_negative_carbon_rejected():
    with pytest.raises(ValidationError):
        CarbonSignal(
            timestamp=datetime(2026, 7, 8, tzinfo=timezone.utc), carbon_intensity_gco2_kwh=-1.0
        )


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        CarbonSignal(
            timestamp=datetime(2026, 7, 8, tzinfo=timezone.utc),
            carbon_intensity_gco2_kwh=10.0,
            typo_field=1,
        )


def test_green_window_duration_and_json_roundtrip():
    w = GreenWindow(
        start=datetime(2026, 7, 7, 22, tzinfo=timezone.utc),
        end=datetime(2026, 7, 8, 2, tzinfo=timezone.utc),
        window_type=WindowType.low_carbon_window,
        carbon_score=0.82,
        mean_carbon_intensity_gco2_kwh=38.0,
        source_model=ModelName.climatology,
    )
    assert w.duration_hours == 4.0
    restored = GreenWindow.model_validate_json(w.model_dump_json())
    assert restored.carbon_score == 0.82
    assert restored.window_type is WindowType.low_carbon_window


def test_forecast_records_issued_at_for_leakage_audit():
    fc = CarbonForecast(
        issued_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
        target_time=datetime(2026, 7, 8, 12, tzinfo=timezone.utc),
        horizon_hours=12,
        model=ModelName.persistence,
        predicted_carbon_intensity_gco2_kwh=30.0,
    )
    assert fc.issued_at < fc.target_time
