import pandas as pd

from green_observatory.providers.rte_system_forecast import (
    RteSystemForecastProvider,
)


def test_normalize_unavailability_preserves_publication_versions_and_intervals():
    records = [
        {
            "identifier": "event-1",
            "message_id": "message-2",
            "version": 2,
            "creation_date": "2026-02-01T08:00:00Z",
            "publication_date": "2026-02-01T08:05:00Z",
            "start_date": "2026-02-02T00:00:00Z",
            "end_date": "2026-02-03T00:00:00Z",
            "fuel_type": "NUCLEAR",
            "event_status": "ACTIVE",
            "affected_asset_or_unit_installed_capacity": 1_300,
            "values": [
                {
                    "start_date": "2026-02-02T00:00:00Z",
                    "end_date": "2026-02-02T12:00:00Z",
                    "available_capacity": 300,
                    "unavailable_capacity": 1_000,
                },
                {
                    "start_date": "2026-02-02T12:00:00Z",
                    "end_date": "2026-02-03T00:00:00Z",
                    "available_capacity": 800,
                    "unavailable_capacity": 500,
                },
            ],
        }
    ]
    frame = RteSystemForecastProvider.normalize_unavailability(records)
    assert len(frame) == 2
    assert frame["version"].tolist() == [2, 2]
    assert frame["unavailable_capacity_mw"].tolist() == [1_000, 500]
    assert str(frame["publication_date"].dt.tz) == "UTC"


def test_normalize_generation_forecast_is_flat_and_parquet_safe():
    records = [
        {
            "start_date": "2026-02-02T00:00:00Z",
            "end_date": "2026-02-03T00:00:00Z",
            "type": "D-1",
            "production_type": "WIND_ONSHORE",
            "sub_type": "NORMAL",
            "values": [
                {
                    "start_date": "2026-02-02T00:00:00Z",
                    "end_date": "2026-02-02T01:00:00Z",
                    "updated_date": "2026-02-01T12:00:00Z",
                    "value": 4_200,
                }
            ],
        }
    ]
    frame = RteSystemForecastProvider.normalize_generation_forecast(records)
    assert frame.loc[0, "forecast_type"] == "D-1"
    assert frame.loc[0, "production_type"] == "WIND_ONSHORE"
    assert frame.loc[0, "value_mw"] == 4_200
    assert frame.loc[0, "updated_date"] < frame.loc[0, "target_start"]


def test_provider_reads_only_rte_credentials_from_ignored_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "RTE_CLIENT_ID=test-id\n"
        "RTE_CLIENT_SECRET=test-secret\n"
        "UNRELATED=value\n"
    )
    provider = RteSystemForecastProvider.from_env(dotenv_path=dotenv)
    assert provider.client_id == "test-id"
    assert provider.client_secret == "test-secret"
