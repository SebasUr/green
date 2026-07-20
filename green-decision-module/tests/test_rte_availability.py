import pandas as pd

from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore


def _intervals():
    return pd.DataFrame(
        [
            {
                "identifier": "outage-a",
                "message_id": "a-v1",
                "version": 1,
                "publication_date": "2026-01-01T00:00:00Z",
                "event_status": "ACTIVE",
                "unavailability_type": "PLANNED",
                "fuel_type": "NUCLEAR",
                "interval_start": "2026-01-02T00:00:00Z",
                "interval_end": "2026-01-04T00:00:00Z",
                "unavailable_capacity_mw": 1_000,
            },
            {
                "identifier": "outage-a",
                "message_id": "a-v2",
                "version": 2,
                "publication_date": "2026-01-02T12:00:00Z",
                "event_status": "ACTIVE",
                "unavailability_type": "PLANNED",
                "fuel_type": "NUCLEAR",
                "interval_start": "2026-01-02T00:00:00Z",
                "interval_end": "2026-01-04T00:00:00Z",
                "unavailable_capacity_mw": 500,
            },
            {
                "identifier": "gas-a",
                "message_id": "gas-v1",
                "version": 1,
                "publication_date": "2026-01-01T00:00:00Z",
                "event_status": "ACTIVE",
                "unavailability_type": "UNPLANNED",
                "fuel_type": "FOSSIL_GAS",
                "interval_start": "2026-01-02T00:00:00Z",
                "interval_end": "2026-01-04T00:00:00Z",
                "unavailable_capacity_mw": 200,
            },
        ]
    )


def test_availability_replay_uses_only_version_published_by_origin():
    store = RteAvailabilityFeatureStore(_intervals())
    origins = pd.DatetimeIndex(
        ["2026-01-02T00:00:00Z", "2026-01-02T18:00:00Z"]
    )
    features = store.features_by_horizon(origins, [12])[12]
    assert features["rte_tgt_nuclear_unavailable_mw"].tolist() == [1_000, 500]
    assert features["rte_tgt_gas_unavailable_mw"].tolist() == [200, 200]
    assert features["rte_tgt_total_unavailable_mw"].tolist() == [1_200, 700]


def test_availability_replay_expires_events_before_target():
    store = RteAvailabilityFeatureStore(_intervals())
    origin = pd.DatetimeIndex(["2026-01-04T00:00:00Z"])
    features = store.features_by_horizon(origin, [1])[1]
    assert features["rte_tgt_total_unavailable_mw"].iloc[0] == 0.0
