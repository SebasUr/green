import numpy as np
import pandas as pd

from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)


def test_rte_d1_features_respect_publication_time_at_origin():
    target = pd.Timestamp("2026-02-02T12:00:00Z")
    rows = []
    for production_type, value in (
        ("WIND_ONSHORE", 4_000),
        ("WIND_OFFSHORE", 1_000),
        ("SOLAR", 3_000),
    ):
        rows.append(
            {
                "forecast_type": "D-1",
                "production_type": production_type,
                "target_start": target,
                "updated_date": target - pd.Timedelta(hours=18),
                "value_mw": value,
            }
        )
    store = RteGenerationForecastFeatureStore(pd.DataFrame(rows))
    origins = pd.DatetimeIndex(
        [target - pd.Timedelta(hours=24), target - pd.Timedelta(hours=12)]
    )
    features = store.features_by_horizon(origins, [24, 12])
    assert np.isnan(features[24]["rte_tgt_solar_d1_mw"].iloc[0])
    assert features[12]["rte_tgt_solar_d1_mw"].iloc[1] == 3_000
    assert features[12]["rte_tgt_wind_d1_mw"].iloc[1] == 5_000
    assert features[12]["rte_tgt_variable_renewables_d1_mw"].iloc[1] == 8_000


def test_rte_d1_features_drop_revisions_after_target():
    frame = pd.DataFrame(
        [
            {
                "forecast_type": "D-1",
                "production_type": "SOLAR",
                "target_start": "2026-02-02T12:00:00Z",
                "updated_date": "2026-02-03T12:00:00Z",
                "value_mw": 9_999,
            }
        ]
    )
    store = RteGenerationForecastFeatureStore(frame)
    origin = pd.DatetimeIndex(["2026-02-02T00:00:00Z"])
    assert np.isnan(store.features_by_horizon(origin, [12])[12].iloc[0]).all()


def test_rte_d1_source_selection_does_not_emit_empty_columns():
    target = pd.Timestamp("2026-02-02T12:00:00Z")
    frame = pd.DataFrame(
        [
            {
                "forecast_type": "D-1",
                "production_type": "SOLAR",
                "target_start": target,
                "updated_date": target - pd.Timedelta(hours=18),
                "value_mw": 3_000,
            }
        ]
    )
    store = RteGenerationForecastFeatureStore(
        frame, production_types=("SOLAR",)
    )
    origin = pd.DatetimeIndex([target - pd.Timedelta(hours=12)])
    features = store.features_by_horizon(origin, [12])[12]

    assert list(features.columns) == ["rte_tgt_solar_d1_mw"]
    assert features.iloc[0, 0] == 3_000
