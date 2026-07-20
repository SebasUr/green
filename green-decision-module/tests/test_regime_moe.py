import numpy as np
import pandas as pd

from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON


def test_day_ahead_features_are_masked_after_local_delivery_day() -> None:
    index = pd.date_range("2025-12-01", "2026-08-01", freq="1h", tz="UTC")
    forecasts = pd.DataFrame(
        {
            "day_ahead_price_eur_mwh": 50.0,
            "load_day_ahead_forecast_mw": 40_000.0,
            "ordinary_weather_feature": 1.0,
        },
        index=index,
    )
    observations = pd.DataFrame({CARBON: np.arange(len(index))}, index=index)
    builder = RegimeMoEFeatureBuilder(forecasts)

    winter, _ = builder.build(
        observations,
        pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC")]),
        supervised=False,
    )
    summer, _ = builder.build(
        observations,
        pd.DatetimeIndex([pd.Timestamp("2026-07-02", tz="UTC")]),
        supervised=False,
    )

    assert winter.loc[:21, "fc_day_ahead_price_eur_mwh"].notna().all()
    assert winter.loc[22:, "fc_day_ahead_price_eur_mwh"].isna().all()
    assert summer.loc[:20, "fc_day_ahead_price_eur_mwh"].notna().all()
    assert summer.loc[21:, "fc_day_ahead_price_eur_mwh"].isna().all()
    assert winter.loc[22:, "fc_load_day_ahead_forecast_mw"].isna().all()
    assert summer.loc[21:, "fc_load_day_ahead_forecast_mw"].isna().all()
    assert winter["fc_ordinary_weather_feature"].notna().all()
    assert summer["fc_ordinary_weather_feature"].notna().all()


def test_target_aligned_day_lag_uses_each_targets_previous_day() -> None:
    index = pd.date_range("2025-12-01", "2026-02-01", freq="1h", tz="UTC")
    observations = pd.DataFrame({CARBON: np.arange(len(index), dtype=float)}, index=index)
    forecasts = pd.DataFrame({"known_feature": 1.0}, index=index)
    origin = pd.Timestamp("2026-01-02", tz="UTC")

    features, _ = RegimeMoEFeatureBuilder(forecasts).build(
        observations,
        pd.DatetimeIndex([origin]),
        supervised=False,
    )

    expected = observations[CARBON].reindex(
        origin + pd.to_timedelta(np.arange(1, 25) - 24, unit="h")
    ).to_numpy(dtype=float, copy=True)
    expected[-1] = np.nan
    np.testing.assert_array_equal(
        features[f"tgtlag24_{CARBON}"].to_numpy(), expected
    )


def test_origin_state_uses_last_fully_closed_hour() -> None:
    index = pd.date_range("2025-12-01", "2026-02-01", freq="1h", tz="UTC")
    observations = pd.DataFrame({CARBON: np.arange(len(index), dtype=float)}, index=index)
    forecasts = pd.DataFrame({"known_feature": 1.0}, index=index)
    origin = pd.Timestamp("2026-01-02", tz="UTC")

    features, _ = RegimeMoEFeatureBuilder(forecasts).build(
        observations, pd.DatetimeIndex([origin]), supervised=False
    )

    expected = observations.loc[origin - pd.Timedelta(hours=1), CARBON]
    assert features[f"origin_{CARBON}"].eq(expected).all()


def test_multiscale_state_is_opt_in_and_causal() -> None:
    index = pd.date_range("2025-12-01", "2026-01-04", freq="1h", tz="UTC")
    origin = pd.Timestamp("2026-01-02", tz="UTC")
    observations = pd.DataFrame(
        {
            CARBON: np.arange(len(index), dtype=float),
            "gas_mw": 1_000.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )
    forecasts = pd.DataFrame({"known_feature": 1.0}, index=index)

    ordinary, _ = RegimeMoEFeatureBuilder(forecasts).build(
        observations, pd.DatetimeIndex([origin]), supervised=False
    )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        include_multiscale_state=True,
        multiscale_state_columns=(CARBON, "gas_mw"),
        multiscale_windows_hours=(3, 24),
    )
    features, _ = builder.build(
        observations, pd.DatetimeIndex([origin]), supervised=False
    )
    changed_future = observations.copy()
    changed_future.loc[changed_future.index > origin, [CARBON, "gas_mw"]] = 1e9
    changed, _ = builder.build(
        changed_future, pd.DatetimeIndex([origin]), supervised=False
    )

    assert not any(column.startswith("ms_") for column in ordinary)
    multiscale = [column for column in features if column.startswith("ms_")]
    assert len(multiscale) == 2 * 2 * 3
    pd.testing.assert_frame_equal(features[multiscale], changed[multiscale])
    assert features["ms_carbon_intensity_gco2_kwh_w3h_level"].nunique() == 1


def test_detailed_physical_state_is_opt_in() -> None:
    index = pd.date_range("2025-12-01", "2026-01-04", freq="1h", tz="UTC")
    origin = pd.Timestamp("2026-01-02", tz="UTC")
    observations = pd.DataFrame(
        {
            CARBON: np.arange(len(index), dtype=float),
            "bioenergy_mw": 1_200.0,
            "commercial_exchange_es_mw": -500.0,
        },
        index=index,
    )
    forecasts = pd.DataFrame({"known_feature": 1.0}, index=index)

    ordinary, _ = RegimeMoEFeatureBuilder(forecasts).build(
        observations, pd.DatetimeIndex([origin]), supervised=False
    )
    detailed, _ = RegimeMoEFeatureBuilder(
        forecasts, include_detailed_state=True
    ).build(observations, pd.DatetimeIndex([origin]), supervised=False)

    assert "origin_bioenergy_mw" not in ordinary
    assert "origin_bioenergy_mw" in detailed
    assert "tgtlag24_commercial_exchange_es_mw" in detailed


def test_rte_d1_forecasts_are_opt_in_aligned_and_form_residual_load() -> None:
    index = pd.date_range("2026-01-01", "2026-01-04", freq="1h", tz="UTC")
    origin = pd.Timestamp("2026-01-02", tz="UTC")
    forecasts = pd.DataFrame(
        {
            "load_day_ahead_forecast_mw": 40_000.0,
            "wind_onshore_day_ahead_forecast_mw": 4_000.0,
            "wind_offshore_day_ahead_forecast_mw": 1_000.0,
            "solar_day_ahead_forecast_mw": 2_000.0,
        },
        index=index,
    )
    observations = pd.DataFrame({CARBON: 20.0}, index=index)
    rte_rows = []
    for horizon in (1, 2):
        target = origin + pd.Timedelta(hours=horizon)
        for production_type, value in (
            ("WIND_ONSHORE", 4_500.0 + horizon),
            ("WIND_OFFSHORE", 1_500.0 + horizon),
            ("SOLAR", 3_000.0 + horizon),
        ):
            rte_rows.append(
                {
                    "forecast_type": "D-1",
                    "production_type": production_type,
                    "target_start": target,
                    "updated_date": origin - pd.Timedelta(hours=1),
                    "value_mw": value,
                }
            )
    store = RteGenerationForecastFeatureStore(pd.DataFrame(rte_rows))

    ordinary, _ = RegimeMoEFeatureBuilder(
        forecasts, horizons=(1, 2)
    ).build(observations, pd.DatetimeIndex([origin]), supervised=False)
    augmented, _ = RegimeMoEFeatureBuilder(
        forecasts, horizons=(1, 2), rte_forecast_store=store
    ).build(observations, pd.DatetimeIndex([origin]), supervised=False)

    assert not any(column.startswith("rte_tgt_") for column in ordinary)
    np.testing.assert_allclose(
        augmented["rte_tgt_wind_d1_mw"], [6_002.0, 6_004.0]
    )
    np.testing.assert_allclose(
        augmented["rte_tgt_residual_load_d1_mw"],
        40_000.0
        - augmented["rte_tgt_variable_renewables_d1_mw"],
    )
    np.testing.assert_allclose(
        augmented["rte_hybrid_tgt_residual_load_d1_mw"],
        40_000.0 - (5_000.0 + augmented["rte_tgt_solar_d1_mw"]),
    )


def test_rte_d1_builder_never_uses_update_after_origin() -> None:
    index = pd.date_range("2026-01-01", "2026-01-04", freq="1h", tz="UTC")
    origin = pd.Timestamp("2026-01-02", tz="UTC")
    target = origin + pd.Timedelta(hours=1)
    observations = pd.DataFrame({CARBON: 20.0}, index=index)
    forecasts = pd.DataFrame(
        {"load_day_ahead_forecast_mw": 40_000.0}, index=index
    )
    store = RteGenerationForecastFeatureStore(
        pd.DataFrame(
            [
                {
                    "forecast_type": "D-1",
                    "production_type": "SOLAR",
                    "target_start": target,
                    "updated_date": origin + pd.Timedelta(minutes=1),
                    "value_mw": 9_999.0,
                }
            ]
        ),
        production_types=("SOLAR",),
    )
    augmented, _ = RegimeMoEFeatureBuilder(
        forecasts, horizons=(1,), rte_forecast_store=store
    ).build(observations, pd.DatetimeIndex([origin]), supervised=False)

    assert np.isnan(augmented.loc[0, "rte_tgt_solar_d1_mw"])
