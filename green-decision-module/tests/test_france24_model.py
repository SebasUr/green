import numpy as np
import pandas as pd

from green_observatory.carbon.france24 import (
    FranceDayAheadFeatureBuilder,
    FranceDayAheadModel,
)
from green_observatory.carbon.fossil_regime import (
    FossilRegimeModel,
    fossil_regime_labels,
)
from green_observatory.providers.carbon_base import CARBON


def _frame(periods=1600):
    index = pd.date_range("2025-01-01", periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(index.hour)
    load = 50_000 + 6_000 * np.cos(2 * np.pi * hour / 24)
    solar = np.clip(7_000 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    wind = 5_000 + 1_500 * np.sin(2 * np.pi * np.arange(periods) / 72)
    gas = np.clip((load - wind - solar - 38_000) / 2, 200, None)
    coal = np.full(periods, 20.0)
    oil = np.where(load > 55_000, 200.0, 50.0)
    hydro = np.full(periods, 8_000.0)
    nuclear = np.full(periods, 42_000.0)
    bio = np.full(periods, 1_000.0)
    denominator = nuclear + gas + coal + oil + wind + solar + hydro + bio
    carbon = 390 * gas / denominator + 1_030 * coal / denominator + 770 * oil / denominator
    frame = pd.DataFrame(
        {
            CARBON: carbon,
            "consumption_mw": load,
            "nuclear_mw": nuclear,
            "gas_mw": gas,
            "coal_mw": coal,
            "fuel_oil_mw": oil,
            "wind_mw": wind,
            "solar_mw": solar,
            "hydro_mw": hydro,
            "bioenergy_mw": bio,
            "pumped_storage_mw": 0.0,
            "physical_exchange_mw": 0.0,
            "gas_ccg_mw": gas * 0.7,
            "gas_turbine_mw": gas * 0.1,
            "gas_cogeneration_mw": gas * 0.2,
            "gas_other_mw": gas * 0.0,
            "hydro_run_of_river_mw": 3_000.0,
        },
        index=index,
    )
    forecast = pd.DataFrame(
        {
            "load_day_ahead_forecast_mw": load - 500,
            "wind_onshore_day_ahead_forecast_mw": wind * 0.9,
            "wind_offshore_day_ahead_forecast_mw": wind * 0.1,
            "solar_day_ahead_forecast_mw": solar * 0.95,
        },
        index=index,
    )
    return frame, forecast


def _builder(forecast):
    return FranceDayAheadFeatureBuilder(
        forecast_frame=forecast,
        lags_hours=(1, 2, 24),
        rolling_means_hours=(3, 24),
        use_system=(
            "consumption_mw", "nuclear_mw", "gas_mw", "coal_mw",
            "fuel_oil_mw", "wind_mw", "solar_mw", "hydro_mw",
            "bioenergy_mw", "gas_ccg_mw", "gas_turbine_mw",
            "gas_cogeneration_mw", "hydro_run_of_river_mw",
        ),
    )


def test_france_features_are_causal_and_derive_residual_load():
    frame, forecast = _frame(900)
    builder = _builder(forecast)
    full = builder.origin_features(frame)
    cutoff = frame.index[700]
    truncated = builder.origin_features(frame.loc[:cutoff])
    pd.testing.assert_series_equal(full.loc[cutoff], truncated.loc[cutoff])

    target = builder.target_block(
        pd.DatetimeIndex([cutoff + pd.Timedelta(hours=24)]), 24
    )
    assert "fr_tgt_residual_load_day_ahead_mw" in target
    expected = (
        target["fc_load_day_ahead_forecast_mw"]
        - target["fc_wind_onshore_day_ahead_forecast_mw"]
        - target["fc_wind_offshore_day_ahead_forecast_mw"]
        - target["fc_solar_day_ahead_forecast_mw"]
    )
    np.testing.assert_allclose(target["fr_tgt_residual_load_day_ahead_mw"], expected)


def test_france_features_accept_optional_firm_capacity_forecasts():
    frame, forecast = _frame(900)
    forecast["nuclear_available_mw"] = 41_000.0
    forecast["hydro_day_ahead_forecast_mw"] = 7_500.0
    forecast["nuclear_unavailable_mw"] = 20_000.0
    cutoff = frame.index[700]
    target = _builder(forecast).target_block(
        pd.DatetimeIndex([cutoff + pd.Timedelta(hours=24)]), 24
    )
    assert target["fr_tgt_nuclear_available_mw"].iloc[0] == 41_000.0
    assert target["fr_tgt_hydro_available_mw"].iloc[0] == 7_500.0
    assert target["fr_tgt_nuclear_unavailable_mw"].iloc[0] == 20_000.0
    expected = (
        target["fr_tgt_load_day_ahead_mw"]
        - target["fr_tgt_variable_renewables_day_ahead_mw"]
        - 41_000.0
        - 7_500.0
    )
    np.testing.assert_allclose(target["fr_tgt_thermal_requirement_mw"], expected)


def test_france24_model_keeps_point_and_decision_outputs_separate():
    frame, forecast = _frame()
    model = FranceDayAheadModel(
        _builder(forecast),
        horizons=(1, 2, 3, 4),
        params={
            "max_iter": 30,
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "early_stopping": False,
        },
        calibration_fraction=0.20,
        calibration_stride_hours=6,
        recency_halflife_days=180,
        smoothing_windows=(1, 3),
        smoothing_weights=(0.0, 0.5),
        uncertainty_weights=(0.0, 0.25),
    ).fit(frame.iloc[:1300])
    origins = frame.index[[1320, 1400]]
    predictions = model.predict_batch(frame, origins)
    assert len(predictions) == 8
    assert set(predictions["horizon"]) == {1, 2, 3, 4}
    assert predictions["point_prediction"].notna().all()
    assert predictions["decision_prediction"].notna().all()
    assert model.validation_mape_ is not None
    assert model.validation_regret_ is not None
    assert set(model.selector_) == {
        "window", "smoothing_weight", "uncertainty_weight"
    }


def test_fossil_regime_labels_dispatchable_gas():
    frame, _ = _frame(500)
    labels = fossil_regime_labels(
        frame, ccg_threshold_mw=500, peak_threshold_mw=2_500
    )
    assert set(labels.unique()) == {0, 1, 2}
    dispatch = frame[["gas_ccg_mw", "gas_turbine_mw", "gas_other_mw"]].sum(axis=1)
    assert (labels[dispatch < 500] == 0).all()
    assert (labels[dispatch >= 2_500] == 2).all()


def test_fossil_regime_expert_predicts_probabilities_and_carbon():
    frame, forecast = _frame(2400)
    model = FossilRegimeModel(
        _builder(forecast),
        horizons=(1, 2, 3, 4),
        ccg_threshold_mw=500,
        peak_threshold_mw=2_500,
        calibration_fraction=0.25,
        training_stride_hours=6,
        classifier_params={
            "max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 15,
            "early_stopping": False,
        },
        source_params={
            "max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 15,
            "early_stopping": False,
        },
        residual_params={
            "loss": "absolute_error", "max_iter": 20, "learning_rate": 0.1,
            "max_leaf_nodes": 15, "early_stopping": False,
        },
        ranker_params={
            "max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 15,
            "early_stopping": False,
        },
    ).fit(frame.iloc[:2200])
    out = model.predict_batch(frame, frame.index[[2220, 2300]])
    assert len(out) == 8
    assert out["point_prediction"].notna().all()
    assert out["decision_prediction"].notna().all()
    assert out["ranked_prediction"].notna().all()
    assert out["ranking_score"].between(0.0, 1.0).all()
    np.testing.assert_allclose(
        out[["prob_baseload", "prob_ccg", "prob_peak"]].sum(axis=1), 1.0
    )
    assert 0.0 <= model.validation_peak_recall_ <= 1.0
    assert model.ranking_weight_ in model.ranking_weight_grid
    assert model.validation_ranked_regret_ is not None
