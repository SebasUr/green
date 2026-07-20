"""Tests for the modular EnsembleCI adaptation."""

import numpy as np
import pandas as pd

from green_observatory.carbon.ensemble_ci import (
    _greedy_ensemble_weights,
    train_ensemble_ci_model,
)
from green_observatory.providers.carbon_base import CARBON


def _frame(periods=1600):
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(index.hour)
    frame = pd.DataFrame(index=index)
    frame["wind_mw"] = 6000 + 2000 * np.sin(np.arange(periods) / 30)
    frame["solar_mw"] = np.maximum(0, 7000 * np.sin(np.pi * (hour - 6) / 13))
    frame["gas_mw"] = 3500 - 0.15 * frame["wind_mw"]
    frame["coal_mw"] = 100.0
    frame["fuel_oil_mw"] = 50.0
    frame["nuclear_mw"] = 42000.0
    frame["hydro_mw"] = 5000.0
    frame["bioenergy_mw"] = 1000.0
    frame["consumption_mw"] = frame.drop(columns=[]).sum(axis=1)
    frame[CARBON] = 8 + 0.004 * frame["gas_mw"] - 0.0005 * frame["wind_mw"]
    return frame


def _config():
    return {
        "model": {"horizons_hours": [1, 6]},
        "features": {
            "recent_signal": {"lags_hours": [1, 24], "rolling_means_hours": [3, 24]},
            "electricity_system": {"use": ["consumption_mw", "wind_mw", "solar_mw"]},
        },
        "ensemble_ci": {
            "history_columns": [CARBON, "wind_mw", "solar_mw"],
            "history_hours": 4,
            "sublearners": ["hist_gradient_boosting", "extra_trees"],
            "ensemble_iterations": 5,
            "sublearner_params": {
                "hist_gradient_boosting": {"max_iter": 15, "early_stopping": False},
                "extra_trees": {"n_estimators": 10, "n_jobs": 1},
            },
        },
    }


def test_ensemble_ci_trains_two_layers_and_predicts_each_horizon():
    frame = _frame()
    model = train_ensemble_ci_model(frame.iloc[:1400], _config())
    origins = frame.index[1450:1510:10]
    prediction = model.predict_batch(frame, origins)
    assert len(prediction) == len(origins) * 2
    assert prediction["prediction"].notna().all()
    assert set(model.base_models_[1]) == {"hist_gradient_boosting", "extra_trees"}
    assert abs(sum(model.weights_[1].values()) - 1.0) < 1e-12


def test_greedy_selection_prefers_the_accurate_sublearner():
    actual = pd.Series([1.0, 2.0, 3.0])
    predictions = pd.DataFrame({"good": [1.0, 2.1, 2.9], "bad": [9.0, 9.0, 9.0]})
    weights = _greedy_ensemble_weights(predictions, actual, iterations=10)
    assert weights["good"] == 1.0
