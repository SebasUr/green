"""Phase-D regret-ranking model tests."""

import numpy as np
import pandas as pd

from green_observatory.carbon.ranking import RegretRankingModel, train_ranking_model
from green_observatory.providers.carbon_base import CARBON


def _frame(periods=1200):
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(index.hour)
    day = np.arange(periods) / 24.0
    carbon = 20 + 9 * np.sin(2 * np.pi * hour / 24) + 3 * np.sin(2 * np.pi * day / 7)
    frame = pd.DataFrame(
        {CARBON: carbon, "consumption_mw": 50000 + 1000 * np.cos(2 * np.pi * hour / 24)},
        index=index,
    )
    forecast = pd.DataFrame(
        {
            "wind_onshore_day_ahead_forecast_mw": 8000 - 250 * carbon,
            "solar_day_ahead_forecast_mw": np.maximum(0, 7000 * np.sin(np.pi * (hour - 6) / 13)),
            "load_day_ahead_forecast_mw": frame["consumption_mw"],
        },
        index=index,
    )
    return frame, forecast


def _config():
    return {
        "model": {
            "horizons_hours": [1, 3, 6, 12],
            "algorithm": "hist_gradient_boosting",
            "hist_gradient_boosting": {"max_iter": 30, "early_stopping": False},
        },
        "features": {
            "recent_signal": {"lags_hours": [1, 24], "rolling_means_hours": [3, 24]},
            "electricity_system": {"use": ["consumption_mw"]},
            "target_forecasts": {"consumption_maxlead_h": 24},
        },
        "ranking_model": {
            "calibration_fraction": 0.20,
            "calibration_stride_hours": 6,
            "hist_gradient_boosting": {"max_iter": 30, "early_stopping": False},
        },
    }


def test_ranking_model_uses_oos_pairs_and_preserves_point_value_distribution():
    frame, forecast = _frame()
    train = frame.iloc[:1000]
    model = train_ranking_model(train, _config(), forecast_frame=forecast)
    origins = frame.index[1020:1100:10]
    direct = model.predict_batch(frame, origins, apply_ranking=False)
    ranked = model.predict_batch(frame, origins, apply_ranking=True)
    assert model.calibration_origins_ > 20
    assert model.calibration_pairs_ > 100
    assert model.ranking_weight_ in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert ranked["ranking_score"].between(0, 1).all()
    for origin in origins:
        expected = np.sort(direct.loc[direct.origin == origin, "prediction"])
        got = np.sort(ranked.loc[ranked.origin == origin, "prediction"])
        np.testing.assert_allclose(got, expected)


def test_pair_weights_reflect_realized_regret_gap():
    model = RegretRankingModel(horizons=(1, 3, 6), regret_weight_cap=30)
    features = pd.DataFrame({"signal": [0.0, 1.0, 2.0]})
    predictions = pd.DataFrame(
        {
            "origin": [pd.Timestamp("2026-01-01T00:00Z")] * 3,
            "actual": [10.0, 11.0, 30.0],
        }
    )
    _, _, weights = model._pairwise_dataset(features, predictions)
    assert weights.max() > weights.min()
