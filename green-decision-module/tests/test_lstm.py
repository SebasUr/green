"""LSTM forecaster smoke test (skipped if torch is not installed)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from green_observatory.carbon.features import FeatureBuilder  # noqa: E402
from green_observatory.carbon.lstm import LSTMForecaster  # noqa: E402
from green_observatory.providers.carbon_base import CARBON  # noqa: E402


def _frame(periods=400):
    idx = pd.date_range("2025-01-01", periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(idx.tz_convert("Europe/Paris").hour)
    carbon = 25 + 12 * np.sin(2 * np.pi * hour / 24.0) + np.random.default_rng(0).normal(0, 1, periods)
    return pd.DataFrame({CARBON: np.clip(carbon, 0, None)}, index=idx)


def test_lstm_fit_predict_shapes():
    frame = _frame()
    fb = FeatureBuilder(climatology=None)
    model = LSTMForecaster(
        fb, horizons=(1, 3), seq_len=12, hidden=8, epochs=3, batch_size=128
    ).fit(frame)
    origin = frame.index[300]
    out = model.predict(frame.loc[:origin], origin, [1, 3])
    assert list(out["horizon_hours"]) == [1, 3]
    assert (out["prediction"] >= 0).all()
    assert np.isfinite(out["prediction"]).all()
