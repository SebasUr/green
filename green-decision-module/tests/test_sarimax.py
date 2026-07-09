"""SARIMAX forecaster tests (skipped if statsmodels is not installed)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from green_observatory.carbon.sarimax import SarimaxForecaster, fourier_exog  # noqa: E402
from green_observatory.providers.carbon_base import CARBON  # noqa: E402


def _frame(periods=1000):
    idx = pd.date_range("2025-01-01", periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(idx.tz_convert("Europe/Paris").hour)
    carbon = 25 + 12 * np.sin(2 * np.pi * hour / 24.0) + np.random.default_rng(0).normal(0, 1, periods)
    return pd.DataFrame({CARBON: np.clip(carbon, 0, None)}, index=idx)


def test_fourier_shape_and_bounds():
    idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    x = fourier_exog(idx, daily_k=4, weekly_k=2)
    assert x.shape == (10, 12)  # (4 + 2) harmonics x (sin, cos)
    assert (x.abs() <= 1.0 + 1e-9).to_numpy().all()


def test_sarimax_fit_predict_tracks_diurnal():
    frame = _frame()
    sar = SarimaxForecaster(
        order=(1, 1, 1), train_window_days=None, apply_window_hours=240, maxiter=30
    ).fit(frame)
    origin = frame.index[900]
    out = sar.predict(frame.loc[:origin], origin, [1, 6, 24])
    assert list(out["horizon_hours"]) == [1, 6, 24]
    assert (out["prediction"] >= 0).all()
    target = origin + pd.Timedelta(hours=24)
    err = abs(out.loc[target, "prediction"] - frame[CARBON].loc[target])
    assert err < 8.0  # within a diurnal swing of the same-phase actual
