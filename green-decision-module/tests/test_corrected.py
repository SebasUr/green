"""Corrected-climatology tests: recent residual pulls short horizons, decays out."""

import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import ClimatologyModel
from green_observatory.carbon.corrected_climatology import CorrectedClimatologyForecaster
from green_observatory.providers.carbon_base import CARBON


def _constant_frame(value, start="2025-01-01", periods=1440):
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame({CARBON: np.full(periods, float(value))}, index=idx)


def test_correction_pulls_toward_recent_actual_and_decays():
    # Climatology says 30 everywhere.
    train = _constant_frame(30.0)
    clim = ClimatologyModel().fit(train)
    fc = CorrectedClimatologyForecaster(
        clim, residual_halflife_hours=12, correction_decay_halflife_hours=24
    )

    # Recent history (up to origin) sits at 20 -> residual = -10.
    hist_idx = pd.date_range("2025-06-01", periods=48, freq="1h", tz="UTC")
    hist = pd.DataFrame({CARBON: np.full(48, 20.0)}, index=hist_idx)
    origin = hist_idx[-1]

    out = fc.predict(hist, origin, [1, 240])
    near = out.iloc[0]["prediction"]  # h = 1
    far = out.iloc[1]["prediction"]   # h = 240 (correction decayed away)

    assert near < 25.0            # pulled down toward the recent 20
    assert far > 29.0             # relaxed back toward climatology 30
    assert near < far


def test_correction_is_zero_without_history():
    train = _constant_frame(30.0)
    clim = ClimatologyModel().fit(train)
    fc = CorrectedClimatologyForecaster(clim)
    empty = pd.DataFrame({CARBON: []}, index=pd.DatetimeIndex([], tz="UTC"))
    out = fc.predict(empty, pd.Timestamp("2025-06-01T00:00:00Z"), [1, 24])
    assert np.allclose(out["prediction"].to_numpy(), 30.0)
