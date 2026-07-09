"""SARIMAX forecaster (optional) - the classical statistical comparison.

Design choices that make SARIMAX both principled and tractable here:

* **Seasonality via deterministic Fourier terms** (daily + weekly harmonics)
  passed as exogenous regressors, instead of a ``seasonal_order`` with period
  24/168 (computationally infeasible in statsmodels). This is the standard way
  to handle multiple/long seasonalities, and Fourier terms are deterministic so
  they are known for all future horizons - no leakage, no need to forecast the
  exogenous inputs.
* **Fit once, roll by conditioning** - the model is fit a single time on recent
  training data; at each origin it is re-applied (Kalman filtering, no refit) to
  the trailing window and then forecast. ``d=1`` lets it track level/regime
  shifts (important: 2026 runs far below the 2021-2025 mean).

Requires ``statsmodels`` (an optional dependency). Note the deliberate contrast
with the electricity-mix features of the project model: SARIMAX here is a purely
*temporal* model, a fair stand-in for "a conventional time-series model".
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import DEFAULT_LOCAL_TZ, _target_index
from green_observatory.models import ModelName
from green_observatory.providers.carbon_base import CARBON


def fourier_exog(
    index: pd.DatetimeIndex, tz: str = DEFAULT_LOCAL_TZ, daily_k: int = 4, weekly_k: int = 2
) -> pd.DataFrame:
    """Daily + weekly Fourier harmonics (deterministic; known for any future)."""
    loc = index.tz_convert(tz)
    hod = np.asarray(loc.hour) + np.asarray(loc.minute) / 60.0
    dow = np.asarray(loc.dayofweek)
    t_daily = 2 * np.pi * hod / 24.0
    t_weekly = 2 * np.pi * (dow * 24 + hod) / 168.0
    cols: dict[str, np.ndarray] = {}
    for k in range(1, daily_k + 1):
        cols[f"d_sin{k}"] = np.sin(k * t_daily)
        cols[f"d_cos{k}"] = np.cos(k * t_daily)
    for k in range(1, weekly_k + 1):
        cols[f"w_sin{k}"] = np.sin(k * t_weekly)
        cols[f"w_cos{k}"] = np.cos(k * t_weekly)
    return pd.DataFrame(cols, index=index)


def _regular_hourly(series: pd.Series, start=None, end=None) -> pd.Series:
    start = start if start is not None else series.index.min()
    end = end if end is not None else series.index.max()
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    return series.reindex(idx).interpolate("linear").ffill().bfill()


class SarimaxForecaster:
    """ARIMA(order) + Fourier-seasonal exogenous, fit once, rolled by ``apply``."""

    name = ModelName.sarimax

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 1),
        *,
        daily_k: int = 4,
        weekly_k: int = 2,
        local_tz: str = DEFAULT_LOCAL_TZ,
        train_window_days: int | None = 365,
        apply_window_hours: int = 720,
        maxiter: int = 50,
    ) -> None:
        self.order = order
        self.daily_k = daily_k
        self.weekly_k = weekly_k
        self.local_tz = local_tz
        self.train_window_days = train_window_days
        self.apply_window_hours = apply_window_hours
        self.maxiter = maxiter
        self.res_ = None

    def _exog(self, index: pd.DatetimeIndex) -> np.ndarray:
        return fourier_exog(index, self.local_tz, self.daily_k, self.weekly_k).to_numpy()

    def fit(self, train_frame: pd.DataFrame) -> SarimaxForecaster:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SarimaxForecaster needs statsmodels (pip install statsmodels)."
            ) from exc

        s = pd.to_numeric(train_frame[CARBON], errors="coerce").dropna()
        if self.train_window_days:
            s = s.loc[s.index.max() - pd.Timedelta(days=self.train_window_days):]
        s = _regular_hourly(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_ = SARIMAX(
                s.to_numpy(),
                exog=self._exog(s.index),
                order=self.order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(
        self, history: pd.DataFrame, origin: pd.Timestamp, horizons_hours: Sequence[float]
    ) -> pd.DataFrame:
        if self.res_ is None:
            raise RuntimeError("SarimaxForecaster.predict called before fit")
        max_h = int(max(horizons_hours))
        win = _regular_hourly(
            history[CARBON], start=origin - pd.Timedelta(hours=self.apply_window_hours), end=origin
        )
        fut_idx = pd.date_range(
            origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=max_h), freq="h", tz="UTC"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.res_.apply(win.to_numpy(), exog=self._exog(win.index))
            fc = np.clip(np.asarray(applied.forecast(max_h, exog=self._exog(fut_idx))), 0.0, None)

        targets = _target_index(origin, horizons_hours)
        preds = [float(fc[int(h) - 1]) for h in horizons_hours]  # position h-1 == origin+h
        return pd.DataFrame(
            {"prediction": preds, "lower": np.nan, "upper": np.nan,
             "horizon_hours": list(horizons_hours)},
            index=targets,
        )
