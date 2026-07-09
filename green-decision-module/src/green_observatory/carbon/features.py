"""Leakage-safe feature construction for the project carbon model.

We use a **direct multi-horizon** setup: one model per horizon ``h``. For an
origin ``t0`` predicting target ``tt = t0 + h``, the feature vector combines:

* **target calendar** (``tgt_*``) - deterministic, known in advance, so using
  the target's hour/day/month/holiday is *not* leakage;
* **origin recent-signal** - lags, rolling means and a short slope of the carbon
  series, plus its residual from climatology, all as-of ``t0``;
* **origin system state** - the observed electricity mix at ``t0`` and a few
  derived shares.

The single invariant: **no feature may use information after ``t0``**. Recent
and system features come from ``shift(k>=0)`` / trailing ``rolling`` / the
current row only; the future value ``y(t0+h)`` is used solely as the label.
Because none of the origin transforms look ahead, origin features computed on
the full series equal those computed on ``history <= t0`` - which the evaluation
layer exploits for speed without any leakage.
"""

from __future__ import annotations

from collections.abc import Sequence

import holidays as holidays_lib
import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import DEFAULT_LOCAL_TZ, ClimatologyModel
from green_observatory.providers.carbon_base import CARBON, MIX_COLUMNS

DEFAULT_LAGS_HOURS = (1, 2, 3, 24, 168)
DEFAULT_ROLLING_MEANS_HOURS = (3, 6, 24)
DEFAULT_ROLLING_SLOPE_HOURS = 6


class FeatureBuilder:
    """Builds leakage-safe feature matrices for the project carbon model."""

    def __init__(
        self,
        climatology: ClimatologyModel | None = None,
        *,
        local_tz: str = DEFAULT_LOCAL_TZ,
        holidays_country: str = "FR",
        lags_hours: Sequence[int] = DEFAULT_LAGS_HOURS,
        rolling_means_hours: Sequence[int] = DEFAULT_ROLLING_MEANS_HOURS,
        rolling_slope_hours: int = DEFAULT_ROLLING_SLOPE_HOURS,
        use_system: Sequence[str] = tuple(MIX_COLUMNS),
        residual_from_climatology: bool = True,
    ) -> None:
        self.climatology = climatology
        self.local_tz = local_tz
        self.holidays_country = holidays_country
        self.lags_hours = tuple(lags_hours)
        self.rolling_means_hours = tuple(rolling_means_hours)
        self.rolling_slope_hours = int(rolling_slope_hours)
        self.use_system = tuple(use_system)
        self.residual_from_climatology = residual_from_climatology
        self._holidays_cache: dict[tuple[int, int], object] = {}

    # -- calendar (deterministic; safe for the target time) ------------- #
    def _holidays_for(self, years: range):
        key = (years.start, years.stop)
        if key not in self._holidays_cache:
            self._holidays_cache[key] = holidays_lib.country_holidays(
                self.holidays_country, years=list(years)
            )
        return self._holidays_cache[key]

    def calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        loc = index.tz_convert(self.local_tz)
        hour = np.asarray(loc.hour)
        dow = np.asarray(loc.dayofweek)
        month = np.asarray(loc.month)
        doy = np.asarray(loc.dayofyear)
        hol = self._holidays_for(range(int(loc.year.min()), int(loc.year.max()) + 2))
        is_holiday = np.fromiter((d in hol for d in loc.date), dtype=int, count=len(loc))
        return pd.DataFrame(
            {
                "hour_of_day": hour,
                "day_of_week": dow,
                "month": month,
                "is_weekend": (dow >= 5).astype(int),
                "is_holiday": is_holiday,
                "hour_sin": np.sin(2 * np.pi * hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * hour / 24.0),
                "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            },
            index=index,
        )

    # -- origin features (as-of t0; never look ahead) ------------------- #
    def origin_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        s = pd.to_numeric(frame[CARBON], errors="coerce")
        feats: dict[str, pd.Series] = {"carbon_now": s}
        for k in self.lags_hours:
            feats[f"carbon_lag_{k}h"] = s.shift(k)
        for w in self.rolling_means_hours:
            feats[f"carbon_rollmean_{w}h"] = s.rolling(w, min_periods=max(2, w // 2)).mean()
        sw = self.rolling_slope_hours
        feats[f"carbon_slope_{sw}h"] = s.diff().rolling(sw, min_periods=2).mean()
        if self.climatology is not None and self.residual_from_climatology:
            clim = self.climatology.predict_carbon(frame.index).to_numpy()
            feats["carbon_resid_clim"] = pd.Series(s.to_numpy() - clim, index=frame.index)

        x = pd.DataFrame(feats, index=frame.index)

        sys_cols = [c for c in self.use_system if c in frame.columns]
        x_sys = frame[sys_cols].apply(pd.to_numeric, errors="coerce")
        x = pd.concat([x, x_sys], axis=1)

        # derived shares (guarded against missing columns / zero consumption)
        cons = pd.to_numeric(frame.get("consumption_mw"), errors="coerce")
        if cons is not None and cons.notna().any():
            cons_safe = cons.where(cons > 0)
            ren_cols = [c for c in ("wind_mw", "solar_mw", "hydro_mw") if c in frame.columns]
            if ren_cols:
                x["renewable_share"] = frame[ren_cols].sum(axis=1) / cons_safe
            if "nuclear_mw" in frame.columns:
                x["nuclear_share"] = pd.to_numeric(frame["nuclear_mw"], errors="coerce") / cons_safe
        if "physical_exchange_mw" in frame.columns:
            x["net_export_mw"] = -pd.to_numeric(frame["physical_exchange_mw"], errors="coerce")
        return x

    # -- supervised assembly -------------------------------------------- #
    def build_supervised(
        self, frame: pd.DataFrame, horizon: int, *, origin_features: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``(X, y)`` for one horizon: origin features at ``t0`` +
        target calendar at ``t0+h``, label ``y(t0+h)``. Rows lacking a label are
        dropped (tree models tolerate remaining feature NaNs)."""
        xo = self.origin_features(frame) if origin_features is None else origin_features
        target_times = frame.index + pd.Timedelta(hours=horizon)
        cal = self.calendar_features(target_times)
        cal.index = frame.index
        x = pd.concat([cal.add_prefix("tgt_"), xo], axis=1)
        y = pd.to_numeric(frame[CARBON], errors="coerce").shift(-horizon)
        mask = y.notna()
        return x.loc[mask], y.loc[mask]

    def build_inference_row(
        self, origin_features: pd.DataFrame, origin: pd.Timestamp, horizon: int
    ) -> pd.DataFrame:
        """Assemble the single feature row to predict ``origin + horizon``."""
        target_time = origin + pd.Timedelta(hours=horizon)
        cal = self.calendar_features(pd.DatetimeIndex([target_time]))
        cal.index = pd.DatetimeIndex([origin])
        xo = origin_features.loc[[origin]]
        return pd.concat([cal.add_prefix("tgt_"), xo], axis=1)


def feature_builder_from_config(
    carbon_cfg: dict, climatology: ClimatologyModel | None = None
) -> FeatureBuilder:
    """Build a :class:`FeatureBuilder` from a carbon-model config dict."""
    feats = carbon_cfg.get("features", {})
    recent = feats.get("recent_signal", {})
    cal = carbon_cfg.get("calendar", {})
    system = feats.get("electricity_system", {}).get("use", list(MIX_COLUMNS))
    return FeatureBuilder(
        climatology=climatology,
        local_tz=cal.get("local_timezone", DEFAULT_LOCAL_TZ),
        holidays_country=cal.get("holidays_country", "FR"),
        lags_hours=recent.get("lags_hours", DEFAULT_LAGS_HOURS),
        rolling_means_hours=recent.get("rolling_means_hours", DEFAULT_ROLLING_MEANS_HOURS),
        rolling_slope_hours=recent.get("rolling_slope_hours", DEFAULT_ROLLING_SLOPE_HOURS),
        use_system=system,
        residual_from_climatology=recent.get("residual_from_climatology", True),
    )
