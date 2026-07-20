"""Persistence baseline and historical climatology (baseline ladder rungs 1-2).

Climatology groups historical carbon intensity by local calendar buckets and
summarizes each bucket with a center and quantiles. Grouping uses
**Europe/Paris local time** (patterns follow the local clock incl. DST) while
every stored instant stays in UTC.

All forecasters share the :class:`Forecaster` interface so the evaluation
backtest can treat persistence, climatology, corrected climatology and the
project model uniformly. A forecaster receives an *as-of* ``history`` frame
(only fully closed rows with ``timestamp < origin``) and must not look into the future.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from green_observatory.models import ModelName
from green_observatory.providers.carbon_base import CARBON

DEFAULT_LOCAL_TZ = "Europe/Paris"
DEFAULT_GROUP_BY = ["month", "day_of_week", "hour_of_day"]
DEFAULT_FALLBACK_GROUP_BY = ["day_of_week", "hour_of_day"]
DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def local_calendar(index: pd.DatetimeIndex, tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """Return calendar fields (month/day_of_week/hour_of_day/is_weekend) in ``tz``.

    ``day_of_week`` is 0=Monday..6=Sunday. The returned frame keeps the original
    (UTC) index so results stay aligned with the source series.
    """
    if index.tz is None:
        raise ValueError("local_calendar requires a tz-aware (UTC) DatetimeIndex")
    loc = index.tz_convert(tz)
    dow = np.asarray(loc.dayofweek)
    return pd.DataFrame(
        {
            "month": np.asarray(loc.month),
            "day_of_week": dow,
            "hour_of_day": np.asarray(loc.hour),
            "is_weekend": (dow >= 5).astype(int),
        },
        index=index,
    )


@runtime_checkable
class Forecaster(Protocol):
    """Uniform forecasting interface used by the evaluation backtest."""

    name: ModelName

    def predict(
        self,
        history: pd.DataFrame,
        origin: pd.Timestamp,
        horizons_hours: Sequence[float],
    ) -> pd.DataFrame:
        """Forecast carbon intensity for ``origin + h`` for each ``h``.

        Returns a frame indexed by target time (UTC) with columns
        ``prediction`` (gCO2/kWh), ``lower``, ``upper`` and ``horizon_hours``.
        """
        ...


def _target_index(origin: pd.Timestamp, horizons_hours: Sequence[float]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [origin + pd.Timedelta(hours=float(h)) for h in horizons_hours], name="target_time"
    )


class ClimatologyModel:
    """Historical climatology with a coarse-bucket fallback for sparse cells.

    Fit on a training frame only; the caller is responsible for ensuring the
    training frame precedes any evaluation period (no look-ahead).
    """

    def __init__(
        self,
        group_by: Sequence[str] = DEFAULT_GROUP_BY,
        fallback_group_by: Sequence[str] = DEFAULT_FALLBACK_GROUP_BY,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        local_tz: str = DEFAULT_LOCAL_TZ,
        center: str = "median",
        min_samples: int = 8,
    ) -> None:
        self.group_by = list(group_by)
        self.fallback_group_by = list(fallback_group_by)
        self.quantiles = list(quantiles)
        self.local_tz = local_tz
        self.center = center
        self.min_samples = int(min_samples)
        self.full_: pd.DataFrame | None = None
        self.fallback_: pd.DataFrame | None = None
        self.global_: dict[str, float] | None = None

    # -- fitting -------------------------------------------------------- #
    def _quantile_cols(self) -> list[str]:
        return [f"p{int(round(q * 100))}" for q in self.quantiles]

    def _aggregate(self, frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        g = frame.groupby(keys, observed=True)["y"]
        stats = pd.DataFrame({"count": g.size(), "std": g.std()})
        stats["center"] = g.median() if self.center == "median" else g.mean()
        for q in self.quantiles:
            stats[f"p{int(round(q * 100))}"] = g.quantile(q)
        return stats

    def fit(self, df: pd.DataFrame) -> ClimatologyModel:
        y = pd.to_numeric(df[CARBON], errors="coerce")
        frame = local_calendar(df.index, self.local_tz)
        frame["y"] = y.to_numpy()
        frame = frame.dropna(subset=["y"])
        if frame.empty:
            raise ValueError("cannot fit climatology on an empty/all-NaN carbon series")

        self.full_ = self._aggregate(frame, self.group_by)
        self.fallback_ = self._aggregate(frame, self.fallback_group_by)
        yy = frame["y"]
        self.global_ = {
            "count": float(len(yy)),
            "std": float(yy.std()),
            "center": float(yy.median() if self.center == "median" else yy.mean()),
        }
        for q in self.quantiles:
            self.global_[f"p{int(round(q * 100))}"] = float(yy.quantile(q))
        return self

    # -- prediction ----------------------------------------------------- #
    def _lookup(self, cal: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if len(keys) == 1:
            key_index: pd.Index = pd.Index(cal[keys[0]].to_numpy(), name=keys[0])
        else:
            key_index = pd.MultiIndex.from_frame(cal[keys])
        out = table.reindex(key_index)
        out.index = cal.index
        return out

    def predict(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Climatology stats for each timestamp in ``index`` (UTC, tz-aware)."""
        if self.full_ is None or self.fallback_ is None or self.global_ is None:
            raise RuntimeError("ClimatologyModel.predict called before fit")
        cal = local_calendar(index, self.local_tz)
        full = self._lookup(cal, self.full_, self.group_by)
        fallback = self._lookup(cal, self.fallback_, self.fallback_group_by)

        use_full = full["count"].fillna(0) >= self.min_samples
        result = full.where(use_full, fallback)
        result = result.fillna(pd.Series(self.global_))

        source = np.where(
            use_full.to_numpy(),
            "full",
            np.where(fallback["count"].notna().to_numpy(), "fallback", "global"),
        )
        result["source"] = source
        result.index = index
        return result

    def predict_carbon(self, index: pd.DatetimeIndex) -> pd.Series:
        return self.predict(index)["center"].rename(CARBON)


class ClimatologyForecaster:
    """Wrap a fitted :class:`ClimatologyModel` as a :class:`Forecaster`."""

    name = ModelName.climatology

    def __init__(self, model: ClimatologyModel) -> None:
        self.model = model

    def predict(
        self, history: pd.DataFrame, origin: pd.Timestamp, horizons_hours: Sequence[float]
    ) -> pd.DataFrame:
        targets = _target_index(origin, horizons_hours)
        stats = self.model.predict(targets)
        lower = stats["p10"] if "p10" in stats else np.nan
        upper = stats["p90"] if "p90" in stats else np.nan
        return pd.DataFrame(
            {
                "prediction": np.asarray(stats["center"], dtype=float),
                "lower": np.asarray(lower, dtype=float),
                "upper": np.asarray(upper, dtype=float),
                "horizon_hours": list(horizons_hours),
            },
            index=targets,
        )


class PersistenceForecaster:
    """Rung 1: forecast = last observed value (carried flat over the horizon)."""

    name = ModelName.persistence

    def predict(
        self, history: pd.DataFrame, origin: pd.Timestamp, horizons_hours: Sequence[float]
    ) -> pd.DataFrame:
        past = history.loc[history.index < origin, CARBON].dropna()
        if past.empty:
            raise ValueError(f"persistence has no observed value at/before {origin}")
        last_val = float(past.iloc[-1])
        targets = _target_index(origin, horizons_hours)
        return pd.DataFrame(
            {
                "prediction": last_val,
                "lower": np.nan,
                "upper": np.nan,
                "horizon_hours": list(horizons_hours),
            },
            index=targets,
        )


def climatology_from_config(df: pd.DataFrame, carbon_cfg: dict) -> ClimatologyModel:
    """Build and fit a :class:`ClimatologyModel` from a carbon-model config dict."""
    clim = carbon_cfg.get("climatology", {})
    cal = carbon_cfg.get("calendar", {})
    model = ClimatologyModel(
        group_by=clim.get("group_by", DEFAULT_GROUP_BY),
        fallback_group_by=clim.get("fallback_group_by", DEFAULT_FALLBACK_GROUP_BY),
        quantiles=clim.get("quantiles", DEFAULT_QUANTILES),
        local_tz=cal.get("local_timezone", DEFAULT_LOCAL_TZ),
        center=clim.get("center", "median"),
        min_samples=clim.get("min_samples_per_bucket", 8),
    )
    return model.fit(df)
