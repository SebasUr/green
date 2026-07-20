"""French daily-curve expert for causal 24-hour carbon forecasting.

The older dense models mainly combine state at the forecast origin with one
target-time forecast value.  This module treats the day as a curve instead.  In
particular, every target hour receives:

* carbon and generation observed at the *same target hour* one/two/seven days
  earlier (all are known for horizons 1..24);
* the complete day-ahead load/wind/solar curve, its level and ramps;
* publication-safe RTE generation unavailability, including absolute outage
  levels as well as changes from the origin;
* the usual leakage-safe state at the forecast origin.

One horizon-conditioned learner shares information across all 24 hours.  A
small temporally separated ensemble combines objectives that behave differently
on France's low-carbon distribution: direct MAPE, log-MAE, and change relative
to yesterday's aligned carbon curve.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.france24 import (
    DENSE_DAY_AHEAD_HORIZONS,
    FranceDayAheadFeatureBuilder,
)
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON


ALIGNED_LAG_COLUMNS = (
    CARBON,
    "consumption_mw",
    "nuclear_mw",
    "gas_mw",
    "coal_mw",
    "fuel_oil_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "bioenergy_mw",
    "gas_cogeneration_mw",
    "gas_ccg_mw",
    "gas_turbine_mw",
    "hydro_run_of_river_mw",
    "hydro_reservoir_mw",
)


def _curve_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_key: pd.Series,
    order_key: pd.Series,
) -> pd.DataFrame:
    """Add within-origin summaries and ramps for known 24-hour curves."""
    out = frame.copy()
    ordered = pd.DataFrame(
        {"group": group_key.to_numpy(), "order": order_key.to_numpy()},
        index=frame.index,
    )
    for column in columns:
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        grouped = values.groupby(ordered["group"], sort=False)
        mean = grouped.transform("mean")
        minimum = grouped.transform("min")
        maximum = grouped.transform("max")
        out[f"{column}_day_mean"] = mean
        out[f"{column}_day_min"] = minimum
        out[f"{column}_day_max"] = maximum
        out[f"{column}_day_range"] = maximum - minimum
        out[f"{column}_from_day_mean"] = values - mean
        # Input rows are assembled horizon-first, not day-first.  Sort once per
        # column so a ramp always means h minus h-1 within the same origin.
        temp = pd.DataFrame(
            {"value": values, "group": ordered["group"], "order": ordered["order"]}
        ).sort_values(["group", "order"])
        ramp = temp.groupby("group", sort=False)["value"].diff()
        out[f"{column}_ramp"] = ramp.reindex(out.index).fillna(0.0)
    return out


class FranceDailyCurveFeatureBuilder:
    """Build a shared, origin-safe matrix for daily 1..24-hour curves."""

    def __init__(
        self,
        base_builder: FranceDayAheadFeatureBuilder,
        *,
        horizons: Sequence[int] = DENSE_DAY_AHEAD_HORIZONS,
        aligned_lag_days: Sequence[int] = (1, 2, 7),
        availability_store: RteAvailabilityFeatureStore | None = None,
        rte_forecast_store: RteGenerationForecastFeatureStore | None = None,
    ) -> None:
        self.base_builder = base_builder
        self.horizons = tuple(int(value) for value in horizons)
        self.aligned_lag_days = tuple(int(value) for value in aligned_lag_days)
        self.availability_store = availability_store
        self.rte_forecast_store = rte_forecast_store

    def matrix(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        *,
        supervised: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        origins = pd.DatetimeIndex(origins)
        if origins.tz is None:
            origins = origins.tz_localize("UTC")
        else:
            origins = origins.tz_convert("UTC")
        origin_features = self.base_builder.origin_features(frame).reindex(origins)
        availability = (
            self.availability_store.features_by_horizon(origins, self.horizons)
            if self.availability_store is not None
            else {}
        )
        rte_forecasts = (
            self.rte_forecast_store.features_by_horizon(origins, self.horizons)
            if self.rte_forecast_store is not None
            else {}
        )

        x_parts: list[pd.DataFrame] = []
        meta_parts: list[pd.DataFrame] = []
        for horizon in self.horizons:
            target_times = origins + pd.Timedelta(hours=horizon)
            target = self.base_builder.target_block(
                target_times, horizon, index=origins
            )
            x = pd.concat([target, origin_features], axis=1)
            for days in self.aligned_lag_days:
                lag_times = target_times - pd.Timedelta(days=days)
                for column in ALIGNED_LAG_COLUMNS:
                    if column in frame:
                        values = pd.to_numeric(
                            frame[column], errors="coerce"
                        ).reindex(lag_times).to_numpy(dtype=float)
                        values = values.copy()
                        values[lag_times >= origins] = np.nan
                        x[f"aligned_{column}_d{days}"] = values
            if horizon in availability:
                x = pd.concat([x, availability[horizon]], axis=1)
            if horizon in rte_forecasts:
                x = pd.concat([x, rte_forecasts[horizon]], axis=1)

            x["curve_horizon"] = float(horizon)
            x["curve_horizon_sin"] = np.sin(2.0 * np.pi * horizon / 24.0)
            x["curve_horizon_cos"] = np.cos(2.0 * np.pi * horizon / 24.0)
            if (
                "fr_tgt_load_day_ahead_mw" in x
                and "fr_tgt_variable_renewables_day_ahead_mw" in x
            ):
                x["curve_residual_load_day_ahead_mw"] = (
                    x["fr_tgt_load_day_ahead_mw"]
                    - x["fr_tgt_variable_renewables_day_ahead_mw"]
                )
            if (
                "rte_tgt_nuclear_unavailable_delta_mw" in x
                and "nuclear_mw" in origin_features
            ):
                x["curve_nuclear_output_delta_proxy_mw"] = (
                    origin_features["nuclear_mw"]
                    - x["rte_tgt_nuclear_unavailable_delta_mw"]
                )
            x_parts.append(x.reset_index(drop=True))

            meta = pd.DataFrame(
                {
                    "origin": origins,
                    "horizon": horizon,
                    "target_time": target_times,
                }
            )
            if supervised:
                meta["actual"] = pd.to_numeric(
                    frame[CARBON], errors="coerce"
                ).reindex(target_times).to_numpy()
            meta_parts.append(meta)

        x_all = pd.concat(x_parts, ignore_index=True)
        meta_all = pd.concat(meta_parts, ignore_index=True)

        # Curve features operate on values that are genuinely known for the
        # whole day at the origin: D-1 forecasts and aligned past observations.
        curve_columns = [
            column
            for column in (
                "fr_tgt_load_day_ahead_mw",
                "fr_tgt_wind_day_ahead_mw",
                "fr_tgt_solar_day_ahead_mw",
                "fr_tgt_variable_renewables_day_ahead_mw",
                "curve_residual_load_day_ahead_mw",
                f"aligned_{CARBON}_d1",
                "aligned_gas_mw_d1",
                "aligned_solar_mw_d1",
                "aligned_nuclear_mw_d1",
                "rte_tgt_nuclear_unavailable_mw",
            )
            if column in x_all
        ]
        x_all = _curve_features(
            x_all,
            curve_columns,
            group_key=meta_all["origin"],
            order_key=meta_all["horizon"],
        )
        x_all = x_all.replace([np.inf, -np.inf], np.nan)
        if supervised:
            valid = meta_all["actual"].notna()
            x_all = x_all.loc[valid].reset_index(drop=True)
            meta_all = meta_all.loc[valid].reset_index(drop=True)
        return x_all, meta_all


def daily_training_origins(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizons: Sequence[int] = DENSE_DAY_AHEAD_HORIZONS,
    warmup_days: int = 8,
) -> pd.DatetimeIndex:
    """UTC daily origins whose complete labels are known at ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    start = frame.index.min().ceil("D") + pd.Timedelta(days=warmup_days)
    last = cutoff - pd.Timedelta(hours=max(map(int, horizons)))
    if last < start:
        return pd.DatetimeIndex([], tz="UTC")
    origins = pd.date_range(start, last.floor("D"), freq="1D")
    return origins[frame[CARBON].reindex(origins).notna().to_numpy()]


def _mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(np.asarray(prediction) - np.asarray(actual))
            / np.clip(np.abs(np.asarray(actual)), 1e-9, None)
        )
    )


def _greedy_mape_weights(
    predictions: pd.DataFrame,
    actual: np.ndarray,
    *,
    iterations: int,
) -> dict[str, float]:
    values = predictions.to_numpy(dtype=float)
    actual = np.asarray(actual, dtype=float)
    counts = np.zeros(values.shape[1], dtype=int)
    running = np.zeros(len(actual), dtype=float)
    for step in range(iterations):
        losses = [
            _mape(actual, (running + values[:, candidate]) / (step + 1))
            for candidate in range(values.shape[1])
        ]
        selected = int(np.argmin(losses))
        counts[selected] += 1
        running += values[:, selected]
    return {
        column: float(count / iterations)
        for column, count in zip(predictions.columns, counts)
    }


class FranceDispatchEnsemble:
    """Compact EnsembleCI-style expert adapted to French daily dispatch."""

    def __init__(
        self,
        feature_builder: FranceDailyCurveFeatureBuilder,
        *,
        validation_days: int = 60,
        ensemble_iterations: int = 30,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.feature_builder = feature_builder
        self.validation_days = int(validation_days)
        self.ensemble_iterations = int(ensemble_iterations)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.feature_names_: list[str] = []
        self.models_: dict[str, object] = {}
        self.level_feature_names_: list[str] = []
        self.level_models_: dict[str, object] = {}
        self.shape_model_: object | None = None
        self.weights_: dict[str, float] = {}
        self.validation_mape_: dict[str, float] = {}

    def _new_model(self, name: str):
        from lightgbm import LGBMRegressor

        common = {
            "n_estimators": 450,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 45,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "verbosity": -1,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }
        objective = {
            "direct_mape": "mape",
            "log_l1": "regression_l1",
            "relative_d1_l1": "regression_l1",
        }[name]
        return LGBMRegressor(objective=objective, **common)

    def _new_level_model(self, name: str):
        from lightgbm import LGBMRegressor

        objective = "mape" if name == "direct_mape" else "regression_l1"
        return LGBMRegressor(
            objective=objective,
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=40,
            colsample_bytree=0.8,
            reg_lambda=4.0,
            verbosity=-1,
            n_jobs=self.n_jobs,
            random_state=self.random_state + 100,
        )

    def _new_shape_model(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=380,
            learning_rate=0.03,
            num_leaves=23,
            min_child_samples=50,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            verbosity=-1,
            n_jobs=self.n_jobs,
            random_state=self.random_state + 200,
        )

    @staticmethod
    def _level_matrix(x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        """One compact feature row per daily origin."""
        grouping = pd.Series(
            pd.to_datetime(meta["origin"], utc=True).to_numpy(), index=x.index
        )
        level = x.groupby(grouping, sort=False).mean(numeric_only=True)
        key_columns = [
            column
            for column in x
            if any(
                token in column
                for token in (
                    "day_ahead",
                    "residual_load",
                    "unavailable",
                    f"aligned_{CARBON}_d1",
                    "aligned_gas_mw_d1",
                )
            )
        ]
        if key_columns:
            grouped = x[key_columns].groupby(grouping, sort=False)
            spread = grouped.agg(["min", "max", "std"])
            spread.columns = [f"{column}_{stat}" for column, stat in spread.columns]
            level = level.join(spread)
        level.index = pd.DatetimeIndex(level.index, tz="UTC", name="origin")
        return level.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _daily_levels(meta: pd.DataFrame) -> pd.Series:
        return (
            meta.assign(origin=pd.to_datetime(meta["origin"], utc=True))
            .groupby("origin", sort=False)["actual"]
            .median()
        )

    @staticmethod
    def _level_target(
        name: str, level_x: pd.DataFrame, level: np.ndarray
    ) -> np.ndarray:
        if name == "direct_mape":
            return level
        if name == "log_l1":
            return np.log(np.clip(level, 0.0, None) + 1.0)
        if name == "relative_d1_l1":
            column = f"aligned_{CARBON}_d1"
            previous = level_x[column].to_numpy(dtype=float)
            return np.log(np.clip(level, 1e-3, None) / np.clip(previous, 1e-3, None))
        raise ValueError(f"unknown level learner {name!r}")

    @staticmethod
    def _level_prediction(name: str, model, level_x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict(level_x), dtype=float)
        if name == "direct_mape":
            prediction = raw
        elif name == "log_l1":
            prediction = np.expm1(raw)
        elif name == "relative_d1_l1":
            previous = level_x[f"aligned_{CARBON}_d1"].to_numpy(dtype=float)
            prediction = previous * np.exp(raw)
        else:
            raise ValueError(f"unknown level learner {name!r}")
        return np.clip(prediction, 1e-3, None)

    @staticmethod
    def _normalized_shape_prediction(
        shape_model, x: pd.DataFrame, meta: pd.DataFrame
    ) -> np.ndarray:
        shape = np.exp(np.asarray(shape_model.predict(x), dtype=float))
        shape = np.clip(shape, 0.25, 4.0)
        frame = pd.DataFrame(
            {
                "origin": pd.to_datetime(meta["origin"], utc=True).to_numpy(),
                "shape": shape,
            },
            index=x.index,
        )
        median = frame.groupby("origin", sort=False)["shape"].transform("median")
        return shape / np.clip(median.to_numpy(dtype=float), 1e-6, None)

    def _level_shape_components(
        self,
        x: pd.DataFrame,
        meta: pd.DataFrame,
        level_models: dict[str, object],
        shape_model,
        *,
        level_feature_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        level_x = self._level_matrix(x, meta).reindex(columns=level_feature_names)
        normalized_shape = self._normalized_shape_prediction(shape_model, x, meta)
        row_origins = pd.DatetimeIndex(pd.to_datetime(meta["origin"], utc=True))
        components: dict[str, np.ndarray] = {}
        for name, model in level_models.items():
            levels = pd.Series(
                self._level_prediction(name, model, level_x), index=level_x.index
            )
            row_level = levels.reindex(row_origins).to_numpy(dtype=float)
            components[f"level_shape_{name}"] = np.clip(
                row_level * normalized_shape, 0.0, None
            )
        return components

    @staticmethod
    def _training_target(
        name: str, x: pd.DataFrame, actual: np.ndarray
    ) -> np.ndarray:
        if name == "direct_mape":
            return actual
        if name == "log_l1":
            return np.log(np.clip(actual, 0.0, None) + 1.0)
        if name == "relative_d1_l1":
            previous = x[f"aligned_{CARBON}_d1"].to_numpy(dtype=float)
            return np.log(
                np.clip(actual, 1e-3, None) / np.clip(previous, 1e-3, None)
            )
        raise ValueError(f"unknown dispatch learner {name!r}")

    @staticmethod
    def _prediction(name: str, model, x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict(x), dtype=float)
        if name == "direct_mape":
            prediction = raw
        elif name == "log_l1":
            prediction = np.expm1(raw)
        elif name == "relative_d1_l1":
            previous = x[f"aligned_{CARBON}_d1"].to_numpy(dtype=float)
            prediction = previous * np.exp(raw)
        else:
            raise ValueError(f"unknown dispatch learner {name!r}")
        return np.clip(prediction, 0.0, None)

    def fit(self, x: pd.DataFrame, meta: pd.DataFrame) -> FranceDispatchEnsemble:
        required = {"origin", "actual"}
        if missing := required - set(meta):
            raise ValueError(f"missing training metadata: {sorted(missing)}")
        self.feature_names_ = list(x.columns)
        origins = pd.DatetimeIndex(meta["origin"])
        split_time = origins.max() - pd.Timedelta(days=self.validation_days - 1)
        base_mask = np.asarray(origins < split_time, dtype=bool)
        validation_mask = ~base_mask
        if base_mask.sum() < 5000 or validation_mask.sum() < 24 * 14:
            raise ValueError("not enough temporally separated dispatch rows")
        actual = meta["actual"].to_numpy(dtype=float)
        names = ("direct_mape", "log_l1", "relative_d1_l1")
        validation_predictions: dict[str, np.ndarray] = {}
        for name in names:
            tuning = self._new_model(name)
            tuning.fit(
                x.loc[base_mask],
                self._training_target(name, x.loc[base_mask], actual[base_mask]),
            )
            prediction = self._prediction(name, tuning, x.loc[validation_mask])
            validation_predictions[name] = prediction
            self.validation_mape_[name] = 100.0 * _mape(
                actual[validation_mask], prediction
            )

        # Explicit daily level / intraday shape branch.  This prevents the
        # many easy low-variance hours from overwhelming a rare daily regime
        # shift in one shared point loss.
        level_x = self._level_matrix(x, meta)
        self.level_feature_names_ = list(level_x.columns)
        levels = self._daily_levels(meta)
        base_origins = pd.DatetimeIndex(origins[base_mask]).unique()
        validation_origins = pd.DatetimeIndex(origins[validation_mask]).unique()
        base_level_x = level_x.reindex(base_origins)
        validation_level_x = level_x.reindex(validation_origins)
        tuning_level_models: dict[str, object] = {}
        for name in names:
            level_model = self._new_level_model(name)
            level_model.fit(
                base_level_x,
                self._level_target(
                    name,
                    base_level_x,
                    levels.reindex(base_origins).to_numpy(dtype=float),
                ),
            )
            tuning_level_models[name] = level_model
        tuning_shape = self._new_shape_model()
        base_level_by_row = levels.reindex(origins[base_mask]).to_numpy(dtype=float)
        shape_target = np.log(
            np.clip(actual[base_mask], 1e-3, None)
            / np.clip(base_level_by_row, 1e-3, None)
        )
        tuning_shape.fit(x.loc[base_mask], shape_target)
        level_shape = self._level_shape_components(
            x.loc[validation_mask],
            meta.loc[validation_mask].reset_index(drop=True),
            tuning_level_models,
            tuning_shape,
            level_feature_names=self.level_feature_names_,
        )
        validation_predictions.update(level_shape)
        for name, prediction in level_shape.items():
            self.validation_mape_[name] = 100.0 * _mape(
                actual[validation_mask], prediction
            )
        validation_frame = pd.DataFrame(validation_predictions)
        self.weights_ = _greedy_mape_weights(
            validation_frame,
            actual[validation_mask],
            iterations=self.ensemble_iterations,
        )
        ensemble = sum(
            validation_frame[name].to_numpy() * weight
            for name, weight in self.weights_.items()
        )
        self.validation_mape_["ensemble"] = 100.0 * _mape(
            actual[validation_mask], ensemble
        )

        self.models_ = {}
        for name in names:
            model = self._new_model(name)
            model.fit(x, self._training_target(name, x, actual))
            self.models_[name] = model
        self.level_models_ = {}
        for name in names:
            level_model = self._new_level_model(name)
            level_model.fit(
                level_x,
                self._level_target(
                    name, level_x, levels.reindex(level_x.index).to_numpy(dtype=float)
                ),
            )
            self.level_models_[name] = level_model
        self.shape_model_ = self._new_shape_model()
        level_by_row = levels.reindex(origins).to_numpy(dtype=float)
        self.shape_model_.fit(
            x,
            np.log(
                np.clip(actual, 1e-3, None)
                / np.clip(level_by_row, 1e-3, None)
            ),
        )
        return self

    def predict(self, x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("FranceDispatchEnsemble.predict called before fit")
        x = x.reindex(columns=self.feature_names_)
        components = {
            name: self._prediction(name, model, x)
            for name, model in self.models_.items()
        }
        if self.shape_model_ is not None and self.level_models_:
            components.update(
                self._level_shape_components(
                    x,
                    meta.reset_index(drop=True),
                    self.level_models_,
                    self.shape_model_,
                    level_feature_names=self.level_feature_names_,
                )
            )
        prediction = sum(
            components[name] * self.weights_.get(name, 0.0)
            for name in components
        )
        out = meta.copy()
        out["prediction"] = np.clip(prediction, 0.0, None)
        out["model"] = "france_dispatch_ensemble"
        for name, values in components.items():
            out[f"prediction_{name}"] = values
        return out


__all__ = [
    "FranceDailyCurveFeatureBuilder",
    "FranceDispatchEnsemble",
    "daily_training_origins",
]
