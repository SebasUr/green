"""France-specific dense day-ahead carbon specialist.

This experiment is intentionally isolated from the normal sparse-horizon
models.  It predicts every hour in ``t0 + 1 .. t0 + 24`` and adds only signals
that are available at the forecast origin:

* French emitting/dispatchable generation state and residual load;
* target-time day-ahead load, wind and solar forecasts;
* trailing, observed errors of those forecasts (causal bias correction);
* a temporally calibrated point scale and a robust dense-hour selector.

The selector never changes the point model used for MAPE reporting.  It exposes
a second ``decision_prediction`` which is calibrated on a pre-test block to
reduce green-window regret without hiding the point-forecast trade-off.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.features import FeatureBuilder
from green_observatory.carbon.model import _make_estimator
from green_observatory.providers.carbon_base import CARBON

DENSE_DAY_AHEAD_HORIZONS = tuple(range(1, 25))


def _sum_available(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series | None:
    present = [column for column in columns if column in frame]
    if not present:
        return None
    return frame[present].sum(axis=1, min_count=1)


def _first_available(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series | None:
    for column in columns:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce")
    return None


class FranceDayAheadFeatureBuilder(FeatureBuilder):
    """Leakage-safe French physical and forecast-regime features."""

    def origin_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = super().origin_features(frame)
        known = frame.shift(self.hourly_state_lag_hours)

        fossil = _sum_available(known, ("gas_mw", "coal_mw", "fuel_oil_mw"))
        variable = _sum_available(known, ("wind_mw", "solar_mw"))
        run_of_river = pd.to_numeric(
            known.get("hydro_run_of_river_mw"), errors="coerce"
        )
        consumption = pd.to_numeric(known.get("consumption_mw"), errors="coerce")
        if fossil is not None:
            x["fr_fossil_mw"] = fossil
        if variable is not None and consumption is not None:
            x["fr_residual_load_mw"] = consumption - variable
            if run_of_river is not None:
                x["fr_residual_load_after_ror_mw"] = consumption - variable - run_of_river

        dispatchable_gas = _sum_available(known, ("gas_ccg_mw", "gas_turbine_mw"))
        if dispatchable_gas is not None:
            x["fr_dispatchable_gas_mw"] = dispatchable_gas
        if "gas_cogeneration_mw" in known:
            x["fr_gas_cogeneration_mw"] = pd.to_numeric(
                known["gas_cogeneration_mw"], errors="coerce"
            )

        generation = _sum_available(
            known,
            (
                "nuclear_mw", "gas_mw", "coal_mw", "fuel_oil_mw", "wind_mw",
                "solar_mw", "hydro_mw", "bioenergy_mw",
            ),
        )
        if generation is not None:
            denominator = generation.where(generation > 0.0)
            if fossil is not None:
                x["fr_fossil_generation_share"] = fossil / denominator
            for source in ("gas_mw", "nuclear_mw", "hydro_mw", "wind_mw", "solar_mw"):
                if source in known:
                    x[f"fr_{source}_generation_share"] = (
                        pd.to_numeric(known[source], errors="coerce") / denominator
                    )

        self._add_forecast_error_features(x, known)
        return x

    def _add_forecast_error_features(
        self, x: pd.DataFrame, frame: pd.DataFrame
    ) -> None:
        forecast_frame = getattr(self, "forecast_frame", None)
        if forecast_frame is None:
            return
        forecasts = forecast_frame.reindex(frame.index).shift(
            self.hourly_state_lag_hours
        )
        pairs: dict[str, tuple[pd.Series | None, pd.Series | None]] = {
            "load": (
                _first_available(
                    forecasts,
                    ("load_day_ahead_forecast_mw", "consumption_forecast_mw"),
                ),
                pd.to_numeric(frame.get("consumption_mw"), errors="coerce"),
            ),
            "wind": (
                _sum_available(
                    forecasts,
                    (
                        "wind_onshore_day_ahead_forecast_mw",
                        "wind_offshore_day_ahead_forecast_mw",
                    ),
                ),
                pd.to_numeric(frame.get("wind_mw"), errors="coerce"),
            ),
            "solar": (
                _sum_available(forecasts, ("solar_day_ahead_forecast_mw",)),
                pd.to_numeric(frame.get("solar_mw"), errors="coerce"),
            ),
        }
        for name, (forecast, actual) in pairs.items():
            if forecast is None or actual is None:
                continue
            # Both operands refer to the last completely closed hour.
            error = forecast - actual
            x[f"fr_{name}_forecast_error_now_mw"] = error
            for window in (24, 168, 720):
                x[f"fr_{name}_forecast_bias_{window}h_mw"] = error.rolling(
                    window, min_periods=max(6, window // 4)
                ).mean()

    def target_block(
        self,
        target_times: pd.DatetimeIndex,
        horizon: int,
        *,
        index: pd.Index | None = None,
    ) -> pd.DataFrame:
        block = super().target_block(target_times, horizon, index=index)
        load = _first_available(
            block,
            ("fc_load_day_ahead_forecast_mw", "fc_consumption_forecast_mw"),
        )
        wind = _sum_available(
            block,
            (
                "fc_wind_onshore_day_ahead_forecast_mw",
                "fc_wind_offshore_day_ahead_forecast_mw",
            ),
        )
        solar = _sum_available(block, ("fc_solar_day_ahead_forecast_mw",))
        nuclear = _first_available(
            block,
            (
                "fc_nuclear_day_ahead_forecast_mw",
                "fc_nuclear_available_mw",
            ),
        )
        hydro = _first_available(
            block,
            (
                "fc_hydro_day_ahead_forecast_mw",
                "fc_hydro_available_mw",
            ),
        )
        thermal = _sum_available(
            block,
            (
                "fc_gas_day_ahead_forecast_mw",
                "fc_coal_day_ahead_forecast_mw",
                "fc_fuel_oil_day_ahead_forecast_mw",
            ),
        )
        if wind is not None:
            block["fr_tgt_wind_day_ahead_mw"] = wind
        if solar is not None:
            block["fr_tgt_solar_day_ahead_mw"] = solar
        if wind is not None and solar is not None:
            variable = wind + solar
            block["fr_tgt_variable_renewables_day_ahead_mw"] = variable
            if load is not None:
                load_safe = load.where(load > 0.0)
                block["fr_tgt_residual_load_day_ahead_mw"] = load - variable
                block["fr_tgt_variable_renewables_share"] = variable / load_safe
        if load is not None:
            block["fr_tgt_load_day_ahead_mw"] = load
        if nuclear is not None:
            block["fr_tgt_nuclear_available_mw"] = nuclear
        if hydro is not None:
            block["fr_tgt_hydro_available_mw"] = hydro
        for source in ("nuclear", "hydro", "thermal"):
            unavailable = _first_available(
                block, (f"fc_{source}_unavailable_mw",)
            )
            if unavailable is not None:
                block[f"fr_tgt_{source}_unavailable_mw"] = unavailable
        if thermal is not None:
            block["fr_tgt_thermal_day_ahead_mw"] = thermal
        if nuclear is not None and hydro is not None:
            firm_low_carbon = nuclear + hydro
            block["fr_tgt_firm_low_carbon_mw"] = firm_low_carbon
            if load is not None and wind is not None and solar is not None:
                block["fr_tgt_thermal_requirement_mw"] = (
                    load - wind - solar - firm_low_carbon
                )
        return block


class FranceDayAheadModel:
    """Dense 24-hour point model plus validation-calibrated robust selector."""

    def __init__(
        self,
        feature_builder: FranceDayAheadFeatureBuilder,
        *,
        horizons: Sequence[int] = DENSE_DAY_AHEAD_HORIZONS,
        algorithm: str = "hist_gradient_boosting",
        params: dict | None = None,
        calibration_fraction: float = 0.20,
        calibration_stride_hours: int = 6,
        recency_halflife_days: float | None = 180.0,
        scale_grid: Sequence[float] = (0.90, 0.95, 1.0, 1.05, 1.10),
        smoothing_windows: Sequence[int] = (1, 3, 5),
        smoothing_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        uncertainty_weights: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
        random_state: int = 42,
    ) -> None:
        if not 0.10 <= calibration_fraction <= 0.40:
            raise ValueError("calibration_fraction must be in [0.10, 0.40]")
        self.feature_builder = feature_builder
        self.horizons = tuple(int(h) for h in horizons)
        self.algorithm = algorithm
        self.params = params or {}
        self.calibration_fraction = float(calibration_fraction)
        self.calibration_stride_hours = int(calibration_stride_hours)
        self.recency_halflife_days = recency_halflife_days
        self.scale_grid = tuple(float(value) for value in scale_grid)
        self.smoothing_windows = tuple(int(value) for value in smoothing_windows)
        self.smoothing_weights = tuple(float(value) for value in smoothing_weights)
        self.uncertainty_weights = tuple(float(value) for value in uncertainty_weights)
        self.random_state = int(random_state)

        self.estimators_: dict[int, object] = {}
        self.feature_names_: dict[int, list[str]] = {}
        self.point_scales_: dict[int, float] = {}
        self.horizon_mae_: dict[int, float] = {}
        self.selector_: dict[str, float | int] = {
            "window": 1, "smoothing_weight": 0.0, "uncertainty_weight": 0.0
        }
        self.validation_mape_: float | None = None
        self.validation_regret_: float | None = None

    def _sample_weight(self, index: pd.DatetimeIndex) -> np.ndarray | None:
        if self.recency_halflife_days is None:
            return None
        age_days = (index.max() - index).total_seconds() / 86400.0
        return np.exp(-np.log(2.0) * age_days / self.recency_halflife_days)

    def _fit_estimators(self, frame: pd.DataFrame) -> tuple[dict, dict]:
        origin = self.feature_builder.origin_features(frame)
        estimators: dict[int, object] = {}
        names: dict[int, list[str]] = {}
        for horizon in self.horizons:
            x, y = self.feature_builder.build_supervised(
                frame, horizon, origin_features=origin
            )
            estimator = _make_estimator(
                self.algorithm, self.params, self.random_state + horizon
            )
            estimator.fit(x, y, sample_weight=self._sample_weight(x.index))
            estimators[horizon] = estimator
            names[horizon] = list(x.columns)
        return estimators, names

    def _raw_predictions(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        estimators: dict[int, object],
        feature_names: dict[int, list[str]],
    ) -> pd.DataFrame:
        origin_features = self.feature_builder.origin_features(frame).reindex(origins)
        parts: list[pd.DataFrame] = []
        for horizon in self.horizons:
            target_times = origins + pd.Timedelta(hours=horizon)
            target = self.feature_builder.target_block(
                target_times, horizon, index=origins
            )
            x = pd.concat([target, origin_features], axis=1).reindex(
                columns=feature_names[horizon]
            )
            parts.append(
                pd.DataFrame(
                    {
                        "origin": origins,
                        "horizon": horizon,
                        "target_time": target_times,
                        "raw_prediction": np.clip(
                            estimators[horizon].predict(x), 0.0, None
                        ),
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _mape(actual: np.ndarray, prediction: np.ndarray) -> float:
        return float(np.mean(np.abs(prediction - actual) / np.clip(np.abs(actual), 1e-9, None)))

    @staticmethod
    def _mean_regret(predictions: pd.DataFrame, column: str) -> float:
        regrets: list[float] = []
        for _, group in predictions.groupby("origin", sort=False):
            selected = group[column].idxmin()
            regrets.append(
                float(predictions.at[selected, "actual"] - group["actual"].min())
            )
        return float(np.mean(regrets))

    @staticmethod
    def _decision_transform(
        predictions: pd.DataFrame,
        *,
        window: int,
        smoothing_weight: float,
        uncertainty_weight: float,
        horizon_mae: dict[int, float],
    ) -> pd.Series:
        out = pd.Series(index=predictions.index, dtype=float)
        for _, group in predictions.groupby("origin", sort=False):
            ordered = group.sort_values("horizon")
            point = ordered["point_prediction"]
            if window > 1:
                smooth = point.rolling(window, center=True, min_periods=1).median()
            else:
                smooth = point
            uncertainty = ordered["horizon"].map(horizon_mae).fillna(0.0)
            score = (
                (1.0 - smoothing_weight) * point
                + smoothing_weight * smooth
                + uncertainty_weight * uncertainty
            )
            out.loc[ordered.index] = score.to_numpy()
        return out

    def fit(self, train_frame: pd.DataFrame) -> FranceDayAheadModel:
        n = len(train_frame)
        split = int(n * (1.0 - self.calibration_fraction))
        if split < 1000 or n - split < 200:
            raise ValueError("not enough rows for France-24 calibration")
        base = train_frame.iloc[:split]
        base_estimators, base_names = self._fit_estimators(base)
        last_origin = train_frame.index[-1] - pd.Timedelta(hours=max(self.horizons))
        origins = train_frame.index[split::self.calibration_stride_hours]
        origins = origins[origins <= last_origin]
        calibration = self._raw_predictions(
            train_frame, origins, base_estimators, base_names
        )
        calibration["actual"] = train_frame[CARBON].reindex(
            pd.DatetimeIndex(calibration["target_time"])
        ).to_numpy()
        calibration = calibration.dropna(subset=["actual", "raw_prediction"])

        for horizon, group in calibration.groupby("horizon"):
            actual = group["actual"].to_numpy()
            raw = group["raw_prediction"].to_numpy()
            self.point_scales_[int(horizon)] = min(
                self.scale_grid,
                key=lambda scale: (self._mape(actual, scale * raw), abs(scale - 1.0)),
            )
        calibration["point_prediction"] = calibration["raw_prediction"] * calibration[
            "horizon"
        ].map(self.point_scales_)
        self.validation_mape_ = 100.0 * self._mape(
            calibration["actual"].to_numpy(),
            calibration["point_prediction"].to_numpy(),
        )
        self.horizon_mae_ = (
            calibration.assign(
                absolute_error=(
                    calibration["point_prediction"] - calibration["actual"]
                ).abs()
            )
            .groupby("horizon")["absolute_error"]
            .mean()
            .to_dict()
        )

        candidates: list[tuple[float, float, int, float, float]] = []
        for window in self.smoothing_windows:
            for smoothing_weight in self.smoothing_weights:
                for uncertainty_weight in self.uncertainty_weights:
                    transformed = self._decision_transform(
                        calibration,
                        window=window,
                        smoothing_weight=smoothing_weight,
                        uncertainty_weight=uncertainty_weight,
                        horizon_mae=self.horizon_mae_,
                    )
                    scored = calibration.assign(decision_prediction=transformed)
                    regret = self._mean_regret(scored, "decision_prediction")
                    # Prefer the less invasive transform when regret ties.
                    complexity = smoothing_weight + uncertainty_weight
                    candidates.append(
                        (regret, complexity, window, smoothing_weight, uncertainty_weight)
                    )
        best = min(candidates)
        self.validation_regret_ = float(best[0])
        self.selector_ = {
            "window": int(best[2]),
            "smoothing_weight": float(best[3]),
            "uncertainty_weight": float(best[4]),
        }

        self.estimators_, self.feature_names_ = self._fit_estimators(train_frame)
        return self

    def predict_batch(
        self, frame: pd.DataFrame, origins: pd.DatetimeIndex
    ) -> pd.DataFrame:
        if not self.estimators_:
            raise RuntimeError("FranceDayAheadModel.predict_batch called before fit")
        predictions = self._raw_predictions(
            frame, origins, self.estimators_, self.feature_names_
        )
        predictions["point_prediction"] = predictions["raw_prediction"] * predictions[
            "horizon"
        ].map(self.point_scales_).fillna(1.0)
        predictions["decision_prediction"] = self._decision_transform(
            predictions,
            window=int(self.selector_["window"]),
            smoothing_weight=float(self.selector_["smoothing_weight"]),
            uncertainty_weight=float(self.selector_["uncertainty_weight"]),
            horizon_mae=self.horizon_mae_,
        )
        return predictions


def france24_feature_builder_from_config(
    carbon_cfg: dict,
    *,
    climatology=None,
    forecast_frame: pd.DataFrame | None = None,
) -> FranceDayAheadFeatureBuilder:
    cfg = carbon_cfg.get("france24_model", {})
    cal = carbon_cfg.get("calendar", {})
    recent = carbon_cfg.get("features", {}).get("recent_signal", {})
    return FranceDayAheadFeatureBuilder(
        climatology=climatology,
        local_tz=cal.get("local_timezone", "Europe/Paris"),
        holidays_country=cal.get("holidays_country", "FR"),
        lags_hours=cfg.get(
            "carbon_lags_hours", recent.get("lags_hours", (1, 2, 3, 24, 168))
        ),
        rolling_means_hours=cfg.get(
            "carbon_rolling_means_hours",
            recent.get("rolling_means_hours", (3, 6, 24)),
        ),
        rolling_slope_hours=recent.get("rolling_slope_hours", 6),
        use_system=cfg.get(
            "origin_features",
            carbon_cfg.get("features", {}).get("electricity_system", {}).get("use", ()),
        ),
        residual_from_climatology=recent.get("residual_from_climatology", True),
        forecast_frame=forecast_frame,
        forecast_maxlead_h=24,
    )


def train_france24_model(
    train_frame: pd.DataFrame,
    carbon_cfg: dict,
    *,
    climatology=None,
    forecast_frame: pd.DataFrame | None = None,
) -> FranceDayAheadModel:
    """Train the isolated dense French day-ahead experiment."""
    cfg = carbon_cfg.get("france24_model", {})
    model = FranceDayAheadModel(
        france24_feature_builder_from_config(
            carbon_cfg, climatology=climatology, forecast_frame=forecast_frame
        ),
        horizons=cfg.get("horizons_hours", DENSE_DAY_AHEAD_HORIZONS),
        algorithm=cfg.get("algorithm", "hist_gradient_boosting"),
        params=cfg.get("hist_gradient_boosting", {}),
        calibration_fraction=cfg.get("calibration_fraction", 0.20),
        calibration_stride_hours=cfg.get("calibration_stride_hours", 6),
        recency_halflife_days=cfg.get("recency_halflife_days", 180.0),
        scale_grid=cfg.get("scale_grid", (0.90, 0.95, 1.0, 1.05, 1.10)),
        smoothing_windows=cfg.get("smoothing_windows", (1, 3, 5)),
        smoothing_weights=cfg.get(
            "smoothing_weights", (0.0, 0.25, 0.5, 0.75, 1.0)
        ),
        uncertainty_weights=cfg.get(
            "uncertainty_weights", (0.0, 0.25, 0.5, 1.0)
        ),
        random_state=cfg.get("random_state", 42),
    )
    return model.fit(train_frame)


__all__ = [
    "DENSE_DAY_AHEAD_HORIZONS",
    "FranceDayAheadFeatureBuilder",
    "FranceDayAheadModel",
    "france24_feature_builder_from_config",
    "train_france24_model",
]
