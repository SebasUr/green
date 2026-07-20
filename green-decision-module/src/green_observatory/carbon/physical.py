"""Physics-guided carbon forecast (Phase C).

The RTE production-based carbon signal is almost entirely determined by the
future generation mix.  This module therefore separates the task into three
auditable stages:

1. forecast the future shares of the emitting generation sources;
2. reconstruct carbon intensity with a non-negative linear physical map;
3. correct the remaining error with a small residual model.

The residual learner is deliberately trained on a trailing temporal block that
was *not* used to fit the source forecasters producing its inputs.  This avoids
the optimistic stacking error caused by training a residual model on in-sample
source predictions.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.features import FeatureBuilder, feature_builder_from_config
from green_observatory.providers.carbon_base import CARBON

DEFAULT_GENERATION_COLUMNS = (
    "nuclear_mw",
    "gas_mw",
    "coal_mw",
    "fuel_oil_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "bioenergy_mw",
)
DEFAULT_SHARE_COLUMNS = ("gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw")


def generation_shares(
    frame: pd.DataFrame,
    *,
    generation_columns: Sequence[str] = DEFAULT_GENERATION_COLUMNS,
    share_columns: Sequence[str] = DEFAULT_SHARE_COLUMNS,
) -> pd.DataFrame:
    """Return non-negative source shares of domestic generation.

    A row is left missing unless every aggregate generation component is
    present.  This is preferable to silently treating an unavailable source as
    zero and corrupting the denominator.
    """

    generation_columns = tuple(generation_columns)
    share_columns = tuple(share_columns)
    missing = [c for c in generation_columns if c not in frame]
    if missing:
        raise ValueError(f"generation frame is missing columns: {missing}")
    if any(c not in generation_columns for c in share_columns):
        raise ValueError("every share column must also be a generation column")

    generation = frame.loc[:, generation_columns].apply(pd.to_numeric, errors="coerce")
    generation = generation.clip(lower=0.0)
    valid = generation.notna().all(axis=1)
    total = generation.sum(axis=1).where(valid)
    total = total.where(total > 0.0)
    shares = generation.loc[:, share_columns].div(total, axis=0)
    return shares.add_suffix("_share")


class PhysicalCarbonMapper:
    """Non-negative empirical emission-factor map from source shares to CI."""

    def __init__(self, share_names: Sequence[str]) -> None:
        self.share_names = tuple(share_names)
        self.intercept_: float | None = None
        self.coefficients_: dict[str, float] = {}

    def fit(self, shares: pd.DataFrame, carbon: pd.Series) -> PhysicalCarbonMapper:
        from sklearn.linear_model import LinearRegression

        x = shares.reindex(columns=self.share_names)
        y = pd.to_numeric(carbon, errors="coerce")
        mask = x.notna().all(axis=1) & y.notna()
        if int(mask.sum()) < max(20, len(self.share_names) * 3):
            raise ValueError("not enough complete rows to fit the physical carbon map")
        reg = LinearRegression(positive=True).fit(x.loc[mask], y.loc[mask])
        self.intercept_ = float(reg.intercept_)
        self.coefficients_ = {
            name: float(value) for name, value in zip(self.share_names, reg.coef_)
        }
        return self

    def predict(self, shares: pd.DataFrame) -> np.ndarray:
        if self.intercept_ is None:
            raise RuntimeError("PhysicalCarbonMapper.predict called before fit")
        x = shares.reindex(columns=self.share_names).clip(lower=0.0)
        # Independent share regressors can very occasionally sum above one.
        # Re-normalize only those invalid rows while preserving valid forecasts.
        row_sum = x.sum(axis=1)
        scale = row_sum.where(row_sum > 1.0, 1.0)
        x = x.div(scale, axis=0)
        coef = pd.Series(self.coefficients_).reindex(self.share_names)
        return np.clip(self.intercept_ + x.to_numpy() @ coef.to_numpy(), 0.0, None)


def _make_hgb(params: dict, random_state: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    clean = {k: v for k, v in (params or {}).items() if v is not None}
    return HistGradientBoostingRegressor(random_state=random_state, **clean)


class PhysicalCarbonModel:
    """Per-horizon source-share forecasts plus physical and residual stages."""

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        *,
        horizons: Sequence[int],
        generation_columns: Sequence[str] = DEFAULT_GENERATION_COLUMNS,
        share_columns: Sequence[str] = DEFAULT_SHARE_COLUMNS,
        source_params: dict | None = None,
        residual_params: dict | None = None,
        residual_holdout_fraction: float = 0.20,
        share_lags_hours: Sequence[int] = (1, 2, 3, 24, 48, 72, 168),
        share_rolling_means_hours: Sequence[int] = (6, 24),
        random_state: int = 42,
    ) -> None:
        if not 0.05 <= residual_holdout_fraction <= 0.5:
            raise ValueError("residual_holdout_fraction must be in [0.05, 0.5]")
        self.feature_builder = feature_builder
        self.horizons = tuple(int(h) for h in horizons)
        self.generation_columns = tuple(generation_columns)
        self.source_columns = tuple(share_columns)
        self.share_names = tuple(f"{c}_share" for c in self.source_columns)
        self.source_params = source_params or {}
        self.residual_params = residual_params or {}
        self.residual_holdout_fraction = float(residual_holdout_fraction)
        self.share_lags_hours = tuple(int(lag) for lag in share_lags_hours)
        self.share_rolling_means_hours = tuple(
            int(window) for window in share_rolling_means_hours
        )
        self.random_state = int(random_state)

        self.mapper = PhysicalCarbonMapper(self.share_names)
        self.source_estimators_: dict[int, dict[str, object]] = {}
        self.source_feature_names_: dict[int, list[str]] = {}
        self.residual_estimators_: dict[int, object] = {}
        self.residual_feature_names_: dict[int, list[str]] = {}
        self.residual_calibration_rows_: dict[int, int] = {}

    def _supervised(
        self,
        frame: pd.DataFrame,
        horizon: int,
        *,
        origin_features: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        x, carbon = self.feature_builder.build_supervised(
            frame, horizon, origin_features=origin_features
        )
        history_shares = generation_shares(
            frame,
            generation_columns=self.generation_columns,
            share_columns=self.source_columns,
        )
        x = self._add_share_history(x, history_shares, horizon)
        targets = history_shares.shift(-horizon).reindex(x.index)
        mask = targets.notna().all(axis=1) & carbon.notna()
        return x.loc[mask], targets.loc[mask], carbon.loc[mask]

    def _inference_matrix(
        self,
        frame: pd.DataFrame,
        origin_features: pd.DataFrame,
        origins: pd.DatetimeIndex,
        horizon: int,
    ) -> pd.DataFrame:
        target = self.feature_builder.target_block(
            origins + pd.Timedelta(hours=horizon), horizon, index=origins
        )
        x = pd.concat([target, origin_features.reindex(origins)], axis=1)
        shares = generation_shares(
            frame,
            generation_columns=self.generation_columns,
            share_columns=self.source_columns,
        )
        return self._add_share_history(x, shares, horizon)

    def _add_share_history(
        self, x: pd.DataFrame, shares: pd.DataFrame, horizon: int
    ) -> pd.DataFrame:
        """Add causal source dynamics and target-aligned seasonal lags."""
        extra: dict[str, pd.Series] = {}
        for name in self.share_names:
            series = shares[name]
            extra[f"mix_{name}_now"] = series.reindex(x.index)
            for lag in self.share_lags_hours:
                extra[f"mix_{name}_origin_lag_{lag}h"] = series.shift(lag).reindex(x.index)
            for window in self.share_rolling_means_hours:
                extra[f"mix_{name}_rollmean_{window}h"] = (
                    series.rolling(window, min_periods=max(2, window // 2)).mean().reindex(x.index)
                )
            # For target T=t0+h, T-L is known at t0 exactly when L>=h.
            for lag in (24, 48, 72, 168):
                if lag >= horizon:
                    extra[f"mix_{name}_target_lag_{lag}h"] = (
                        series.shift(lag - horizon).reindex(x.index)
                    )
        return pd.concat([x, pd.DataFrame(extra, index=x.index)], axis=1)

    def _fit_sources(
        self, x: pd.DataFrame, targets: pd.DataFrame
    ) -> dict[str, object]:
        estimators: dict[str, object] = {}
        for name in self.share_names:
            estimator = _make_hgb(self.source_params, self.random_state)
            estimator.fit(x, targets[name])
            estimators[name] = estimator
        return estimators

    @staticmethod
    def _predict_sources(
        estimators: dict[str, object], x: pd.DataFrame
    ) -> pd.DataFrame:
        predicted = {
            name: np.clip(estimator.predict(x), 0.0, 1.0)
            for name, estimator in estimators.items()
        }
        return pd.DataFrame(predicted, index=x.index)

    @staticmethod
    def _residual_matrix(
        x: pd.DataFrame, predicted_shares: pd.DataFrame, physical: np.ndarray
    ) -> pd.DataFrame:
        extra = predicted_shares.add_prefix("predicted_")
        extra["physical_prediction_gco2_kwh"] = physical
        return pd.concat([x, extra], axis=1)

    def fit(self, train_frame: pd.DataFrame) -> PhysicalCarbonModel:
        n = len(train_frame)
        split = int(n * (1.0 - self.residual_holdout_fraction))
        min_base = max(200, max(self.horizons) * 4)
        if split < min_base or n - split < 100:
            raise ValueError("not enough rows for source training plus residual calibration")

        base = train_frame.iloc[:split]
        base_shares = generation_shares(
            base,
            generation_columns=self.generation_columns,
            share_columns=self.source_columns,
        )
        base_mapper = PhysicalCarbonMapper(self.share_names).fit(base_shares, base[CARBON])
        base_origin_features = self.feature_builder.origin_features(base)
        all_origin_features = self.feature_builder.origin_features(train_frame)

        # Train the residual stage from genuinely out-of-sample source forecasts.
        for horizon in self.horizons:
            base_x, base_targets, _ = self._supervised(
                base, horizon, origin_features=base_origin_features
            )
            base_estimators = self._fit_sources(base_x, base_targets)

            max_origin_pos = n - horizon
            calibration_origins = train_frame.index[split:max_origin_pos]
            calibration_x = self._inference_matrix(
                train_frame, all_origin_features, calibration_origins, horizon
            ).reindex(columns=base_x.columns)
            predicted_shares = self._predict_sources(base_estimators, calibration_x)
            physical = base_mapper.predict(predicted_shares)
            target_times = calibration_origins + pd.Timedelta(hours=horizon)
            actual = pd.to_numeric(
                train_frame[CARBON].reindex(target_times), errors="coerce"
            ).to_numpy()
            residual_y = actual - physical
            residual_x = self._residual_matrix(calibration_x, predicted_shares, physical)
            valid = np.isfinite(residual_y)
            residual_estimator = _make_hgb(self.residual_params, self.random_state)
            residual_estimator.fit(residual_x.loc[valid], residual_y[valid])
            self.residual_estimators_[horizon] = residual_estimator
            self.residual_feature_names_[horizon] = list(residual_x.columns)
            self.residual_calibration_rows_[horizon] = int(valid.sum())

        # Refit the physical map and source forecasters on every pre-test row.
        full_shares = generation_shares(
            train_frame,
            generation_columns=self.generation_columns,
            share_columns=self.source_columns,
        )
        self.mapper.fit(full_shares, train_frame[CARBON])
        full_origin_features = self.feature_builder.origin_features(train_frame)
        for horizon in self.horizons:
            x, targets, _ = self._supervised(
                train_frame, horizon, origin_features=full_origin_features
            )
            self.source_estimators_[horizon] = self._fit_sources(x, targets)
            self.source_feature_names_[horizon] = list(x.columns)
        return self

    def predict_batch(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        horizons: Sequence[int] | None = None,
        *,
        residual_correction: bool = True,
    ) -> pd.DataFrame:
        origin_features = self.feature_builder.origin_features(frame)
        frames: list[pd.DataFrame] = []
        for horizon in (self.horizons if horizons is None else tuple(map(int, horizons))):
            if horizon not in self.source_estimators_:
                continue
            x = self._inference_matrix(frame, origin_features, origins, horizon)
            x = x.reindex(columns=self.source_feature_names_[horizon])
            predicted_shares = self._predict_sources(self.source_estimators_[horizon], x)
            physical = self.mapper.predict(predicted_shares)
            correction = np.zeros(len(x), dtype=float)
            if residual_correction and horizon in self.residual_estimators_:
                residual_x = self._residual_matrix(x, predicted_shares, physical)
                residual_x = residual_x.reindex(columns=self.residual_feature_names_[horizon])
                correction = self.residual_estimators_[horizon].predict(residual_x)
            frames.append(
                pd.DataFrame(
                    {
                        "origin": origins,
                        "horizon": horizon,
                        "target_time": origins + pd.Timedelta(hours=horizon),
                        "physical_prediction": physical,
                        "residual_correction": correction,
                        "prediction": np.clip(physical + correction, 0.0, None),
                    }
                )
            )
        if not frames:
            return pd.DataFrame(
                columns=[
                    "origin", "horizon", "target_time", "physical_prediction",
                    "residual_correction", "prediction",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def save(self, path) -> None:
        import pathlib

        import joblib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> PhysicalCarbonModel:
        import joblib

        return joblib.load(path)


def train_physical_model(
    train_frame: pd.DataFrame,
    carbon_cfg: dict,
    *,
    climatology=None,
    forecast_frame=None,
) -> PhysicalCarbonModel:
    """Construct and fit the Phase-C model from ``carbon_model.yaml``."""

    cfg = carbon_cfg.get("physical_model", {})
    model_cfg = carbon_cfg.get("model", {})
    physical_carbon_cfg = copy.deepcopy(carbon_cfg)
    origin_features = cfg.get("origin_features")
    if origin_features is not None:
        physical_carbon_cfg.setdefault("features", {}).setdefault(
            "electricity_system", {}
        )["use"] = origin_features
    feature_builder = feature_builder_from_config(
        physical_carbon_cfg, climatology=climatology, forecast_frame=forecast_frame
    )
    model = PhysicalCarbonModel(
        feature_builder,
        horizons=model_cfg.get("horizons_hours", (1, 3, 6, 12, 24, 48)),
        generation_columns=cfg.get("generation_columns", DEFAULT_GENERATION_COLUMNS),
        share_columns=cfg.get("share_columns", DEFAULT_SHARE_COLUMNS),
        source_params=cfg.get("source_hist_gradient_boosting", {}),
        residual_params=cfg.get("residual_hist_gradient_boosting", {}),
        residual_holdout_fraction=cfg.get("residual_holdout_fraction", 0.20),
        share_lags_hours=cfg.get("share_lags_hours", (1, 2, 3, 24, 48, 72, 168)),
        share_rolling_means_hours=cfg.get("share_rolling_means_hours", (6, 24)),
        random_state=cfg.get("random_state", 42),
    )
    return model.fit(train_frame)


__all__ = [
    "PhysicalCarbonMapper",
    "PhysicalCarbonModel",
    "generation_shares",
    "train_physical_model",
]
