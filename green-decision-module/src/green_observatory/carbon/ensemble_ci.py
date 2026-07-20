"""Leakage-free, opt-in adaptation of the EnsembleCI architecture.

The paper's core design is preserved:

* diverse base sublearners receive raw history/weather/calendar features;
* a second layer receives the raw features plus every base prediction;
* greedy ensemble selection learns static weights for the stack predictions.

LightGBM and CatBoost are used when installed.  In the lean default environment
they fall back to HistGradientBoosting and ExtraTrees respectively, while the
third learner remains an MLP.  The selected backend is recorded on the model so
benchmark output never confuses the lightweight adaptation with an exact
AutoGluon reproduction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.features import FeatureBuilder, feature_builder_from_config
from green_observatory.providers.carbon_base import CARBON


def _make_sublearner(name: str, params: dict, random_state: int):
    """Return ``(estimator, concrete_backend_name)`` for one learner family."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if name in {"lightgbm", "hist_gradient_boosting"}:
        if name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                defaults = {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31}
                return (
                    LGBMRegressor(random_state=random_state, verbosity=-1, **(defaults | params)),
                    "lightgbm",
                )
            except ImportError:
                pass
        from sklearn.ensemble import HistGradientBoostingRegressor

        allowed = {
            "max_iter", "learning_rate", "max_leaf_nodes", "max_depth",
            "min_samples_leaf", "l2_regularization", "early_stopping",
            "validation_fraction", "n_iter_no_change",
        }
        clean = {key: value for key, value in params.items() if key in allowed}
        defaults = {"max_iter": 140, "learning_rate": 0.05, "max_leaf_nodes": 31}
        return (
            HistGradientBoostingRegressor(
                random_state=random_state, **(defaults | clean)
            ),
            "hist_gradient_boosting",
        )

    if name in {"catboost", "extra_trees"}:
        if name == "catboost":
            try:
                from catboost import CatBoostRegressor

                defaults = {
                    "iterations": 250, "learning_rate": 0.04, "depth": 7,
                    "verbose": False, "allow_writing_files": False,
                }
                return (
                    CatBoostRegressor(random_seed=random_state, **(defaults | params)),
                    "catboost",
                )
            except ImportError:
                pass
        from sklearn.ensemble import ExtraTreesRegressor

        allowed = {
            "n_estimators", "max_depth", "min_samples_leaf", "max_features",
            "n_jobs", "bootstrap",
        }
        clean = {key: value for key, value in params.items() if key in allowed}
        defaults = {
            "n_estimators": 100, "min_samples_leaf": 3,
            "max_features": 0.8, "n_jobs": -1,
        }
        return (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("model", ExtraTreesRegressor(
                        random_state=random_state, **(defaults | clean)
                    )),
                ]
            ),
            "extra_trees",
        )

    if name in {"neural_network", "mlp"}:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        allowed = {
            "hidden_layer_sizes", "activation", "solver", "alpha", "batch_size",
            "learning_rate", "learning_rate_init", "max_iter", "early_stopping",
            "validation_fraction", "n_iter_no_change",
        }
        clean = {key: value for key, value in params.items() if key in allowed}
        defaults = {
            "hidden_layer_sizes": (64, 32), "alpha": 0.001, "batch_size": 256,
            "learning_rate_init": 0.001, "max_iter": 60, "early_stopping": True,
            "n_iter_no_change": 6,
        }
        return (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    ("model", MLPRegressor(
                        random_state=random_state, **(defaults | clean)
                    )),
                ]
            ),
            "mlp",
        )
    raise ValueError(f"unknown EnsembleCI sublearner: {name!r}")


def _greedy_ensemble_weights(
    predictions: pd.DataFrame, actual: pd.Series, iterations: int
) -> dict[str, float]:
    """Caruana-style ensemble selection minimizing validation MAE."""
    values = predictions.to_numpy(dtype=float)
    target = actual.to_numpy(dtype=float)
    counts = np.zeros(values.shape[1], dtype=int)
    running = np.zeros(len(target), dtype=float)
    for step in range(iterations):
        errors = [
            np.mean(np.abs((running + values[:, candidate]) / (step + 1) - target))
            for candidate in range(values.shape[1])
        ]
        selected = int(np.argmin(errors))
        counts[selected] += 1
        running += values[:, selected]
    return {
        column: float(count / iterations)
        for column, count in zip(predictions.columns, counts)
    }


class EnsembleCIModel:
    """Per-horizon two-layer stacking ensemble with temporal training blocks."""

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        *,
        horizons: Sequence[int],
        history_columns: Sequence[str],
        history_hours: int = 24,
        sublearners: Sequence[str] = ("lightgbm", "catboost", "neural_network"),
        sublearner_params: dict | None = None,
        base_fraction: float = 0.60,
        stack_fraction: float = 0.20,
        ensemble_iterations: int = 30,
        random_state: int = 42,
    ) -> None:
        if base_fraction <= 0 or stack_fraction <= 0 or base_fraction + stack_fraction >= 1:
            raise ValueError("base and stack fractions must be positive and sum below 1")
        self.feature_builder = feature_builder
        self.horizons = tuple(map(int, horizons))
        self.history_columns = tuple(history_columns)
        self.history_hours = int(history_hours)
        self.sublearners = tuple(sublearners)
        self.sublearner_params = sublearner_params or {}
        self.base_fraction = float(base_fraction)
        self.stack_fraction = float(stack_fraction)
        self.ensemble_iterations = int(ensemble_iterations)
        self.random_state = int(random_state)

        self.base_models_: dict[int, dict[str, object]] = {}
        self.stack_models_: dict[int, dict[str, object]] = {}
        self.feature_names_: dict[int, list[str]] = {}
        self.stack_feature_names_: dict[int, list[str]] = {}
        self.weights_: dict[int, dict[str, float]] = {}
        self.backends_: dict[str, str] = {}
        self.validation_mae_: dict[int, float] = {}

    def _add_history(self, x: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        history: dict[str, pd.Series] = {}
        for column in self.history_columns:
            if column not in frame:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            for lag in range(self.history_hours):
                history[f"hist_{column}_lag_{lag}h"] = series.shift(lag).reindex(x.index)
        return pd.concat([x, pd.DataFrame(history, index=x.index)], axis=1)

    def _supervised(
        self, frame: pd.DataFrame, horizon: int, origin_features: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        x, y = self.feature_builder.build_supervised(
            frame, horizon, origin_features=origin_features
        )
        return self._add_history(x, frame), y

    def _inference_matrix(
        self, frame: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int,
        origin_features: pd.DataFrame,
    ) -> pd.DataFrame:
        target = self.feature_builder.target_block(
            origins + pd.Timedelta(hours=horizon), horizon, index=origins
        )
        x = pd.concat([target, origin_features.reindex(origins)], axis=1)
        return self._add_history(x, frame)

    def _new_learner(self, name: str, stage_offset: int = 0):
        params = self.sublearner_params.get(name, {})
        estimator, backend = _make_sublearner(
            name, params, self.random_state + stage_offset
        )
        self.backends_[name] = backend
        return estimator

    def fit(self, train_frame: pd.DataFrame) -> EnsembleCIModel:
        origin_features = self.feature_builder.origin_features(train_frame)
        for horizon in self.horizons:
            x, y = self._supervised(train_frame, horizon, origin_features)
            n = len(x)
            base_end = int(n * self.base_fraction)
            stack_end = int(n * (self.base_fraction + self.stack_fraction))
            if base_end < 500 or stack_end - base_end < 200 or n - stack_end < 200:
                raise ValueError(f"not enough temporal rows for EnsembleCI horizon {horizon}")
            self.feature_names_[horizon] = list(x.columns)

            base_models: dict[str, object] = {}
            base_stack_predictions: dict[str, np.ndarray] = {}
            base_validation_predictions: dict[str, np.ndarray] = {}
            for name in self.sublearners:
                model = self._new_learner(name)
                model.fit(x.iloc[:base_end], y.iloc[:base_end])
                base_models[name] = model
                base_stack_predictions[f"base_{name}"] = model.predict(
                    x.iloc[base_end:stack_end]
                )
                base_validation_predictions[f"base_{name}"] = model.predict(
                    x.iloc[stack_end:]
                )

            stack_train = pd.concat(
                [
                    x.iloc[base_end:stack_end],
                    pd.DataFrame(base_stack_predictions, index=x.index[base_end:stack_end]),
                ],
                axis=1,
            )
            stack_validation = pd.concat(
                [
                    x.iloc[stack_end:],
                    pd.DataFrame(base_validation_predictions, index=x.index[stack_end:]),
                ],
                axis=1,
            )
            self.stack_feature_names_[horizon] = list(stack_train.columns)
            stack_models: dict[str, object] = {}
            stack_predictions: dict[str, np.ndarray] = {}
            for name in self.sublearners:
                model = self._new_learner(name, stage_offset=1000)
                model.fit(stack_train, y.iloc[base_end:stack_end])
                stack_models[name] = model
                stack_predictions[name] = np.clip(
                    model.predict(stack_validation), 0.0, None
                )
            validation_predictions = pd.DataFrame(
                stack_predictions, index=x.index[stack_end:]
            )
            weights = _greedy_ensemble_weights(
                validation_predictions, y.iloc[stack_end:], self.ensemble_iterations
            )
            ensemble = sum(
                validation_predictions[name].to_numpy() * weight
                for name, weight in weights.items()
            )
            self.validation_mae_[horizon] = float(
                np.mean(np.abs(ensemble - y.iloc[stack_end:].to_numpy()))
            )

            # Refit after model/weight selection so deployment is not anchored
            # to the oldest 60% of history. The base layer sees the first 80%;
            # its genuinely forward predictions on the newest 20% train the
            # final stack layer. The held-out external test remains untouched.
            final_base_models: dict[str, object] = {}
            final_base_predictions: dict[str, np.ndarray] = {}
            for name in self.sublearners:
                model = self._new_learner(name, stage_offset=2000)
                model.fit(x.iloc[:stack_end], y.iloc[:stack_end])
                final_base_models[name] = model
                final_base_predictions[f"base_{name}"] = model.predict(
                    x.iloc[stack_end:]
                )
            final_stack_train = pd.concat(
                [
                    x.iloc[stack_end:],
                    pd.DataFrame(final_base_predictions, index=x.index[stack_end:]),
                ],
                axis=1,
            ).reindex(columns=self.stack_feature_names_[horizon])
            final_stack_models: dict[str, object] = {}
            for name in self.sublearners:
                model = self._new_learner(name, stage_offset=3000)
                model.fit(final_stack_train, y.iloc[stack_end:])
                final_stack_models[name] = model

            self.base_models_[horizon] = final_base_models
            self.stack_models_[horizon] = final_stack_models
            self.weights_[horizon] = weights
        return self

    def predict_batch(
        self, frame: pd.DataFrame, origins: pd.DatetimeIndex,
        horizons: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        origin_features = self.feature_builder.origin_features(frame)
        parts: list[pd.DataFrame] = []
        for horizon in self.horizons if horizons is None else tuple(map(int, horizons)):
            if horizon not in self.base_models_:
                continue
            x = self._inference_matrix(
                frame, origins, horizon, origin_features
            ).reindex(columns=self.feature_names_[horizon])
            base_predictions = {
                f"base_{name}": model.predict(x)
                for name, model in self.base_models_[horizon].items()
            }
            stack_x = pd.concat(
                [x, pd.DataFrame(base_predictions, index=x.index)], axis=1
            ).reindex(columns=self.stack_feature_names_[horizon])
            stack_predictions = {
                name: np.clip(model.predict(stack_x), 0.0, None)
                for name, model in self.stack_models_[horizon].items()
            }
            prediction = sum(
                stack_predictions[name] * weight
                for name, weight in self.weights_[horizon].items()
            )
            parts.append(
                pd.DataFrame(
                    {
                        "origin": origins,
                        "horizon": horizon,
                        "target_time": origins + pd.Timedelta(hours=horizon),
                        "prediction": np.clip(prediction, 0.0, None),
                    }
                )
            )
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def save(self, path) -> None:
        import pathlib

        import joblib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> EnsembleCIModel:
        import joblib

        return joblib.load(path)


def train_ensemble_ci_model(
    train_frame: pd.DataFrame, carbon_cfg: dict, *, climatology=None,
    forecast_frame: pd.DataFrame | None = None,
) -> EnsembleCIModel:
    """Build and fit the modular EnsembleCI adaptation from configuration."""
    cfg = carbon_cfg.get("ensemble_ci", {})
    model_cfg = carbon_cfg.get("model", {})
    builder = feature_builder_from_config(
        carbon_cfg, climatology=climatology, forecast_frame=forecast_frame
    )
    model = EnsembleCIModel(
        builder,
        horizons=model_cfg.get("horizons_hours", (1, 3, 6, 12, 24, 48)),
        history_columns=cfg.get(
            "history_columns",
            [CARBON, "nuclear_mw", "gas_mw", "coal_mw", "fuel_oil_mw",
             "wind_mw", "solar_mw", "hydro_mw", "bioenergy_mw"],
        ),
        history_hours=cfg.get("history_hours", 24),
        sublearners=cfg.get(
            "sublearners", ("lightgbm", "catboost", "neural_network")
        ),
        sublearner_params=cfg.get("sublearner_params", {}),
        base_fraction=cfg.get("base_fraction", 0.60),
        stack_fraction=cfg.get("stack_fraction", 0.20),
        ensemble_iterations=cfg.get("ensemble_iterations", 30),
        random_state=cfg.get("random_state", 42),
    )
    return model.fit(train_frame)


__all__ = ["EnsembleCIModel", "train_ensemble_ci_model"]
