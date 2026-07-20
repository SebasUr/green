"""Compact direct regressors for a leakage-safe French carbon stack.

The models in this module deliberately consume an already constructed causal
feature matrix.  They do not know how the forecasts were collected and cannot
silently replace missing day-ahead values with realised observations.  This
makes them suitable as diverse second opinions next to ``DirectRegimeMoE``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PooledCarbonRegressor:
    """A pooled tree regressor optimised for absolute percentage error.

    ``target_transform='identity'`` together with ``inverse_level_weight=True``
    minimises a weighted L1 objective equivalent to empirical MAPE (apart from
    the configurable denominator floor).  The log target is a robust
    alternative whose absolute error approximates relative error for small
    residuals.
    """

    def __init__(
        self,
        *,
        backend: str = "lightgbm",
        target_transform: str = "identity",
        inverse_level_weight: bool = True,
        inverse_level_floor: float = 8.0,
        params: dict | None = None,
        random_state: int = 42,
    ) -> None:
        if backend not in {"lightgbm", "catboost"}:
            raise ValueError("backend must be 'lightgbm' or 'catboost'")
        if target_transform not in {"identity", "log"}:
            raise ValueError("target_transform must be 'identity' or 'log'")
        self.backend = backend
        self.target_transform = target_transform
        self.inverse_level_weight = bool(inverse_level_weight)
        self.inverse_level_floor = float(inverse_level_floor)
        self.params = params or {}
        self.random_state = int(random_state)
        self.feature_names_: list[str] = []
        self.model_: object | None = None

    def _new_model(self):
        if self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            defaults = {
                "objective": "regression_l1",
                "n_estimators": 350,
                "learning_rate": 0.035,
                "num_leaves": 23,
                "min_child_samples": 25,
                "feature_fraction": 0.85,
                "reg_lambda": 5.0,
                "verbosity": -1,
                "n_jobs": 1,
            }
            return LGBMRegressor(
                random_state=self.random_state, **(defaults | self.params)
            )

        from catboost import CatBoostRegressor

        defaults = {
            "loss_function": "MAE",
            "iterations": 350,
            "learning_rate": 0.04,
            "depth": 7,
            "l2_leaf_reg": 5.0,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
        }
        return CatBoostRegressor(
            random_seed=self.random_state, **(defaults | self.params)
        )

    def fit(self, x: pd.DataFrame, actual: np.ndarray | pd.Series):
        self.feature_names_ = list(x.columns)
        actual = np.asarray(actual, dtype=float)
        target = (
            np.log(np.clip(actual, 1e-6, None))
            if self.target_transform == "log"
            else actual
        )
        sample_weight = None
        if self.inverse_level_weight and self.target_transform == "identity":
            sample_weight = 1.0 / np.clip(
                actual, self.inverse_level_floor, None
            )
        self.model_ = self._new_model()
        self.model_.fit(x, target, sample_weight=sample_weight)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("PooledCarbonRegressor.predict called before fit")
        raw = np.asarray(
            self.model_.predict(x.reindex(columns=self.feature_names_)), dtype=float
        )
        prediction = np.exp(raw) if self.target_transform == "log" else raw
        return np.clip(prediction, 0.0, None)


class PerHorizonCarbonRegressor:
    """Independent small regressors for each forecast horizon.

    This is intentionally a single controlled ablation.  It removes the
    pooled model's requirement to learn all horizon interactions, at the cost
    of giving each tree only about one twenty-fourth as many samples.
    """

    def __init__(self, **pooled_kwargs) -> None:
        self.pooled_kwargs = pooled_kwargs
        self.models_: dict[int, PooledCarbonRegressor] = {}

    def fit(
        self,
        x: pd.DataFrame,
        actual: np.ndarray | pd.Series,
        horizon: np.ndarray | pd.Series,
    ):
        actual = np.asarray(actual, dtype=float)
        horizon = np.asarray(horizon, dtype=int)
        self.models_ = {}
        for value in sorted(np.unique(horizon)):
            mask = horizon == value
            model = PooledCarbonRegressor(**self.pooled_kwargs)
            model.fit(x.loc[mask], actual[mask])
            self.models_[int(value)] = model
        return self

    def predict(
        self, x: pd.DataFrame, horizon: np.ndarray | pd.Series
    ) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("PerHorizonCarbonRegressor.predict called before fit")
        horizon = np.asarray(horizon, dtype=int)
        prediction = np.full(len(x), np.nan, dtype=float)
        for value, model in self.models_.items():
            mask = horizon == value
            if mask.any():
                prediction[mask] = model.predict(x.loc[mask])
        if not np.isfinite(prediction).all():
            raise ValueError("prediction contains a horizon absent during training")
        return prediction


def percentage_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return the compact metric set used during temporal model selection."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mape": float(100.0 * np.mean(absolute / np.clip(np.abs(actual), 1e-9, None))),
        "wape": float(100.0 * absolute.sum() / np.abs(actual).sum()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "n": int(len(actual)),
    }


__all__ = [
    "PerHorizonCarbonRegressor",
    "PooledCarbonRegressor",
    "percentage_metrics",
]
