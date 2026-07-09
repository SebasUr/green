"""Project carbon forecast model (baseline ladder rung 4).

A **direct multi-horizon** regressor: one scikit-learn estimator per horizon,
trained on the leakage-safe features from :mod:`green_observatory.carbon.features`.
``HistGradientBoostingRegressor`` is the primary algorithm (fast on tabular
data, native NaN handling, easy to explain); ``random_forest`` is available for
comparison (wrapped with median imputation).

The trained model exposes a :class:`ProjectModelForecaster` matching the shared
``Forecaster`` interface so it drops straight into the evaluation backtest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import _target_index
from green_observatory.carbon.features import FeatureBuilder, feature_builder_from_config
from green_observatory.models import ModelName
from green_observatory.providers.carbon_base import CARBON

DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48)


def _make_estimator(algorithm: str, params: dict, random_state: int):
    params = {k: v for k, v in (params or {}).items() if v is not None}
    if algorithm == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=random_state, **params)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(random_state=random_state, **params)),
            ]
        )
    raise ValueError(f"unknown algorithm: {algorithm!r}")


class ProjectCarbonModel:
    """Per-horizon regressors trained on leakage-safe features."""

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        algorithm: str = "hist_gradient_boosting",
        params: dict | None = None,
        random_state: int = 42,
    ) -> None:
        self.feature_builder = feature_builder
        self.horizons = tuple(int(h) for h in horizons)
        self.algorithm = algorithm
        self.params = params or {}
        self.random_state = random_state
        self.estimators_: dict[int, object] = {}
        self.feature_names_: dict[int, list[str]] = {}

    def fit(self, train_frame: pd.DataFrame) -> ProjectCarbonModel:
        origin_feats = self.feature_builder.origin_features(train_frame)
        for h in self.horizons:
            x, y = self.feature_builder.build_supervised(
                train_frame, h, origin_features=origin_feats
            )
            if len(x) == 0:
                raise ValueError(f"no training rows for horizon {h}h")
            est = _make_estimator(self.algorithm, self.params, self.random_state)
            est.fit(x, y)
            self.estimators_[h] = est
            self.feature_names_[h] = list(x.columns)
        return self

    def make_forecaster(self, predict_frame: pd.DataFrame) -> ProjectModelForecaster:
        """Bind the trained model to a series to predict over.

        Origin features are precomputed on ``predict_frame`` (leakage-free: no
        origin transform looks ahead), so per-origin prediction is a fast lookup.
        """
        origin_feats = self.feature_builder.origin_features(predict_frame)
        return ProjectModelForecaster(self, origin_feats)

    # -- persistence ---------------------------------------------------- #
    def save(self, path) -> None:
        import pathlib

        import joblib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> ProjectCarbonModel:
        import joblib

        return joblib.load(path)


class ProjectModelForecaster:
    """Adapts a trained :class:`ProjectCarbonModel` to the ``Forecaster`` API."""

    name = ModelName.project_model

    def __init__(self, model: ProjectCarbonModel, origin_features: pd.DataFrame) -> None:
        self.model = model
        self.origin_features = origin_features

    def _origin_row(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        if origin in self.origin_features.index:
            return self.origin_features
        # Fallback for origins outside the bound series (e.g. fresh live data).
        return self.model.feature_builder.origin_features(history)

    def predict(
        self, history: pd.DataFrame, origin: pd.Timestamp, horizons_hours: Sequence[float]
    ) -> pd.DataFrame:
        origin_feats = self._origin_row(history, origin)
        preds: list[float] = []
        for h in horizons_hours:
            hi = int(h)
            if hi not in self.model.estimators_:
                raise ValueError(f"model was not trained for horizon {hi}h")
            x = self.model.feature_builder.build_inference_row(origin_feats, origin, hi)
            x = x.reindex(columns=self.model.feature_names_[hi])
            yhat = float(self.model.estimators_[hi].predict(x)[0])
            preds.append(max(yhat, 0.0))
        targets = _target_index(origin, horizons_hours)
        return pd.DataFrame(
            {
                "prediction": preds,
                "lower": np.nan,
                "upper": np.nan,
                "horizon_hours": list(horizons_hours),
            },
            index=targets,
        )


def train_project_model(
    train_frame: pd.DataFrame, carbon_cfg: dict, climatology=None
) -> ProjectCarbonModel:
    """Build and fit a :class:`ProjectCarbonModel` from a carbon-model config dict."""
    model_cfg = carbon_cfg.get("model", {})
    algorithm = model_cfg.get("algorithm", "hist_gradient_boosting")
    params = model_cfg.get(algorithm, {})
    builder = feature_builder_from_config(carbon_cfg, climatology=climatology)
    model = ProjectCarbonModel(
        builder,
        horizons=model_cfg.get("horizons_hours", DEFAULT_HORIZONS),
        algorithm=algorithm,
        params=params,
        random_state=model_cfg.get("random_state", 42),
    )
    return model.fit(train_frame)
