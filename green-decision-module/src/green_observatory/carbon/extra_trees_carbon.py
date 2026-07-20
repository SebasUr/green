"""Leakage-safe Extremely Randomized Trees head for carbon intensity.

This module intentionally contains only the estimator.  Feature visibility and
temporal slicing remain the responsibility of :class:`RegimeMoEFeatureBuilder`
and the standalone evaluator.  In particular, missing-value imputation is fit
on the training matrix and is never recomputed from validation or holdout rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ExtraTreesCarbonRegressor:
    """Direct ERT regressor with train-only median imputation.

    The squared-error split criterion is paired with ``1 / max(y, floor)``
    sample weights.  This keeps fitting computationally practical while moving
    the optimization pressure toward relative error (MAPE) at France's common
    low-carbon operating levels.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_features: float | int | str | None = 0.7,
        min_samples_leaf: int = 6,
        max_depth: int | None = None,
        inverse_level_floor: float = 8.0,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.inverse_level_floor = float(inverse_level_floor)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.feature_names_: list[str] = []
        self.imputer_: object | None = None
        self.estimator_: object | None = None

    @staticmethod
    def _dependencies():
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.impute import SimpleImputer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ExtraTreesCarbonRegressor requires scikit-learn"
            ) from exc
        return ExtraTreesRegressor, SimpleImputer

    def _coerce(self, x: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        matrix = x.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if fitting:
            # Columns absent throughout the fit period have no train-only
            # statistic.  Dropping them is preferable to inventing a constant
            # using knowledge of future availability.
            self.feature_names_ = [
                column for column in matrix if matrix[column].notna().any()
            ]
            if not self.feature_names_:
                raise ValueError("ERT training matrix has no usable features")
        elif not self.feature_names_:
            raise RuntimeError("ExtraTreesCarbonRegressor used before fit")
        return matrix.reindex(columns=self.feature_names_)

    def sample_weight(self, actual: np.ndarray) -> np.ndarray:
        """Return the fixed relative-error approximation used during fit."""

        actual = np.asarray(actual, dtype=float)
        return 1.0 / np.maximum(actual, self.inverse_level_floor)

    def fit(
        self, x: pd.DataFrame, actual: pd.Series | np.ndarray
    ) -> "ExtraTreesCarbonRegressor":
        ExtraTreesRegressor, SimpleImputer = self._dependencies()
        y = np.asarray(actual, dtype=float)
        if len(x) != len(y):
            raise ValueError("ERT features and target must have equal length")
        valid = np.isfinite(y) & (y > 0.0)
        if not valid.any():
            raise ValueError("ERT training target has no finite positive rows")
        x_fit = self._coerce(x.loc[valid].reset_index(drop=True), fitting=True)
        y_fit = y[valid]
        self.imputer_ = SimpleImputer(strategy="median")
        transformed = self.imputer_.fit_transform(x_fit)
        self.estimator_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion="squared_error",
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            bootstrap=False,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.estimator_.fit(
            transformed, y_fit, sample_weight=self.sample_weight(y_fit)
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.imputer_ is None or self.estimator_ is None:
            raise RuntimeError("ExtraTreesCarbonRegressor used before fit")
        matrix = self._coerce(x, fitting=False)
        transformed = self.imputer_.transform(matrix)
        return np.clip(self.estimator_.predict(transformed), 0.0, None)


__all__ = ["ExtraTreesCarbonRegressor"]
