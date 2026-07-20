"""Consolidated French carbon model based on detailed source shares.

The seven emitting technology components are divided by observed domestic
generation before forecasting.  This removes the independent denominator
forecast from :mod:`consolidated_physical`.  Effective emission factors are
still estimated by positive, no-intercept OLS on the train-only physical
identity ``actual * generation = components @ factors``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.consolidated_physical import (
    EMITTING_COMPONENTS,
    GENERATION_COLUMNS,
)


DETAILED_SHARE_COLUMNS = tuple(f"{column}_share" for column in EMITTING_COMPONENTS)
TOTAL_GENERATION = "total_generation_mw"
GAS_COMPONENTS = (
    "gas_turbine_mw",
    "gas_cogeneration_mw",
    "gas_ccg_mw",
    "gas_other_mw",
)


def detailed_generation_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Return seven detailed emitting components over domestic generation."""

    missing = sorted(set(GENERATION_COLUMNS + EMITTING_COMPONENTS) - set(frame.columns))
    if missing:
        raise ValueError(f"detailed carbon data miss columns: {missing}")
    generation = frame.loc[:, GENERATION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).clip(lower=0.0)
    valid = generation.notna().all(axis=1)
    denominator = generation.sum(axis=1).where(valid)
    denominator = denominator.where(denominator > 0.0)
    components = frame.loc[:, EMITTING_COMPONENTS].apply(
        pd.to_numeric, errors="coerce"
    ).clip(lower=0.0)
    return components.div(denominator, axis=0).rename(
        columns={column: f"{column}_share" for column in EMITTING_COMPONENTS}
    )


def detailed_share_targets(
    frame: pd.DataFrame, target_times: Sequence,
) -> pd.DataFrame:
    target_times = pd.DatetimeIndex(target_times)
    out = detailed_generation_shares(frame).reindex(target_times).copy()
    generation = frame.loc[:, GENERATION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).clip(lower=0.0)
    out[TOTAL_GENERATION] = generation.sum(axis=1).where(
        generation.notna().all(axis=1)
    ).reindex(target_times)
    return out.reset_index(drop=True)


def add_causal_detailed_share_features(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    hourly_state_lag_hours: int = 1,
    target_lags_hours: Sequence[int] = (24, 168),
) -> pd.DataFrame:
    """Attach origin/D-1/D-7 detailed shares under closed-hour semantics."""

    if hourly_state_lag_hours < 1:
        raise ValueError("hourly_state_lag_hours must be at least 1")
    if len(x) != len(meta):
        raise ValueError("share feature matrix and metadata must have equal length")
    shares = detailed_generation_shares(frame)
    origins = pd.DatetimeIndex(pd.to_datetime(meta["origin"], utc=True))
    targets = pd.DatetimeIndex(pd.to_datetime(meta["target_time"], utc=True))
    origin_times = origins - pd.Timedelta(hours=hourly_state_lag_hours)
    out = x.reset_index(drop=True).copy()
    for name in DETAILED_SHARE_COLUMNS:
        origin = shares[name].reindex(origin_times).to_numpy(dtype=float)
        out[f"detail_share_origin_{name}"] = origin
        for lag in target_lags_hours:
            lag_times = targets - pd.Timedelta(hours=int(lag))
            values = shares[name].reindex(lag_times).to_numpy(dtype=float).copy()
            values[lag_times >= origins] = np.nan
            out[f"detail_share_tgtlag{int(lag)}_{name}"] = values
            out[f"detail_share_delta_tgtlag{int(lag)}_origin_{name}"] = (
                values - origin
            )
    return out.replace([np.inf, -np.inf], np.nan)


class ConsolidatedShareRegressor:
    """Pooled share regressors followed by train-only positive OLS factors."""

    def __init__(
        self,
        *,
        source_params: dict | None = None,
        inverse_level_floor: float = 12.0,
        random_state: int = 42,
    ) -> None:
        self.source_params = source_params or {}
        self.inverse_level_floor = float(inverse_level_floor)
        self.random_state = int(random_state)
        self.feature_names_: list[str] = []
        self.emission_factors_: np.ndarray | None = None
        self.regressors_: dict[str, object] = {}
        self.gas_total_regressor_: object | None = None

    @staticmethod
    def _dependencies():
        try:
            from lightgbm import LGBMRegressor
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ConsolidatedShareRegressor requires LightGBM and scikit-learn"
            ) from exc
        return LGBMRegressor, LinearRegression

    def fit(
        self, x: pd.DataFrame, meta: pd.DataFrame
    ) -> "ConsolidatedShareRegressor":
        required = {"actual", TOTAL_GENERATION, *DETAILED_SHARE_COLUMNS}
        missing = sorted(required.difference(meta.columns))
        if missing:
            raise ValueError(f"consolidated share metadata miss columns: {missing}")
        valid = meta.loc[:, list(required)].notna().all(axis=1)
        valid &= meta["actual"].gt(0.0)
        x_fit = x.loc[valid].reset_index(drop=True)
        meta_fit = meta.loc[valid].reset_index(drop=True)
        self.feature_names_ = list(x_fit.columns)

        LGBMRegressor, LinearRegression = self._dependencies()
        total = meta_fit[TOTAL_GENERATION].to_numpy(dtype=float)
        design = (
            meta_fit.loc[:, DETAILED_SHARE_COLUMNS].to_numpy(dtype=float)
            * total[:, None]
        )
        mapper = LinearRegression(fit_intercept=False, positive=True)
        mapper.fit(design, meta_fit["actual"].to_numpy(dtype=float) * total)
        self.emission_factors_ = np.asarray(mapper.coef_, dtype=float)

        defaults = {
            "objective": "regression_l1",
            "n_estimators": 350,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "min_child_samples": 20,
            "reg_lambda": 5.0,
            "verbosity": -1,
            "n_jobs": 1,
        }
        defaults.update(self.source_params)
        sample_weight = 1.0 / np.clip(
            meta_fit["actual"].to_numpy(dtype=float),
            self.inverse_level_floor,
            None,
        )
        self.regressors_ = {}
        for offset, (name, factor) in enumerate(
            zip(DETAILED_SHARE_COLUMNS, self.emission_factors_)
        ):
            # Contribution units avoid poorly scaled targets around 1e-4.
            scale = float(factor) if factor > 1e-6 else 1_000.0
            target = scale * meta_fit[name].to_numpy(dtype=float)
            estimator = LGBMRegressor(
                random_state=self.random_state + offset, **defaults
            )
            estimator.fit(x_fit, target, sample_weight=sample_weight)
            self.regressors_[name] = (estimator, scale)
        gas_total_share = meta_fit[
            [f"{column}_share" for column in GAS_COMPONENTS]
        ].sum(axis=1).to_numpy(dtype=float)
        self.gas_total_regressor_ = LGBMRegressor(
            random_state=self.random_state + 20, **defaults
        )
        self.gas_total_regressor_.fit(
            x_fit, 400.0 * gas_total_share, sample_weight=sample_weight
        )
        return self

    def predict_matrix(self, x: pd.DataFrame) -> pd.DataFrame:
        if (
            self.emission_factors_ is None
            or not self.regressors_
            or self.gas_total_regressor_ is None
        ):
            raise RuntimeError("ConsolidatedShareRegressor used before fit")
        x = x.reindex(columns=self.feature_names_)
        predicted_shares = {}
        for name, (estimator, scale) in self.regressors_.items():
            predicted_shares[name] = np.clip(estimator.predict(x) / scale, 0.0, 1.0)
        matrix = np.column_stack(
            [predicted_shares[name] for name in DETAILED_SHARE_COLUMNS]
        )
        prediction = matrix @ self.emission_factors_
        gas_positions = [EMITTING_COMPONENTS.index(column) for column in GAS_COMPONENTS]
        gas_matrix = matrix[:, gas_positions]
        gas_sum = gas_matrix.sum(axis=1)
        gas_total_share = np.clip(
            self.gas_total_regressor_.predict(x) / 400.0, 0.0, 1.0
        )
        gas_proportions = gas_matrix / np.clip(gas_sum[:, None], 1e-12, None)
        gas_factors = self.emission_factors_[gas_positions]
        old_gas_contribution = gas_matrix @ gas_factors
        gas_total_contribution = (
            gas_proportions * gas_total_share[:, None]
        ) @ gas_factors
        prediction_gas_total = prediction - old_gas_contribution + gas_total_contribution
        out = pd.DataFrame(
            {
                "prediction": np.clip(prediction, 0.0, None),
                "prediction_gas_total": np.clip(prediction_gas_total, 0.0, None),
                "predicted_gas_total_share": gas_total_share,
            },
            index=x.index,
        )
        for name in DETAILED_SHARE_COLUMNS:
            out[f"predicted_{name}"] = predicted_shares[name]
        return out


__all__ = [
    "ConsolidatedShareRegressor",
    "DETAILED_SHARE_COLUMNS",
    "TOTAL_GENERATION",
    "GAS_COMPONENTS",
    "add_causal_detailed_share_features",
    "detailed_generation_shares",
    "detailed_share_targets",
]
