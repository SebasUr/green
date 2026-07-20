"""Operational carbon forecast through directly predicted generation shares.

This experiment is deliberately isolated from :mod:`realtime_proxy`.  The
existing physical head predicts four emitting sources and total generation in
MW, then divides two independently forecast quantities.  Here each target is
instead the source's share of total domestic generation, so the fixed RTE
identity is linear in the model outputs::

    CI = 986*coal_share + 777*fuel_oil_share
         + 429*gas_share + 494*bioenergy_share

The feature helper only exposes the last closed origin hour and target-aligned
D-1/D-7 shares.  In particular, target D-1 at horizon 24 is masked because it
lands on the still-open hour labelled by the forecast origin.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.physical import generation_shares
from green_observatory.carbon.realtime_proxy import EMISSION_FACTORS_GCO2_KWH


SHARE_SOURCE_COLUMNS = ("gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw")
SHARE_TARGET_COLUMNS = tuple(f"{column}_share" for column in SHARE_SOURCE_COLUMNS)


def source_share_targets(
    frame: pd.DataFrame, target_times: Sequence,
) -> pd.DataFrame:
    """Return physical emitting-source shares at ``target_times``."""

    shares = generation_shares(frame, share_columns=SHARE_SOURCE_COLUMNS)
    return shares.reindex(pd.DatetimeIndex(target_times)).reset_index(drop=True)


def add_causal_share_features(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    hourly_state_lag_hours: int = 1,
    target_lags_hours: Sequence[int] = (24, 168),
) -> pd.DataFrame:
    """Add closed-hour source-share state without changing the base builder.

    ``frame`` uses RTE hour-start labels for hourly averages.  Consequently
    the row at origin ``t`` is not complete at issue time and the origin state
    is read at ``t-1h``.  Target-aligned lags are retained only when their
    timestamp is strictly earlier than the corresponding origin.
    """

    if hourly_state_lag_hours < 1:
        raise ValueError("hourly_state_lag_hours must be at least 1")
    required = {"origin", "target_time"}
    missing = sorted(required.difference(meta.columns))
    if missing:
        raise ValueError(f"share feature metadata is missing: {missing}")
    if len(x) != len(meta):
        raise ValueError("share feature matrix and metadata must have equal length")

    shares = generation_shares(frame, share_columns=SHARE_SOURCE_COLUMNS)
    origins = pd.DatetimeIndex(pd.to_datetime(meta["origin"], utc=True))
    targets = pd.DatetimeIndex(pd.to_datetime(meta["target_time"], utc=True))
    complete_times = origins - pd.Timedelta(hours=hourly_state_lag_hours)
    out = x.reset_index(drop=True).copy()

    origin_values: dict[str, np.ndarray] = {}
    lag_values: dict[tuple[int, str], np.ndarray] = {}
    for name in SHARE_TARGET_COLUMNS:
        origin = shares[name].reindex(complete_times).to_numpy(dtype=float)
        origin_values[name] = origin
        out[f"share_origin_{name}"] = origin
        for lag in target_lags_hours:
            lag_times = targets - pd.Timedelta(hours=int(lag))
            values = shares[name].reindex(lag_times).to_numpy(dtype=float)
            values = values.copy()
            values[lag_times >= origins] = np.nan
            lag_values[(int(lag), name)] = values
            out[f"share_tgtlag{int(lag)}_{name}"] = values
            out[f"share_delta_tgtlag{int(lag)}_origin_{name}"] = values - origin

    # Compact physical summaries make the relation to the final fixed formula
    # explicit while preserving each individual source for nonlinear trees.
    for prefix, values_by_source in (
        ("origin", origin_values),
        *(
            (
                f"tgtlag{int(lag)}",
                {
                    name: lag_values[(int(lag), name)]
                    for name in SHARE_TARGET_COLUMNS
                },
            )
            for lag in target_lags_hours
        ),
    ):
        emitting = np.zeros(len(out), dtype=float)
        carbon = np.zeros(len(out), dtype=float)
        valid = np.ones(len(out), dtype=bool)
        for source, name in zip(SHARE_SOURCE_COLUMNS, SHARE_TARGET_COLUMNS):
            values = values_by_source[name]
            valid &= np.isfinite(values)
            emitting += np.nan_to_num(values, nan=0.0)
            carbon += EMISSION_FACTORS_GCO2_KWH[source] * np.nan_to_num(
                values, nan=0.0
            )
        emitting[~valid] = np.nan
        carbon[~valid] = np.nan
        out[f"share_{prefix}_emitting_total"] = emitting
        out[f"share_{prefix}_fixed_ci"] = carbon
    return out.replace([np.inf, -np.inf], np.nan)


class SourceShareProxyMoE:
    """Forecast emitting generation shares and apply the fixed RTE formula.

    Gas retains the three regime experts used by the MW model because it is
    France's dominant marginal emitting source.  A pooled gas regressor is
    fitted as a robustness alternative.  Coal, fuel oil and bioenergy use one
    pooled regressor each.  Regressors learn carbon-contribution units
    (``factor * share``), which is merely a numerically better-scaled version
    of directly learning the share; outputs are converted back to shares
    before applying the formula.
    """

    def __init__(
        self,
        *,
        classifier_params: dict | None = None,
        source_params: dict | None = None,
        inverse_level_floor: float = 12.0,
        warm_season_coal_persistence: bool = False,
        random_state: int = 42,
    ) -> None:
        self.classifier_params = classifier_params or {}
        self.source_params = source_params or {}
        self.inverse_level_floor = float(inverse_level_floor)
        self.warm_season_coal_persistence = bool(warm_season_coal_persistence)
        self.random_state = int(random_state)
        self.feature_names_: list[str] = []
        self.classifier_: object | None = None
        self.gas_experts_: dict[int, object] = {}
        self.gas_pooled_: object | None = None
        self.other_regressors_: dict[str, object] = {}

    @staticmethod
    def _lightgbm():
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SourceShareProxyMoE requires LightGBM") from exc
        return LGBMClassifier, LGBMRegressor

    def fit(self, x: pd.DataFrame, meta: pd.DataFrame) -> "SourceShareProxyMoE":
        required = {"actual", "regime", *SHARE_TARGET_COLUMNS}
        missing = sorted(required.difference(meta.columns))
        if missing:
            raise ValueError(f"source-share metadata is missing: {missing}")
        valid = meta.loc[:, list(required)].notna().all(axis=1)
        x_fit = x.loc[valid].reset_index(drop=True)
        meta_fit = meta.loc[valid].reset_index(drop=True)
        self.feature_names_ = list(x_fit.columns)
        regime = meta_fit["regime"].astype(int).to_numpy()
        if set(np.unique(regime)) != {0, 1, 2}:
            raise ValueError("SourceShareProxyMoE needs all three fossil regimes")

        LGBMClassifier, LGBMRegressor = self._lightgbm()
        classifier_defaults = {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 30,
            "reg_lambda": 5.0,
            "verbosity": -1,
            "n_jobs": 1,
            "class_weight": "balanced",
        }
        classifier_defaults.update(self.classifier_params)
        self.classifier_ = LGBMClassifier(
            random_state=self.random_state, **classifier_defaults
        )
        self.classifier_.fit(x_fit, regime)

        source_defaults = {
            "objective": "regression_l1",
            "n_estimators": 300,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "min_child_samples": 20,
            "reg_lambda": 5.0,
            "verbosity": -1,
            "n_jobs": 1,
        }
        source_defaults.update(self.source_params)
        weight = 1.0 / np.clip(
            meta_fit["actual"].to_numpy(dtype=float),
            self.inverse_level_floor,
            None,
        )

        gas_target = (
            EMISSION_FACTORS_GCO2_KWH["gas_mw"]
            * meta_fit["gas_mw_share"].to_numpy(dtype=float)
        )
        self.gas_experts_ = {}
        for label in (0, 1, 2):
            mask = regime == label
            estimator = LGBMRegressor(
                random_state=self.random_state + 10 + label,
                **source_defaults,
            )
            estimator.fit(x_fit.loc[mask], gas_target[mask], sample_weight=weight[mask])
            self.gas_experts_[label] = estimator
        self.gas_pooled_ = LGBMRegressor(
            random_state=self.random_state + 13, **source_defaults
        )
        self.gas_pooled_.fit(x_fit, gas_target, sample_weight=weight)

        self.other_regressors_ = {}
        for offset, source in enumerate(
            ("coal_mw", "fuel_oil_mw", "bioenergy_mw")
        ):
            target = (
                EMISSION_FACTORS_GCO2_KWH[source]
                * meta_fit[f"{source}_share"].to_numpy(dtype=float)
            )
            estimator = LGBMRegressor(
                random_state=self.random_state + 20 + offset,
                **source_defaults,
            )
            estimator.fit(x_fit, target, sample_weight=weight)
            self.other_regressors_[source] = estimator
        return self

    def predict_matrix(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.classifier_ is None or self.gas_pooled_ is None:
            raise RuntimeError("SourceShareProxyMoE.predict_matrix called before fit")
        x = x.reindex(columns=self.feature_names_)
        raw_probability = self.classifier_.predict_proba(x)
        probability = np.zeros((len(x), 3), dtype=float)
        for position, label in enumerate(self.classifier_.classes_):
            probability[:, int(label)] = raw_probability[:, position]
        gas_by_regime = np.column_stack(
            [
                np.clip(self.gas_experts_[label].predict(x), 0.0, 429.0)
                for label in (0, 1, 2)
            ]
        )
        gas = np.sum(probability * gas_by_regime, axis=1)
        gas_pooled = np.clip(self.gas_pooled_.predict(x), 0.0, 429.0)
        contribution = {
            source: np.clip(
                estimator.predict(x), 0.0, EMISSION_FACTORS_GCO2_KWH[source]
            )
            for source, estimator in self.other_regressors_.items()
        }

        coal_model = contribution["coal_mw"].copy()
        if (
            self.warm_season_coal_persistence
            and "share_tgtlag24_coal_mw_share" in x
            and "target_month" in x
        ):
            month = pd.to_numeric(x["target_month"], errors="coerce").to_numpy()
            coal_d1_share = pd.to_numeric(
                x["share_tgtlag24_coal_mw_share"], errors="coerce"
            ).to_numpy()
            warm = np.isin(month, np.arange(2, 11)) & np.isfinite(coal_d1_share)
            contribution["coal_mw"] = np.where(
                warm,
                EMISSION_FACTORS_GCO2_KWH["coal_mw"]
                * np.clip(coal_d1_share, 0.0, 1.0),
                coal_model,
            )

        non_gas = (
            contribution["coal_mw"]
            + contribution["fuel_oil_mw"]
            + contribution["bioenergy_mw"]
        )
        prediction = gas + non_gas
        prediction_pooled = gas_pooled + non_gas
        out = pd.DataFrame(
            {
                "prediction": np.clip(prediction, 0.0, None),
                "prediction_pooled": np.clip(prediction_pooled, 0.0, None),
                "predicted_gas_share": gas / 429.0,
                "predicted_gas_pooled_share": gas_pooled / 429.0,
                "predicted_coal_share": contribution["coal_mw"] / 986.0,
                "predicted_coal_model_share": coal_model / 986.0,
                "predicted_fuel_oil_share": contribution["fuel_oil_mw"] / 777.0,
                "predicted_bioenergy_share": contribution["bioenergy_mw"] / 494.0,
                "prob_baseload": probability[:, 0],
                "prob_ccg": probability[:, 1],
                "prob_peak": probability[:, 2],
            },
            index=x.index,
        )
        for label, name in ((0, "baseload"), (1, "ccg"), (2, "peak")):
            out[f"gas_expert_{name}_contribution"] = gas_by_regime[:, label]
        return out


def source_share_variants(matrix: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return soft, sharpened, hard and pooled gas-mixture predictions."""

    probability = matrix[["prob_baseload", "prob_ccg", "prob_peak"]].to_numpy()
    experts = matrix[
        [
            "gas_expert_baseload_contribution",
            "gas_expert_ccg_contribution",
            "gas_expert_peak_contribution",
        ]
    ].to_numpy()
    base = matrix["prediction"].to_numpy(dtype=float)
    base_gas = matrix["predicted_gas_share"].to_numpy(dtype=float) * 429.0
    variants = {
        "share_physical": base,
        "share_physical_pooled": matrix["prediction_pooled"].to_numpy(dtype=float),
    }
    for alpha in (2.0, 3.0, 5.0):
        sharpened = probability**alpha
        sharpened /= np.clip(sharpened.sum(axis=1, keepdims=True), 1e-12, None)
        gas = np.sum(sharpened * experts, axis=1)
        variants[f"share_physical_alpha{int(alpha)}"] = np.clip(
            base + gas - base_gas, 0.0, None
        )
    hard = np.zeros_like(probability)
    hard[np.arange(len(hard)), np.argmax(probability, axis=1)] = 1.0
    gas = np.sum(hard * experts, axis=1)
    variants["share_physical_hard"] = np.clip(base + gas - base_gas, 0.0, None)
    return variants


__all__ = [
    "SHARE_SOURCE_COLUMNS",
    "SHARE_TARGET_COLUMNS",
    "SourceShareProxyMoE",
    "add_causal_share_features",
    "source_share_targets",
    "source_share_variants",
]
