"""Operational RTE real-time carbon target and a physics-explicit forecast head.

The consolidated eCO2mix ``taux_co2`` history is not numerically equivalent to
the provisional value published by the real-time dataset.  The latter is
reconstructed from the simultaneously published French generation mix as::

    (986 * coal + 777 * fuel_oil + 429 * gas + 494 * bioenergy)
    / total_domestic_generation

MW cancel in the ratio and the emission factors are expressed in gCO2/kWh.
This module keeps that operational target separate from the consolidated
target used by the older experiments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


GENERATION_COLUMNS = (
    "nuclear_mw",
    "gas_mw",
    "coal_mw",
    "fuel_oil_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "bioenergy_mw",
)
EMISSION_FACTORS_GCO2_KWH: Mapping[str, float] = {
    "coal_mw": 986.0,
    "fuel_oil_mw": 777.0,
    "gas_mw": 429.0,
    "bioenergy_mw": 494.0,
}
PHYSICAL_TARGET_COLUMNS = (
    "gas_mw",
    "coal_mw",
    "fuel_oil_mw",
    "bioenergy_mw",
    "total_generation_mw",
)


def rte_realtime_carbon_proxy(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct RTE's provisional production-based intensity in gCO2/kWh.

    A value is returned only when all eight domestic generation aggregates are
    present and their non-negative sum is positive.  Missing fuels are never
    silently interpreted as zero.
    """

    missing = [column for column in GENERATION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"RTE proxy frame is missing generation columns: {missing}")
    generation = frame.loc[:, GENERATION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    generation = generation.clip(lower=0.0)
    valid = generation.notna().all(axis=1)
    denominator = generation.sum(axis=1).where(valid)
    denominator = denominator.where(denominator > 0.0)
    numerator = sum(
        factor * generation[column]
        for column, factor in EMISSION_FACTORS_GCO2_KWH.items()
    )
    return (numerator / denominator).rename("rte_realtime_proxy_gco2_kwh")


def proxy_training_frame(frame: pd.DataFrame, *, carbon_column: str) -> pd.DataFrame:
    """Return a copy whose carbon target is the operational RTE proxy."""

    out = frame.copy()
    out[carbon_column] = rte_realtime_carbon_proxy(out)
    return out


def physical_targets(frame: pd.DataFrame, target_times: Sequence) -> pd.DataFrame:
    """Return target-time fuel MW and total domestic generation for training."""

    generation = frame.loc[:, GENERATION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).clip(lower=0.0)
    valid = generation.notna().all(axis=1)
    generation["total_generation_mw"] = generation.sum(axis=1).where(valid)
    targets = generation.reindex(pd.DatetimeIndex(target_times))
    return targets.loc[:, PHYSICAL_TARGET_COLUMNS].reset_index(drop=True)


class PhysicalProxyMoE:
    """Regime-aware fuel-MW forecast followed by the fixed RTE formula.

    Gas is forecast by three specialists mixed with probabilities from the
    same baseload/CCG/peak classifier as the direct MoE.  Coal, fuel oil,
    bioenergy and the generation denominator use pooled regressors.  No
    empirical carbon mapper is fitted: the final head is the official fixed
    formula above.
    """

    def __init__(
        self,
        *,
        classifier_params: dict | None = None,
        source_params: dict | None = None,
        inverse_level_floor: float = 12.0,
        recency_half_life_days: float | None = None,
        warm_season_coal_persistence: bool = False,
        include_pooled_gas: bool = False,
        random_state: int = 42,
    ) -> None:
        self.classifier_params = classifier_params or {}
        self.source_params = source_params or {}
        self.inverse_level_floor = float(inverse_level_floor)
        if recency_half_life_days is not None and recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive or None")
        self.recency_half_life_days = (
            None
            if recency_half_life_days is None
            else float(recency_half_life_days)
        )
        self.warm_season_coal_persistence = bool(warm_season_coal_persistence)
        self.include_pooled_gas = bool(include_pooled_gas)
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
            raise ImportError("PhysicalProxyMoE requires LightGBM") from exc
        return LGBMClassifier, LGBMRegressor

    def fit(self, x: pd.DataFrame, meta: pd.DataFrame) -> "PhysicalProxyMoE":
        required = {"actual", "regime", *PHYSICAL_TARGET_COLUMNS}
        if self.recency_half_life_days is not None:
            required.add("target_time")
        missing = sorted(required.difference(meta.columns))
        if missing:
            raise ValueError(f"physical proxy metadata is missing: {missing}")
        valid = meta.loc[:, list(required)].notna().all(axis=1)
        x_fit = x.loc[valid].reset_index(drop=True)
        meta_fit = meta.loc[valid].reset_index(drop=True)
        self.feature_names_ = list(x_fit.columns)
        regime = meta_fit["regime"].astype(int).to_numpy()
        if set(np.unique(regime)) != {0, 1, 2}:
            raise ValueError("PhysicalProxyMoE needs all three fossil regimes")

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
        sample_weight = 1.0 / np.clip(
            meta_fit["actual"].to_numpy(dtype=float),
            self.inverse_level_floor,
            None,
        )
        # A daily expanding refit adds very little mass to a multi-year fit.
        # This optional causal weighting makes recent dispatch behaviour count
        # more without discarding older regimes completely.  The reference
        # date is the newest target in the *training* slice only.
        if self.recency_half_life_days is not None:
            target_time = pd.to_datetime(meta_fit["target_time"], utc=True)
            age_days = (
                target_time.max() - target_time
            ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
            recency_weight = np.exp2(
                -np.clip(age_days, 0.0, None) / self.recency_half_life_days
            )
            sample_weight *= recency_weight
        self.gas_experts_ = {}
        for label in (0, 1, 2):
            mask = regime == label
            estimator = LGBMRegressor(
                random_state=self.random_state + 10 + label, **source_defaults
            )
            estimator.fit(
                x_fit.loc[mask],
                meta_fit.loc[mask, "gas_mw"],
                sample_weight=sample_weight[mask],
            )
            self.gas_experts_[label] = estimator

        # A pooled gas head is kept alongside the regime specialists.  It is
        # deliberately not the default prediction: it is an opt-in ablation
        # that tests whether avoiding classifier errors transfers better to a
        # new dispatch season.
        self.gas_pooled_ = None
        if self.include_pooled_gas:
            self.gas_pooled_ = LGBMRegressor(
                random_state=self.random_state + 13, **source_defaults
            )
            self.gas_pooled_.fit(
                x_fit,
                meta_fit["gas_mw"],
                sample_weight=sample_weight,
            )

        self.other_regressors_ = {}
        for offset, column in enumerate(
            ("coal_mw", "fuel_oil_mw", "bioenergy_mw", "total_generation_mw")
        ):
            estimator = LGBMRegressor(
                random_state=self.random_state + 20 + offset, **source_defaults
            )
            estimator.fit(
                x_fit,
                meta_fit[column],
                sample_weight=sample_weight,
            )
            self.other_regressors_[column] = estimator
        return self

    def predict_matrix(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.classifier_ is None or len(self.gas_experts_) != 3:
            raise RuntimeError("PhysicalProxyMoE.predict_matrix called before fit")
        x = x.reindex(columns=self.feature_names_)
        raw_probability = self.classifier_.predict_proba(x)
        probability = np.zeros((len(x), 3), dtype=float)
        for position, label in enumerate(self.classifier_.classes_):
            probability[:, int(label)] = raw_probability[:, position]
        gas_by_regime = np.column_stack(
            [
                np.clip(self.gas_experts_[label].predict(x), 0.0, None)
                for label in (0, 1, 2)
            ]
        )
        gas = np.sum(probability * gas_by_regime, axis=1)
        components = {
            column: np.clip(estimator.predict(x), 0.0, None)
            for column, estimator in self.other_regressors_.items()
        }
        coal_model = components["coal_mw"].copy()
        # French coal has a strongly seasonal hurdle: outside the cold season
        # the few-MW background is vastly more persistent than a pooled level
        # regressor.  The target-aligned D-1 value is known for every h<=24.
        # This switch is opt-in because it was introduced after the first live
        # diagnostic and must therefore be reported as exploratory there.
        if (
            self.warm_season_coal_persistence
            and "tgtlag24_coal_mw" in x
            and "target_month" in x
        ):
            month = pd.to_numeric(x["target_month"], errors="coerce").to_numpy()
            coal_d1 = pd.to_numeric(
                x["tgtlag24_coal_mw"], errors="coerce"
            ).to_numpy()
            warm = np.isin(month, np.arange(2, 11)) & np.isfinite(coal_d1)
            components["coal_mw"] = np.where(warm, np.clip(coal_d1, 0.0, None), coal_model)
        emitting_sum = (
            gas
            + components["coal_mw"]
            + components["fuel_oil_mw"]
            + components["bioenergy_mw"]
        )
        denominator = np.maximum(components["total_generation_mw"], emitting_sum)
        denominator = np.clip(denominator, 1.0, None)
        prediction = (
            EMISSION_FACTORS_GCO2_KWH["gas_mw"] * gas
            + EMISSION_FACTORS_GCO2_KWH["coal_mw"] * components["coal_mw"]
            + EMISSION_FACTORS_GCO2_KWH["fuel_oil_mw"]
            * components["fuel_oil_mw"]
            + EMISSION_FACTORS_GCO2_KWH["bioenergy_mw"]
            * components["bioenergy_mw"]
        ) / denominator
        out = pd.DataFrame(
            {
                "prediction": np.clip(prediction, 0.0, None),
                "predicted_gas_mw": gas,
                "predicted_coal_mw": components["coal_mw"],
                "predicted_coal_model_mw": coal_model,
                "predicted_fuel_oil_mw": components["fuel_oil_mw"],
                "predicted_bioenergy_mw": components["bioenergy_mw"],
                "predicted_total_generation_mw": denominator,
                "prob_baseload": probability[:, 0],
                "prob_ccg": probability[:, 1],
                "prob_peak": probability[:, 2],
            },
            index=x.index,
        )
        if self.gas_pooled_ is not None:
            gas_pooled = np.clip(self.gas_pooled_.predict(x), 0.0, None)
            pooled_denominator = np.maximum(
                components["total_generation_mw"],
                gas_pooled
                + components["coal_mw"]
                + components["fuel_oil_mw"]
                + components["bioenergy_mw"],
            )
            pooled_denominator = np.clip(pooled_denominator, 1.0, None)
            prediction_pooled = (
                EMISSION_FACTORS_GCO2_KWH["gas_mw"] * gas_pooled
                + EMISSION_FACTORS_GCO2_KWH["coal_mw"] * components["coal_mw"]
                + EMISSION_FACTORS_GCO2_KWH["fuel_oil_mw"]
                * components["fuel_oil_mw"]
                + EMISSION_FACTORS_GCO2_KWH["bioenergy_mw"]
                * components["bioenergy_mw"]
            ) / pooled_denominator
            out["prediction_pooled"] = np.clip(prediction_pooled, 0.0, None)
            out["predicted_gas_pooled_mw"] = gas_pooled
        for label, name in ((0, "baseload"), (1, "ccg"), (2, "peak")):
            out[f"gas_expert_{name}_mw"] = gas_by_regime[:, label]
        return out


__all__ = [
    "EMISSION_FACTORS_GCO2_KWH",
    "GENERATION_COLUMNS",
    "PHYSICAL_TARGET_COLUMNS",
    "PhysicalProxyMoE",
    "physical_targets",
    "proxy_training_frame",
    "rte_realtime_carbon_proxy",
]
