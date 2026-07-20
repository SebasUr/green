"""Physics-explicit model for RTE's consolidated French carbon intensity.

Unlike the provisional proxy, the consolidated target uses technology-level
factors and revisions.  This model learns non-negative effective factors from
training data only, forecasts the seven emitting components independently,
and divides their predicted emissions by predicted domestic generation.
"""

from __future__ import annotations

from collections.abc import Sequence

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
EMITTING_COMPONENTS = (
    "coal_mw",
    "fuel_oil_mw",
    "gas_turbine_mw",
    "gas_cogeneration_mw",
    "gas_ccg_mw",
    "gas_other_mw",
    "bioenergy_waste_mw",
)
GAS_COMPONENTS = (
    "gas_turbine_mw",
    "gas_cogeneration_mw",
    "gas_ccg_mw",
    "gas_other_mw",
)
TOTAL_GENERATION = "total_generation_mw"
PHYSICAL_TARGETS = (*EMITTING_COMPONENTS, TOTAL_GENERATION)


def detailed_physical_targets(frame: pd.DataFrame, target_times: Sequence) -> pd.DataFrame:
    """Align detailed source targets and domestic generation to target times."""

    missing = sorted(set(GENERATION_COLUMNS + EMITTING_COMPONENTS) - set(frame.columns))
    if missing:
        raise ValueError(f"detailed carbon data miss columns: {missing}")
    out = frame.loc[:, EMITTING_COMPONENTS].apply(pd.to_numeric, errors="coerce")
    out = out.clip(lower=0.0)
    generation = frame.loc[:, GENERATION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    generation = generation.clip(lower=0.0)
    # Keep the aggregate gas target available for the optional reconciliation
    # head.  It is almost exactly the sum of RTE's four detailed gas series,
    # but forecasting it directly can reduce mutually cancelling component
    # errors.  The default model does not consume this extra metadata column.
    out["gas_mw"] = generation["gas_mw"]
    out[TOTAL_GENERATION] = generation.sum(axis=1).where(generation.notna().all(axis=1))
    return out.reindex(pd.DatetimeIndex(target_times)).reset_index(drop=True)


class ConsolidatedPhysicalRegressor:
    """Pooled LightGBM source regressors plus train-only positive OLS factors."""

    def __init__(
        self,
        *,
        source_params: dict | None = None,
        inverse_level_floor: float = 12.0,
        ccg_moe: bool = False,
        gas_total_reconciliation: bool = False,
        random_state: int = 42,
    ) -> None:
        self.source_params = source_params or {}
        self.inverse_level_floor = float(inverse_level_floor)
        self.ccg_moe = bool(ccg_moe)
        self.gas_total_reconciliation = bool(gas_total_reconciliation)
        self.random_state = int(random_state)
        self.feature_names_: list[str] = []
        self.emission_factors_: np.ndarray | None = None
        self.regressors_: dict[str, object] = {}
        self.ccg_classifier_: object | None = None
        self.ccg_experts_: dict[int, object] = {}
        self.gas_component_shares_: np.ndarray | None = None

    @staticmethod
    def _model_classes():
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ConsolidatedPhysicalRegressor requires LightGBM") from exc
        return LGBMClassifier, LGBMRegressor

    @staticmethod
    def _positive_ols(design: np.ndarray, emissions: np.ndarray) -> np.ndarray:
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError("positive OLS requires scikit-learn") from exc
        estimator = LinearRegression(fit_intercept=False, positive=True)
        estimator.fit(design, emissions)
        return np.asarray(estimator.coef_, dtype=float)

    def fit(self, x: pd.DataFrame, meta: pd.DataFrame) -> "ConsolidatedPhysicalRegressor":
        required = {"actual", *PHYSICAL_TARGETS}
        if self.gas_total_reconciliation:
            required.add("gas_mw")
        if self.ccg_moe:
            required.add("regime")
        missing = sorted(required - set(meta.columns))
        if missing:
            raise ValueError(f"consolidated physical metadata miss columns: {missing}")
        valid = meta.loc[:, list(required)].notna().all(axis=1)
        valid &= meta["actual"].gt(0.0) & meta[TOTAL_GENERATION].gt(0.0)
        x_fit = x.loc[valid].reset_index(drop=True)
        meta_fit = meta.loc[valid].reset_index(drop=True)
        self.feature_names_ = list(x_fit.columns)
        design = meta_fit.loc[:, EMITTING_COMPONENTS].to_numpy(dtype=float)
        emissions = (
            meta_fit["actual"].to_numpy(dtype=float)
            * meta_fit[TOTAL_GENERATION].to_numpy(dtype=float)
        )
        self.emission_factors_ = self._positive_ols(design, emissions)
        gas_component_matrix = meta_fit.loc[:, GAS_COMPONENTS].to_numpy(dtype=float)
        gas_component_totals = gas_component_matrix.sum(axis=0)
        if gas_component_totals.sum() > 0.0:
            self.gas_component_shares_ = gas_component_totals / gas_component_totals.sum()
        else:  # pragma: no cover - RTE gas is positive throughout the real dataset
            self.gas_component_shares_ = np.full(len(GAS_COMPONENTS), 1.0 / len(GAS_COMPONENTS))

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
            meta_fit["actual"].to_numpy(dtype=float), self.inverse_level_floor, None
        )
        LGBMClassifier, LGBMRegressor = self._model_classes()
        self.ccg_classifier_ = None
        self.ccg_experts_ = {}
        if self.ccg_moe:
            regime = meta_fit["regime"].astype(int).to_numpy()
            classifier = LGBMClassifier(
                random_state=self.random_state + 100,
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=30,
                reg_lambda=5.0,
                verbosity=-1,
                n_jobs=1,
                class_weight="balanced",
            )
            classifier.fit(x_fit, regime)
            self.ccg_classifier_ = classifier
            for label in (0, 1, 2):
                mask = regime == label
                expert = LGBMRegressor(
                    random_state=self.random_state + 110 + label,
                    **defaults,
                )
                expert.fit(
                    x_fit.loc[mask],
                    meta_fit.loc[mask, "gas_ccg_mw"],
                    sample_weight=sample_weight[mask],
                )
                self.ccg_experts_[label] = expert
        self.regressors_ = {}
        regressor_targets = list(PHYSICAL_TARGETS)
        if self.gas_total_reconciliation:
            # Append this target so enabling reconciliation does not change the
            # random seeds (and therefore fits) of any incumbent source head.
            regressor_targets.append("gas_mw")
        for offset, column in enumerate(regressor_targets):
            if self.ccg_moe and column == "gas_ccg_mw":
                continue
            estimator = LGBMRegressor(
                random_state=self.random_state + offset,
                **defaults,
            )
            estimator.fit(
                x_fit,
                meta_fit[column],
                sample_weight=sample_weight,
            )
            self.regressors_[column] = estimator
        return self

    def predict_matrix(self, x: pd.DataFrame) -> pd.DataFrame:
        expected_regressors = (
            len(PHYSICAL_TARGETS)
            - int(self.ccg_moe)
            + int(self.gas_total_reconciliation)
        )
        if self.emission_factors_ is None or len(self.regressors_) != expected_regressors:
            raise RuntimeError("ConsolidatedPhysicalRegressor used before fit")
        x = x.reindex(columns=self.feature_names_)
        predicted = {
            column: np.clip(estimator.predict(x), 0.0, None)
            for column, estimator in self.regressors_.items()
        }
        probability = None
        ccg_by_regime = None
        if self.ccg_moe:
            if self.ccg_classifier_ is None or len(self.ccg_experts_) != 3:
                raise RuntimeError("CCG MoE used before fit")
            raw_probability = self.ccg_classifier_.predict_proba(x)
            probability = np.zeros((len(x), 3), dtype=float)
            for position, label in enumerate(self.ccg_classifier_.classes_):
                probability[:, int(label)] = raw_probability[:, position]
            ccg_by_regime = np.column_stack(
                [
                    np.clip(self.ccg_experts_[label].predict(x), 0.0, None)
                    for label in (0, 1, 2)
                ]
            )
            predicted["gas_ccg_mw"] = np.sum(probability * ccg_by_regime, axis=1)
        raw_components = np.column_stack(
            [predicted[column] for column in EMITTING_COMPONENTS]
        )
        raw_emitting_sum = raw_components.sum(axis=1)
        raw_denominator = np.maximum(
            predicted[TOTAL_GENERATION], raw_emitting_sum
        )
        raw_denominator = np.clip(raw_denominator, 1.0, None)
        raw_intensity = raw_components @ self.emission_factors_ / raw_denominator

        raw_gas_sum = None
        if self.gas_total_reconciliation:
            if self.gas_component_shares_ is None:
                raise RuntimeError("gas reconciliation used before fit")
            gas_total = predicted["gas_mw"]
            raw_gas = np.column_stack([predicted[column] for column in GAS_COMPONENTS])
            raw_gas_sum = raw_gas.sum(axis=1)
            reconciled_gas = np.empty_like(raw_gas)
            usable = raw_gas_sum > 1e-9
            reconciled_gas[usable] = raw_gas[usable] * (
                gas_total[usable] / raw_gas_sum[usable]
            )[:, None]
            reconciled_gas[~usable] = (
                gas_total[~usable, None] * self.gas_component_shares_[None, :]
            )
            for position, column in enumerate(GAS_COMPONENTS):
                predicted[column] = reconciled_gas[:, position]

        components = np.column_stack([predicted[column] for column in EMITTING_COMPONENTS])
        emitting_sum = components.sum(axis=1)
        denominator = np.maximum(predicted[TOTAL_GENERATION], emitting_sum)
        denominator = np.clip(denominator, 1.0, None)
        intensity = components @ self.emission_factors_ / denominator
        out = pd.DataFrame(
            {
                "prediction": np.clip(intensity, 0.0, None),
                "predicted_total_generation_mw": denominator,
            },
            index=x.index,
        )
        if self.gas_total_reconciliation:
            out["prediction_unreconciled"] = np.clip(raw_intensity, 0.0, None)
            out["predicted_gas_total_mw"] = predicted["gas_mw"]
            out["predicted_gas_components_raw_sum_mw"] = raw_gas_sum
            for column in GAS_COMPONENTS:
                raw_position = EMITTING_COMPONENTS.index(column)
                out[f"predicted_raw_{column}"] = raw_components[:, raw_position]
        for column in EMITTING_COMPONENTS:
            out[f"predicted_{column}"] = predicted[column]
        if probability is not None and ccg_by_regime is not None:
            for label, name in ((0, "baseload"), (1, "ccg"), (2, "peak")):
                out[f"ccg_prob_{name}"] = probability[:, label]
                out[f"ccg_expert_{name}_mw"] = ccg_by_regime[:, label]
        return out


__all__ = [
    "ConsolidatedPhysicalRegressor",
    "EMITTING_COMPONENTS",
    "GAS_COMPONENTS",
    "GENERATION_COLUMNS",
    "PHYSICAL_TARGETS",
    "TOTAL_GENERATION",
    "detailed_physical_targets",
]
