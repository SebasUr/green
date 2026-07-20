"""Probabilistic French fossil-regime expert for dense day-ahead forecasting.

The model targets the information bottleneck found in the France-24 analysis:
future dispatchable gas.  It learns three operational regimes from observed
CCG/TAC/other gas, predicts their probabilities from origin-safe day-ahead
features, estimates emitting generation shares conditionally on the regime,
and maps those shares back to carbon intensity.

One shared horizon-conditioned expert is used for all 24 target hours.  This is
both faster and statistically more efficient than training 24 independent sets
of regime/source models.  Residual and decision-risk calibration use predictions
from a trailing block that the component models did not see.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.france24 import (
    DENSE_DAY_AHEAD_HORIZONS,
    FranceDayAheadFeatureBuilder,
    france24_feature_builder_from_config,
)
from green_observatory.carbon.model import _make_estimator
from green_observatory.carbon.physical import PhysicalCarbonMapper, generation_shares
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON

REGIME_NAMES = {0: "baseload", 1: "ccg", 2: "peak"}
DISPATCHABLE_GAS_COLUMNS = ("gas_ccg_mw", "gas_turbine_mw", "gas_other_mw")


def dispatchable_gas(frame: pd.DataFrame) -> pd.Series:
    """Observed CCG/TAC/other gas MW used only to construct supervised labels."""
    missing = [column for column in DISPATCHABLE_GAS_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"fossil-regime frame is missing columns: {missing}")
    return frame.loc[:, DISPATCHABLE_GAS_COLUMNS].sum(axis=1, min_count=1)


def fossil_regime_labels(
    frame: pd.DataFrame, *, ccg_threshold_mw: float, peak_threshold_mw: float
) -> pd.Series:
    if not 0.0 <= ccg_threshold_mw < peak_threshold_mw:
        raise ValueError("regime thresholds must satisfy 0 <= ccg < peak")
    gas = dispatchable_gas(frame)
    labels = pd.Series(0, index=frame.index, dtype="int8", name="fossil_regime")
    labels.loc[gas >= ccg_threshold_mw] = 1
    labels.loc[gas >= peak_threshold_mw] = 2
    return labels.where(gas.notna())


class FossilRegimeModel:
    """Shared 1..24h probabilistic regime and emitting-share expert."""

    def __init__(
        self,
        feature_builder: FranceDayAheadFeatureBuilder,
        *,
        horizons: Sequence[int] = DENSE_DAY_AHEAD_HORIZONS,
        generation_columns: Sequence[str] = (
            "nuclear_mw", "gas_mw", "coal_mw", "fuel_oil_mw", "wind_mw",
            "solar_mw", "hydro_mw", "bioenergy_mw",
        ),
        share_columns: Sequence[str] = (
            "gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw",
        ),
        ccg_threshold_mw: float = 500.0,
        peak_threshold_mw: float = 2500.0,
        calibration_fraction: float = 0.25,
        training_stride_hours: int = 6,
        classifier_params: dict | None = None,
        source_params: dict | None = None,
        residual_params: dict | None = None,
        ranker_params: dict | None = None,
        point_scale_grid: Sequence[float] = (0.90, 0.95, 1.0, 1.05, 1.10),
        risk_weight_grid: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
        ranking_weight_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        availability_store: RteAvailabilityFeatureStore | None = None,
        availability_feature_mode: str = "delta",
        rte_forecast_store: RteGenerationForecastFeatureStore | None = None,
        random_state: int = 42,
    ) -> None:
        if not 0.15 <= calibration_fraction <= 0.40:
            raise ValueError("calibration_fraction must be in [0.15, 0.40]")
        self.feature_builder = feature_builder
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.generation_columns = tuple(generation_columns)
        self.source_columns = tuple(share_columns)
        self.share_names = tuple(f"{column}_share" for column in self.source_columns)
        self.ccg_threshold_mw = float(ccg_threshold_mw)
        self.peak_threshold_mw = float(peak_threshold_mw)
        self.calibration_fraction = float(calibration_fraction)
        self.training_stride_hours = int(training_stride_hours)
        self.classifier_params = classifier_params or {}
        self.source_params = source_params or {}
        self.residual_params = residual_params or {}
        self.ranker_params = ranker_params or {}
        self.point_scale_grid = tuple(float(value) for value in point_scale_grid)
        self.risk_weight_grid = tuple(float(value) for value in risk_weight_grid)
        self.ranking_weight_grid = tuple(float(value) for value in ranking_weight_grid)
        self.availability_store = availability_store
        if availability_feature_mode not in {"delta", "all"}:
            raise ValueError("availability_feature_mode must be delta or all")
        self.availability_feature_mode = availability_feature_mode
        self.rte_forecast_store = rte_forecast_store
        self.random_state = int(random_state)

        self.feature_names_: list[str] = []
        self.classifier_: object | None = None
        self.gas_estimators_: dict[int, object] = {}
        self.other_estimators_: dict[str, object] = {}
        self.residual_estimator_: object | None = None
        self.residual_feature_names_: list[str] = []
        self.ranker_: object | None = None
        self.candidate_feature_names_: list[str] = []
        self.ranker_feature_names_: list[str] = []
        self.mapper = PhysicalCarbonMapper(self.share_names)
        self.point_scale_: float = 1.0
        self.risk_weight_: float = 0.0
        self.ranking_weight_: float = 0.0
        self.validation_regime_accuracy_: float | None = None
        self.validation_peak_recall_: float | None = None
        self.validation_mape_: float | None = None
        self.validation_regret_: float | None = None
        self.validation_ranked_regret_: float | None = None
        self.regime_counts_: dict[str, int] = {}

    def _origin_grid(self, frame: pd.DataFrame) -> pd.DatetimeIndex:
        warmup = max(720, max(self.horizons))
        last = len(frame) - max(self.horizons)
        return frame.index[warmup:last:self.training_stride_hours]

    def _matrix(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        *,
        supervised: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        origin_features = self.feature_builder.origin_features(frame).reindex(origins)
        availability = (
            self.availability_store.features_by_horizon(origins, self.horizons)
            if self.availability_store is not None
            else {}
        )
        rte_forecasts = (
            self.rte_forecast_store.features_by_horizon(origins, self.horizons)
            if self.rte_forecast_store is not None
            else {}
        )
        shares = generation_shares(
            frame,
            generation_columns=self.generation_columns,
            share_columns=self.source_columns,
        )
        regimes = fossil_regime_labels(
            frame,
            ccg_threshold_mw=self.ccg_threshold_mw,
            peak_threshold_mw=self.peak_threshold_mw,
        )
        x_parts: list[pd.DataFrame] = []
        meta_parts: list[pd.DataFrame] = []
        for horizon in self.horizons:
            target_times = origins + pd.Timedelta(hours=horizon)
            target = self.feature_builder.target_block(
                target_times, horizon, index=origins
            )
            x = pd.concat([target, origin_features], axis=1)
            if horizon in availability:
                availability_block = availability[horizon]
                if self.availability_feature_mode == "delta":
                    availability_block = availability_block.loc[
                        :,
                        [
                            column
                            for column in availability_block
                            if column.endswith("_delta_mw")
                        ],
                    ]
                x = pd.concat([x, availability_block], axis=1)
                nuclear_delta = "rte_tgt_nuclear_unavailable_delta_mw"
                if nuclear_delta in x and "nuclear_mw" in origin_features:
                    x["rte_tgt_nuclear_output_proxy_mw"] = (
                        origin_features["nuclear_mw"] - x[nuclear_delta]
                    )
            if horizon in rte_forecasts:
                x = pd.concat([x, rte_forecasts[horizon]], axis=1)
                if (
                    "fr_tgt_load_day_ahead_mw" in x
                    and "rte_tgt_variable_renewables_d1_mw" in x
                ):
                    x["rte_tgt_residual_load_d1_mw"] = (
                        x["fr_tgt_load_day_ahead_mw"]
                        - x["rte_tgt_variable_renewables_d1_mw"]
                    )
            x["fr_regime_horizon_hours"] = float(horizon)
            x["fr_regime_horizon_sin"] = np.sin(2 * np.pi * horizon / 24.0)
            x["fr_regime_horizon_cos"] = np.cos(2 * np.pi * horizon / 24.0)
            x_parts.append(x.reset_index(drop=True))
            meta = pd.DataFrame(
                {
                    "origin": origins,
                    "horizon": horizon,
                    "target_time": target_times,
                }
            )
            if supervised:
                meta["actual"] = frame[CARBON].reindex(target_times).to_numpy()
                meta["regime"] = regimes.reindex(target_times).to_numpy()
                for name in self.share_names:
                    meta[name] = shares[name].reindex(target_times).to_numpy()
            meta_parts.append(meta)
        x_all = pd.concat(x_parts, ignore_index=True)
        meta_all = pd.concat(meta_parts, ignore_index=True)
        if supervised:
            valid = (
                meta_all[["actual", "regime", *self.share_names]].notna().all(axis=1)
            )
            x_all = x_all.loc[valid].reset_index(drop=True)
            meta_all = meta_all.loc[valid].reset_index(drop=True)
        return x_all, meta_all

    def _make_classifier(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = {
            key: value
            for key, value in self.classifier_params.items()
            if value is not None
        }
        return HistGradientBoostingClassifier(
            random_state=self.random_state, **params
        )

    @staticmethod
    def _classification_weight(regime: np.ndarray, actual: np.ndarray) -> np.ndarray:
        values, counts = np.unique(regime, return_counts=True)
        balance = {value: len(regime) / (len(values) * count) for value, count in zip(values, counts)}
        class_weight = np.asarray([balance[value] for value in regime], dtype=float)
        carbon_weight = 1.0 + np.clip((actual - np.median(actual)) / 15.0, 0.0, 3.0)
        return class_weight * carbon_weight

    def _fit_components(
        self, x: pd.DataFrame, meta: pd.DataFrame
    ) -> tuple[object, dict[int, object], dict[str, object]]:
        classifier = self._make_classifier()
        regime = meta["regime"].astype(int).to_numpy()
        classifier.fit(
            x,
            regime,
            sample_weight=self._classification_weight(
                regime, meta["actual"].to_numpy(dtype=float)
            ),
        )

        gas_estimators: dict[int, object] = {}
        gas_name = "gas_mw_share"
        for label in sorted(REGIME_NAMES):
            mask = regime == label
            estimator = _make_estimator(
                "hist_gradient_boosting",
                self.source_params,
                self.random_state + 10 + label,
            )
            estimator.fit(x.loc[mask], meta.loc[mask, gas_name])
            gas_estimators[label] = estimator

        other_estimators: dict[str, object] = {}
        for offset, name in enumerate(self.share_names):
            if name == gas_name:
                continue
            estimator = _make_estimator(
                "hist_gradient_boosting",
                self.source_params,
                self.random_state + 20 + offset,
            )
            estimator.fit(x, meta[name])
            other_estimators[name] = estimator
        return classifier, gas_estimators, other_estimators

    def _component_predictions(
        self,
        x: pd.DataFrame,
        classifier,
        gas_estimators: dict[int, object],
        other_estimators: dict[str, object],
        mapper: PhysicalCarbonMapper,
    ) -> pd.DataFrame:
        probabilities = classifier.predict_proba(x)
        probability_frame = pd.DataFrame(0.0, index=x.index, columns=[0, 1, 2])
        for position, label in enumerate(classifier.classes_):
            probability_frame[int(label)] = probabilities[:, position]
        gas_by_regime = pd.DataFrame(
            {
                label: np.clip(estimator.predict(x), 0.0, 1.0)
                for label, estimator in gas_estimators.items()
            },
            index=x.index,
        )
        gas_expected = (probability_frame * gas_by_regime).sum(axis=1)
        gas_variance = (
            probability_frame
            * gas_by_regime.sub(gas_expected, axis=0).pow(2)
        ).sum(axis=1)
        shares = pd.DataFrame(index=x.index)
        shares["gas_mw_share"] = gas_expected
        for name, estimator in other_estimators.items():
            shares[name] = np.clip(estimator.predict(x), 0.0, 1.0)
        physical = mapper.predict(shares)
        gas_factor = mapper.coefficients_.get("gas_mw_share", 390.0)
        out = shares.add_prefix("predicted_")
        out["prob_baseload"] = probability_frame[0]
        out["prob_ccg"] = probability_frame[1]
        out["prob_peak"] = probability_frame[2]
        out["regime_prediction"] = probability_frame.idxmax(axis=1).astype(int)
        out["physical_prediction"] = physical
        out["regime_uncertainty_gco2"] = np.sqrt(gas_variance) * gas_factor
        return out

    @staticmethod
    def _residual_matrix(x: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([x.reset_index(drop=True), components.reset_index(drop=True)], axis=1)

    @staticmethod
    def _mape(actual: np.ndarray, prediction: np.ndarray) -> float:
        return float(np.mean(np.abs(prediction - actual) / np.clip(np.abs(actual), 1e-9, None)))

    @staticmethod
    def _mean_regret(frame: pd.DataFrame, score_column: str) -> float:
        regrets: list[float] = []
        for _, group in frame.groupby("origin", sort=False):
            selected = group[score_column].idxmin()
            regrets.append(float(frame.at[selected, "actual"] - group["actual"].min()))
        return float(np.mean(regrets))

    @staticmethod
    def _candidate_features(
        x: pd.DataFrame,
        meta: pd.DataFrame,
        components: pd.DataFrame,
        point_prediction: np.ndarray,
        decision_prediction: np.ndarray,
    ) -> pd.DataFrame:
        """Compact candidate state used by the within-day pairwise ranker.

        The ranker intentionally sees target-time forecasts and the fossil
        expert's own outputs, rather than the full high-dimensional lag matrix.
        That keeps the pairwise problem statistically tractable and ensures
        every feature is available at the forecast origin.
        """
        features = pd.DataFrame(index=x.index)
        features["point_prediction"] = point_prediction
        features["direct_prediction"] = decision_prediction
        features["horizon"] = meta["horizon"].to_numpy(dtype=float)
        for column in (
            "prob_baseload",
            "prob_ccg",
            "prob_peak",
            "predicted_gas_mw_share",
            "predicted_coal_mw_share",
            "predicted_fuel_oil_mw_share",
            "predicted_bioenergy_mw_share",
            "physical_prediction",
            "regime_uncertainty_gco2",
        ):
            if column in components:
                features[column] = components[column].to_numpy(dtype=float)
        target_tokens = (
            "fr_tgt_",
            "rte_tgt_",
            "target_hour_sin",
            "target_hour_cos",
            "target_dow_sin",
            "target_dow_cos",
            "fr_regime_horizon_sin",
            "fr_regime_horizon_cos",
        )
        for column in x:
            if column not in features and any(token in column for token in target_tokens):
                features[column] = x[column].to_numpy(dtype=float)
        return features.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _pairwise_dataset(
        features: pd.DataFrame, meta: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for _, group in meta.groupby("origin", sort=False):
            for left, right in itertools.combinations(list(group.index), 2):
                gap = float(meta.at[right, "actual"] - meta.at[left, "actual"])
                if not np.isfinite(gap) or abs(gap) < 1e-9:
                    continue
                difference = (
                    features.loc[left].to_numpy(dtype=float)
                    - features.loc[right].to_numpy(dtype=float)
                )
                weight = float(np.clip(abs(gap), 1.0, 30.0))
                label = int(gap > 0.0)  # left candidate is greener
                rows.extend((difference, -difference))
                labels.extend((label, 1 - label))
                weights.extend((weight, weight))
        columns = [f"delta_{column}" for column in features.columns]
        return (
            pd.DataFrame(rows, columns=columns),
            np.asarray(labels, dtype=int),
            np.asarray(weights, dtype=float),
        )

    def _pair_scores(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
        classifier=None,
    ) -> pd.Series:
        classifier = classifier or self.ranker_
        if classifier is None:
            raise RuntimeError("fossil-regime ranker has not been fitted")
        differences: list[np.ndarray] = []
        pairs: list[tuple[int, int]] = []
        for _, group in meta.groupby("origin", sort=False):
            for left, right in itertools.combinations(list(group.index), 2):
                difference = (
                    features.loc[left].to_numpy(dtype=float)
                    - features.loc[right].to_numpy(dtype=float)
                )
                differences.extend((difference, -difference))
                pairs.append((left, right))
        scores = pd.Series(0.0, index=meta.index)
        counts = pd.Series(0, index=meta.index, dtype=int)
        if not differences:
            return scores + 0.5
        pair_x = pd.DataFrame(differences, columns=self.ranker_feature_names_)
        probabilities = classifier.predict_proba(pair_x)
        positive = list(classifier.classes_).index(1)
        for pair_index, (left, right) in enumerate(pairs):
            forward = float(probabilities[2 * pair_index, positive])
            reverse = float(probabilities[2 * pair_index + 1, positive])
            probability = 0.5 * (forward + 1.0 - reverse)
            scores[left] += probability
            scores[right] += 1.0 - probability
            counts[left] += 1
            counts[right] += 1
        return scores / counts.clip(lower=1)

    @staticmethod
    def _direct_scores(meta: pd.DataFrame, decision: np.ndarray) -> pd.Series:
        scores = pd.Series(index=meta.index, dtype=float)
        direct = pd.Series(decision, index=meta.index)
        for _, group in meta.groupby("origin", sort=False):
            n = len(group)
            rank = direct.loc[group.index].rank(method="average", ascending=True)
            scores.loc[group.index] = 1.0 - (rank - 1.0) / max(1, n - 1)
        return scores

    @staticmethod
    def _regret_from_greener_score(
        meta: pd.DataFrame, greener_score: pd.Series
    ) -> float:
        regrets: list[float] = []
        for _, group in meta.groupby("origin", sort=False):
            selected = greener_score.loc[group.index].idxmax()
            regrets.append(
                float(meta.at[selected, "actual"] - group["actual"].min())
            )
        return float(np.mean(regrets))

    @staticmethod
    def _ranked_values(
        meta: pd.DataFrame, point: np.ndarray, greener_score: pd.Series
    ) -> np.ndarray:
        """Reorder point values so the decision metric follows the ranker."""
        ranked = np.asarray(point, dtype=float).copy()
        for _, group in meta.groupby("origin", sort=False):
            positions = list(group.index)
            greener_order = sorted(
                positions, key=lambda position: greener_score[position], reverse=True
            )
            point_values = np.sort(np.asarray(point)[positions])
            for rank, position in enumerate(greener_order):
                ranked[position] = point_values[rank]
        return ranked

    def fit(self, train_frame: pd.DataFrame) -> FossilRegimeModel:
        origins = self._origin_grid(train_frame)
        x, meta = self._matrix(train_frame, origins, supervised=True)
        self.feature_names_ = list(x.columns)
        split_time = train_frame.index[int(len(train_frame) * (1.0 - self.calibration_fraction))]
        base_mask = meta["origin"] < split_time
        calibration_mask = ~base_mask
        min_base = max(500, len(self.horizons) * 100)
        min_calibration = max(200, len(self.horizons) * 40)
        if int(base_mask.sum()) < min_base or int(calibration_mask.sum()) < min_calibration:
            raise ValueError("not enough stacked rows for fossil-regime calibration")

        base_x = x.loc[base_mask].reset_index(drop=True)
        base_meta = meta.loc[base_mask].reset_index(drop=True)
        calibration_x = x.loc[calibration_mask].reset_index(drop=True)
        calibration_meta = meta.loc[calibration_mask].reset_index(drop=True)
        base_mapper = PhysicalCarbonMapper(self.share_names).fit(
            base_meta.loc[:, self.share_names], base_meta["actual"]
        )
        base_classifier, base_gas, base_other = self._fit_components(base_x, base_meta)
        calibration_components = self._component_predictions(
            calibration_x, base_classifier, base_gas, base_other, base_mapper
        )
        residual_x = self._residual_matrix(calibration_x, calibration_components)
        residual_y = (
            calibration_meta["actual"].to_numpy()
            - calibration_components["physical_prediction"].to_numpy()
        )
        calibration_origins = pd.Index(calibration_meta["origin"].drop_duplicates())
        residual_fit_origins = set(calibration_origins[: len(calibration_origins) // 2])
        decision_origins = calibration_origins[len(calibration_origins) // 2 :]
        validation_size = max(10, len(decision_origins) // 3)
        validation_size = min(validation_size, len(decision_origins) // 2)
        ranking_fit_origins = set(decision_origins[:-validation_size])
        validation_origins = set(decision_origins[-validation_size:])
        residual_fit_mask = calibration_meta["origin"].isin(residual_fit_origins).to_numpy()
        ranking_fit_mask = calibration_meta["origin"].isin(ranking_fit_origins).to_numpy()
        validation_mask = calibration_meta["origin"].isin(validation_origins).to_numpy()
        residual_estimator = _make_estimator(
            "hist_gradient_boosting", self.residual_params, self.random_state + 100
        )
        # Inverse-level weighting approximates MAPE while clipping low French
        # carbon hours to avoid a tiny-denominator objective.
        residual_estimator.fit(
            residual_x.loc[residual_fit_mask],
            residual_y[residual_fit_mask],
            sample_weight=1.0
            / np.clip(
                calibration_meta.loc[residual_fit_mask, "actual"].to_numpy(),
                8.0,
                None,
            ),
        )
        calibration_point = np.clip(
            calibration_components["physical_prediction"].to_numpy()
            + residual_estimator.predict(residual_x),
            0.0,
            None,
        )

        # Tune scale and uncertainty aversion on an OOS decision block.  Its
        # trailing slice remains untouched for ranker/blend validation.
        actual = calibration_meta["actual"].to_numpy(dtype=float)
        self.point_scale_ = min(
            self.point_scale_grid,
            key=lambda scale: (
                self._mape(
                    actual[ranking_fit_mask],
                    scale * calibration_point[ranking_fit_mask],
                ),
                abs(scale - 1.0),
            ),
        )
        scaled_point = self.point_scale_ * calibration_point
        scored = calibration_meta.loc[
            ranking_fit_mask, ["origin", "horizon", "actual"]
        ].copy()
        scored["point_prediction"] = scaled_point[ranking_fit_mask]
        scored["uncertainty"] = calibration_components.loc[
            ranking_fit_mask, "regime_uncertainty_gco2"
        ].to_numpy()
        regrets: dict[float, float] = {}
        for weight in self.risk_weight_grid:
            scored["decision_score"] = (
                scored["point_prediction"] + weight * scored["uncertainty"]
            )
            regrets[weight] = self._mean_regret(scored, "decision_score")
        self.risk_weight_ = min(regrets, key=lambda weight: (regrets[weight], weight))
        decision = scaled_point + self.risk_weight_ * calibration_components[
            "regime_uncertainty_gco2"
        ].to_numpy()

        # Pairwise greener-than comparisons are weighted by the realized carbon
        # gap.  Alpha=0 remains available, so validation can reject the ranker.
        from sklearn.ensemble import HistGradientBoostingClassifier

        candidate_features = self._candidate_features(
            calibration_x,
            calibration_meta,
            calibration_components,
            scaled_point,
            decision,
        )
        self.candidate_feature_names_ = list(candidate_features.columns)
        ranking_meta = calibration_meta.loc[ranking_fit_mask]
        pair_x, pair_y, pair_weight = self._pairwise_dataset(
            candidate_features.loc[ranking_fit_mask], ranking_meta
        )
        if len(pair_x) < 100 or len(np.unique(pair_y)) < 2:
            raise ValueError("not enough non-tied pairs for fossil-regime ranking")
        self.ranker_feature_names_ = list(pair_x.columns)
        ranker_params = {
            key: value for key, value in self.ranker_params.items() if value is not None
        }
        tuning_ranker = HistGradientBoostingClassifier(
            random_state=self.random_state + 200, **ranker_params
        )
        tuning_ranker.fit(pair_x, pair_y, sample_weight=pair_weight)
        validation_meta = calibration_meta.loc[validation_mask]
        validation_features = candidate_features.loc[validation_mask]
        pair_score = self._pair_scores(
            validation_features, validation_meta, tuning_ranker
        )
        direct_score = self._direct_scores(
            validation_meta, decision[validation_mask]
        )
        ranking_regrets: dict[float, float] = {}
        for weight in self.ranking_weight_grid:
            combined_score = (1.0 - weight) * direct_score + weight * pair_score
            ranking_regrets[weight] = self._regret_from_greener_score(
                validation_meta, combined_score
            )
        self.ranking_weight_ = min(
            ranking_regrets,
            key=lambda weight: (ranking_regrets[weight], weight),
        )
        self.validation_regret_ = ranking_regrets[0.0]
        self.validation_ranked_regret_ = ranking_regrets[self.ranking_weight_]
        self.validation_mape_ = 100.0 * self._mape(
            actual[validation_mask], scaled_point[validation_mask]
        )
        regime_actual = calibration_meta.loc[
            validation_mask, "regime"
        ].astype(int).to_numpy()
        regime_predicted = calibration_components.loc[
            validation_mask, "regime_prediction"
        ].astype(int).to_numpy()
        self.validation_regime_accuracy_ = float(np.mean(regime_actual == regime_predicted))
        peak = regime_actual == 2
        self.validation_peak_recall_ = float(
            np.mean(regime_predicted[peak] == 2) if peak.any() else np.nan
        )

        # The deployable residual may now use the complete OOS calibration block;
        # scale/risk choices above remain based on the untouched later half.
        final_residual = _make_estimator(
            "hist_gradient_boosting", self.residual_params, self.random_state + 100
        )
        final_residual.fit(
            residual_x,
            residual_y,
            sample_weight=1.0 / np.clip(actual, 8.0, None),
        )

        decision_mask = ranking_fit_mask | validation_mask
        final_pair_x, final_pair_y, final_pair_weight = self._pairwise_dataset(
            candidate_features.loc[decision_mask],
            calibration_meta.loc[decision_mask],
        )
        self.ranker_ = HistGradientBoostingClassifier(
            random_state=self.random_state + 200, **ranker_params
        )
        self.ranker_.fit(
            final_pair_x, final_pair_y, sample_weight=final_pair_weight
        )

        # Refit component models and physical map on every pre-test stacked row.
        self.mapper.fit(meta.loc[:, self.share_names], meta["actual"])
        self.classifier_, self.gas_estimators_, self.other_estimators_ = self._fit_components(x, meta)
        self.residual_estimator_ = final_residual
        self.residual_feature_names_ = list(residual_x.columns)
        counts = meta["regime"].astype(int).value_counts().to_dict()
        self.regime_counts_ = {
            REGIME_NAMES[label]: int(counts.get(label, 0)) for label in REGIME_NAMES
        }
        return self

    def predict_batch(
        self, frame: pd.DataFrame, origins: pd.DatetimeIndex
    ) -> pd.DataFrame:
        if (
            self.classifier_ is None
            or self.residual_estimator_ is None
            or self.ranker_ is None
        ):
            raise RuntimeError("FossilRegimeModel.predict_batch called before fit")
        x, meta = self._matrix(frame, origins, supervised=False)
        x = x.reindex(columns=self.feature_names_)
        components = self._component_predictions(
            x, self.classifier_, self.gas_estimators_, self.other_estimators_, self.mapper
        )
        residual_x = self._residual_matrix(x, components).reindex(
            columns=self.residual_feature_names_
        )
        point = np.clip(
            self.point_scale_
            * (
                components["physical_prediction"].to_numpy()
                + self.residual_estimator_.predict(residual_x)
            ),
            0.0,
            None,
        )
        decision = (
            point
            + self.risk_weight_
            * components["regime_uncertainty_gco2"].to_numpy()
        )
        candidate_features = self._candidate_features(
            x, meta, components, point, decision
        ).reindex(columns=self.candidate_feature_names_)
        pair_score = self._pair_scores(candidate_features, meta)
        direct_score = self._direct_scores(meta, decision)
        ranking_score = (
            (1.0 - self.ranking_weight_) * direct_score
            + self.ranking_weight_ * pair_score
        )

        out = meta.copy()
        out["point_prediction"] = point
        out["decision_prediction"] = decision
        out["ranked_prediction"] = self._ranked_values(
            meta, point, ranking_score
        )
        out["ranking_score"] = ranking_score.to_numpy()
        output_columns = [
            "prob_baseload", "prob_ccg", "prob_peak", "regime_prediction",
            "physical_prediction",
            "regime_uncertainty_gco2",
        ]
        output_columns.extend(
            column
            for column in components.columns
            if column.startswith("predicted_")
        )
        for column in output_columns:
            out[column] = components[column].to_numpy()
        return out

    def save(self, path) -> None:
        import pathlib

        import joblib

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> FossilRegimeModel:
        import joblib

        model = joblib.load(path)
        if not isinstance(model, FossilRegimeModel):
            raise TypeError("artifact is not a FossilRegimeModel")
        return model


def train_fossil_regime_model(
    train_frame: pd.DataFrame,
    carbon_cfg: dict,
    *,
    climatology=None,
    forecast_frame: pd.DataFrame | None = None,
    availability_store: RteAvailabilityFeatureStore | None = None,
    availability_feature_mode: str | None = None,
    rte_forecast_store: RteGenerationForecastFeatureStore | None = None,
) -> FossilRegimeModel:
    cfg = carbon_cfg.get("fossil_regime_model", {})
    model = FossilRegimeModel(
        france24_feature_builder_from_config(
            carbon_cfg, climatology=climatology, forecast_frame=forecast_frame
        ),
        horizons=cfg.get("horizons_hours", DENSE_DAY_AHEAD_HORIZONS),
        ccg_threshold_mw=cfg.get("ccg_threshold_mw", 500.0),
        peak_threshold_mw=cfg.get("peak_threshold_mw", 2500.0),
        calibration_fraction=cfg.get("calibration_fraction", 0.25),
        training_stride_hours=cfg.get("training_stride_hours", 6),
        classifier_params=cfg.get("classifier_hist_gradient_boosting", {}),
        source_params=cfg.get("source_hist_gradient_boosting", {}),
        residual_params=cfg.get("residual_hist_gradient_boosting", {}),
        ranker_params=cfg.get("ranker_hist_gradient_boosting", {}),
        point_scale_grid=cfg.get("point_scale_grid", (0.90, 0.95, 1.0, 1.05, 1.10)),
        risk_weight_grid=cfg.get("risk_weight_grid", (0.0, 0.25, 0.5, 1.0)),
        ranking_weight_grid=cfg.get(
            "ranking_weight_grid", (0.0, 0.25, 0.5, 0.75, 1.0)
        ),
        availability_store=availability_store,
        availability_feature_mode=(
            availability_feature_mode
            or cfg.get("availability_feature_mode", "delta")
        ),
        rte_forecast_store=rte_forecast_store,
        random_state=cfg.get("random_state", 42),
    )
    return model.fit(train_frame)


__all__ = [
    "FossilRegimeModel",
    "dispatchable_gas",
    "fossil_regime_labels",
    "train_fossil_regime_model",
]
