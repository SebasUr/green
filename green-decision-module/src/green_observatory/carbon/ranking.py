"""Decision-trained Phase-D ranking model.

The point forecaster remains a normal direct multi-horizon carbon model.  A
separate pairwise selector then learns which of two candidate hours is greener.
Pairs are weighted by their realized carbon gap, so confusing 10 with 30
gCO2/kWh matters much more than confusing 10 with 11.  The selector is trained
only on a trailing calibration block using out-of-sample point predictions.

At inference, the selector changes only the ordering: it reassigns the point
model's own predicted values according to the learned pairwise ranking.  This
keeps predictions on the carbon scale and makes the Phase-D strategy directly
comparable with the untouched baseline.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.model import ProjectCarbonModel, train_project_model
from green_observatory.providers.carbon_base import CARBON


def _point_batch(
    model: ProjectCarbonModel,
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    origin_features = model.feature_builder.origin_features(frame).reindex(origins)
    parts: list[pd.DataFrame] = []
    use_horizons = model.horizons if horizons is None else tuple(map(int, horizons))
    for horizon in use_horizons:
        if horizon not in model.estimators_:
            continue
        target = model.feature_builder.target_block(
            origins + pd.Timedelta(hours=horizon), horizon, index=origins
        )
        x = pd.concat([target, origin_features], axis=1)
        x = x.reindex(columns=model.feature_names_[horizon])
        parts.append(
            pd.DataFrame(
                {
                    "origin": origins,
                    "horizon": horizon,
                    "target_time": origins + pd.Timedelta(hours=horizon),
                    "direct_prediction": np.clip(
                        model.estimators_[horizon].predict(x), 0.0, None
                    ),
                }
            )
        )
    if not parts:
        return pd.DataFrame(
            columns=["origin", "horizon", "target_time", "direct_prediction"]
        )
    return pd.concat(parts, ignore_index=True)


class RegretRankingModel:
    """Direct point model plus pairwise regret-weighted candidate selector."""

    def __init__(
        self,
        *,
        horizons: Sequence[int],
        classifier_params: dict | None = None,
        calibration_fraction: float = 0.20,
        calibration_stride_hours: int = 3,
        ranking_validation_fraction: float = 0.25,
        ranking_weight_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        regret_weight_floor: float = 1.0,
        regret_weight_cap: float = 30.0,
        random_state: int = 42,
    ) -> None:
        if not 0.10 <= calibration_fraction <= 0.5:
            raise ValueError("calibration_fraction must be in [0.10, 0.5]")
        if not 0.10 <= ranking_validation_fraction <= 0.5:
            raise ValueError("ranking_validation_fraction must be in [0.10, 0.5]")
        if not ranking_weight_grid or any(
            value < 0.0 or value > 1.0 for value in ranking_weight_grid
        ):
            raise ValueError("ranking_weight_grid must contain values in [0, 1]")
        self.horizons = tuple(map(int, horizons))
        self.classifier_params = classifier_params or {}
        self.calibration_fraction = float(calibration_fraction)
        self.calibration_stride_hours = int(calibration_stride_hours)
        self.ranking_validation_fraction = float(ranking_validation_fraction)
        self.ranking_weight_grid = tuple(float(value) for value in ranking_weight_grid)
        self.regret_weight_floor = float(regret_weight_floor)
        self.regret_weight_cap = float(regret_weight_cap)
        self.random_state = int(random_state)

        self.point_model: ProjectCarbonModel | None = None
        self.classifier_: object | None = None
        self.candidate_feature_names_: list[str] = []
        self.pairwise_feature_names_: list[str] = []
        self.calibration_origins_: int = 0
        self.calibration_pairs_: int = 0
        self.ranking_weight_: float = 0.0
        self.validation_regret_by_weight_: dict[float, float] = {}

    @staticmethod
    def _candidate_features(
        predictions: pd.DataFrame, model: ProjectCarbonModel
    ) -> pd.DataFrame:
        """Common candidate representation across all direct horizons."""
        blocks: list[pd.DataFrame] = []
        for horizon, rows in predictions.groupby("horizon", sort=False):
            target_times = pd.DatetimeIndex(rows["target_time"])
            block = model.feature_builder.target_block(
                target_times, int(horizon), index=rows.index
            )
            block = block.apply(pd.to_numeric, errors="coerce")
            block["direct_prediction"] = rows["direct_prediction"].to_numpy()
            block["horizon_hours"] = float(horizon)
            block["horizon_log1p"] = np.log1p(float(horizon))
            blocks.append(block)
        features = pd.concat(blocks).sort_index()

        wind_cols = [c for c in features if "wind" in c and "forecast" in c]
        solar_cols = [c for c in features if "solar" in c and "forecast" in c]
        load_cols = [
            c
            for c in features
            if ("load" in c or "consumption" in c) and "forecast" in c
        ]
        if wind_cols:
            features["derived_wind_forecast_mw"] = features[wind_cols].sum(
                axis=1, min_count=1
            )
        if solar_cols:
            features["derived_solar_forecast_mw"] = features[solar_cols].sum(
                axis=1, min_count=1
            )
        renewable_cols = [
            c
            for c in ("derived_wind_forecast_mw", "derived_solar_forecast_mw")
            if c in features
        ]
        if renewable_cols:
            features["derived_variable_renewables_mw"] = features[
                renewable_cols
            ].sum(axis=1, min_count=1)
        if load_cols and "derived_variable_renewables_mw" in features:
            load = features[load_cols].mean(axis=1)
            features["derived_residual_load_mw"] = (
                load - features["derived_variable_renewables_mw"]
            )
        return features

    def _pairwise_dataset(
        self, features: pd.DataFrame, predictions: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for _, group in predictions.groupby("origin", sort=False):
            positions = list(group.index)
            for left, right in itertools.combinations(positions, 2):
                actual_left = float(predictions.at[left, "actual"])
                actual_right = float(predictions.at[right, "actual"])
                if not np.isfinite(actual_left) or not np.isfinite(actual_right):
                    continue
                gap = actual_right - actual_left
                if abs(gap) < 1e-9:
                    continue
                difference = (
                    features.loc[left].to_numpy(dtype=float)
                    - features.loc[right].to_numpy(dtype=float)
                )
                weight = np.clip(
                    abs(gap), self.regret_weight_floor, self.regret_weight_cap
                )
                label = int(gap > 0.0)  # left candidate has lower carbon
                rows.extend([difference, -difference])
                labels.extend([label, 1 - label])
                weights.extend([weight, weight])
        columns = [f"delta_{name}" for name in features.columns]
        return (
            pd.DataFrame(rows, columns=columns),
            np.asarray(labels, dtype=int),
            np.asarray(weights, dtype=float),
        )

    def fit(
        self,
        train_frame: pd.DataFrame,
        carbon_cfg: dict,
        *,
        climatology=None,
        forecast_frame: pd.DataFrame | None = None,
    ) -> RegretRankingModel:
        from sklearn.ensemble import HistGradientBoostingClassifier

        n = len(train_frame)
        split = int(n * (1.0 - self.calibration_fraction))
        max_horizon = max(self.horizons)
        if split < 500 or n - split < max(200, max_horizon * 4):
            raise ValueError("not enough rows for point training plus ranking calibration")

        base = train_frame.iloc[:split]
        base_point_model = train_project_model(
            base, carbon_cfg, climatology=climatology, forecast_frame=forecast_frame
        )
        last_origin_position = n - max_horizon
        origins = train_frame.index[
            split:last_origin_position:self.calibration_stride_hours
        ]
        predictions = _point_batch(
            base_point_model, train_frame, origins, self.horizons
        )
        predictions["actual"] = train_frame[CARBON].reindex(
            pd.DatetimeIndex(predictions["target_time"])
        ).to_numpy()
        features = self._candidate_features(predictions, base_point_model)
        features = features.replace([np.inf, -np.inf], np.nan)
        pair_x, pair_y, pair_weight = self._pairwise_dataset(features, predictions)
        if len(pair_x) < 100 or len(np.unique(pair_y)) < 2:
            raise ValueError("not enough non-tied candidate pairs for ranking calibration")

        params = {
            k: v for k, v in self.classifier_params.items() if v is not None
        }
        self.candidate_feature_names_ = list(features.columns)
        self.pairwise_feature_names_ = list(pair_x.columns)

        # Select how strongly the ranker may override the direct ordering on a
        # later temporal slice of the OOS calibration block. Alpha=0 is always
        # available, so a ranker that does not generalize cannot hurt by design.
        unique_origins = pd.Index(predictions["origin"].drop_duplicates())
        validation_size = max(
            20, int(len(unique_origins) * self.ranking_validation_fraction)
        )
        validation_size = min(validation_size, len(unique_origins) // 2)
        tuning_origins = set(unique_origins[:-validation_size])
        validation_origins = set(unique_origins[-validation_size:])
        tuning_predictions = predictions[
            predictions["origin"].isin(tuning_origins)
        ]
        validation_predictions = predictions[
            predictions["origin"].isin(validation_origins)
        ]
        tuning_x, tuning_y, tuning_weight = self._pairwise_dataset(
            features, tuning_predictions
        )
        tuning_classifier = HistGradientBoostingClassifier(
            random_state=self.random_state, **params
        )
        tuning_classifier.fit(tuning_x, tuning_y, sample_weight=tuning_weight)
        validation_features = features.loc[validation_predictions.index]
        pair_score = self._pair_scores(
            validation_features, validation_predictions, tuning_classifier
        )
        direct_score = self._direct_scores(validation_predictions)
        regrets: dict[float, float] = {}
        for weight in self.ranking_weight_grid:
            combined = (1.0 - weight) * direct_score + weight * pair_score
            regrets[weight] = self._mean_selection_regret(
                validation_predictions, combined
            )
        self.ranking_weight_ = min(regrets, key=lambda weight: (regrets[weight], weight))
        self.validation_regret_by_weight_ = regrets

        classifier = HistGradientBoostingClassifier(
            random_state=self.random_state, **params
        )
        classifier.fit(pair_x, pair_y, sample_weight=pair_weight)
        self.classifier_ = classifier
        self.calibration_origins_ = int(predictions["origin"].nunique())
        self.calibration_pairs_ = int(len(pair_x) // 2)

        # The deployable point stage sees all pre-test rows; only the ranking
        # stage remains calibrated from genuinely OOS point predictions.
        self.point_model = train_project_model(
            train_frame, carbon_cfg, climatology=climatology,
            forecast_frame=forecast_frame
        )
        return self

    def _pair_scores(
        self,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        classifier=None,
    ) -> pd.Series:
        """Vectorized antisymmetric pairwise greener scores in ``[0, 1]``."""
        classifier = classifier or self.classifier_
        if classifier is None:
            raise RuntimeError("ranking classifier has not been fitted")
        differences: list[np.ndarray] = []
        pairs: list[tuple[int, int]] = []
        for _, group in predictions.groupby("origin", sort=False):
            for left, right in itertools.combinations(list(group.index), 2):
                difference = (
                    features.loc[left].to_numpy(dtype=float)
                    - features.loc[right].to_numpy(dtype=float)
                )
                differences.extend([difference, -difference])
                pairs.append((left, right))
        scores = pd.Series(0.0, index=predictions.index)
        counts = pd.Series(0, index=predictions.index, dtype=int)
        if not differences:
            return scores + 0.5
        pair_x = pd.DataFrame(differences, columns=self.pairwise_feature_names_)
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
    def _direct_scores(predictions: pd.DataFrame) -> pd.Series:
        """Greener score implied by the direct point forecast ordering."""
        scores = pd.Series(index=predictions.index, dtype=float)
        for _, group in predictions.groupby("origin", sort=False):
            n = len(group)
            rank = group["direct_prediction"].rank(method="average", ascending=True)
            scores.loc[group.index] = 1.0 - (rank - 1.0) / max(1, n - 1)
        return scores

    @staticmethod
    def _mean_selection_regret(
        predictions: pd.DataFrame, greener_score: pd.Series
    ) -> float:
        regrets: list[float] = []
        for _, group in predictions.groupby("origin", sort=False):
            selected = greener_score.loc[group.index].idxmax()
            actual = group["actual"]
            regrets.append(float(predictions.at[selected, "actual"] - actual.min()))
        return float(np.mean(regrets))

    def predict_batch(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        horizons: Sequence[int] | None = None,
        *,
        apply_ranking: bool = True,
    ) -> pd.DataFrame:
        if self.point_model is None or self.classifier_ is None:
            raise RuntimeError("RegretRankingModel.predict_batch called before fit")
        predictions = _point_batch(
            self.point_model, frame, origins, horizons or self.horizons
        )
        features = self._candidate_features(predictions, self.point_model)
        features = features.reindex(columns=self.candidate_feature_names_)
        features = features.replace([np.inf, -np.inf], np.nan)
        predictions["ranking_score"] = 0.5
        predictions["prediction"] = predictions["direct_prediction"]
        if not apply_ranking:
            return predictions

        pair_score = self._pair_scores(features, predictions)
        direct_score = self._direct_scores(predictions)
        combined_score = (
            (1.0 - self.ranking_weight_) * direct_score
            + self.ranking_weight_ * pair_score
        )
        for _, group in predictions.groupby("origin", sort=False):
            positions = list(group.index)
            greener_order = sorted(
                positions, key=lambda position: combined_score[position], reverse=True
            )
            point_values = np.sort(
                predictions.loc[positions, "direct_prediction"].to_numpy()
            )
            for rank, position in enumerate(greener_order):
                predictions.at[position, "ranking_score"] = combined_score[position]
                predictions.at[position, "prediction"] = point_values[rank]
        return predictions

    def save(self, path) -> None:
        import pathlib

        import joblib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> RegretRankingModel:
        import joblib

        return joblib.load(path)


def train_ranking_model(
    train_frame: pd.DataFrame,
    carbon_cfg: dict,
    *,
    climatology=None,
    forecast_frame: pd.DataFrame | None = None,
) -> RegretRankingModel:
    """Build and fit the modular Phase-D selector from configuration."""
    cfg = carbon_cfg.get("ranking_model", {})
    model_cfg = carbon_cfg.get("model", {})
    model = RegretRankingModel(
        horizons=model_cfg.get("horizons_hours", (1, 3, 6, 12, 24, 48)),
        classifier_params=cfg.get("hist_gradient_boosting", {}),
        calibration_fraction=cfg.get("calibration_fraction", 0.20),
        calibration_stride_hours=cfg.get("calibration_stride_hours", 3),
        ranking_validation_fraction=cfg.get("ranking_validation_fraction", 0.25),
        ranking_weight_grid=cfg.get(
            "ranking_weight_grid", (0.0, 0.25, 0.5, 0.75, 1.0)
        ),
        regret_weight_floor=cfg.get("regret_weight_floor", 1.0),
        regret_weight_cap=cfg.get("regret_weight_cap", 30.0),
        random_state=cfg.get("random_state", 42),
    )
    return model.fit(
        train_frame, carbon_cfg, climatology=climatology,
        forecast_frame=forecast_frame
    )


__all__ = ["RegretRankingModel", "train_ranking_model"]
