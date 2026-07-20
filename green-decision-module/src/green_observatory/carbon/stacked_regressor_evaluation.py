"""Temporal evaluation for compact pooled/stacked French carbon regressors.

The command has an explicit development stage that ends before March 2026.
Only after the candidate and stack are fixed should ``--stage holdout`` be run.
Every learner consumes the exact same causal matrix as ``DirectRegimeMoE``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _forecast_frame, _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import (
    DirectRegimeMoE,
    RegimeMoEFeatureBuilder,
    select_mape_scale,
)
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.stacked_regressor import (
    PerHorizonCarbonRegressor,
    PooledCarbonRegressor,
    percentage_metrics,
)
from green_observatory.providers.carbon_odre import OdreCarbonProvider


MODEL_SPECS = {
    "lgbm_relative": {
        "backend": "lightgbm",
        "target_transform": "identity",
        "inverse_level_weight": True,
    },
    "lgbm_log": {
        "backend": "lightgbm",
        "target_transform": "log",
        "inverse_level_weight": False,
    },
    "catboost_relative": {
        "backend": "catboost",
        "target_transform": "identity",
        "inverse_level_weight": True,
    },
    "catboost_log": {
        "backend": "catboost",
        "target_transform": "log",
        "inverse_level_weight": False,
    },
}

PER_HORIZON_SPEC = {
    "backend": "lightgbm",
    "target_transform": "log",
    "inverse_level_weight": False,
    "params": {
        "n_estimators": 220,
        "learning_rate": 0.035,
        "num_leaves": 9,
        "min_child_samples": 20,
    },
}


def _greedy_mape_stack(
    predictions: pd.DataFrame, actual: np.ndarray, iterations: int = 20
) -> dict[str, float]:
    """Caruana ensemble selection, with MAPE as the held-out loss."""
    values = predictions.to_numpy(dtype=float)
    actual = np.asarray(actual, dtype=float)
    counts = np.zeros(values.shape[1], dtype=int)
    running = np.zeros(len(actual), dtype=float)
    for step in range(iterations):
        losses = []
        for candidate in range(values.shape[1]):
            trial = (running + values[:, candidate]) / (step + 1)
            losses.append(
                np.mean(
                    np.abs(trial - actual)
                    / np.clip(np.abs(actual), 1e-9, None)
                )
            )
        selected = int(np.argmin(losses))
        counts[selected] += 1
        running += values[:, selected]
    return {
        column: float(count / iterations)
        for column, count in zip(predictions.columns, counts)
        if count
    }


def _weighted_prediction(
    predictions: pd.DataFrame, weights: dict[str, float]
) -> np.ndarray:
    return sum(
        predictions[column].to_numpy(dtype=float) * weight
        for column, weight in weights.items()
    )


def _fit_scaled_pooled(
    spec: dict,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    scale_train_end: pd.Timestamp,
    final_train_end: pd.Timestamp,
    evaluation: pd.Series,
) -> tuple[np.ndarray, float, float, float]:
    tuning_train = meta["target_time"] < scale_train_end
    calibration = (meta["origin"] >= scale_train_end) & (
        meta["origin"] < final_train_end
    )
    started = time.perf_counter()
    tuning = PooledCarbonRegressor(**spec)
    tuning.fit(x.loc[tuning_train], meta.loc[tuning_train, "actual"])
    calibration_prediction = tuning.predict(x.loc[calibration])
    scale, calibration_mape = select_mape_scale(
        meta.loc[calibration, "actual"].to_numpy(), calibration_prediction
    )
    model = PooledCarbonRegressor(**spec)
    final_train = meta["target_time"] < final_train_end
    model.fit(x.loc[final_train], meta.loc[final_train, "actual"])
    prediction = scale * model.predict(x.loc[evaluation])
    return prediction, scale, calibration_mape, time.perf_counter() - started


def _fit_scaled_per_horizon(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    scale_train_end: pd.Timestamp,
    final_train_end: pd.Timestamp,
    evaluation: pd.Series,
) -> tuple[np.ndarray, float, float, float]:
    tuning_train = meta["target_time"] < scale_train_end
    calibration = (meta["origin"] >= scale_train_end) & (
        meta["origin"] < final_train_end
    )
    started = time.perf_counter()
    tuning = PerHorizonCarbonRegressor(**PER_HORIZON_SPEC)
    tuning.fit(
        x.loc[tuning_train],
        meta.loc[tuning_train, "actual"],
        meta.loc[tuning_train, "horizon"],
    )
    calibration_prediction = tuning.predict(
        x.loc[calibration], meta.loc[calibration, "horizon"]
    )
    scale, calibration_mape = select_mape_scale(
        meta.loc[calibration, "actual"].to_numpy(), calibration_prediction
    )
    final_train = meta["target_time"] < final_train_end
    model = PerHorizonCarbonRegressor(**PER_HORIZON_SPEC)
    model.fit(
        x.loc[final_train],
        meta.loc[final_train, "actual"],
        meta.loc[final_train, "horizon"],
    )
    prediction = scale * model.predict(
        x.loc[evaluation], meta.loc[evaluation, "horizon"]
    )
    return prediction, scale, calibration_mape, time.perf_counter() - started


def _fit_scaled_moe(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    scale_train_end: pd.Timestamp,
    final_train_end: pd.Timestamp,
    evaluation: pd.Series,
) -> tuple[np.ndarray, float, float, float]:
    tuning_train = meta["target_time"] < scale_train_end
    calibration = (meta["origin"] >= scale_train_end) & (
        meta["origin"] < final_train_end
    )
    started = time.perf_counter()
    tuning = DirectRegimeMoE()
    tuning.fit(
        x.loc[tuning_train].reset_index(drop=True),
        meta.loc[tuning_train].reset_index(drop=True),
    )
    calibration_prediction = tuning.predict_matrix(
        x.loc[calibration].reset_index(drop=True)
    )["prediction"].to_numpy()
    scale, calibration_mape = select_mape_scale(
        meta.loc[calibration, "actual"].to_numpy(), calibration_prediction
    )
    model = DirectRegimeMoE(point_scale=scale)
    final_train = meta["target_time"] < final_train_end
    model.fit(
        x.loc[final_train].reset_index(drop=True),
        meta.loc[final_train].reset_index(drop=True),
    )
    prediction = model.predict_matrix(
        x.loc[evaluation].reset_index(drop=True)
    )["prediction"].to_numpy()
    return prediction, scale, calibration_mape, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    stage_end = "2026-02-28" if args.stage == "dev" else args.eval_end
    carbon = OdreCarbonProvider.load_snapshot(args.carbon)
    full = regularize_hourly(carbon)
    forecasts = _forecast_frame([args.mix_forecast, args.price_forecast])
    availability = RteAvailabilityFeatureStore.from_parquet(
        args.rte_unavailability
    )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        availability_store=availability,
        availability_feature_mode="all",
        include_curve_summaries=True,
    )
    origins = pd.date_range(
        _utc(args.feature_start), _utc(stage_end), freq="1D"
    )
    x, meta = builder.build(full, origins, supervised=True)
    scale_train_end = _utc("2025-01-01")
    final_train_end = _utc("2026-01-01")
    evaluation = (meta["origin"] >= final_train_end) & (
        meta["origin"] <= _utc(stage_end)
    )
    actual = meta.loc[evaluation, "actual"].to_numpy(dtype=float)
    evaluation_meta = meta.loc[
        evaluation, ["origin", "horizon", "target_time", "actual"]
    ].reset_index(drop=True)

    predictions: dict[str, np.ndarray] = {}
    fit_report: dict[str, dict] = {}
    direct, scale, scale_mape, seconds = _fit_scaled_moe(
        x,
        meta,
        scale_train_end=scale_train_end,
        final_train_end=final_train_end,
        evaluation=evaluation,
    )
    predictions["direct_regime_moe"] = direct
    fit_report["direct_regime_moe"] = {
        "scale": scale,
        "scale_calibration_mape": scale_mape,
        "fit_seconds": seconds,
    }
    for name, spec in MODEL_SPECS.items():
        prediction, scale, scale_mape, seconds = _fit_scaled_pooled(
            spec,
            x,
            meta,
            scale_train_end=scale_train_end,
            final_train_end=final_train_end,
            evaluation=evaluation,
        )
        predictions[name] = prediction
        fit_report[name] = {
            "scale": scale,
            "scale_calibration_mape": scale_mape,
            "fit_seconds": seconds,
        }
    prediction, scale, scale_mape, seconds = _fit_scaled_per_horizon(
        x,
        meta,
        scale_train_end=scale_train_end,
        final_train_end=final_train_end,
        evaluation=evaluation,
    )
    predictions["per_horizon_lgbm_log"] = prediction
    fit_report["per_horizon_lgbm_log"] = {
        "scale": scale,
        "scale_calibration_mape": scale_mape,
        "fit_seconds": seconds,
    }

    prediction_frame = pd.DataFrame(predictions)
    dev = evaluation_meta["origin"] < _utc("2026-03-01")
    metrics_dev = {
        name: percentage_metrics(actual[dev], values[dev])
        for name, values in predictions.items()
    }
    direct_candidates = [*MODEL_SPECS, "per_horizon_lgbm_log"]
    best_pooled = min(
        direct_candidates, key=lambda name: metrics_dev[name]["mape"]
    )
    blend_candidates = []
    for weight in np.arange(0.0, 1.001, 0.05):
        values = (
            weight * prediction_frame[best_pooled]
            + (1.0 - weight) * prediction_frame["direct_regime_moe"]
        ).to_numpy()
        blend_candidates.append(
            {
                "pooled_weight": float(weight),
                "dev_mape": percentage_metrics(actual[dev], values[dev])["mape"],
            }
        )
    selected_blend = min(
        blend_candidates, key=lambda row: (row["dev_mape"], row["pooled_weight"])
    )
    blend_weights = {
        best_pooled: selected_blend["pooled_weight"],
        "direct_regime_moe": 1.0 - selected_blend["pooled_weight"],
    }
    stack_weights = _greedy_mape_stack(
        prediction_frame.loc[dev], actual[dev], iterations=20
    )
    derived = {
        "best_pooled_direct_blend": _weighted_prediction(
            prediction_frame, blend_weights
        ),
        "greedy_stack": _weighted_prediction(prediction_frame, stack_weights),
    }
    for name, values in derived.items():
        prediction_frame[name] = values
        metrics_dev[name] = percentage_metrics(actual[dev], values[dev])

    selection_pool = [best_pooled, "best_pooled_direct_blend", "greedy_stack"]
    selected_architecture = min(
        selection_pool, key=lambda name: metrics_dev[name]["mape"]
    )
    report: dict = {
        "stage": args.stage,
        "protocol": {
            "component_scale_train_target_before": str(scale_train_end),
            "component_scale_calibration_origins": [
                str(scale_train_end),
                str(final_train_end),
            ],
            "final_component_train_target_before": str(final_train_end),
            "architecture_and_stack_selection": "origins before 2026-03-01",
            "holdout": (
                None
                if args.stage == "dev"
                else ["2026-03-01", str(_utc(stage_end))]
            ),
            "day_ahead_visibility": (
                "masked when target local date is after origin local date"
            ),
            "features": "price + mix + curve summaries + availability + aligned lags",
        },
        "rows": len(x),
        "features": len(x.columns),
        "fit": fit_report,
        "metrics_dev": metrics_dev,
        "best_pooled": best_pooled,
        "blend_weights": blend_weights,
        "blend_candidates": blend_candidates,
        "greedy_stack_weights": stack_weights,
        "selected_architecture": selected_architecture,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.stage == "holdout":
        holdout = evaluation_meta["origin"] >= _utc("2026-03-01")
        report["metrics_holdout"] = {
            name: percentage_metrics(actual[holdout], prediction_frame.loc[holdout, name])
            for name in [
                "direct_regime_moe",
                best_pooled,
                "best_pooled_direct_blend",
                "greedy_stack",
            ]
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    long_predictions = pd.concat(
        [
            evaluation_meta.assign(model=name, prediction=values)
            for name, values in prediction_frame.items()
        ],
        ignore_index=True,
    )
    long_predictions.to_parquet(output.with_suffix(".predictions.parquet"), index=False)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "holdout"), required=True)
    parser.add_argument(
        "--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet"
    )
    parser.add_argument(
        "--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet"
    )
    parser.add_argument(
        "--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet"
    )
    parser.add_argument(
        "--rte-unavailability",
        default="data/cache/rte_unavailability_messages.parquet",
    )
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--eval-end", default="2026-04-29")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
