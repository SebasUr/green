"""Evaluate the frozen compact stack on the June-July 2026 live snapshot.

Architecture and weights are not selected here.  They were fixed using only
January-February 2026: 75% DirectRegimeMoE, 15% pooled LightGBM log-target and
10% per-horizon LightGBM log-target.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.stacked_regressor import percentage_metrics
from green_observatory.carbon.stacked_regressor_evaluation import (
    MODEL_SPECS,
    _fit_scaled_moe,
    _fit_scaled_per_horizon,
    _fit_scaled_pooled,
)
from green_observatory.providers.carbon_odre import OdreCarbonProvider


FROZEN_WEIGHTS = {
    "direct_regime_moe": 0.75,
    "lgbm_log": 0.15,
    "per_horizon_lgbm_log": 0.10,
}


def _indexed_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    index = pd.DatetimeIndex(frame.index)
    frame.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return frame


def _combine_indexed(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames).sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    historical = OdreCarbonProvider.load_snapshot(args.carbon)
    live = _indexed_parquet(args.carbon_holdout)
    full = regularize_hourly(_combine_indexed(historical, live))

    mix = _combine_indexed(
        _indexed_parquet(args.mix_forecast),
        _indexed_parquet(args.mix_forecast_holdout),
    )
    price = _combine_indexed(
        _indexed_parquet(args.price_forecast),
        _indexed_parquet(args.price_forecast_holdout),
    )
    forecasts = mix.join(price, how="outer")
    availability_intervals = pd.concat(
        [
            pd.read_parquet(args.rte_unavailability),
            pd.read_parquet(args.rte_unavailability_holdout),
        ],
        ignore_index=True,
    ).drop_duplicates()
    availability = RteAvailabilityFeatureStore(availability_intervals)
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        availability_store=availability,
        availability_feature_mode="all",
        include_curve_summaries=True,
    )

    historical_origins = pd.date_range(
        _utc(args.feature_start), _utc("2026-04-29"), freq="1D"
    )
    holdout_origins = pd.date_range(
        _utc(args.eval_start), _utc(args.eval_end), freq="1D"
    )
    origins = historical_origins.append(holdout_origins)
    x, meta = builder.build(full, origins, supervised=True)
    scale_train_end = _utc("2026-01-01")
    final_train_end = _utc("2026-05-01")
    evaluation = (meta["origin"] >= _utc(args.eval_start)) & (
        meta["origin"] <= _utc(args.eval_end)
    )
    actual = meta.loc[evaluation, "actual"].to_numpy(dtype=float)

    predictions: dict[str, np.ndarray] = {}
    fit_report: dict[str, dict] = {}
    direct, scale, calibration_mape, seconds = _fit_scaled_moe(
        x,
        meta,
        scale_train_end=scale_train_end,
        final_train_end=final_train_end,
        evaluation=evaluation,
    )
    predictions["direct_regime_moe"] = direct
    fit_report["direct_regime_moe"] = {
        "scale": scale,
        "scale_calibration_mape": calibration_mape,
        "fit_seconds": seconds,
    }
    pooled, scale, calibration_mape, seconds = _fit_scaled_pooled(
        MODEL_SPECS["lgbm_log"],
        x,
        meta,
        scale_train_end=scale_train_end,
        final_train_end=final_train_end,
        evaluation=evaluation,
    )
    predictions["lgbm_log"] = pooled
    fit_report["lgbm_log"] = {
        "scale": scale,
        "scale_calibration_mape": calibration_mape,
        "fit_seconds": seconds,
    }
    per_horizon, scale, calibration_mape, seconds = _fit_scaled_per_horizon(
        x,
        meta,
        scale_train_end=scale_train_end,
        final_train_end=final_train_end,
        evaluation=evaluation,
    )
    predictions["per_horizon_lgbm_log"] = per_horizon
    fit_report["per_horizon_lgbm_log"] = {
        "scale": scale,
        "scale_calibration_mape": calibration_mape,
        "fit_seconds": seconds,
    }
    stack = sum(predictions[name] * weight for name, weight in FROZEN_WEIGHTS.items())
    predictions["frozen_greedy_stack"] = stack

    evaluation_meta = meta.loc[
        evaluation, ["origin", "horizon", "target_time", "actual"]
    ].reset_index(drop=True)
    warm = evaluation_meta["origin"] >= _utc(args.warm_start)
    report = {
        "protocol": {
            "architecture_selection": "January-February 2026 only",
            "frozen_weights": FROZEN_WEIGHTS,
            "scale_component_train_target_before": str(scale_train_end),
            "scale_calibration_origins": [str(scale_train_end), str(final_train_end)],
            "final_component_train_target_before": str(final_train_end),
            "holdout_origins": [args.eval_start, args.eval_end],
            "warm_origins": [args.warm_start, args.eval_end],
            "note": (
                "Static pre-holdout refit. Early origins lack some aligned D7 "
                "lags because the live carbon snapshot starts on June 16."
            ),
        },
        "rows": len(x),
        "features": len(x.columns),
        "fit": fit_report,
        "metrics_all": {
            name: percentage_metrics(actual, prediction)
            for name, prediction in predictions.items()
        },
        "metrics_warm": {
            name: percentage_metrics(actual[warm], prediction[warm])
            for name, prediction in predictions.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.concat(
        [
            evaluation_meta.assign(model=name, prediction=prediction)
            for name, prediction in predictions.items()
        ],
        ignore_index=True,
    ).to_parquet(output.with_suffix(".predictions.parquet"), index=False)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet")
    parser.add_argument("--carbon-holdout", default="data/cache/carbon_fr_realtime_holdout.parquet")
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument("--mix-forecast-holdout", default="data/cache/mix_day_ahead_fr_holdout.parquet")
    parser.add_argument("--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet")
    parser.add_argument("--price-forecast-holdout", default="data/cache/day_ahead_price_fr_holdout.parquet")
    parser.add_argument("--rte-unavailability", default="data/cache/rte_unavailability_messages.parquet")
    parser.add_argument(
        "--rte-unavailability-holdout",
        default="data/cache/rte_unavailability_messages_holdout.parquet",
    )
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--eval-start", default="2026-06-17")
    parser.add_argument("--eval-end", default="2026-07-15")
    parser.add_argument("--warm-start", default="2026-06-24")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
