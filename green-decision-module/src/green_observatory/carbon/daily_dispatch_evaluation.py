"""Walk-forward evaluation of the modular French daily-curve expert.

The expensive learner is refitted at a configurable cadence (weekly by
default), while every origin is still predicted separately with only features
available at that origin.  This matches a realistic deployment where a daily
calibrator can move faster than the backbone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.adaptive_ensemble import causal_scaled_expert
from green_observatory.carbon.annual_evaluation import _forecast_frame, _utc
from green_observatory.carbon.climatology import climatology_from_config
from green_observatory.carbon.france24 import france24_feature_builder_from_config
from green_observatory.carbon.france_dispatch import (
    FranceDailyCurveFeatureBuilder,
    FranceDispatchEnsemble,
    daily_training_origins,
)
from green_observatory.carbon.protocols import aggregate_metrics, regularize_hourly
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.config import load_named
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider
from green_observatory.windows.oracle import window_selection_metrics


HORIZONS = tuple(range(1, 25))


def _evaluation_origins(
    frame: pd.DataFrame, start: str, end: str
) -> pd.DatetimeIndex:
    candidates = pd.date_range(_utc(start), _utc(end), freq="1D")
    valid: list[pd.Timestamp] = []
    for origin in candidates:
        targets = origin + pd.to_timedelta(HORIZONS, unit="h")
        if frame[CARBON].reindex(targets).notna().all():
            valid.append(origin)
    return pd.DatetimeIndex(valid)


def _tidy_predictions(wide: pd.DataFrame) -> pd.DataFrame:
    common = ["origin", "horizon", "target_time", "actual"]
    parts = [
        wide[common + ["prediction"]]
        .rename(columns={"prediction": "prediction"})
        .assign(model="france_dispatch_ensemble")
    ]
    for column in (
        "prediction_direct_mape",
        "prediction_log_l1",
        "prediction_relative_d1_l1",
    ):
        if column in wide:
            parts.append(
                wide[common + [column]]
                .rename(columns={column: "prediction"})
                .assign(model=column.removeprefix("prediction_"))
            )
    return pd.concat(parts, ignore_index=True)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.round(6).to_json(orient="records"))


def run(args: argparse.Namespace) -> dict:
    consolidated = OdreCarbonProvider.load_snapshot(args.carbon)
    full = regularize_hourly(consolidated)
    forecasts = _forecast_frame(
        [args.weather, args.consumption_forecast, args.mix_forecast]
    )
    availability_store = (
        RteAvailabilityFeatureStore.from_parquet(args.rte_unavailability)
        if Path(args.rte_unavailability).exists()
        else None
    )
    rte_store = (
        RteGenerationForecastFeatureStore.from_parquet(
            args.rte_generation_forecast,
            production_types=("WIND_ONSHORE", "WIND_OFFSHORE", "SOLAR"),
        )
        if Path(args.rte_generation_forecast).exists()
        else None
    )
    cfg = load_named("carbon_model")
    origins = _evaluation_origins(full, args.start, args.end)
    if origins.empty:
        raise ValueError("no complete evaluation origins")
    if args.refit_days <= 0:
        raise ValueError("refit_days must be positive")

    prediction_parts: list[pd.DataFrame] = []
    fit_rows: list[dict] = []
    model: FranceDispatchEnsemble | None = None
    curve_builder: FranceDailyCurveFeatureBuilder | None = None
    last_refit_position = -args.refit_days
    for position, origin in enumerate(origins):
        if model is None or position - last_refit_position >= args.refit_days:
            started = time.monotonic()
            observed = consolidated.loc[consolidated.index < origin]
            climatology = climatology_from_config(observed, cfg)
            base_builder = france24_feature_builder_from_config(
                cfg, climatology=climatology, forecast_frame=forecasts
            )
            curve_builder = FranceDailyCurveFeatureBuilder(
                base_builder,
                availability_store=availability_store,
                rte_forecast_store=rte_store,
            )
            train_origins = daily_training_origins(full, cutoff=origin)
            train_x, train_meta = curve_builder.matrix(
                full, train_origins, supervised=True
            )
            # The matrix helper already restricts targets to known labels; this
            # assertion documents and enforces the operational cutoff.
            if (pd.to_datetime(train_meta["target_time"], utc=True) > origin).any():
                raise AssertionError("dispatch training target crosses forecast origin")
            model = FranceDispatchEnsemble(
                curve_builder,
                validation_days=args.validation_days,
                ensemble_iterations=args.ensemble_iterations,
                n_jobs=args.n_jobs,
            ).fit(train_x, train_meta)
            elapsed = time.monotonic() - started
            fit_rows.append(
                {
                    "origin": origin,
                    "training_days": int(train_meta["origin"].nunique()),
                    "training_rows": int(len(train_meta)),
                    "fit_seconds": elapsed,
                    "weights": model.weights_,
                    "validation_mape": model.validation_mape_,
                }
            )
            last_refit_position = position
            print(
                f"refit {origin.date()} rows={len(train_meta)} "
                f"seconds={elapsed:.1f} weights={model.weights_}"
            )

        assert curve_builder is not None and model is not None
        predict_x, predict_meta = curve_builder.matrix(
            full, pd.DatetimeIndex([origin]), supervised=True
        )
        prediction_parts.append(model.predict(predict_x, predict_meta))

    wide = pd.concat(prediction_parts, ignore_index=True)
    predictions = _tidy_predictions(wide)
    scaled = causal_scaled_expert(
        predictions,
        lookback_days=args.calibration_days,
        candidates=("france_dispatch_ensemble",),
        default_expert="france_dispatch_ensemble",
        scale_grid=np.arange(0.70, 1.401, 0.005),
        name=f"france_dispatch_scaled_{args.calibration_days}d",
    )
    predictions = pd.concat([predictions, scaled], ignore_index=True)

    aggregate = aggregate_metrics(predictions)
    selection = window_selection_metrics(predictions, full).reset_index()
    report = {
        "protocol": "daily_origin_periodic_expanding_refit",
        "start": str(origins.min()),
        "end": str(origins.max()),
        "origins": int(len(origins)),
        "refit_days": int(args.refit_days),
        "calibration_days": int(args.calibration_days),
        "aggregate_metrics": _records(aggregate),
        "window_selection": _records(selection),
        "fits": fit_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output.with_suffix(".predictions.parquet"), index=False)
    wide.to_parquet(output.with_suffix(".components.parquet"), index=False)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(selection.to_string(index=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet"
    )
    parser.add_argument("--weather", default="data/cache/weather_fr_hourly.parquet")
    parser.add_argument(
        "--consumption-forecast",
        default="data/cache/consumption_forecast_fr_hourly.parquet",
    )
    parser.add_argument(
        "--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet"
    )
    parser.add_argument(
        "--rte-generation-forecast",
        default="data/cache/rte_generation_forecast.parquet",
    )
    parser.add_argument(
        "--rte-unavailability",
        default="data/cache/rte_unavailability_messages.parquet",
    )
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-29")
    parser.add_argument("--refit-days", type=int, default=7)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--ensemble-iterations", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
