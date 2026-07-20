"""Daily causal refit evaluation for the selected French 24-hour models.

For every evaluated UTC day, the complete baseline/fossil backbone is rebuilt
using only observations strictly earlier than that origin.  Individual daily
prediction files are checkpoints, so long evaluations can resume safely and
date chunks can run in separate processes.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon import evaluation as ev
from green_observatory.carbon.annual_evaluation import _forecast_frame, _utc
from green_observatory.carbon.climatology import climatology_from_config
from green_observatory.carbon.fossil_regime import (
    FossilRegimeModel,
    train_fossil_regime_model,
)
from green_observatory.carbon.france24 import france24_feature_builder_from_config
from green_observatory.carbon.model import train_project_model
from green_observatory.carbon.physical import PhysicalCarbonMapper, generation_shares
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.config import load_named
from green_observatory.providers.carbon_odre import OdreCarbonProvider


HORIZONS = tuple(range(1, 25))


def _origins(args: argparse.Namespace, frame: pd.DataFrame) -> pd.DatetimeIndex:
    if args.origin_stride_days <= 0:
        raise ValueError("origin_stride_days must be positive")
    if args.dates:
        candidates = pd.DatetimeIndex(
            [_utc(value.strip()) for value in args.dates.split(",") if value.strip()]
        )
    else:
        candidates = pd.date_range(_utc(args.start), _utc(args.end), freq="1D")
    candidates = candidates[:: args.origin_stride_days]
    valid = []
    target = frame["carbon_intensity_gco2_kwh"]
    for origin in candidates:
        required = pd.DatetimeIndex(
            [origin, *(origin + pd.Timedelta(hours=h) for h in HORIZONS)]
        )
        if target.reindex(required).notna().all():
            valid.append(origin)
    return pd.DatetimeIndex(valid)


def _prediction_path(output_dir: Path, origin: pd.Timestamp) -> Path:
    return output_dir / f"{origin:%Y-%m-%d}.parquet"


def run(args: argparse.Namespace) -> None:
    consolidated = OdreCarbonProvider.load_snapshot(args.carbon)
    full = regularize_hourly(consolidated)
    forecasts = _forecast_frame(
        [args.weather, args.consumption_forecast, args.mix_forecast]
    )
    rte_store = RteGenerationForecastFeatureStore(
        pd.read_parquet(args.rte_generation_forecast),
        production_types=("WIND_ONSHORE", "WIND_OFFSHORE", "SOLAR"),
    )
    availability_path = Path(args.rte_unavailability)
    availability_store = (
        RteAvailabilityFeatureStore.from_parquet(availability_path)
        if availability_path.exists()
        else None
    )
    cfg = load_named("carbon_model")
    availability_feature_mode = (
        args.availability_feature_mode
        or cfg.get("fossil_regime_model", {}).get(
            "availability_feature_mode", "delta"
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    origins = _origins(args, full)

    for position, origin in enumerate(origins, start=1):
        output_path = _prediction_path(output_dir, origin)
        if output_path.exists() and not args.overwrite:
            print(f"[{position}/{len(origins)}] skip {origin.date()} (checkpoint exists)")
            continue
        train = consolidated.loc[consolidated.index < origin]
        if args.window_years:
            train = train.loc[train.index >= origin - pd.DateOffset(years=args.window_years)]
        started = time.monotonic()
        print(
            f"[{position}/{len(origins)}] refit {origin.date()} "
            f"window={'all' if not args.window_years else str(args.window_years) + 'y'} "
            f"rows={len(train)}"
        )
        climatology = climatology_from_config(train, cfg)
        calibration = full.loc[
            (origin - pd.Timedelta(days=args.calibration_days) <= full.index)
            & (full.index < origin)
        ]
        shares = generation_shares(calibration)
        recent_mapper = PhysicalCarbonMapper(shares.columns).fit(
            shares, calibration["carbon_intensity_gco2_kwh"]
        )

        if args.source_only:
            fossil_cfg = cfg.get("fossil_regime_model", {})
            fossil = FossilRegimeModel(
                france24_feature_builder_from_config(
                    cfg, climatology=climatology, forecast_frame=forecasts
                ),
                horizons=fossil_cfg.get("horizons_hours", HORIZONS),
                ccg_threshold_mw=fossil_cfg.get("ccg_threshold_mw", 500.0),
                peak_threshold_mw=fossil_cfg.get("peak_threshold_mw", 2500.0),
                calibration_fraction=fossil_cfg.get("calibration_fraction", 0.25),
                training_stride_hours=fossil_cfg.get("training_stride_hours", 6),
                classifier_params=fossil_cfg.get(
                    "classifier_hist_gradient_boosting", {}
                ),
                source_params=fossil_cfg.get("source_hist_gradient_boosting", {}),
                residual_params=fossil_cfg.get(
                    "residual_hist_gradient_boosting", {}
                ),
                ranker_params=fossil_cfg.get("ranker_hist_gradient_boosting", {}),
                availability_store=availability_store,
                availability_feature_mode=availability_feature_mode,
                rte_forecast_store=rte_store,
                random_state=fossil_cfg.get("random_state", 42),
            )
            training_origins = fossil._origin_grid(train)
            training_x, training_meta = fossil._matrix(
                train, training_origins, supervised=True
            )
            classifier, gas_estimators, other_estimators = fossil._fit_components(
                training_x, training_meta
            )
            prediction_x, prediction_meta = fossil._matrix(
                full, pd.DatetimeIndex([origin]), supervised=False
            )
            prediction_x = prediction_x.reindex(columns=training_x.columns)
            components = fossil._component_predictions(
                prediction_x,
                classifier,
                gas_estimators,
                other_estimators,
                recent_mapper,
            )
            predictions = prediction_meta.copy()
            predictions["prediction"] = components["physical_prediction"].to_numpy()
            predictions["model"] = "fossil_regime_recent_mapper"
            predictions["actual"] = full["carbon_intensity_gco2_kwh"].reindex(
                pd.DatetimeIndex(predictions["target_time"])
            ).to_numpy()
            predictions["train_start"] = train.index.min()
            predictions["train_end"] = train.index.max()
            predictions["window_years"] = args.window_years or 0
            predictions["fit_seconds"] = time.monotonic() - started
            predictions.to_parquet(output_path, index=False)
            print(
                f"  saved {output_path.name} in "
                f"{predictions['fit_seconds'].iloc[0]:.1f}s"
            )
            continue

        baseline_cfg = copy.deepcopy(cfg)
        # The hybrid consumes the direct baseline only at h1 and h2.
        baseline_cfg.setdefault("model", {})["horizons_hours"] = [1, 2]
        baseline = train_project_model(
            train,
            baseline_cfg,
            climatology=climatology,
            forecast_frame=forecasts,
        )
        fossil = train_fossil_regime_model(
            train,
            cfg,
            climatology=climatology,
            forecast_frame=forecasts,
            availability_store=availability_store,
            availability_feature_mode=availability_feature_mode,
            rte_forecast_store=rte_store,
        )

        origin_index = pd.DatetimeIndex([origin])
        baseline_prediction = ev._project_batch(
            baseline, full, origin_index, (1, 2)
        )
        regime = fossil.predict_batch(full, origin_index)
        fossil_point = regime[
            ["origin", "horizon", "target_time", "point_prediction"]
        ].rename(columns={"point_prediction": "prediction"})
        fossil_point_output = fossil_point.copy()
        fossil_point_output["model"] = "fossil_regime_point"
        fossil_decision = regime[
            ["origin", "horizon", "target_time", "decision_prediction"]
        ].rename(columns={"decision_prediction": "prediction"})
        fossil_decision["model"] = "fossil_regime_decision"
        fossil_ranked = regime[
            ["origin", "horizon", "target_time", "ranked_prediction"]
        ].rename(columns={"ranked_prediction": "prediction"})
        fossil_ranked["model"] = "fossil_regime_ranked"

        predicted_shares = pd.DataFrame(
            {
                name: regime[f"predicted_{name}"].to_numpy()
                for name in recent_mapper.share_names
            },
            index=regime.index,
        )
        recent_physical = regime[["origin", "horizon", "target_time"]].copy()
        recent_physical["prediction"] = recent_mapper.predict(predicted_shares)

        mapper_delta = fossil_point.copy()
        mapper_delta["prediction"] = np.clip(
            fossil_point["prediction"].to_numpy(dtype=float)
            + float(fossil.point_scale_)
            * (
                recent_physical["prediction"].to_numpy(dtype=float)
                - regime["physical_prediction"].to_numpy(dtype=float)
            ),
            0.0,
            None,
        )

        hybrid = pd.concat(
            [
                baseline_prediction,
                fossil_point[fossil_point["horizon"] > 2],
            ],
            ignore_index=True,
        )
        hybrid["model"] = "hybrid_h2"
        hybrid_delta = pd.concat(
            [
                baseline_prediction,
                mapper_delta[mapper_delta["horizon"] > 2],
            ],
            ignore_index=True,
        )
        hybrid_delta["model"] = "hybrid_h2_mapper_delta"
        recent_physical["model"] = "fossil_regime_recent_mapper"
        predictions = pd.concat(
            [
                hybrid,
                hybrid_delta,
                recent_physical,
                fossil_point_output,
                fossil_decision,
                fossil_ranked,
            ],
            ignore_index=True,
        )
        predictions["actual"] = full["carbon_intensity_gco2_kwh"].reindex(
            pd.DatetimeIndex(predictions["target_time"])
        ).to_numpy()
        predictions["train_start"] = train.index.min()
        predictions["train_end"] = train.index.max()
        predictions["window_years"] = args.window_years or 0
        predictions["fit_seconds"] = time.monotonic() - started
        predictions.to_parquet(output_path, index=False)
        print(
            f"  saved {output_path.name} in "
            f"{predictions['fit_seconds'].iloc[0]:.1f}s"
        )


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
        help="Publication-versioned RTE generation unavailability snapshot.",
    )
    parser.add_argument(
        "--availability-feature-mode",
        choices=("delta", "all"),
        default=None,
        help="Override the fossil expert's RTE outage feature set.",
    )
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-29")
    parser.add_argument("--dates", default=None)
    parser.add_argument("--window-years", type=int, default=0)
    parser.add_argument(
        "--origin-stride-days",
        type=int,
        default=1,
        help="Evaluate every Nth daily origin (useful for causal pilots).",
    )
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Refit only the exact source-share head used by recent_mapper.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
