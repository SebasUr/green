"""Causal annual evaluation for the selected dense French carbon architecture.

The expensive baseline and fossil-regime backbone is frozen before the annual
evaluation window.  At the start of each month, only the lightweight recent
physical map and optional multiplicative scales are calibrated using the six
preceding days.  No outcome from the evaluated month is used by that month's
forecast.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from green_observatory.carbon.climatology import climatology_from_config
from green_observatory.carbon.fossil_regime import train_fossil_regime_model
from green_observatory.carbon.france24 import DENSE_DAY_AHEAD_HORIZONS
from green_observatory.carbon.model import train_project_model
from green_observatory.carbon.physical import PhysicalCarbonMapper, generation_shares
from green_observatory.carbon.protocols import (
    DAILY_UTC,
    ROLLING_6H,
    aggregate_metrics,
    evaluate_protocol,
    fit_mape_scales,
    model_predictions,
    protocol_origins,
    regularize_hourly,
)
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.carbon import evaluation as ev
from green_observatory.config import load_named
from green_observatory.providers.carbon_odre import OdreCarbonProvider
from green_observatory.windows.oracle import window_selection_metrics


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.round(5).to_json(orient="records"))


def _forecast_frame(paths: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        part = pd.read_parquet(path).copy()
        if part.index.tz is None:
            part.index = part.index.tz_localize("UTC")
        else:
            part.index = part.index.tz_convert("UTC")
        merged = part if merged is None else merged.join(part, how="outer")
    if merged is None:
        raise FileNotFoundError("at least one forecast snapshot is required")
    return merged


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    first = start.to_period("M").to_timestamp().tz_localize("UTC")
    last = end.to_period("M").to_timestamp().tz_localize("UTC")
    return pd.date_range(first, last, freq="MS")


def run(args: argparse.Namespace) -> dict:
    consolidated = OdreCarbonProvider.load_snapshot(args.carbon)
    full = regularize_hourly(consolidated)
    train_cutoff = _utc(args.train_end)
    eval_start = _utc(args.eval_start)
    eval_end = _utc(args.eval_end)
    # CLI dates denote complete evaluation days.  18:00 is the last origin on
    # the six-hour rolling grid; daily UTC still selects that day's 00:00 row.
    eval_last_origin = eval_end + pd.Timedelta(hours=18)
    train = consolidated.loc[consolidated.index < train_cutoff]

    forecasts = _forecast_frame(
        [args.weather, args.consumption_forecast, args.mix_forecast]
    )
    rte_frame = pd.read_parquet(args.rte_generation_forecast)
    rte_store = RteGenerationForecastFeatureStore(
        rte_frame,
        production_types=("WIND_ONSHORE", "WIND_OFFSHORE", "SOLAR"),
    )

    cfg = load_named("carbon_model")
    climatology = climatology_from_config(train, cfg)
    dense_cfg = copy.deepcopy(cfg)
    dense_cfg.setdefault("model", {})["horizons_hours"] = list(
        DENSE_DAY_AHEAD_HORIZONS
    )

    print(
        f"training frozen backbone on {len(train)} rows through "
        f"{train.index.max()}"
    )
    baseline = train_project_model(
        train, dense_cfg, climatology=climatology, forecast_frame=forecasts
    )
    fossil = train_fossil_regime_model(
        train,
        cfg,
        climatology=climatology,
        forecast_frame=forecasts,
        rte_forecast_store=rte_store,
    )

    specs = (DAILY_UTC, ROLLING_6H)
    predictions: dict[str, list[pd.DataFrame]] = {spec.name: [] for spec in specs}
    report: dict[str, object] = {
        "method": "frozen_backbone_monthly_causal_calibration",
        "train_end": str(train_cutoff),
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "calibration_days": int(args.calibration_days),
        "short_horizon_cutoff": int(args.short_horizon_cutoff),
        "months": {},
    }

    for month_start in _month_starts(eval_start, eval_end):
        next_month = month_start + pd.offsets.MonthBegin(1)
        month_end = min(
            eval_last_origin,
            next_month - pd.Timedelta(hours=6),
        )
        calibration_start = month_start - pd.Timedelta(days=args.calibration_days)
        calibration_end = month_start - pd.Timedelta(days=1)
        calibration_rows = full.loc[
            (calibration_start <= full.index) & (full.index < month_start)
        ]
        recent_shares = generation_shares(calibration_rows)
        recent_mapper = PhysicalCarbonMapper(recent_shares.columns).fit(
            recent_shares,
            calibration_rows["carbon_intensity_gco2_kwh"],
        )
        calibration_origins = protocol_origins(
            full, calibration_start, calibration_end, ROLLING_6H
        )
        calibration_predictions = model_predictions(
            full,
            calibration_origins,
            baseline=baseline,
            fossil_regime=fossil,
            short_horizon_cutoff=args.short_horizon_cutoff,
            recent_mapper=recent_mapper,
        )
        scales = fit_mape_scales(calibration_predictions)

        month_key = month_start.strftime("%Y-%m")
        print(
            f"evaluating {month_key}: calibration={calibration_start.date()}.."
            f"{calibration_end.date()} ({len(calibration_rows)} rows)"
        )
        month_report: dict[str, object] = {
            "calibration_start": str(calibration_start),
            "calibration_end": str(calibration_end),
            "calibration_rows": int(len(calibration_rows)),
            "recent_map": {
                "intercept": recent_mapper.intercept_,
                "coefficients": recent_mapper.coefficients_,
            },
            "scales": scales,
            "protocols": {},
        }
        for spec in specs:
            origins = protocol_origins(full, month_start, month_end, spec)
            result = evaluate_protocol(
                full,
                origins,
                baseline=baseline,
                fossil_regime=fossil,
                short_horizon_cutoff=args.short_horizon_cutoff,
                calibration_scales=scales,
                recent_mapper=recent_mapper,
            )
            part = result["predictions"].copy()
            part["evaluation_month"] = month_key
            predictions[spec.name].append(part)
            month_report["protocols"][spec.name] = {
                "origins": int(len(origins)),
                "aggregate_metrics": _records(result["aggregate"]),
                "window_selection": _records(
                    result["selection"].reset_index()
                ),
            }
        report["months"][month_key] = month_report

    report["annual"] = {}
    output_path = Path(args.output)
    prediction_path = output_path.with_suffix(".predictions.parquet")
    all_prediction_parts: list[pd.DataFrame] = []
    for spec in specs:
        annual_predictions = pd.concat(predictions[spec.name], ignore_index=True)
        annual_predictions["protocol"] = spec.name
        all_prediction_parts.append(annual_predictions)
        annual_aggregate = aggregate_metrics(annual_predictions)
        annual_point = ev.point_metrics(annual_predictions)
        annual_selection = window_selection_metrics(annual_predictions, full)
        report["annual"][spec.name] = {
            "origins": int(annual_predictions["origin"].nunique()),
            "aggregate_metrics": _records(annual_aggregate),
            "point_metrics": _records(annual_point),
            "window_selection": _records(annual_selection.reset_index()),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.concat(all_prediction_parts, ignore_index=True).to_parquet(
        prediction_path, index=False
    )
    print(f"saved report -> {output_path}")
    print(f"saved predictions -> {prediction_path}")
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
    parser.add_argument("--train-end", default="2025-05-01")
    parser.add_argument("--eval-start", default="2025-05-01")
    parser.add_argument("--eval-end", default="2026-04-29")
    parser.add_argument("--calibration-days", type=int, default=6)
    parser.add_argument("--short-horizon-cutoff", type=int, default=2)
    parser.add_argument(
        "--output", default="runs/annual_walk_forward/annual_metrics.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
