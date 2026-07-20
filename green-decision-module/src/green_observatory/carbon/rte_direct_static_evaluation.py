"""Static causal-clean Direct-MoE ablation for RTE D-1 generation forecasts.

The MAPE scale is frozen from the already completed pre-2026 experiment.  The
component model is fitted once before January-February and refitted using only
targets before March for the March-April evaluation.  This deliberately avoids
the expensive daily-refit protocol until the added data show a static gain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.consolidated_physical_calibration import (
    _window_metrics,
)
from green_observatory.carbon.consolidated_physical_evaluation import (
    KEYS,
    _indexed_parquet,
    _metrics,
    _paired_comparison,
)
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import DirectRegimeMoE, RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_odre import OdreCarbonProvider


def _fit_predict(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    train_before: pd.Timestamp,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
    point_scale: float,
    threads: int,
) -> pd.DataFrame:
    train = meta["target_time"] < train_before
    evaluation = meta["origin"].between(origin_start, origin_end)
    model = DirectRegimeMoE(
        point_scale=point_scale,
        classifier_params={"n_jobs": int(threads)},
        expert_params={"n_jobs": int(threads)},
    )
    model.fit(
        x.loc[train].reset_index(drop=True),
        meta.loc[train].reset_index(drop=True),
    )
    return model.predict(
        x.loc[evaluation].reset_index(drop=True),
        meta.loc[evaluation].reset_index(drop=True),
    )


def _reference(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    reference = pd.read_parquet(path).copy()
    for column in ("origin", "target_time"):
        reference[column] = pd.to_datetime(reference[column], utc=True)
    prediction_column = (
        "prediction" if "prediction" in reference else "direct_rte_d1"
    )
    return frame.merge(
        reference[KEYS + [prediction_column]].rename(
            columns={prediction_column: "reference_prediction"}
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    forecasts = _indexed_parquet(args.mix_forecast).join(
        _indexed_parquet(args.price_forecast), how="outer"
    )
    rte_store = (
        RteGenerationForecastFeatureStore.from_parquet(
            args.rte_generation_forecast,
            production_types=args.rte_production_type or None,
        )
        if args.rte_generation_forecast
        else None
    )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        availability_store=RteAvailabilityFeatureStore.from_parquet(
            args.rte_unavailability
        ),
        availability_feature_mode="all",
        include_curve_summaries=True,
        rte_forecast_store=rte_store,
    )
    origins = pd.date_range(_utc(args.feature_start), _utc("2026-04-29"), freq="1D")
    x, meta = builder.build(frame, origins, supervised=True)
    dev = _fit_predict(
        x,
        meta,
        train_before=_utc("2026-01-01"),
        origin_start=_utc("2026-01-01"),
        origin_end=_utc("2026-02-28"),
        point_scale=args.point_scale,
        threads=args.model_threads,
    ).rename(columns={"prediction": "direct_rte_d1"})
    holdout = _fit_predict(
        x,
        meta,
        train_before=_utc("2026-03-01"),
        origin_start=_utc("2026-03-01"),
        origin_end=_utc("2026-04-29"),
        point_scale=args.point_scale,
        threads=args.model_threads,
    ).rename(columns={"prediction": "direct_rte_d1"})
    dev["actual"] = meta.loc[
        meta["origin"].between(_utc("2026-01-01"), _utc("2026-02-28")),
        "actual",
    ].to_numpy(dtype=float)
    holdout["actual"] = meta.loc[
        meta["origin"].between(_utc("2026-03-01"), _utc("2026-04-29")),
        "actual",
    ].to_numpy(dtype=float)

    if args.reference_predictions:
        dev = _reference(dev, args.reference_predictions)
        holdout = _reference(holdout, args.reference_predictions)

    report_columns = ["direct_rte_d1"]
    if "reference_prediction" in dev:
        report_columns.append("reference_prediction")
    february = dev["origin"] >= _utc("2026-02-01")
    report = {
        "protocol": {
            "target": "RTE consolidated production intensity",
            "issue_state": "last fully closed hourly bin",
            "dev": "fit target_time < 2026-01-01; origins Jan-Feb",
            "holdout": "refit target_time < 2026-03-01; origins Mar-Apr",
            "point_scale": args.point_scale,
            "point_scale_provenance": "frozen from pre-2026 calibration",
            "rte_generation_forecast": (
                {
                    "path": args.rte_generation_forecast,
                    "production_types": args.rte_production_type or "all supported",
                    "visibility": "latest updated_date <= origin",
                }
                if args.rte_generation_forecast
                else None
            ),
        },
        "dev_jan_feb": {
            column: _metrics(dev["actual"], dev[column]) for column in report_columns
        },
        "february_validation": {
            column: _metrics(dev.loc[february, "actual"], dev.loc[february, column])
            for column in report_columns
        },
        "holdout_mar_apr": {
            column: _metrics(holdout["actual"], holdout[column])
            for column in report_columns
        },
        "holdout_window_selection": {
            column: _window_metrics(
                holdout, column, actual_by_time=frame["carbon_intensity_gco2_kwh"]
            )
            for column in report_columns
        },
    }
    if "reference_prediction" in holdout:
        report["paired_rte_vs_reference"] = _paired_comparison(
            holdout.rename(columns={"direct_rte_d1": "candidate"}),
            "candidate",
            "reference_prediction",
        )
    predictions = pd.concat(
        [dev.assign(split="dev_jan_feb"), holdout.assign(split="holdout_mar_apr")],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Direct MoE + RTE D-1 — ablación estática",
        "",
        "| Señal | Dev MAPE | Feb MAPE | Mar-Abr MAPE | Oracle Mar-Abr |",
        "|---|---:|---:|---:|---:|",
    ]
    for column in report_columns:
        lines.append(
            f"| `{column}` | {report['dev_jan_feb'][column]['mape']:.3f}% | "
            f"{report['february_validation'][column]['mape']:.3f}% | "
            f"{report['holdout_mar_apr'][column]['mape']:.3f}% | "
            f"{report['holdout_window_selection'][column]['pct_oracle_potential']:.1f}% |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_detailed.parquet")
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument("--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet")
    parser.add_argument("--rte-unavailability", default="data/cache/rte_unavailability_messages.parquet")
    parser.add_argument("--rte-generation-forecast", default="")
    parser.add_argument(
        "--rte-production-type",
        action="append",
        choices=("WIND_ONSHORE", "WIND_OFFSHORE", "SOLAR"),
        default=[],
    )
    parser.add_argument("--reference-predictions", default="")
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--point-scale", type=float, default=0.92)
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/rte_forecast_ablation/direct",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
