"""Expanding daily refit of DirectRegimeMoE on the live 2026 holdout.

This is an operational-protocol evaluation, not a tuning command.  Model
hyperparameters, the stable scale and the causal scale rule are fixed before
the June-July holdout is read:

* refit all model components with rows whose ``target_time < origin``;
* estimate one global recent scale from prior raw predictions in 14 days;
* require ``target_time < origin`` for every scale-calibration observation;
* shrink 75% from the stable 0.93 scale toward the recent scale.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import (
    DirectRegimeMoE,
    RegimeMoEFeatureBuilder,
    select_mape_scale,
)
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_odre import OdreCarbonProvider
from green_observatory.carbon.thermal_margin import day_ahead_thermal_margin_features
from green_observatory.providers.entsoe_fms import a71_day_ahead_features
from green_observatory.providers.rte_exchange_schedule import day_ahead_hourly_features


def _indexed_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    index = pd.DatetimeIndex(frame.index)
    frame.index = (
        index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    )
    return frame


def _combine_indexed(*frames: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(frames).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]


def _metrics(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mape": float(100.0 * np.mean(absolute / np.abs(actual))),
        "wape": float(100.0 * absolute.sum() / np.abs(actual).sum()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(prediction.mean()),
        "n": int(len(actual)),
    }


def _load_completed(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    return frame.sort_values(["origin", "horizon"]).drop_duplicates(
        ["origin", "horizon"], keep="last"
    )


def _recent_scale(
    prior: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    stable_scale: float,
    lookback_days: int,
    recent_weight: float,
) -> tuple[float, float, int]:
    if prior.empty:
        return stable_scale, stable_scale, 0
    usable = prior[
        (prior["origin"] >= origin - pd.Timedelta(days=lookback_days))
        & (prior["origin"] < origin)
        & (prior["target_time"] < origin)
    ]
    if usable.empty:
        return stable_scale, stable_scale, 0
    recent, _ = select_mape_scale(
        usable["actual"].to_numpy(dtype=float),
        usable["prediction_raw"].to_numpy(dtype=float),
    )
    effective = stable_scale + recent_weight * (recent - stable_scale)
    return float(recent), float(effective), int(len(usable))


def _build_inputs(args: argparse.Namespace):
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
    availability_intervals = pd.concat(
        [
            pd.read_parquet(args.rte_unavailability),
            pd.read_parquet(args.rte_unavailability_holdout),
        ],
        ignore_index=True,
    ).drop_duplicates()
    forecast_frame = mix.join(price, how="outer")
    # Normalize datetime unit before feature joins (ms/ns parquet mix).
    forecast_frame.index = pd.DatetimeIndex(forecast_frame.index).as_unit("ns")
    if args.exchange_schedule:
        # Opt-in vintage-safe DA exchange programs; the builder's day_ahead
        # mask hides target hours on the next local delivery day.
        forecast_frame = forecast_frame.join(
            day_ahead_hourly_features(pd.read_parquet(args.exchange_schedule)),
            how="outer",
        )
    if args.thermal_margin:
        forecast_frame = forecast_frame.join(
            day_ahead_thermal_margin_features(forecast_frame, availability_intervals),
            how="left",
        )
    if args.entsoe_a71:
        forecast_frame = forecast_frame.join(
            a71_day_ahead_features(pd.read_parquet(args.entsoe_a71)), how="outer"
        )
    builder = RegimeMoEFeatureBuilder(
        forecast_frame,
        availability_store=RteAvailabilityFeatureStore(availability_intervals),
        availability_feature_mode="all",
        include_curve_summaries=True,
    )
    historical_origins = pd.date_range(
        _utc(args.feature_start), _utc("2026-04-29"), freq="1D"
    )
    evaluation_origins = pd.date_range(
        _utc(args.eval_start), _utc(args.eval_end), freq="1D"
    )
    # Evaluation windows before the historical cutoff (e.g. a winter backtest)
    # overlap historical_origins; dedupe so each origin is built exactly once.
    build_origins = historical_origins.append(evaluation_origins).drop_duplicates()
    x, meta = builder.build(full, build_origins, supervised=True)
    return full, x, meta, evaluation_origins


def _summarize(
    predictions: pd.DataFrame,
    full: pd.DataFrame,
    *,
    static_report: str | None,
    static_predictions: str | None,
) -> dict:
    report: dict = {"daily_refit": _metrics(predictions["actual"], predictions["prediction"])}
    raw = _metrics(predictions["actual"], predictions["prediction_raw"])
    report["daily_refit_raw_unscaled"] = raw
    d1_time = pd.DatetimeIndex(predictions["target_time"]) - pd.Timedelta(hours=24)
    d1 = full["carbon_intensity_gco2_kwh"].reindex(d1_time).to_numpy(dtype=float).copy()
    d1[d1_time >= pd.DatetimeIndex(predictions["origin"])] = np.nan
    valid = np.isfinite(d1)
    report["d1_persistence"] = _metrics(
        predictions.loc[valid, "actual"], d1[valid]
    )
    report["daily_refit_on_d1_common_rows"] = _metrics(
        predictions.loc[valid, "actual"], predictions.loc[valid, "prediction"]
    )
    if static_report and Path(static_report).exists():
        static = json.loads(Path(static_report).read_text(encoding="utf-8"))
        report["static_direct_reference"] = static["metrics_all"][
            "direct_regime_moe"
        ]
    if static_predictions and Path(static_predictions).exists():
        old = pd.read_parquet(static_predictions)
        old = old[old["model"].eq("direct_regime_moe")].copy()
        keys = ["origin", "horizon", "target_time"]
        paired = predictions.merge(
            old[keys + ["prediction"]].rename(
                columns={"prediction": "static_prediction"}
            ),
            on=keys,
            validate="one_to_one",
        )
        paired["delta_ape"] = 100.0 * (
            np.abs(paired["prediction"] - paired["actual"])
            - np.abs(paired["static_prediction"] - paired["actual"])
        ) / paired["actual"]
        daily_delta = paired.groupby("origin")["delta_ape"].mean().to_numpy()
        rng = np.random.default_rng(42)
        n_days = len(daily_delta)
        bootstrap = np.mean(
            daily_delta[
                rng.integers(0, n_days, size=(20_000, n_days))
            ],
            axis=1,
        )
        block_means = []
        n_blocks = int(np.ceil(n_days / 7))
        for _ in range(20_000):
            starts = rng.integers(0, n_days, size=n_blocks)
            sample = np.concatenate(
                [
                    daily_delta[(start + np.arange(7)) % n_days]
                    for start in starts
                ]
            )[:n_days]
            block_means.append(sample.mean())
        report["paired_vs_static"] = {
            "mape_delta_points": float(daily_delta.mean()),
            "days_better_percent": float(100.0 * np.mean(daily_delta < 0.0)),
            "day_bootstrap_95ci": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "circular_7d_block_bootstrap_95ci": [
                float(value)
                for value in np.quantile(block_means, [0.025, 0.975])
            ],
        }
    report["by_origin_month"] = {
        str(month): _metrics(group["actual"], group["prediction"])
        for month, group in predictions.groupby(predictions["origin"].dt.strftime("%Y-%m"))
    }
    report["by_horizon"] = {
        str(int(horizon)): _metrics(group["actual"], group["prediction"])
        for horizon, group in predictions.groupby("horizon")
    }
    report["by_regime"] = {
        str(int(regime)): _metrics(group["actual"], group["prediction"])
        for regime, group in predictions.groupby("regime")
    }
    report["scale_by_origin"] = [
        {
            "origin": str(origin),
            "recent_scale": float(group["recent_scale"].iloc[0]),
            "effective_scale": float(group["effective_scale"].iloc[0]),
            "scale_rows": int(group["scale_rows"].iloc[0]),
        }
        for origin, group in predictions.groupby("origin", sort=True)
    ]
    return report


def _fit_origin_raw(
    origin: pd.Timestamp, *, x: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Fit one origin from scratch and return its unscaled prediction frame."""
    train = meta["target_time"] < origin
    evaluation = meta["origin"] == origin
    if evaluation.sum() != 24:
        raise ValueError(
            f"origin {origin} has {int(evaluation.sum())} complete targets, expected 24"
        )
    model = DirectRegimeMoE(point_scale=1.0)
    model.fit(
        x.loc[train].reset_index(drop=True),
        meta.loc[train].reset_index(drop=True),
    )
    raw = model.predict_matrix(x.loc[evaluation].reset_index(drop=True))[
        "prediction"
    ].to_numpy(dtype=float)
    day = meta.loc[
        evaluation,
        ["origin", "horizon", "target_time", "actual", "regime"],
    ].reset_index(drop=True)
    day["prediction_raw"] = raw
    return day


_WORKER_STATE: dict = {}


def _init_origin_worker(x, meta) -> None:
    _WORKER_STATE.update(x=x, meta=meta)


def _fit_origin_raw_task(origin: pd.Timestamp) -> pd.DataFrame:
    return _fit_origin_raw(origin, **_WORKER_STATE)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full, x, meta, origins = _build_inputs(args)
    completed = _load_completed(output_dir)
    done = set(completed["origin"].unique()) if not completed.empty else set()
    # Phase 1 — raw fits. Each origin trains from scratch on data strictly
    # before it, so fits are independent and parallelizable. The recent-scale
    # layer only reads *raw* predictions of prior days, so it can run as a
    # cheap sequential second phase with identical results.
    pending = [origin for origin in origins if origin not in done]
    for origin in origins:
        if origin in done:
            print(f"reuse {origin.date()}", flush=True)
    raw_days: dict[pd.Timestamp, pd.DataFrame] = {}
    if args.parallel_origins > 1 and pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.parallel_origins,
            mp_context=context,
            initializer=_init_origin_worker,
            initargs=(x, meta),
        ) as executor:
            futures = {
                executor.submit(_fit_origin_raw_task, origin): origin
                for origin in pending
            }
            fit_started = time.perf_counter()
            for finished, future in enumerate(as_completed(futures), start=1):
                origin = futures[future]
                raw_days[origin] = future.result()
                print(
                    f"[fit {finished}/{len(pending)}] {origin.date()} "
                    f"elapsed={time.perf_counter()-fit_started:.0f}s",
                    flush=True,
                )
    else:
        for position, origin in enumerate(pending, start=1):
            day_started = time.perf_counter()
            raw_days[origin] = _fit_origin_raw(origin, x=x, meta=meta)
            print(
                f"[fit {position}/{len(pending)}] {origin.date()} "
                f"seconds={time.perf_counter()-day_started:.1f}",
                flush=True,
            )

    # Phase 2 — chronological scale pass (needs prior days' raw predictions).
    for origin in sorted(raw_days):
        day = raw_days[origin]
        recent_scale, effective_scale, scale_rows = _recent_scale(
            completed,
            origin,
            stable_scale=args.stable_scale,
            lookback_days=args.scale_lookback_days,
            recent_weight=args.recent_scale_weight,
        )
        day["prediction"] = effective_scale * day["prediction_raw"]
        day["recent_scale"] = recent_scale
        day["effective_scale"] = effective_scale
        day["scale_rows"] = scale_rows
        day["model"] = "direct_regime_moe_daily_expanding_refit"
        day.to_parquet(output_dir / f"{origin.date()}.parquet", index=False)
        completed = pd.concat([completed, day], ignore_index=True)
        print(
            f"scaled {origin.date()} scale={effective_scale:.4f} ({scale_rows} prior)",
            flush=True,
        )

    predictions = _load_completed(output_dir)
    predictions = predictions[predictions["origin"].isin(origins)].copy()
    metrics = _summarize(
        predictions,
        full,
        static_report=args.static_report,
        static_predictions=args.static_predictions,
    )
    report = {
        "protocol": {
            "training": "daily expanding; target_time strictly before origin",
            "origins": [args.eval_start, args.eval_end],
            "features": (
                "causal mix + price + curve summaries + versioned availability "
                "+ origin state + aligned D1/D7"
            ),
            "stable_scale": args.stable_scale,
            "scale_lookback_days": args.scale_lookback_days,
            "recent_scale_weight": args.recent_scale_weight,
            "scale_formula": (
                "stable_scale + recent_weight * (recent_scale - stable_scale)"
            ),
            "scale_cutoff": "origin prior 14d and target_time strictly before origin",
        },
        "days": int(predictions["origin"].nunique()),
        "rows": len(predictions),
        "features": len(x.columns),
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_parquet(output.with_suffix(".predictions.parquet"), index=False)
    print(json.dumps(report, indent=2), flush=True)
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
    parser.add_argument(
        "--entsoe-a71",
        default=None,
        help=(
            "Optional ENTSO-E A71 hourly snapshot parquet; adds D-1 total "
            "scheduled generation features (FMS export)"
        ),
    )
    parser.add_argument(
        "--thermal-margin",
        action="store_true",
        help=(
            "Add D-1 implied thermal-tightness features (residual demand + "
            "publication-versioned outage state)"
        ),
    )
    parser.add_argument(
        "--exchange-schedule",
        default=None,
        help=(
            "Optional RTE Exchange Schedule DA parquet; adds first-version "
            "day-ahead net scheduled exchange features per border"
        ),
    )
    parser.add_argument(
        "--parallel-origins",
        type=int,
        default=1,
        help=(
            "Worker processes fitting independent origins concurrently "
            "(raw fits parallel; the recent-scale pass stays sequential)"
        ),
    )
    parser.add_argument("--stable-scale", type=float, default=0.93)
    parser.add_argument("--scale-lookback-days", type=int, default=14)
    parser.add_argument("--recent-scale-weight", type=float, default=0.75)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--static-report",
        default="runs/daily_refit_2026/stacked_experiment/true_holdout.json",
    )
    parser.add_argument(
        "--static-predictions",
        default=(
            "runs/daily_refit_2026/stacked_experiment/"
            "true_holdout.predictions.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
