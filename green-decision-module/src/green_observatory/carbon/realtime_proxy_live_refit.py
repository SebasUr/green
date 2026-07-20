"""Daily expanding refit for the operational RTE real-time carbon proxy.

Every live origin is treated exactly as it would be in production: the direct
and/or physical proxy model is fitted with rows whose target timestamp is
strictly earlier than that origin, then the next 24 hours are predicted.  Each
day is checkpointed independently so a long run is resumable.

No level scale from the consolidated ``taux_co2`` experiments is reused.  All
reported predictions are raw outputs in the provisional proxy definition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from green_observatory.carbon.protocols import _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.realtime_proxy import (
    PhysicalProxyMoE,
    physical_targets,
    proxy_training_frame,
)
from green_observatory.carbon.protocols import fossil_regime_labels
from green_observatory.carbon.regime_moe import DirectRegimeMoE, RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_base import CARBON
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
    return frame.sort_index()


def _combine_indexed(*frames: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(frames).sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def _build_state_features_with_rte_supervision(
    builder: RegimeMoEFeatureBuilder,
    state_frame: pd.DataFrame,
    rte_label_frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build state/lags from one frame while accepting labels only from RTE."""

    x, meta = builder.build(state_frame, origins, supervised=False)
    target_times = pd.DatetimeIndex(meta["target_time"])
    meta["actual"] = rte_label_frame[CARBON].reindex(target_times).to_numpy(
        dtype=float
    )
    regimes = fossil_regime_labels(
        rte_label_frame, ccg_threshold_mw=500.0, peak_threshold_mw=2500.0
    )
    meta["regime"] = regimes.reindex(target_times).to_numpy()
    valid = meta[["actual", "regime"]].notna().all(axis=1)
    return (
        x.loc[valid].reset_index(drop=True),
        meta.loc[valid].reset_index(drop=True),
    )


def _metrics(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
    actual = actual[valid]
    prediction = prediction[valid]
    if len(actual) == 0:
        return {
            "mape": float("nan"),
            "wape": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "actual_mean": float("nan"),
            "prediction_mean": float("nan"),
            "n": 0,
        }
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mape": float(100.0 * np.mean(absolute / actual)),
        "wape": float(100.0 * absolute.sum() / actual.sum()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(prediction.mean()),
        "n": int(len(actual)),
    }


def _physical_variants(matrix: pd.DataFrame) -> dict[str, np.ndarray]:
    probability = matrix[["prob_baseload", "prob_ccg", "prob_peak"]].to_numpy()
    experts = matrix[
        ["gas_expert_baseload_mw", "gas_expert_ccg_mw", "gas_expert_peak_mw"]
    ].to_numpy()
    base = matrix["prediction"].to_numpy(dtype=float)
    base_gas = matrix["predicted_gas_mw"].to_numpy(dtype=float)
    denominator = matrix["predicted_total_generation_mw"].to_numpy(dtype=float)
    variants = {"physical": base}
    if "prediction_pooled" in matrix:
        variants["physical_pooled"] = matrix["prediction_pooled"].to_numpy(dtype=float)
    for alpha in (2.0, 3.0, 5.0):
        sharpened = probability**alpha
        sharpened /= np.clip(sharpened.sum(axis=1, keepdims=True), 1e-12, None)
        gas = np.sum(sharpened * experts, axis=1)
        variants[f"physical_alpha{int(alpha)}"] = np.clip(
            base + 429.0 * (gas - base_gas) / denominator, 0.0, None
        )
    hard = np.zeros_like(probability)
    hard[np.arange(len(hard)), np.argmax(probability, axis=1)] = 1.0
    gas = np.sum(hard * experts, axis=1)
    variants["physical_hard"] = np.clip(
        base + 429.0 * (gas - base_gas) / denominator, 0.0, None
    )
    return variants


def _load_completed(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("2026-*.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    return frame.sort_values(["origin", "horizon"]).drop_duplicates(
        ["origin", "horizon"], keep="last"
    )


def _build_inputs(args: argparse.Namespace):
    historical_published = regularize_hourly(
        OdreCarbonProvider.load_snapshot(args.carbon)
    )
    live_published = _indexed_parquet(args.carbon_live)
    combined_published = regularize_hourly(
        _combine_indexed(historical_published, live_published)
    )
    proxy = proxy_training_frame(combined_published, carbon_column=CARBON)
    state_published = combined_published
    if args.carbon_gap:
        # Energy-Charts is a different source/vintage.  RTE wins anywhere the
        # two overlap, and the bridge is used only by the feature builder.
        # ``proxy`` above remains the label/component-training frame.
        gap = _indexed_parquet(args.carbon_gap)
        state_published = regularize_hourly(
            _combine_indexed(historical_published, gap, live_published)
        )
    state_proxy = proxy_training_frame(state_published, carbon_column=CARBON)
    forecasts = _combine_indexed(
        _indexed_parquet(args.mix_forecast),
        _indexed_parquet(args.mix_forecast_live),
    ).join(
        _combine_indexed(
            _indexed_parquet(args.price_forecast),
            _indexed_parquet(args.price_forecast_live),
        ),
        how="outer",
    )
    # Parquet round-trips mix ms/ns datetime units; joining mismatched units
    # silently degrades the index to object. Normalize once at the boundary.
    forecasts.index = pd.DatetimeIndex(forecasts.index).as_unit("ns")
    if args.exchange_schedule:
        # Opt-in vintage-safe DA exchange programs; the builder's day_ahead
        # mask hides target hours on the next local delivery day.
        forecasts = forecasts.join(
            day_ahead_hourly_features(pd.read_parquet(args.exchange_schedule)),
            how="outer",
        )
    if args.entsoe_a71:
        forecasts = forecasts.join(
            a71_day_ahead_features(pd.read_parquet(args.entsoe_a71)), how="outer"
        )
    intervals = pd.concat(
        [
            pd.read_parquet(args.rte_unavailability),
            pd.read_parquet(args.rte_unavailability_live),
        ],
        ignore_index=True,
    ).drop_duplicates()
    if args.thermal_margin:
        # Opt-in D-1 implied-tightness columns (residual demand + outage
        # state at Paris-midnight vintage); day_ahead names get masked.
        forecasts = forecasts.join(
            day_ahead_thermal_margin_features(forecasts, intervals), how="left"
        )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        availability_store=RteAvailabilityFeatureStore(intervals),
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=True,
    )
    historical_origins = pd.date_range(
        _utc(args.feature_start), _utc("2026-04-29"), freq="1D"
    )
    evaluation_origins = pd.date_range(
        _utc(args.eval_start), _utc(args.eval_end), freq="1D"
    )
    # June 16 contributes any labels already observed before the first formal
    # origin.  It is never itself reported as an evaluation day.
    adaptation_start = _utc(args.eval_start) - pd.Timedelta(days=1)
    origins = historical_origins.append(
        pd.date_range(adaptation_start, _utc(args.eval_end), freq="1D")
    ).drop_duplicates()
    # Build state and target-aligned lags from the optional bridge, then attach
    # supervision exclusively from RTE.  This prevents synthetic gap carbon or
    # Energy-Charts fuel values from becoming labels/training targets.
    x, meta = _build_state_features_with_rte_supervision(
        builder, state_proxy, proxy, origins
    )
    return proxy, state_proxy, combined_published, x, meta, evaluation_origins


def _summarize(predictions: pd.DataFrame) -> dict:
    model_columns = [
        column
        for column in (
            "direct",
            "physical",
            "physical_pooled",
            "physical_alpha2",
            "physical_alpha3",
            "physical_alpha5",
            "physical_hard",
            "d1",
        )
        if column in predictions
    ]
    report = {
        "overall": {
            column: _metrics(predictions["actual"], predictions[column])
            for column in model_columns
        },
        "by_month": {},
        "by_horizon": {},
    }
    for month, group in predictions.groupby(predictions["origin"].dt.strftime("%Y-%m")):
        report["by_month"][month] = {
            column: _metrics(group["actual"], group[column])
            for column in model_columns
        }
    for horizon, group in predictions.groupby("horizon"):
        report["by_horizon"][str(int(horizon))] = {
            column: _metrics(group["actual"], group[column])
            for column in model_columns
        }
    return report


def _evaluate_origin(
    origin: pd.Timestamp,
    *,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    proxy: pd.DataFrame,
    state_carbon: pd.Series,
    published_carbon: pd.Series,
    model: str,
    pooled_gas: bool,
    model_threads: int,
) -> tuple[pd.DataFrame, int]:
    """Fit from scratch and predict one origin (no cross-origin state)."""
    train = meta["target_time"] < origin
    evaluation = meta["origin"] == origin
    if int(evaluation.sum()) != 24:
        raise ValueError(
            f"origin {origin} has {int(evaluation.sum())} targets, expected 24"
        )
    x_train = x.loc[train].reset_index(drop=True)
    meta_train = meta.loc[train].reset_index(drop=True)
    x_eval = x.loc[evaluation].reset_index(drop=True)
    meta_eval = meta.loc[evaluation].reset_index(drop=True)
    out = meta_eval[["origin", "horizon", "target_time", "actual", "regime"]].copy()
    out = out.rename(columns={"actual": "proxy_actual"})
    out["actual"] = published_carbon.reindex(
        pd.DatetimeIndex(out["target_time"])
    ).to_numpy(dtype=float)
    d1_times = pd.DatetimeIndex(out["target_time"]) - pd.Timedelta(hours=24)
    d1 = state_carbon.reindex(d1_times).to_numpy(dtype=float).copy()
    d1[d1_times >= pd.DatetimeIndex(out["origin"])] = np.nan
    out["d1"] = d1

    if model in {"direct", "both"}:
        direct = DirectRegimeMoE(point_scale=1.0)
        direct.fit(x_train, meta_train)
        out["direct"] = direct.predict_matrix(x_eval)["prediction"].to_numpy()
    if model in {"physical", "both"}:
        target_meta = meta_train.copy()
        components = physical_targets(proxy, target_meta["target_time"])
        for column in components:
            target_meta[column] = components[column].to_numpy()
        physical = PhysicalProxyMoE(
            warm_season_coal_persistence=True,
            include_pooled_gas=pooled_gas,
            classifier_params={"n_jobs": model_threads},
            source_params={"n_jobs": model_threads},
        )
        physical.fit(x_train, target_meta)
        matrix = physical.predict_matrix(x_eval)
        for name, values in _physical_variants(matrix).items():
            out[name] = values
        for column in (
            "predicted_gas_mw",
            "predicted_coal_mw",
            "predicted_coal_model_mw",
            "predicted_fuel_oil_mw",
            "predicted_bioenergy_mw",
            "predicted_total_generation_mw",
            "prob_baseload",
            "prob_ccg",
            "prob_peak",
            "gas_expert_baseload_mw",
            "gas_expert_ccg_mw",
            "gas_expert_peak_mw",
        ):
            out[column] = matrix[column].to_numpy()
        if "predicted_gas_pooled_mw" in matrix:
            out["predicted_gas_pooled_mw"] = matrix[
                "predicted_gas_pooled_mw"
            ].to_numpy()
    return out, int(train.sum())


#: Per-process state for parallel origin evaluation (set by the initializer;
#: sent once per worker instead of once per task).
_WORKER_STATE: dict = {}


def _init_origin_worker(
    x, meta, proxy, state_carbon, published_carbon, model, pooled_gas, model_threads
) -> None:
    _WORKER_STATE.update(
        x=x,
        meta=meta,
        proxy=proxy,
        state_carbon=state_carbon,
        published_carbon=published_carbon,
        model=model,
        pooled_gas=pooled_gas,
        model_threads=model_threads,
    )


def _evaluate_origin_task(origin: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    return _evaluate_origin(origin, **_WORKER_STATE)


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy, state_proxy, published, x, meta, origins = _build_inputs(args)
    completed = _load_completed(output_dir)
    done = set(completed["origin"].unique()) if not completed.empty else set()

    evaluate_kwargs = dict(
        x=x,
        meta=meta,
        proxy=proxy,
        state_carbon=state_proxy[CARBON],
        published_carbon=published[CARBON],
        model=args.model,
        pooled_gas=args.pooled_gas,
        model_threads=args.model_threads,
    )
    pending = [origin for origin in origins if origin not in done]
    for origin in origins:
        if origin in done:
            print(f"reuse {origin.date()}", flush=True)

    if args.parallel_origins > 1 and pending:
        # Each origin trains from scratch on data strictly before it, so
        # origins are embarrassingly parallel. Keep LightGBM threads low
        # (ideally --model-threads 1) to avoid OpenMP oversubscription.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.parallel_origins,
            mp_context=context,
            initializer=_init_origin_worker,
            initargs=(
                x,
                meta,
                proxy,
                state_proxy[CARBON],
                published[CARBON],
                args.model,
                args.pooled_gas,
                args.model_threads,
            ),
        ) as executor:
            futures = {
                executor.submit(_evaluate_origin_task, origin): origin
                for origin in pending
            }
            finished = 0
            started = time.perf_counter()
            for future in as_completed(futures):
                origin = futures[future]
                out, train_rows = future.result()
                path = output_dir / f"{origin.date()}.parquet"
                out.to_parquet(path, index=False)
                finished += 1
                print(
                    f"[{finished}/{len(pending)}] saved {path.name} "
                    f"train={train_rows} elapsed={time.perf_counter()-started:.0f}s",
                    flush=True,
                )
    else:
        for position, origin in enumerate(pending, start=1):
            started = time.perf_counter()
            out, train_rows = _evaluate_origin(origin, **evaluate_kwargs)
            path = output_dir / f"{origin.date()}.parquet"
            out.to_parquet(path, index=False)
            print(
                f"[{position}/{len(pending)}] saved {path.name} "
                f"train={train_rows} in {time.perf_counter()-started:.1f}s",
                flush=True,
            )

    predictions = _load_completed(output_dir)
    predictions = predictions[
        predictions["origin"].isin(origins)
    ].reset_index(drop=True)
    report = {
        "protocol": {
            "fit": "daily expanding, target_time strictly before origin",
            "target": "operational RTE physical proxy",
            "published_evaluation_target": "real-time taux_co2",
            "point_scale": 1.0,
            "coal_head": "D-1 persistence in Feb-Oct; learned regressor in Nov-Jan",
            "coal_head_status": "exploratory_after_first_live_diagnostic",
            "probability_alphas": [1, 2, 3, 5, "hard"],
            "alpha_selection": "none on live; all variants reported",
            "model": args.model,
            "pooled_gas": bool(args.pooled_gas),
            "model_threads": int(args.model_threads),
            "carbon_gap": args.carbon_gap,
            "carbon_gap_role": (
                "Energy-Charts state/lag only; never labels or training origins"
                if args.carbon_gap
                else "disabled"
            ),
            "carbon_gap_status": (
                "retrospective exploratory; timestamp-causal but not "
                "vintage-causal because the snapshot was downloaded later"
                if args.carbon_gap
                else "not applicable"
            ),
            "thermal_margin": bool(args.thermal_margin),
            "entsoe_a71": args.entsoe_a71,
            "exchange_schedule": args.exchange_schedule,
            "exchange_schedule_role": (
                "first-version DA scheduled exchanges, vintage-causal per "
                "publication; day_ahead-masked for next-local-day targets"
                if args.exchange_schedule
                else "disabled"
            ),
            "origins": [args.eval_start, args.eval_end],
        },
        **_summarize(predictions),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_detailed.parquet")
    parser.add_argument("--carbon-live", default="data/cache/carbon_fr_realtime_holdout.parquet")
    parser.add_argument(
        "--carbon-gap",
        default=None,
        help=(
            "Optional Energy-Charts hourly bridge used only for feature state/lags; "
            "it is never used as an RTE label or component-training target"
        ),
    )
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
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument("--mix-forecast-live", default="data/cache/mix_day_ahead_fr_holdout.parquet")
    parser.add_argument("--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet")
    parser.add_argument("--price-forecast-live", default="data/cache/day_ahead_price_fr_holdout.parquet")
    parser.add_argument("--rte-unavailability", default="data/cache/rte_unavailability_messages.parquet")
    parser.add_argument("--rte-unavailability-live", default="data/cache/rte_unavailability_messages_holdout.parquet")
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--eval-start", default="2026-06-17")
    parser.add_argument("--eval-end", default="2026-07-15")
    parser.add_argument("--model", choices=("direct", "physical", "both"), default="both")
    parser.add_argument(
        "--pooled-gas",
        action="store_true",
        help="Fit and report the opt-in pooled gas head alongside regime variants",
    )
    parser.add_argument(
        "--model-threads",
        type=int,
        default=1,
        help="LightGBM threads per component fit",
    )
    parser.add_argument(
        "--parallel-origins",
        type=int,
        default=1,
        help=(
            "Worker processes fitting independent origins concurrently "
            "(each origin trains from scratch; combine with --model-threads 1)"
        ),
    )
    parser.add_argument("--output-dir", default="runs/daily_refit_2026/realtime_proxy_daily_refit")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
