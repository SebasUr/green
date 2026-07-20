"""Reproducible temporal evaluation of the isolated direct regime MoE.

Default temporal cuts are intentionally explicit:

* component tuning train: data whose target is before 2025-01-01;
* point-scale calibration: daily origins in calendar year 2025;
* final component fit: data whose target is before 2026-01-01;
* blend calibration: January-February 2026;
* untouched blend report: March-April 2026.

The evaluator can consume the expensive daily-refit checkpoints already on
disk.  It does not modify or refit any existing fossil model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.adaptive_ensemble import (
    causal_block_scaled_expert,
    causal_scaled_expert,
    rank_consensus,
)
from green_observatory.carbon.annual_evaluation import _forecast_frame, _utc
from green_observatory.carbon.protocols import aggregate_metrics, regularize_hourly
from green_observatory.carbon.regime_moe import (
    DEFAULT_MULTISCALE_STATE_COLUMNS,
    DEFAULT_STATE_COLUMNS,
    DirectRegimeMoE,
    RegimeMoEFeatureBuilder,
    select_mape_scale,
)
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_odre import OdreCarbonProvider
from green_observatory.windows.oracle import window_selection_metrics


def _load_checkpoints(directory: str) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no daily-refit checkpoints in {directory}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    return frame.drop_duplicates(["model", "origin", "horizon"], keep="last")


def _mape(frame: pd.DataFrame) -> float:
    return float(
        100.0
        * np.mean(
            np.abs(frame["prediction"] - frame["actual"])
            / np.clip(np.abs(frame["actual"]), 1e-9, None)
        )
    )


def _mean_regret(frame: pd.DataFrame) -> float:
    regrets = []
    for _, group in frame.groupby("origin", sort=False):
        selected = group["prediction"].idxmin()
        regrets.append(float(frame.at[selected, "actual"] - group["actual"].min()))
    return float(np.mean(regrets))


def _complete_origins(frame: pd.DataFrame) -> set[pd.Timestamp]:
    complete = set()
    for origin, group in frame.groupby("origin"):
        if (
            len(group) == 24
            and set(group["horizon"].astype(int)) == set(range(1, 25))
            and np.isfinite(group[["prediction", "actual"]]).all().all()
        ):
            complete.add(origin)
    return complete


def _select_blend_weight(
    moe: pd.DataFrame,
    adaptive: pd.DataFrame,
    calibration_end: pd.Timestamp,
) -> tuple[float, pd.DataFrame, list[dict]]:
    keys = ["origin", "horizon", "target_time"]
    merged = moe[keys + ["prediction", "actual"]].merge(
        adaptive[keys + ["prediction"]],
        on=keys,
        suffixes=("_moe", "_adaptive"),
        validate="one_to_one",
    )
    calibration = merged["origin"] < calibration_end
    candidates = []
    for weight in np.arange(0.0, 1.01, 0.10):
        prediction = (
            weight * merged["prediction_moe"]
            + (1.0 - weight) * merged["prediction_adaptive"]
        )
        loss = 100.0 * np.mean(
            np.abs(prediction[calibration] - merged.loc[calibration, "actual"])
            / merged.loc[calibration, "actual"]
        )
        candidates.append({"moe_weight": float(weight), "calibration_mape": float(loss)})
    best = min(candidates, key=lambda row: (row["calibration_mape"], row["moe_weight"]))
    weight = best["moe_weight"]
    out = merged[keys + ["actual"]].copy()
    out["prediction"] = (
        weight * merged["prediction_moe"]
        + (1.0 - weight) * merged["prediction_adaptive"]
    )
    out["model"] = f"regime_moe_adaptive_w{weight:.1f}"
    return weight, out, candidates


def _select_level_shape_blend(
    moe: pd.DataFrame,
    level_shape: pd.DataFrame,
    calibration_end: pd.Timestamp,
) -> tuple[float, pd.DataFrame, list[dict]]:
    """Select a compact MoE/causal-shape blend using only prior origins."""
    keys = ["origin", "horizon", "target_time"]
    merged = moe[keys + ["prediction", "actual"]].merge(
        level_shape[keys + ["prediction"]],
        on=keys,
        suffixes=("_moe", "_level_shape"),
        validate="one_to_one",
    )
    calibration = merged["origin"] < calibration_end
    candidates = []
    for weight in np.arange(0.0, 1.001, 0.05):
        prediction = (
            weight * merged["prediction_moe"]
            + (1.0 - weight) * merged["prediction_level_shape"]
        )
        loss = 100.0 * np.mean(
            np.abs(prediction[calibration] - merged.loc[calibration, "actual"])
            / merged.loc[calibration, "actual"]
        )
        candidates.append({"moe_weight": float(weight), "calibration_mape": float(loss)})
    best = min(candidates, key=lambda row: (row["calibration_mape"], row["moe_weight"]))
    weight = best["moe_weight"]
    out = merged[keys + ["actual"]].copy()
    out["prediction"] = (
        weight * merged["prediction_moe"]
        + (1.0 - weight) * merged["prediction_level_shape"]
    )
    out["model"] = f"regime_moe_level_shape_w{weight:.2f}"
    return weight, out, candidates


def _selector_bias_candidate(
    recent: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    method: str,
    shrink: float,
) -> pd.DataFrame:
    calibration = recent[recent["origin"] < fit_end]
    residual = calibration["actual"] - calibration["prediction"]
    grouped = residual.groupby(calibration["horizon"])
    offset = grouped.mean() if method == "mean" else grouped.median()
    out = recent.copy()
    apply = out["origin"] >= fit_end
    out.loc[apply, "prediction"] += shrink * out.loc[apply, "horizon"].map(offset)
    out["model"] = f"recent_mapper_horizon_{method}_s{shrink:.2f}"
    return out


def _select_robust_selector(recent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Choose correction on Jan->Feb, refit Jan-Feb, then apply from March."""
    february = recent[
        (recent["origin"] >= _utc("2026-02-01"))
        & (recent["origin"] < _utc("2026-03-01"))
    ]
    candidates = []
    for method in ("mean", "median"):
        for shrink in (0.0, 0.25, 0.50, 0.75, 1.0):
            candidate = _selector_bias_candidate(
                recent, fit_end=_utc("2026-02-01"), method=method, shrink=shrink
            )
            validation = candidate[candidate["origin"].isin(february["origin"].unique())]
            candidates.append(
                {
                    "method": method,
                    "shrink": float(shrink),
                    "february_regret": _mean_regret(validation),
                }
            )
    selected = min(
        candidates,
        key=lambda row: (row["february_regret"], row["shrink"], row["method"]),
    )
    final = _selector_bias_candidate(
        recent,
        fit_end=_utc("2026-03-01"),
        method=selected["method"],
        shrink=selected["shrink"],
    )
    final["model"] = "recent_mapper_robust_horizon_bias"
    return final, {"selected": selected, "candidates": candidates}


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.round(6).to_json(orient="records"))


def run(args: argparse.Namespace) -> dict:
    carbon = OdreCarbonProvider.load_snapshot(args.carbon)
    full = regularize_hourly(carbon)
    # The Energy-Charts mix forecast is the causal backbone used by the model.
    # Extra forecast snapshots are opt-in: this keeps the published experiment
    # reproducible and avoids silently introducing weather reanalysis as if it
    # had been an operational forecast.  Any numeric column in an extra frame
    # (for example the known day-ahead market price) is picked up generically by
    # RegimeMoEFeatureBuilder as ``fc_<column>``.
    forecast_paths = [args.mix_forecast, *args.extra_forecast]
    forecasts = _forecast_frame(forecast_paths)
    excluded = [
        column for column in args.exclude_forecast_column if column in forecasts
    ]
    if excluded:
        forecasts = forecasts.drop(columns=excluded)
    availability_store = (
        RteAvailabilityFeatureStore.from_parquet(args.rte_unavailability)
        if args.rte_unavailability
        else None
    )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        target_lags_hours=tuple(
            dict.fromkeys((24, 168, *args.extra_target_lag_hour))
        ),
        state_columns=tuple(
            dict.fromkeys((*DEFAULT_STATE_COLUMNS, *args.extra_state_column))
        ),
        availability_store=availability_store,
        availability_feature_mode=args.availability_feature_mode,
        include_curve_summaries=args.curve_summaries,
        include_detailed_state=args.detailed_state,
        include_multiscale_state=args.multiscale_state,
        multiscale_state_columns=(
            args.multiscale_state_column or DEFAULT_MULTISCALE_STATE_COLUMNS
        ),
    )

    feature_origins = pd.date_range(
        _utc(args.feature_start), _utc(args.eval_end), freq="1D"
    )
    x, meta = builder.build(full, feature_origins, supervised=True)
    scale_train_end = _utc(args.scale_train_end)
    final_train_end = _utc(args.eval_start)
    calibration = (meta["origin"] >= scale_train_end) & (
        meta["origin"] < final_train_end
    )
    tuning_train = meta["target_time"] < scale_train_end
    final_train = meta["target_time"] < final_train_end
    evaluation = (meta["origin"] >= final_train_end) & (
        meta["origin"] <= _utc(args.eval_end)
    )

    classifier_params = {
        "class_weight": (
            None if args.classifier_class_weight == "none" else "balanced"
        )
    }
    tuning = DirectRegimeMoE(classifier_params=classifier_params)
    tuning.fit(
        x.loc[tuning_train].reset_index(drop=True),
        meta.loc[tuning_train].reset_index(drop=True),
    )
    calibration_raw = tuning.predict_matrix(x.loc[calibration].reset_index(drop=True))
    point_scale, calibration_mape = select_mape_scale(
        meta.loc[calibration, "actual"].to_numpy(),
        calibration_raw["prediction"].to_numpy(),
    )

    model = DirectRegimeMoE(
        point_scale=point_scale, classifier_params=classifier_params
    )
    model.fit(
        x.loc[final_train].reset_index(drop=True),
        meta.loc[final_train].reset_index(drop=True),
    )
    moe = model.predict(
        x.loc[evaluation].reset_index(drop=True),
        meta.loc[evaluation].reset_index(drop=True),
    )
    moe["actual"] = meta.loc[evaluation, "actual"].to_numpy()

    checkpoints = _load_checkpoints(args.daily_predictions)
    adaptive = causal_scaled_expert(checkpoints, lookback_days=7, name="adaptive_signal_7d")
    level_shape = causal_block_scaled_expert(
        checkpoints,
        lookback_days=21,
        half_life_days=3.0,
        blocks=((1, 6), (7, 16), (17, 21), (22, 24)),
        block_weight=0.25,
        shape_expert="hybrid_h2_mapper_delta",
        shape_weight=0.10,
        name="causal_level_shape_21d",
    )
    ranks = rank_consensus(checkpoints, name="rank_consensus")
    recent = checkpoints[
        checkpoints["model"].eq("fossil_regime_recent_mapper")
    ].copy()
    common = (
        _complete_origins(moe)
        & _complete_origins(adaptive)
        & _complete_origins(level_shape)
        & _complete_origins(ranks)
        & _complete_origins(recent)
    )
    moe = moe[moe["origin"].isin(common)].copy()
    adaptive = adaptive[adaptive["origin"].isin(common)].copy()
    level_shape = level_shape[level_shape["origin"].isin(common)].copy()
    ranks = ranks[ranks["origin"].isin(common)].copy()
    recent = recent[recent["origin"].isin(common)].copy()

    blend_weight, blend, blend_candidates = _select_blend_weight(
        moe, adaptive, _utc(args.blend_holdout_start)
    )
    level_shape_weight, level_shape_blend, level_shape_blend_candidates = (
        _select_level_shape_blend(
            moe, level_shape, _utc(args.blend_holdout_start)
        )
    )
    robust_selector, selector_selection = _select_robust_selector(recent)
    predictions = pd.concat(
        [
            moe[["origin", "horizon", "target_time", "prediction", "actual", "model"]],
            adaptive[["origin", "horizon", "target_time", "prediction", "actual", "model"]],
            blend,
            level_shape[
                ["origin", "horizon", "target_time", "prediction", "actual", "model"]
            ],
            level_shape_blend,
            recent[["origin", "horizon", "target_time", "prediction", "actual", "model"]],
            robust_selector[["origin", "horizon", "target_time", "prediction", "actual", "model"]],
        ],
        ignore_index=True,
    )
    holdout = predictions["origin"] >= _utc(args.blend_holdout_start)
    metrics = aggregate_metrics(predictions)
    holdout_metrics = (
        aggregate_metrics(predictions.loc[holdout])
        if holdout.any()
        else pd.DataFrame()
    )
    selection_predictions = pd.concat(
        [
            predictions,
            ranks[["origin", "horizon", "target_time", "prediction", "actual", "model"]],
        ],
        ignore_index=True,
    )
    selection = window_selection_metrics(selection_predictions, full).reset_index()
    holdout_selection_frame = selection_predictions.loc[
        selection_predictions["origin"] >= _utc(args.blend_holdout_start)
    ]
    holdout_selection = (
        window_selection_metrics(holdout_selection_frame, full).reset_index()
        if not holdout_selection_frame.empty
        else pd.DataFrame()
    )

    report = {
        "protocol": {
            "scale_component_train_target_before": str(scale_train_end),
            "scale_calibration_origins": [str(scale_train_end), str(final_train_end)],
            "final_component_train_target_before": str(final_train_end),
            "blend_calibration_before": args.blend_holdout_start,
            "untouched_blend_holdout": (
                [args.blend_holdout_start, args.eval_end]
                if _utc(args.eval_end) >= _utc(args.blend_holdout_start)
                else []
            ),
            "day_ahead_visibility": (
                "masked when target local date is after origin local date"
            ),
            "multiscale_state": bool(args.multiscale_state),
            "detailed_state": bool(args.detailed_state),
            "extra_state_columns": list(args.extra_state_column),
            "classifier_class_weight": args.classifier_class_weight,
            "days": len(common),
        },
        "point_scale": point_scale,
        "scale_calibration_mape": calibration_mape,
        "blend_weight": blend_weight,
        "blend_candidates": blend_candidates,
        "level_shape_blend_weight": level_shape_weight,
        "level_shape_blend_candidates": level_shape_blend_candidates,
        "selector_selection": selector_selection,
        "metrics_full": _records(metrics),
        "metrics_holdout": _records(holdout_metrics),
        "selection_full": _records(selection),
        "selection_holdout": _records(holdout_selection),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    selection_predictions.to_parquet(
        output.with_suffix(".predictions.parquet"), index=False
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet")
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument(
        "--extra-forecast",
        action="append",
        default=[],
        help=(
            "Optional timestamp-indexed forecast parquet to join (repeatable), "
            "for example data/cache/day_ahead_price_fr_hourly.parquet"
        ),
    )
    parser.add_argument(
        "--exclude-forecast-column",
        action="append",
        default=[],
        help="Forecast column to exclude after joining snapshots (repeatable)",
    )
    parser.add_argument(
        "--rte-unavailability",
        default=None,
        help="Optional versioned RTE unavailability-message parquet",
    )
    parser.add_argument(
        "--availability-feature-mode",
        choices=("all", "delta"),
        default="all",
    )
    parser.add_argument(
        "--curve-summaries",
        action="store_true",
        help="Add day-level forecast summaries and per-hour forecast ramps",
    )
    parser.add_argument(
        "--multiscale-state",
        action="store_true",
        help=(
            "Add causal 3/6/12/24/48/168h level, trend and volatility "
            "summaries for a compact set of physical state variables"
        ),
    )
    parser.add_argument(
        "--detailed-state",
        action="store_true",
        help=(
            "Add opt-in RTE fuel, hydro, bioenergy and commercial-border "
            "state columns at the origin and aligned D-1/D-7 lags"
        ),
    )
    parser.add_argument(
        "--extra-state-column",
        action="append",
        default=[],
        help=(
            "Append one physical state variable at the origin and D-1/D-7 "
            "target-aligned lags (repeatable)"
        ),
    )
    parser.add_argument(
        "--extra-target-lag-hour",
        action="append",
        default=[],
        type=int,
        help=(
            "Add a target-aligned observation lag in hours (repeatable); "
            "24 and 168 remain enabled"
        ),
    )
    parser.add_argument(
        "--multiscale-state-column",
        action="append",
        default=[],
        help=(
            "Restrict multi-scale summaries to this state column "
            "(repeatable; defaults to the compact physical set)"
        ),
    )
    parser.add_argument("--daily-predictions", required=True)
    parser.add_argument(
        "--classifier-class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Regime-classifier weighting; 'none' uses empirical frequencies",
    )
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--scale-train-end", default="2025-01-01")
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--eval-end", default="2026-04-29")
    parser.add_argument("--blend-holdout-start", default="2026-03-01")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
