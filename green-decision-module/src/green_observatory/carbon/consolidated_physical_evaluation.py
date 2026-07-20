"""Strict Jan-Feb / Mar-Apr evaluation of the consolidated physical head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.consolidated_physical import (
    EMITTING_COMPONENTS,
    ConsolidatedPhysicalRegressor,
    detailed_physical_targets,
)
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder, select_mape_scale
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


KEYS = ["origin", "horizon", "target_time"]


def _indexed_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    index = pd.DatetimeIndex(frame.index)
    frame.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return frame.sort_index()


def _metrics(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
    actual = actual[valid]
    prediction = prediction[valid]
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


def _paired_comparison(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
) -> dict:
    actual = frame["actual"].to_numpy(dtype=float)
    delta = 100.0 * (
        np.abs(frame[candidate].to_numpy(dtype=float) - actual)
        - np.abs(frame[reference].to_numpy(dtype=float) - actual)
    ) / actual
    daily = pd.Series(delta, index=frame["origin"]).groupby(level=0).mean().to_numpy()
    rng = np.random.default_rng(42)
    n_days = len(daily)
    bootstrap = np.mean(
        daily[rng.integers(0, n_days, size=(20_000, n_days))], axis=1
    )
    block_means = []
    n_blocks = int(np.ceil(n_days / 7))
    for _ in range(20_000):
        starts = rng.integers(0, n_days, size=n_blocks)
        sample = np.concatenate(
            [daily[(start + np.arange(7)) % n_days] for start in starts]
        )[:n_days]
        block_means.append(sample.mean())
    return {
        "mape_delta_points": float(delta.mean()),
        "days_better_percent": float(100.0 * np.mean(daily < 0.0)),
        "day_bootstrap_95ci": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "circular_7d_block_bootstrap_95ci": [
            float(value) for value in np.quantile(block_means, [0.025, 0.975])
        ],
    }


def _attach_targets(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    physical = detailed_physical_targets(frame, out["target_time"])
    for column in physical:
        out[column] = physical[column].to_numpy()
    return out


def _predict(
    model: ConsolidatedPhysicalRegressor,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    mask: pd.Series,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    x_eval = x.loc[mask].reset_index(drop=True)
    out = meta.loc[mask, KEYS + ["actual"]].reset_index(drop=True)
    matrix = model.predict_matrix(x_eval)
    out["physical_lgbm"] = matrix["prediction"].to_numpy()
    if "prediction_unreconciled" in matrix:
        out["physical_lgbm_unreconciled"] = matrix[
            "prediction_unreconciled"
        ].to_numpy()
    for column in matrix:
        if column != "prediction":
            out[column] = matrix[column].to_numpy()

    # Causal D-1 component ablations.  The hour labelled exactly at origin is
    # not closed, so h24 falls back to the model prediction component-wise.
    targets = pd.DatetimeIndex(out["target_time"])
    origins = pd.DatetimeIndex(out["origin"])
    d1_times = targets - pd.Timedelta(hours=24)
    predicted_components = np.column_stack(
        [out[f"predicted_{column}"].to_numpy(dtype=float) for column in EMITTING_COMPONENTS]
    )
    d1_components = []
    for column in EMITTING_COMPONENTS:
        values = frame[column].reindex(d1_times).to_numpy(dtype=float).copy()
        values[d1_times >= origins] = np.nan
        d1_components.append(values)
    d1_components = np.column_stack(d1_components)
    usable_d1 = np.where(np.isfinite(d1_components), d1_components, predicted_components)
    factors = model.emission_factors_
    denominator = out["predicted_total_generation_mw"].to_numpy(dtype=float)
    sparse = predicted_components.copy()
    sparse_columns = ("coal_mw", "fuel_oil_mw", "bioenergy_waste_mw")
    for column in sparse_columns:
        position = EMITTING_COMPONENTS.index(column)
        sparse[:, position] = usable_d1[:, position]
    out["physical_sparse_d1"] = np.clip(sparse @ factors / denominator, 0.0, None)
    out["physical_all_components_d1"] = np.clip(
        usable_d1 @ factors / denominator, 0.0, None
    )

    actual_components = detailed_physical_targets(frame, out["target_time"])
    if "predicted_gas_total_mw" in out:
        out["actual_gas_mw"] = actual_components["gas_mw"].to_numpy(dtype=float)
    actual_matrix = actual_components.loc[:, EMITTING_COMPONENTS].to_numpy(dtype=float)
    actual_denominator = actual_components["total_generation_mw"].to_numpy(dtype=float)
    out["oracle_learned_factors"] = np.clip(
        actual_matrix @ factors / actual_denominator, 0.0, None
    )
    return out


def _load_reference(path: str, keys: pd.DataFrame, model: str) -> np.ndarray:
    frame = pd.read_parquet(path)
    frame = frame[frame["model"].eq(model)].copy()
    for column in ("origin", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    merged = keys.merge(
        frame[KEYS + ["prediction"]].rename(columns={"prediction": "direct_reference"}),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    return merged["direct_reference"].to_numpy(dtype=float)


def _fit_predict_split(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    train_before: pd.Timestamp,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
    ccg_moe: bool = False,
    gas_total_reconciliation: bool = False,
    threads: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    train = meta["target_time"] < train_before
    evaluation = meta["origin"].between(origin_start, origin_end)
    train_meta = _attach_targets(frame, meta.loc[train])
    model = ConsolidatedPhysicalRegressor(
        ccg_moe=ccg_moe,
        gas_total_reconciliation=gas_total_reconciliation,
        source_params={"n_jobs": int(threads)},
    )
    model.fit(x.loc[train].reset_index(drop=True), train_meta)
    return _predict(model, x, meta, evaluation, frame), model.emission_factors_.copy()


def run(args: argparse.Namespace) -> dict:
    gas_total_reconciliation = bool(
        getattr(args, "gas_total_reconciliation", False)
    )
    default_output_dir = (
        "runs/daily_refit_2026/consolidated_physical_gas_reconciliation"
        if gas_total_reconciliation
        else "runs/daily_refit_2026/consolidated_physical"
    )
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reusable = output_dir / "predictions.parquet"
    previous_report = output_dir / "summary.json"
    if args.reuse_predictions and reusable.exists() and previous_report.exists():
        predictions = pd.read_parquet(reusable)
        for column in ("origin", "target_time"):
            predictions[column] = pd.to_datetime(predictions[column], utc=True)
        dev = predictions[predictions["split"].eq("dev_jan_feb")].copy()
        holdout = predictions[predictions["split"].eq("holdout_mar_apr")].copy()
        old_report = json.loads(previous_report.read_text(encoding="utf-8"))
        dev_factors = np.asarray(
            [old_report["dev_factors"][column] for column in EMITTING_COMPONENTS]
        )
        holdout_factors = np.asarray(
            [old_report["holdout_refit_factors"][column] for column in EMITTING_COMPONENTS]
        )
    else:
        frame = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
        forecasts = _indexed_parquet(args.mix_forecast).join(
            _indexed_parquet(args.price_forecast), how="outer"
        )
        builder = RegimeMoEFeatureBuilder(
            forecasts,
            availability_store=RteAvailabilityFeatureStore(
                pd.read_parquet(args.rte_unavailability)
            ),
            availability_feature_mode="all",
            include_curve_summaries=True,
            include_detailed_state=True,
            rte_forecast_store=(
                RteGenerationForecastFeatureStore.from_parquet(
                    args.rte_generation_forecast,
                    production_types=args.rte_production_type or None,
                )
                if args.rte_generation_forecast
                else None
            ),
        )
        origins = pd.date_range(_utc(args.feature_start), _utc("2026-04-29"), freq="1D")
        x, meta = builder.build(frame, origins, supervised=True)
        dev, dev_factors = _fit_predict_split(
            x,
            meta,
            frame,
            train_before=_utc("2026-01-01"),
            origin_start=_utc("2026-01-01"),
            origin_end=_utc("2026-02-28"),
            ccg_moe=args.ccg_moe,
            gas_total_reconciliation=gas_total_reconciliation,
            threads=args.model_threads,
        )
        holdout, holdout_factors = _fit_predict_split(
            x,
            meta,
            frame,
            train_before=_utc("2026-03-01"),
            origin_start=_utc("2026-03-01"),
            origin_end=_utc("2026-04-29"),
            ccg_moe=args.ccg_moe,
            gas_total_reconciliation=gas_total_reconciliation,
            threads=args.model_threads,
        )
    for split in (dev, holdout):
        split["direct_reference"] = _load_reference(
            args.direct_predictions, split[KEYS], "direct_regime_moe"
        )

    candidate_columns = [
        "physical_lgbm",
        "physical_sparse_d1",
        "physical_all_components_d1",
    ]
    if gas_total_reconciliation:
        candidate_columns.insert(1, "physical_lgbm_unreconciled")
    scales = {}
    january = dev["origin"] < _utc("2026-02-01")
    for column in candidate_columns:
        scale, _ = select_mape_scale(
            dev.loc[january, "actual"].to_numpy(dtype=float),
            dev.loc[january, column].to_numpy(dtype=float),
        )
        scales[column] = scale
        dev[f"{column}_jan_scaled"] = scale * dev[column]
        holdout[f"{column}_jan_scaled"] = scale * holdout[column]

    blend_weights = np.arange(0.0, 1.001, 0.05)
    february = dev["origin"] >= _utc("2026-02-01")
    january_blend_losses = []
    blend_losses_dev = []
    blend_losses_february = []
    for physical_weight in blend_weights:
        prediction = (
            physical_weight * dev["physical_lgbm"]
            + (1.0 - physical_weight) * dev["direct_reference"]
        )
        blend_losses_dev.append(_metrics(dev["actual"], prediction)["mape"])
        january_blend_losses.append(
            _metrics(dev.loc[january, "actual"], prediction[january])["mape"]
        )
        blend_losses_february.append(
            _metrics(dev.loc[february, "actual"], prediction[february])["mape"]
        )
    selected_physical_weight = float(
        blend_weights[int(np.argmin(blend_losses_dev))]
    )
    january_selected_physical_weight = float(
        blend_weights[int(np.argmin(january_blend_losses))]
    )
    january_selected_unreconciled_weight = None
    if gas_total_reconciliation:
        unreconciled_losses = []
        for physical_weight in blend_weights:
            prediction = (
                physical_weight * dev["physical_lgbm_unreconciled"]
                + (1.0 - physical_weight) * dev["direct_reference"]
            )
            unreconciled_losses.append(
                _metrics(dev.loc[january, "actual"], prediction[january])["mape"]
            )
        january_selected_unreconciled_weight = float(
            blend_weights[int(np.argmin(unreconciled_losses))]
        )
    february_selected_physical_weight = float(
        blend_weights[int(np.argmin(blend_losses_february))]
    )
    for split in (dev, holdout):
        split["january_selected_direct_physical_blend"] = (
            january_selected_physical_weight * split["physical_lgbm"]
            + (1.0 - january_selected_physical_weight) * split["direct_reference"]
        )
        if gas_total_reconciliation:
            split["january_selected_direct_unreconciled_blend"] = (
                january_selected_unreconciled_weight
                * split["physical_lgbm_unreconciled"]
                + (1.0 - january_selected_unreconciled_weight)
                * split["direct_reference"]
            )
        split["selected_direct_physical_blend"] = (
            selected_physical_weight * split["physical_lgbm"]
            + (1.0 - selected_physical_weight) * split["direct_reference"]
        )
        split["february_selected_direct_physical_blend"] = (
            february_selected_physical_weight * split["physical_lgbm"]
            + (1.0 - february_selected_physical_weight) * split["direct_reference"]
        )

    threshold_grid = np.arange(5.0, 60.001, 0.5)
    january = dev["origin"] < _utc("2026-02-01")

    def level_gate(
        frame: pd.DataFrame,
        threshold: float,
        physical_column: str = "physical_lgbm",
    ) -> pd.Series:
        return frame[physical_column].where(
            frame[physical_column] < threshold,
            frame["direct_reference"],
        )

    january_threshold_losses = [
        _metrics(dev.loc[january, "actual"], level_gate(dev, threshold)[january])[
            "mape"
        ]
        for threshold in threshold_grid
    ]
    dev_threshold_losses = [
        _metrics(dev["actual"], level_gate(dev, threshold))["mape"]
        for threshold in threshold_grid
    ]
    selected_level_threshold = float(
        threshold_grid[int(np.argmin(january_threshold_losses))]
    )
    dev_selected_level_threshold = float(
        threshold_grid[int(np.argmin(dev_threshold_losses))]
    )
    unreconciled_level_threshold = None
    if gas_total_reconciliation:
        unreconciled_threshold_losses = [
            _metrics(
                dev.loc[january, "actual"],
                level_gate(dev, threshold, "physical_lgbm_unreconciled")[january],
            )["mape"]
            for threshold in threshold_grid
        ]
        unreconciled_level_threshold = float(
            threshold_grid[int(np.argmin(unreconciled_threshold_losses))]
        )
    for split in (dev, holdout):
        split["level_gated_direct_physical"] = level_gate(
            split, selected_level_threshold
        )
        split["dev_selected_level_gated_direct_physical"] = level_gate(
            split, dev_selected_level_threshold
        )
        if gas_total_reconciliation:
            split["level_gated_direct_unreconciled"] = level_gate(
                split,
                unreconciled_level_threshold,
                "physical_lgbm_unreconciled",
            )
        physical_error = np.abs(split["physical_lgbm"] - split["actual"])
        direct_error = np.abs(split["direct_reference"] - split["actual"])
        split["point_oracle_direct_physical"] = split["physical_lgbm"].where(
            physical_error <= direct_error,
            split["direct_reference"],
        )

    report_columns = [
        "direct_reference",
        *candidate_columns,
        *(f"{column}_jan_scaled" for column in candidate_columns),
        "selected_direct_physical_blend",
        "january_selected_direct_physical_blend",
        "february_selected_direct_physical_blend",
        "level_gated_direct_physical",
        "dev_selected_level_gated_direct_physical",
        "point_oracle_direct_physical",
        "oracle_learned_factors",
    ]
    if gas_total_reconciliation:
        report_columns.extend(
            [
                "january_selected_direct_unreconciled_blend",
                "level_gated_direct_unreconciled",
            ]
        )
    report = {
        "status": (
            "gas reconciliation rejected on February validation; Mar-Apr is audit only"
            if gas_total_reconciliation
            else "exploratory architecture; Mar-Apr opened for this first evaluation"
        ),
        "protocol": {
            "target": "RTE consolidated production intensity",
            "issue_state": "last fully closed hourly bin",
            "component_model": "independent LightGBM L1 regressors",
            "gas_ccg_head": (
                "three regime experts with soft classifier mixture"
                if args.ccg_moe
                else "pooled LightGBM regressor"
            ),
            "gas_total_reconciliation": (
                "dedicated LightGBM gas_mw head; four detailed gas predictions "
                "are rescaled to its total while retaining their raw proportions"
                if gas_total_reconciliation
                else None
            ),
            "emission_factor_fit": "positive OLS, no intercept, training rows only",
            "dev": "train target_time < 2026-01-01; origins Jan-Feb",
            "holdout": "refit target_time < 2026-03-01; origins Mar-Apr",
            "scale": "optional January-only multiplicative MAPE scale",
            "rte_generation_forecast": (
                {
                    "path": args.rte_generation_forecast,
                    "production_types": args.rte_production_type or "all supported",
                    "visibility": "latest updated_date <= origin",
                }
                if args.rte_generation_forecast
                else None
            ),
            "blend_selection": (
                "physical_lgbm + causal-clean Direct; the causally clean variant "
                "is selected on January on a predeclared 0.05 grid and validated "
                "on February; legacy Jan-Feb and February-only variants are also "
                "reported for continuity"
            ),
            "level_gate_selection": (
                "if physical_lgbm < threshold use physical, otherwise Direct; "
                "threshold grid 5..60 by 0.5 selected only on January and "
                "validated on February"
            ),
        },
        "emission_components": list(EMITTING_COMPONENTS),
        "dev_factors": dict(zip(EMITTING_COMPONENTS, map(float, dev_factors))),
        "holdout_refit_factors": dict(
            zip(EMITTING_COMPONENTS, map(float, holdout_factors))
        ),
        "january_scales": scales,
        "selected_physical_weight": selected_physical_weight,
        "january_selected_physical_weight": january_selected_physical_weight,
        "january_selected_unreconciled_weight": (
            january_selected_unreconciled_weight
        ),
        "february_selected_physical_weight": february_selected_physical_weight,
        "selected_level_threshold_january": selected_level_threshold,
        "selected_unreconciled_level_threshold_january": (
            unreconciled_level_threshold
        ),
        "diagnostic_level_threshold_jan_feb": dev_selected_level_threshold,
        "dev_jan_feb": {
            column: _metrics(dev["actual"], dev[column]) for column in report_columns
        },
        "holdout_mar_apr": {
            column: _metrics(holdout["actual"], holdout[column])
            for column in report_columns
        },
    }
    report["january_selection"] = {
        column: _metrics(dev.loc[january, "actual"], dev.loc[january, column])
        for column in report_columns
    }
    report["february_validation"] = {
        column: _metrics(dev.loc[february, "actual"], dev.loc[february, column])
        for column in report_columns
    }
    report["holdout_paired_vs_direct"] = _paired_comparison(
        holdout,
        "selected_direct_physical_blend",
        "direct_reference",
    )
    report["holdout_level_gate_paired_vs_direct"] = _paired_comparison(
        holdout,
        "level_gated_direct_physical",
        "direct_reference",
    )
    report["holdout_level_gate_paired_vs_physical"] = _paired_comparison(
        holdout,
        "level_gated_direct_physical",
        "physical_lgbm",
    )
    if gas_total_reconciliation:
        report["holdout_reconciled_paired_vs_unreconciled"] = _paired_comparison(
            holdout,
            "physical_lgbm",
            "physical_lgbm_unreconciled",
        )
        report["holdout_reconciled_gate_paired_vs_unreconciled_gate"] = (
            _paired_comparison(
                holdout,
                "level_gated_direct_physical",
                "level_gated_direct_unreconciled",
            )
        )
        validation_raw_delta = (
            report["february_validation"]["physical_lgbm"]["mape"]
            - report["february_validation"]["physical_lgbm_unreconciled"]["mape"]
        )
        validation_gate_delta = (
            report["february_validation"]["level_gated_direct_physical"]["mape"]
            - report["february_validation"]["level_gated_direct_unreconciled"]["mape"]
        )
        report["decision"] = {
            "status": "reject",
            "selection_basis": "February validation, before consulting Mar-Apr",
            "reason": (
                "reconciliation worsens both raw physical and January-selected "
                "level-gate MAPE on February"
            ),
            "february_raw_mape_delta_points": float(validation_raw_delta),
            "february_gate_mape_delta_points": float(validation_gate_delta),
            "mar_apr_is_post_rejection_audit": True,
        }
        report["gas_total_diagnostics"] = {
            "dev_jan_feb": {
                "dedicated_total": _metrics(
                    dev["actual_gas_mw"], dev["predicted_gas_total_mw"]
                ),
                "raw_component_sum": _metrics(
                    dev["actual_gas_mw"],
                    dev["predicted_gas_components_raw_sum_mw"],
                ),
            },
            "holdout_mar_apr": {
                "dedicated_total": _metrics(
                    holdout["actual_gas_mw"], holdout["predicted_gas_total_mw"]
                ),
                "raw_component_sum": _metrics(
                    holdout["actual_gas_mw"],
                    holdout["predicted_gas_components_raw_sum_mw"],
                ),
            },
        }
    diagnostic_columns = [
        "level_gated_direct_physical",
        "dev_selected_level_gated_direct_physical",
        "selected_direct_physical_blend",
        "january_selected_direct_physical_blend",
        "february_selected_direct_physical_blend",
        "physical_lgbm",
        "direct_reference",
        "point_oracle_direct_physical",
    ]
    if gas_total_reconciliation:
        diagnostic_columns.insert(-2, "physical_lgbm_unreconciled")
        diagnostic_columns.extend(
            [
                "january_selected_direct_unreconciled_blend",
                "level_gated_direct_unreconciled",
            ]
        )
    report["holdout_by_month"] = {
        str(month): {
            column: _metrics(group["actual"], group[column])
            for column in diagnostic_columns
        }
        for month, group in holdout.groupby(holdout["origin"].dt.strftime("%Y-%m"))
    }
    report["holdout_by_horizon"] = {
        str(int(horizon)): {
            column: _metrics(group["actual"], group[column])
            for column in diagnostic_columns
        }
        for horizon, group in holdout.groupby("horizon")
    }
    report["holdout_by_block"] = {
        f"h{start}-{end}": {
            column: _metrics(group["actual"], group[column])
            for column in diagnostic_columns
        }
        for start, end in ((1, 7), (8, 15), (16, 21), (22, 24))
        for group in [holdout[holdout["horizon"].between(start, end)]]
    }
    predictions = pd.concat(
        [dev.assign(split="dev_jan_feb"), holdout.assign(split="holdout_mar_apr")],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Modelo físico consolidado — exploratorio",
        "",
        "Factores efectivos aprendidos sólo en entrenamiento y regresores causales por componente.",
        "",
        "| Modelo | Dev MAPE | Mar–Abr MAPE | Mar–Abr WAPE |",
        "|---|---:|---:|---:|",
    ]
    for column in report_columns:
        dev_metric = report["dev_jan_feb"][column]
        holdout_metric = report["holdout_mar_apr"][column]
        lines.append(
            f"| `{column}` | {dev_metric['mape']:.3f}% | "
            f"{holdout_metric['mape']:.3f}% | {holdout_metric['wape']:.3f}% |"
        )
    lines += [
        "",
        "Marzo–abril es la primera apertura de esta arquitectura, pero el hallazgo del oracle físico que la motivó ya se observó en ese periodo; por eso se reporta como exploratorio.",
    ]
    if gas_total_reconciliation:
        lines += [
            "",
            "## Decisión",
            "",
            "**Rechazada.** La reconciliación empeoró en la validación de febrero, "
            "antes de consultar marzo–abril; el holdout sólo confirma el rechazo.",
        ]
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
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument(
        "--direct-predictions",
        default="runs/daily_refit_2026/regime_moe_price_availability_curves_closed_hour.predictions.parquet",
    )
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--ccg-moe", action="store_true")
    parser.add_argument(
        "--gas-total-reconciliation",
        action="store_true",
        help=(
            "fit an additional aggregate gas head and rescale the four detailed "
            "gas predictions to that total"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "artifact directory (defaults to a separate gas-reconciliation "
            "directory when that opt-in is enabled)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
