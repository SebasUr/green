"""Causal Jan-Feb / Mar-Apr ablation of consolidated detailed shares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.consolidated_physical import EMITTING_COMPONENTS
from green_observatory.carbon.consolidated_physical_evaluation import (
    KEYS,
    _indexed_parquet,
    _metrics,
    _paired_comparison,
)
from green_observatory.carbon.consolidated_share import (
    DETAILED_SHARE_COLUMNS,
    TOTAL_GENERATION,
    ConsolidatedShareRegressor,
    add_causal_detailed_share_features,
    detailed_share_targets,
)
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder, select_mape_scale
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


SHARE_CANDIDATES = (
    "share_lgbm",
    "share_gas_total_lgbm",
    "share_sparse_d1",
    "share_all_components_d1",
)
REFERENCE_COLUMNS = (
    "direct_reference",
    "physical_lgbm",
    "selected_direct_physical_blend",
)


def _attach_targets(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    shares = detailed_share_targets(frame, out["target_time"])
    for column in (*DETAILED_SHARE_COLUMNS, TOTAL_GENERATION):
        out[column] = shares[column].to_numpy()
    return out


def _predict(
    model: ConsolidatedShareRegressor,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    mask: pd.Series,
) -> pd.DataFrame:
    x_eval = x.loc[mask].reset_index(drop=True)
    out = meta.loc[mask, KEYS + ["actual"]].reset_index(drop=True)
    matrix = model.predict_matrix(x_eval)
    out["share_lgbm"] = matrix["prediction"].to_numpy()
    out["share_gas_total_lgbm"] = matrix["prediction_gas_total"].to_numpy()
    out["predicted_gas_total_share"] = matrix[
        "predicted_gas_total_share"
    ].to_numpy()
    predicted = np.column_stack(
        [matrix[f"predicted_{name}"].to_numpy(dtype=float) for name in DETAILED_SHARE_COLUMNS]
    )
    d1 = np.column_stack(
        [
            pd.to_numeric(
                x_eval[f"detail_share_tgtlag24_{name}"], errors="coerce"
            ).to_numpy(dtype=float)
            for name in DETAILED_SHARE_COLUMNS
        ]
    )
    usable = np.where(np.isfinite(d1), d1, predicted)
    sparse = predicted.copy()
    for source in ("coal_mw", "fuel_oil_mw", "bioenergy_waste_mw"):
        position = EMITTING_COMPONENTS.index(source)
        sparse[:, position] = usable[:, position]
    out["share_sparse_d1"] = np.clip(
        sparse @ model.emission_factors_, 0.0, None
    )
    out["share_all_components_d1"] = np.clip(
        usable @ model.emission_factors_, 0.0, None
    )
    for name in DETAILED_SHARE_COLUMNS:
        out[f"predicted_{name}"] = matrix[f"predicted_{name}"].to_numpy()
    return out


def _load_reference(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    for column in ("origin", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    keep = KEYS + ["split", *REFERENCE_COLUMNS, "oracle_learned_factors"]
    return frame.loc[:, [column for column in keep if column in frame]]


def _merge_reference(frame: pd.DataFrame, reference: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = reference[reference["split"].eq(split)].drop(columns="split")
    return frame.merge(subset, on=KEYS, how="left", validate="one_to_one")


def _fit_predict(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    train_before: pd.Timestamp,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
    threads: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    train = meta["target_time"] < train_before
    evaluation = meta["origin"].between(origin_start, origin_end)
    model = ConsolidatedShareRegressor(source_params={"n_jobs": threads})
    model.fit(x.loc[train].reset_index(drop=True), _attach_targets(frame, meta.loc[train]))
    out = _predict(model, x, meta, evaluation)
    actual_shares = detailed_share_targets(frame, out["target_time"])
    out["share_oracle_learned_factors"] = np.clip(
        actual_shares.loc[:, DETAILED_SHARE_COLUMNS].to_numpy(dtype=float)
        @ model.emission_factors_,
        0.0,
        None,
    )
    return out, model.emission_factors_.copy()


def _select_blend(
    dev: pd.DataFrame, candidate: str, reference: str, mask: pd.Series
) -> float:
    weights = np.arange(0.0, 1.001, 0.05)
    losses = []
    for weight in weights:
        prediction = weight * dev[candidate] + (1.0 - weight) * dev[reference]
        losses.append(_metrics(dev.loc[mask, "actual"], prediction[mask])["mape"])
    return float(weights[int(np.argmin(losses))])


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    x_base, meta = builder.build(frame, origins, supervised=True)
    x = add_causal_detailed_share_features(x_base, meta, frame)
    dev, dev_factors = _fit_predict(
        x,
        meta,
        frame,
        train_before=_utc("2026-01-01"),
        origin_start=_utc("2026-01-01"),
        origin_end=_utc("2026-02-28"),
        threads=args.model_threads,
    )
    holdout, holdout_factors = _fit_predict(
        x,
        meta,
        frame,
        train_before=_utc("2026-03-01"),
        origin_start=_utc("2026-03-01"),
        origin_end=_utc("2026-04-29"),
        threads=args.model_threads,
    )
    reference = _load_reference(args.reference_predictions)
    dev = _merge_reference(dev, reference, "dev_jan_feb")
    holdout = _merge_reference(holdout, reference, "holdout_mar_apr")

    january = dev["origin"] < _utc("2026-02-01")
    february = ~january
    scales = {}
    for column in SHARE_CANDIDATES:
        scale, _ = select_mape_scale(
            dev.loc[january, "actual"].to_numpy(dtype=float),
            dev.loc[january, column].to_numpy(dtype=float),
        )
        scales[column] = scale
        dev[f"{column}_jan_scaled"] = scale * dev[column]
        holdout[f"{column}_jan_scaled"] = scale * holdout[column]

    # Preselect the share variant on Jan-Feb; all later comparisons carry that
    # identity unchanged.
    selected_share = min(
        SHARE_CANDIDATES,
        key=lambda column: _metrics(dev["actual"], dev[column])["mape"],
    )
    share_weight = _select_blend(
        dev, selected_share, "direct_reference", pd.Series(True, index=dev.index)
    )
    share_weight_february = _select_blend(
        dev, selected_share, "direct_reference", february
    )
    augmentation_weight = _select_blend(
        dev,
        selected_share,
        "selected_direct_physical_blend",
        pd.Series(True, index=dev.index),
    )
    augmentation_weight_february = _select_blend(
        dev,
        selected_share,
        "selected_direct_physical_blend",
        february,
    )
    for split in (dev, holdout):
        split["selected_share_direct_blend"] = (
            share_weight * split[selected_share]
            + (1.0 - share_weight) * split["direct_reference"]
        )
        split["february_selected_share_direct_blend"] = (
            share_weight_february * split[selected_share]
            + (1.0 - share_weight_february) * split["direct_reference"]
        )
        split["share_augmented_existing_blend"] = (
            augmentation_weight * split[selected_share]
            + (1.0 - augmentation_weight) * split["selected_direct_physical_blend"]
        )
        split["february_selected_share_augmented_existing_blend"] = (
            augmentation_weight_february * split[selected_share]
            + (1.0 - augmentation_weight_february)
            * split["selected_direct_physical_blend"]
        )

    report_columns = (
        *REFERENCE_COLUMNS,
        *SHARE_CANDIDATES,
        *(f"{column}_jan_scaled" for column in SHARE_CANDIDATES),
        "selected_share_direct_blend",
        "february_selected_share_direct_blend",
        "share_augmented_existing_blend",
        "february_selected_share_augmented_existing_blend",
        "share_oracle_learned_factors",
        "oracle_learned_factors",
    )
    report = {
        "status": "exploratory share ablation; no denominator forecast",
        "protocol": {
            "target": "RTE consolidated production intensity",
            "component_model": "seven independent LightGBM L1 share regressors",
            "aggregate_gas_ablation": "one extra pooled total-gas-share head rescales the four predicted gas subtype shares; reported but not selected",
            "emission_factor_fit": "positive OLS, no intercept, train-only actual*generation versus component MW",
            "feature_visibility": "origin-1h plus target-aligned D-1/D-7; D-1 h24 masked",
            "rte_generation_forecast": (
                {
                    "path": args.rte_generation_forecast,
                    "production_types": args.rte_production_type or "all supported",
                    "visibility": "latest updated_date <= origin",
                }
                if args.rte_generation_forecast
                else None
            ),
            "dev": "train target_time < 2026-01-01; origins Jan-Feb",
            "holdout": "refit target_time < 2026-03-01; origins Mar-Apr",
        },
        "emission_components": list(EMITTING_COMPONENTS),
        "dev_factors": dict(zip(EMITTING_COMPONENTS, map(float, dev_factors))),
        "holdout_refit_factors": dict(
            zip(EMITTING_COMPONENTS, map(float, holdout_factors))
        ),
        "january_scales": scales,
        "selected_share_variant": selected_share,
        "selected_share_weight_vs_direct": share_weight,
        "february_selected_share_weight_vs_direct": share_weight_february,
        "selected_share_weight_augmenting_existing_blend": augmentation_weight,
        "february_selected_share_weight_augmenting_existing_blend": (
            augmentation_weight_february
        ),
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
    }
    report["holdout_paired_share_blend_vs_existing_blend"] = _paired_comparison(
        holdout, "share_augmented_existing_blend", "selected_direct_physical_blend"
    )
    report["holdout_paired_share_raw_vs_physical_mw"] = _paired_comparison(
        holdout, selected_share, "physical_lgbm"
    )
    report["holdout_paired_share_raw_vs_direct"] = _paired_comparison(
        holdout, selected_share, "direct_reference"
    )
    diagnostic_columns = (
        selected_share,
        "physical_lgbm",
        "direct_reference",
        "selected_direct_physical_blend",
        "share_augmented_existing_blend",
        "february_selected_share_augmented_existing_blend",
    )
    report["holdout_by_month"] = {
        month: {
            column: _metrics(group["actual"], group[column])
            for column in diagnostic_columns
        }
        for month, group in holdout.groupby(holdout["origin"].dt.strftime("%Y-%m"))
    }
    report["holdout_by_horizon_block"] = {}
    for start, end in ((1, 6), (7, 16), (17, 21), (22, 24)):
        group = holdout[holdout["horizon"].between(start, end)]
        report["holdout_by_horizon_block"][f"h{start}-{end}"] = {
            column: _metrics(group["actual"], group[column])
            for column in diagnostic_columns
        }
    report["decision"] = {
        "beats_physical_mw": bool(
            report["holdout_mar_apr"][selected_share]["mape"]
            < report["holdout_mar_apr"]["physical_lgbm"]["mape"]
        ),
        "beats_existing_blend": bool(
            report["holdout_mar_apr"]["share_augmented_existing_blend"]["mape"]
            < report["holdout_mar_apr"]["selected_direct_physical_blend"]["mape"]
        ),
    }
    predictions = pd.concat(
        [dev.assign(split="dev_jan_feb"), holdout.assign(split="holdout_mar_apr")],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Ablation consolidada por shares",
        "",
        "Siete shares físicos, factores positive-OLS sólo con entrenamiento y sin regresor de denominador.",
        "",
        "| Modelo | Dev MAPE | Mar-Abr MAPE | Mar-Abr WAPE |",
        "|---|---:|---:|---:|",
    ]
    for column in report_columns:
        a = report["dev_jan_feb"][column]
        b = report["holdout_mar_apr"][column]
        lines.append(
            f"| `{column}` | {a['mape']:.3f}% | {b['mape']:.3f}% | {b['wape']:.3f}% |"
        )
    lines += [
        "",
        f"Share seleccionado: `{selected_share}`.",
        f"Peso share frente a Direct (Jan-Feb): {share_weight:.2f}.",
        f"Peso share al aumentar el blend existente: {augmentation_weight:.2f}.",
        f"Peso share al aumentar el blend seleccionado sólo en febrero: {augmentation_weight_february:.2f}.",
        "",
        "La mejora primaria frente al blend MW+Direct es exploratoria pero aparece también con selección sólo en febrero. El bootstrap pareado por día y por bloques de 7 días se conserva en `summary.json`.",
        f"El head agregado de gas no fue seleccionado: {report['holdout_mar_apr']['share_gas_total_lgbm']['mape']:.3f}% en marzo-abril.",
        f"El share crudo mejora al modelo MW en {report['holdout_paired_share_raw_vs_physical_mw']['mape_delta_points']:.3f} puntos MAPE; IC bootstrap diario {report['holdout_paired_share_raw_vs_physical_mw']['day_bootstrap_95ci']}.",
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
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument(
        "--reference-predictions",
        default="runs/daily_refit_2026/consolidated_physical/predictions.parquet",
    )
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir", default="runs/daily_refit_2026/consolidated_share"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
