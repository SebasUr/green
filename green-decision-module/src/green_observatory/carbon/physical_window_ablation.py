"""Causal recency ablation for the operational physical carbon proxy.

This module is intentionally separate from the daily-refit runners.  It asks a
single question: does the physical fuel forecast transfer better when old
dispatch years are either dropped or smoothly down-weighted?

Selection uses January-February 2026 only.  March-April and the June-July
real-time period are reported afterwards as diagnostics and can never change
the selected strategy or prediction variant.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.realtime_proxy import (
    EMISSION_FACTORS_GCO2_KWH,
    PhysicalProxyMoE,
    physical_targets,
    proxy_training_frame,
)
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


KEYS = ["origin", "horizon", "target_time"]
VARIANTS = ("physical", "physical_alpha2", "physical_pooled")


@dataclass(frozen=True)
class RecencyStrategy:
    name: str
    window_years: int | None = None
    half_life_days: float | None = None


STRATEGIES = (
    RecencyStrategy("window_all"),
    RecencyStrategy("window_2y", window_years=2),
    RecencyStrategy("window_1y", window_years=1),
    RecencyStrategy("half_life_180d", half_life_days=180.0),
    RecencyStrategy("half_life_365d", half_life_days=365.0),
    RecencyStrategy("half_life_730d", half_life_days=730.0),
)


def _indexed_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    index = pd.DatetimeIndex(frame.index)
    frame.index = (
        index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    )
    return frame.sort_index()


def _combine_indexed(*frames: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat(frames).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def _metrics(actual, prediction) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (np.abs(actual) > 0.0)
    actual = actual[valid]
    prediction = prediction[valid]
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


def _training_mask(
    meta: pd.DataFrame, cutoff: pd.Timestamp, strategy: RecencyStrategy
) -> pd.Series:
    target_time = pd.to_datetime(meta["target_time"], utc=True)
    mask = target_time < cutoff
    if strategy.window_years is not None:
        lower = cutoff - pd.DateOffset(years=strategy.window_years)
        mask &= target_time >= lower
    return mask


def _fit_physical(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    physical_meta: pd.DataFrame,
    train_mask: pd.Series,
    strategy: RecencyStrategy,
) -> PhysicalProxyMoE:
    model = PhysicalProxyMoE(
        warm_season_coal_persistence=True,
        include_pooled_gas=True,
        recency_half_life_days=strategy.half_life_days,
    )
    model.fit(
        x.loc[train_mask].reset_index(drop=True),
        physical_meta.loc[train_mask].reset_index(drop=True),
    )
    return model


def _physical_predictions(
    model: PhysicalProxyMoE,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    evaluation_mask: pd.Series,
) -> pd.DataFrame:
    x_eval = x.loc[evaluation_mask].reset_index(drop=True)
    meta_eval = meta.loc[evaluation_mask].reset_index(drop=True)
    matrix = model.predict_matrix(x_eval)
    out = meta_eval.loc[:, KEYS + ["actual", "regime"]].copy()
    out["physical"] = matrix["prediction"].to_numpy(dtype=float)
    out["physical_pooled"] = matrix["prediction_pooled"].to_numpy(dtype=float)

    probability = matrix[
        ["prob_baseload", "prob_ccg", "prob_peak"]
    ].to_numpy(dtype=float)
    experts = matrix[
        [
            "gas_expert_baseload_mw",
            "gas_expert_ccg_mw",
            "gas_expert_peak_mw",
        ]
    ].to_numpy(dtype=float)
    sharpened = probability**2.0
    sharpened /= np.clip(sharpened.sum(axis=1, keepdims=True), 1e-12, None)
    gas_alpha2 = np.sum(sharpened * experts, axis=1)
    gas_base = matrix["predicted_gas_mw"].to_numpy(dtype=float)
    denominator = matrix["predicted_total_generation_mw"].to_numpy(dtype=float)
    out["physical_alpha2"] = np.clip(
        out["physical"].to_numpy(dtype=float)
        + EMISSION_FACTORS_GCO2_KWH["gas_mw"]
        * (gas_alpha2 - gas_base)
        / np.clip(denominator, 1.0, None),
        0.0,
        None,
    )
    for column in (
        "predicted_gas_mw",
        "predicted_gas_pooled_mw",
        "predicted_coal_mw",
        "predicted_fuel_oil_mw",
        "predicted_bioenergy_mw",
        "predicted_total_generation_mw",
    ):
        out[column] = matrix[column].to_numpy(dtype=float)
    return out


def _variant_report(frame: pd.DataFrame, actual_column: str = "actual") -> dict:
    return {
        variant: _metrics(frame[actual_column], frame[variant])
        for variant in VARIANTS
    }


def _evaluate_historical_cut(
    *,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    physical_meta: pd.DataFrame,
    cutoff: pd.Timestamp,
    evaluation_mask: pd.Series,
    strategy: RecencyStrategy,
) -> tuple[pd.DataFrame, dict]:
    train_mask = _training_mask(meta, cutoff, strategy)
    model = _fit_physical(x, meta, physical_meta, train_mask, strategy)
    predictions = _physical_predictions(model, x, meta, evaluation_mask)
    predictions.insert(0, "strategy", strategy.name)
    report = {
        "train_rows": int(train_mask.sum()),
        "train_start": str(pd.to_datetime(meta.loc[train_mask, "target_time"], utc=True).min()),
        "train_end": str(pd.to_datetime(meta.loc[train_mask, "target_time"], utc=True).max()),
        "metrics": _variant_report(predictions),
    }
    return predictions, report


def _build_live(
    args: argparse.Namespace,
    historical_published: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    live_published = _indexed_parquet(args.carbon_live)
    combined_published = regularize_hourly(
        _combine_indexed(historical_published, live_published)
    )
    combined_proxy = proxy_training_frame(combined_published, carbon_column=CARBON)
    forecast = _combine_indexed(
        _indexed_parquet(args.mix_forecast),
        _indexed_parquet(args.mix_forecast_live),
    ).join(
        _combine_indexed(
            _indexed_parquet(args.price_forecast),
            _indexed_parquet(args.price_forecast_live),
        ),
        how="outer",
    )
    intervals = pd.concat(
        [
            pd.read_parquet(args.rte_unavailability),
            pd.read_parquet(args.rte_unavailability_live),
        ],
        ignore_index=True,
    ).drop_duplicates()
    builder = RegimeMoEFeatureBuilder(
        forecast,
        availability_store=RteAvailabilityFeatureStore(intervals),
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=True,
        hourly_state_lag_hours=1,
    )
    origins = pd.date_range(_utc(args.live_start), _utc(args.live_end), freq="1D")
    x_live, meta_live = builder.build(combined_proxy, origins, supervised=True)
    return x_live, meta_live, combined_published


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_published = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    historical = proxy_training_frame(historical_published, carbon_column=CARBON)
    forecast = _indexed_parquet(args.mix_forecast).join(
        _indexed_parquet(args.price_forecast), how="outer"
    )
    builder = RegimeMoEFeatureBuilder(
        forecast,
        availability_store=RteAvailabilityFeatureStore.from_parquet(
            args.rte_unavailability
        ),
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=True,
        hourly_state_lag_hours=1,
    )
    origins = pd.date_range(
        _utc(args.feature_start), _utc(args.retrospective_end), freq="1D"
    )
    x, meta = builder.build(historical, origins, supervised=True)
    physical_meta = meta.copy()
    targets = physical_targets(historical, meta["target_time"])
    for column in targets:
        physical_meta[column] = targets[column].to_numpy()

    jan_cutoff = _utc("2026-01-01")
    mar_cutoff = _utc("2026-03-01")
    may_cutoff = _utc("2026-05-01")
    dev_mask = (meta["origin"] >= jan_cutoff) & (meta["origin"] < mar_cutoff)
    retrospective_mask = (meta["origin"] >= mar_cutoff) & (
        meta["origin"] <= _utc(args.retrospective_end)
    )

    all_predictions: list[pd.DataFrame] = []
    dev_report: dict[str, dict] = {}
    for strategy in STRATEGIES:
        predictions, result = _evaluate_historical_cut(
            x=x,
            meta=meta,
            physical_meta=physical_meta,
            cutoff=jan_cutoff,
            evaluation_mask=dev_mask,
            strategy=strategy,
        )
        predictions.insert(1, "split", "dev_jan_feb")
        all_predictions.append(predictions)
        dev_report[strategy.name] = result
        print(
            f"dev {strategy.name}: "
            + ", ".join(
                f"{name}={result['metrics'][name]['mape']:.4f}%"
                for name in VARIANTS
            ),
            flush=True,
        )

    selected_pair = min(
        (
            (strategy.name, variant)
            for strategy in STRATEGIES
            for variant in VARIANTS
        ),
        key=lambda pair: dev_report[pair[0]]["metrics"][pair[1]]["mape"],
    )
    selected_strategy = next(
        strategy for strategy in STRATEGIES if strategy.name == selected_pair[0]
    )

    # Open future periods only after freezing the pair selected above.  We fit
    # the all-history reference too; it is a benchmark, never a new selector.
    diagnostic_strategies = [selected_strategy]
    if selected_strategy.name != "window_all":
        diagnostic_strategies.append(STRATEGIES[0])

    retrospective_report: dict[str, dict] = {}
    for strategy in diagnostic_strategies:
        predictions, result = _evaluate_historical_cut(
            x=x,
            meta=meta,
            physical_meta=physical_meta,
            cutoff=mar_cutoff,
            evaluation_mask=retrospective_mask,
            strategy=strategy,
        )
        predictions.insert(1, "split", "retrospective_mar_apr")
        all_predictions.append(predictions)
        retrospective_report[strategy.name] = result
        print(
            f"retrospective {strategy.name}: "
            + ", ".join(
                f"{name}={result['metrics'][name]['mape']:.4f}%"
                for name in VARIANTS
            ),
            flush=True,
        )

    x_live, meta_live, combined_published = _build_live(args, historical_published)
    live_mask = pd.Series(True, index=meta_live.index)
    live_report: dict[str, dict] = {}
    for strategy in diagnostic_strategies:
        train_mask = _training_mask(meta, may_cutoff, strategy)
        model = _fit_physical(x, meta, physical_meta, train_mask, strategy)
        predictions = _physical_predictions(model, x_live, meta_live, live_mask)
        predictions.insert(0, "strategy", strategy.name)
        predictions.insert(1, "split", "live_jun_jul")
        predictions = predictions.rename(columns={"actual": "proxy_actual"})
        predictions["actual"] = combined_published[CARBON].reindex(
            pd.DatetimeIndex(predictions["target_time"])
        ).to_numpy(dtype=float)
        predictions = predictions.loc[predictions["actual"].notna()].reset_index(
            drop=True
        )
        all_predictions.append(predictions)
        live_report[strategy.name] = {
            "train_rows": int(train_mask.sum()),
            "train_start": str(
                pd.to_datetime(meta.loc[train_mask, "target_time"], utc=True).min()
            ),
            "train_end": str(
                pd.to_datetime(meta.loc[train_mask, "target_time"], utc=True).max()
            ),
            "against_published_realtime": _variant_report(predictions),
            "against_reconstructed_proxy": _variant_report(
                predictions, actual_column="proxy_actual"
            ),
        }
        print(
            f"live {strategy.name}: "
            + ", ".join(
                f"{name}={live_report[strategy.name]['against_published_realtime'][name]['mape']:.4f}%"
                for name in VARIANTS
            ),
            flush=True,
        )

    selected_name, selected_variant = selected_pair
    selection_mape = dev_report[selected_name]["metrics"][selected_variant]["mape"]
    baseline_mape = dev_report["window_all"]["metrics"][selected_variant]["mape"]
    retrospective_delta = (
        retrospective_report[selected_name]["metrics"][selected_variant]["mape"]
        - retrospective_report["window_all"]["metrics"][selected_variant]["mape"]
    )
    live_delta = (
        live_report[selected_name]["against_published_realtime"][selected_variant][
            "mape"
        ]
        - live_report["window_all"]["against_published_realtime"][selected_variant][
            "mape"
        ]
    )
    promote = retrospective_delta < 0.0 and live_delta < 0.0
    summary = {
        "protocol": {
            "target": "RTE provisional physical proxy",
            "feature_visibility": "origin state uses the last fully closed hourly bin (origin minus one hour)",
            "selection_period": "2026-01-01 through 2026-02-28 origins",
            "selection_rule": "minimum aggregate MAPE across the six predeclared recency strategies and three physical variants",
            "retrospective_diagnostic": "2026-03-01 through 2026-04-29 origins",
            "live_diagnostic": [args.live_start, args.live_end],
            "scientific_status": "causal feature/training protocol; exploratory architecture because the live period had been inspected before this ablation",
        },
        "strategies": [strategy.__dict__ for strategy in STRATEGIES],
        "selected_on_dev_only": {
            "strategy": selected_name,
            "variant": selected_variant,
            "dev_mape": selection_mape,
            "same_variant_window_all_dev_mape": baseline_mape,
            "delta_mape_points_vs_window_all": selection_mape - baseline_mape,
        },
        "decision": {
            "promote_recency_strategy": bool(promote),
            "retrospective_delta_mape_points_vs_window_all": retrospective_delta,
            "live_delta_mape_points_vs_window_all": live_delta,
            "reason": (
                "promote only when the dev-selected pair also beats all-history "
                "on both later diagnostics"
            ),
        },
        "dev_jan_feb": dev_report,
        "retrospective_mar_apr": retrospective_report,
        "live_jun_jul": live_report,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.concat(all_predictions, ignore_index=True, sort=False).to_parquet(
        output_dir / "predictions.parquet", index=False
    )
    _write_readme(output_dir / "README.md", summary)
    print(json.dumps(summary["selected_on_dev_only"], indent=2), flush=True)
    return summary


def _write_readme(path: Path, summary: dict) -> None:
    selected = summary["selected_on_dev_only"]
    strategy = selected["strategy"]
    variant = selected["variant"]
    retrospective = summary["retrospective_mar_apr"][strategy]["metrics"][variant]
    live = summary["live_jun_jul"][strategy]["against_published_realtime"][variant]
    reference_retro = summary["retrospective_mar_apr"].get("window_all")
    reference_live = summary["live_jun_jul"].get("window_all")
    lines = [
        "# Ablación causal de recencia del proxy físico",
        "",
        "La selección se hizo únicamente con enero-febrero de 2026. Marzo-abril y junio-julio se abrieron después y no cambian la elección.",
        "",
        f"- Estrategia seleccionada: `{strategy}`.",
        f"- Variante seleccionada: `{variant}`.",
        f"- MAPE enero-febrero: **{selected['dev_mape']:.3f}%**.",
        f"- MAPE marzo-abril diagnóstico: **{retrospective['mape']:.3f}%**.",
        f"- MAPE live contra RTE publicado: **{live['mape']:.3f}%**.",
    ]
    if reference_retro is not None:
        lines.append(
            f"- Referencia all-history marzo-abril, misma variante: **{reference_retro['metrics'][variant]['mape']:.3f}%**."
        )
    if reference_live is not None:
        lines.append(
            f"- Referencia all-history live, misma variante: **{reference_live['against_published_realtime'][variant]['mape']:.3f}%**."
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            (
                "**Decisión: promover la recencia.**"
                if decision["promote_recency_strategy"]
                else "**Decisión: recencia rechazada; conservar all-history.**"
            ),
            (
                "La mejora de desarrollo no fue robusta en ambos diagnósticos "
                "posteriores."
                if not decision["promote_recency_strategy"]
                else "La mejora sobrevivió ambos diagnósticos posteriores."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "Las vidas medias multiplican el peso inverso al nivel por `2**(-edad_días/vida_media)`; las ventanas descartan targets anteriores al límite. La etiqueta, el estado y los lags siguen siendo causales.",
            "",
            "El live es diagnóstico exploratorio: ese periodo ya había sido observado en experimentos anteriores.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carbon", default="data/cache/carbon_fr_hourly_detailed.parquet"
    )
    parser.add_argument(
        "--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet"
    )
    parser.add_argument(
        "--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet"
    )
    parser.add_argument(
        "--rte-unavailability",
        default="data/cache/rte_unavailability_messages.parquet",
    )
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--retrospective-end", default="2026-04-29")
    parser.add_argument(
        "--carbon-live", default="data/cache/carbon_fr_realtime_holdout.parquet"
    )
    parser.add_argument(
        "--mix-forecast-live", default="data/cache/mix_day_ahead_fr_holdout.parquet"
    )
    parser.add_argument(
        "--price-forecast-live", default="data/cache/day_ahead_price_fr_holdout.parquet"
    )
    parser.add_argument(
        "--rte-unavailability-live",
        default="data/cache/rte_unavailability_messages_holdout.parquet",
    )
    parser.add_argument("--live-start", default="2026-06-17")
    parser.add_argument("--live-end", default="2026-07-15")
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/physical_window_ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
