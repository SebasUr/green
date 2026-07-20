"""Strict temporal evaluation of the operational RTE real-time proxy target.

Selection protocol
------------------
* fit component models using targets strictly before 2026-01-01;
* calibrate component scales on January 2026;
* compare fixed simple blends on February 2026;
* select exactly one candidate on January-February;
* only when it improves the D-1 baseline, refit through February and open the
  March-April retrospective confirmation once;
* finally refit on all consolidated history and report June-July real-time as
  a diagnostic holdout (it has already been inspected by prior experiments).

The module writes only to its dedicated output directory and does not mutate
the older consolidated-target experiments.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.realtime_proxy import (
    PhysicalProxyMoE,
    physical_targets,
    proxy_training_frame,
    rte_realtime_carbon_proxy,
)
from green_observatory.carbon.regime_moe import (
    DirectRegimeMoE,
    RegimeMoEFeatureBuilder,
    select_mape_scale,
)
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


KEYS = ["origin", "horizon", "target_time"]
PHYSICAL_VARIANTS = (
    "physical",
    "physical_pooled",
    "physical_alpha2",
    "physical_alpha3",
    "physical_alpha5",
    "physical_hard",
)
REPORT_SOURCE_COLUMNS = ("direct", *PHYSICAL_VARIANTS, "d1")


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


def _metrics(actual, prediction) -> dict[str, float]:
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


def _attach_physical_targets(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    targets = physical_targets(frame, out["target_time"])
    for column in targets:
        out[column] = targets[column].to_numpy()
    return out


def _fit_models(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    train: pd.Series,
) -> tuple[DirectRegimeMoE, PhysicalProxyMoE]:
    x_train = x.loc[train].reset_index(drop=True)
    meta_train = meta.loc[train].reset_index(drop=True)
    direct = DirectRegimeMoE(point_scale=1.0)
    direct.fit(x_train, meta_train)
    physical = PhysicalProxyMoE(
        warm_season_coal_persistence=True,
        include_pooled_gas=True,
    )
    physical.fit(x_train, _attach_physical_targets(frame, meta_train))
    return direct, physical


def _predict_sources(
    direct: DirectRegimeMoE,
    physical: PhysicalProxyMoE,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    evaluation: pd.Series,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    x_eval = x.loc[evaluation].reset_index(drop=True)
    meta_eval = meta.loc[evaluation].reset_index(drop=True)
    out = meta_eval.loc[:, KEYS + ["actual", "regime"]].copy()
    out["direct"] = direct.predict_matrix(x_eval)["prediction"].to_numpy()
    physical_matrix = physical.predict_matrix(x_eval)
    out["physical"] = physical_matrix["prediction"].to_numpy()
    out["physical_pooled"] = physical_matrix["prediction_pooled"].to_numpy()
    d1_times = pd.DatetimeIndex(out["target_time"]) - pd.Timedelta(hours=24)
    d1 = frame[CARBON].reindex(d1_times).to_numpy(dtype=float).copy()
    d1[d1_times >= pd.DatetimeIndex(out["origin"])] = np.nan
    out["d1"] = d1
    for column in (
        "predicted_gas_mw",
        "predicted_gas_pooled_mw",
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
        out[column] = physical_matrix[column].to_numpy()
    probability = out[["prob_baseload", "prob_ccg", "prob_peak"]].to_numpy()
    experts = out[
        ["gas_expert_baseload_mw", "gas_expert_ccg_mw", "gas_expert_peak_mw"]
    ].to_numpy()
    base_gas = out["predicted_gas_mw"].to_numpy(dtype=float)
    denominator = out["predicted_total_generation_mw"].to_numpy(dtype=float)
    for alpha, name in (
        (2.0, "physical_alpha2"),
        (3.0, "physical_alpha3"),
        (5.0, "physical_alpha5"),
    ):
        sharpened = probability**alpha
        sharpened /= np.clip(sharpened.sum(axis=1, keepdims=True), 1e-12, None)
        gas = np.sum(sharpened * experts, axis=1)
        out[name] = out["physical"] + 429.0 * (gas - base_gas) / denominator
    hard = np.zeros_like(probability)
    hard[np.arange(len(hard)), np.argmax(probability, axis=1)] = 1.0
    gas_hard = np.sum(hard * experts, axis=1)
    out["physical_hard"] = (
        out["physical"] + 429.0 * (gas_hard - base_gas) / denominator
    )
    return out


def _candidate_weights(step: float = 0.10) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    weights: list[dict[str, float]] = []
    for direct_units, physical_units in product(range(units + 1), repeat=2):
        d1_units = units - direct_units - physical_units
        if d1_units < 0:
            continue
        weights.append(
            {
                "direct": direct_units / units,
                "physical": physical_units / units,
                "d1": d1_units / units,
            }
        )
    return weights


def _select_candidate(dev: pd.DataFrame) -> tuple[dict, list[dict]]:
    january = dev["origin"] < _utc("2026-02-01")
    february = ~january
    scales: dict[str, float] = {}
    for column in REPORT_SOURCE_COLUMNS:
        scale, _ = select_mape_scale(
            dev.loc[january, "actual"].to_numpy(),
            dev.loc[january, column].to_numpy(),
        )
        scales[column] = scale

    candidates: list[dict] = []
    for physical_column in PHYSICAL_VARIANTS:
        for weights_base in _candidate_weights():
            weights = {
                "direct": weights_base["direct"],
                physical_column: weights_base["physical"],
                "d1": weights_base["d1"],
            }
            for scale_mode in ("raw", "january_scaled"):
                applied_scales = {
                    column: (scales[column] if scale_mode == "january_scaled" else 1.0)
                    for column in weights
                }
                active = [
                    column for column, weight in weights.items() if weight != 0.0
                ]
                prediction = sum(
                    weights[column] * applied_scales[column] * dev[column]
                    for column in active
                )
                january_metrics = _metrics(dev.loc[january, "actual"], prediction[january])
                february_metrics = _metrics(dev.loc[february, "actual"], prediction[february])
                full_metrics = _metrics(dev["actual"], prediction)
                candidates.append(
                    {
                        "weights": weights,
                        "scale_mode": scale_mode,
                        "source_scales_from_january": applied_scales,
                        "january": january_metrics,
                        "february": february_metrics,
                        "jan_feb": full_metrics,
                    }
                )

    # February is the internal validation slice.  Complexity is not rewarded:
    # within numerical ties prefer more direct-model weight and fewer sources.
    def key(row: dict):
        active = sum(value > 0.0 for value in row["weights"].values())
        return (
            row["february"]["mape"],
            row["jan_feb"]["mape"],
            active,
            -row["weights"]["direct"],
        )

    selected = min(candidates, key=key)
    selected = {
        **selected,
        "selection_rule": (
            "minimum February MAPE among raw and January-scaled candidates; "
            "raw candidates were added after the first retrospective opening"
        ),
        "scientific_status": (
            "selection correction is retrospective/exploratory because the "
            "original menu omitted no-scale candidates"
        ),
    }
    return selected, candidates


def _apply_candidate(frame: pd.DataFrame, selected: dict) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=float)
    for column in selected["weights"]:
        if selected["weights"][column] == 0.0:
            continue
        prediction += (
            selected["weights"][column]
            * selected["source_scales_from_january"][column]
            * frame[column].to_numpy(dtype=float)
        )
    return np.clip(prediction, 0.0, None)


def _candidate_report(frame: pd.DataFrame, selected: dict) -> dict:
    prediction = _apply_candidate(frame, selected)
    report = {"selected": _metrics(frame["actual"], prediction)}
    for column in REPORT_SOURCE_COLUMNS:
        scale = selected["source_scales_from_january"].get(column, 1.0)
        scaled = scale * frame[column]
        report[f"{column}_scaled"] = _metrics(frame["actual"], scaled)
        report[f"{column}_raw"] = _metrics(frame["actual"], frame[column])
    return report


def _target_agreement(frame: pd.DataFrame, published: pd.Series) -> dict:
    proxy = rte_realtime_carbon_proxy(frame)
    common = proxy.notna() & published.notna()
    metrics = _metrics(published[common], proxy[common])
    metrics["correlation"] = float(published[common].corr(proxy[common]))
    return metrics


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_published = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    historical = proxy_training_frame(historical_published, carbon_column=CARBON)
    forecast = _indexed_parquet(args.mix_forecast).join(
        _indexed_parquet(args.price_forecast), how="outer"
    )
    availability = RteAvailabilityFeatureStore.from_parquet(args.rte_unavailability)
    builder = RegimeMoEFeatureBuilder(
        forecast,
        availability_store=availability,
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=True,
    )
    historical_origins = pd.date_range(
        _utc(args.feature_start), _utc(args.retrospective_end), freq="1D"
    )
    x, meta = builder.build(historical, historical_origins, supervised=True)

    train_dev = meta["target_time"] < _utc("2026-01-01")
    dev_mask = (meta["origin"] >= _utc("2026-01-01")) & (
        meta["origin"] < _utc("2026-03-01")
    )
    direct_dev, physical_dev = _fit_models(x, meta, historical, train_dev)
    dev = _predict_sources(
        direct_dev, physical_dev, x, meta, dev_mask, historical
    )
    selected, candidates = _select_candidate(dev)
    dev["prediction"] = _apply_candidate(dev, selected)

    # Opening March-April is conditional on a genuine dev improvement over
    # the strongest simple operational baseline, scaled D-1 persistence.
    dev_report = _candidate_report(dev, selected)
    candidate_wins = (
        dev_report["selected"]["mape"]
        < dev_report["d1_scaled"]["mape"]
    )
    retrospective = pd.DataFrame()
    retrospective_report: dict | None = None
    if candidate_wins:
        train_retrospective = meta["target_time"] < _utc("2026-03-01")
        retrospective_mask = (meta["origin"] >= _utc("2026-03-01")) & (
            meta["origin"] <= _utc(args.retrospective_end)
        )
        direct_retro, physical_retro = _fit_models(
            x, meta, historical, train_retrospective
        )
        retrospective = _predict_sources(
            direct_retro,
            physical_retro,
            x,
            meta,
            retrospective_mask,
            historical,
        )
        retrospective["prediction"] = _apply_candidate(retrospective, selected)
        retrospective_report = _candidate_report(retrospective, selected)

    live_report: dict | None = None
    live_predictions = pd.DataFrame()
    if candidate_wins and args.carbon_live:
        live_published = _indexed_parquet(args.carbon_live)
        combined_published = regularize_hourly(
            _combine_indexed(historical_published, live_published)
        )
        combined_proxy = proxy_training_frame(combined_published, carbon_column=CARBON)
        live_forecast = _combine_indexed(
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
        live_builder = RegimeMoEFeatureBuilder(
            live_forecast,
            availability_store=RteAvailabilityFeatureStore(intervals),
            availability_feature_mode="all",
            include_curve_summaries=True,
            include_detailed_state=True,
        )
        live_origins = pd.date_range(
            _utc(args.live_start), _utc(args.live_end), freq="1D"
        )
        x_live, meta_live = live_builder.build(
            combined_proxy, live_origins, supervised=True
        )

        # Static final fit: all historical targets available before May.  The
        # candidate weights/scales remain exactly those selected on Jan-Feb.
        final_train = meta["target_time"] < _utc("2026-05-01")
        direct_final, physical_final = _fit_models(
            x, meta, historical, final_train
        )
        live_mask = pd.Series(True, index=meta_live.index)
        live_predictions = _predict_sources(
            direct_final,
            physical_final,
            x_live,
            meta_live,
            live_mask,
            combined_proxy,
        )
        live_predictions = live_predictions.rename(columns={"actual": "proxy_actual"})
        live_predictions["actual"] = combined_published[CARBON].reindex(
            pd.DatetimeIndex(live_predictions["target_time"])
        ).to_numpy(dtype=float)
        live_predictions["prediction"] = _apply_candidate(
            live_predictions, selected
        )
        valid_live = live_predictions["actual"].notna()
        live_predictions = live_predictions.loc[valid_live].reset_index(drop=True)
        live_report = {
            "status": "diagnostic_not_pristine_prior_live_period_was_already_inspected",
            "against_published_realtime_taux_co2": _candidate_report(
                live_predictions, selected
            ),
            "against_reconstructed_proxy": {
                "selected": _metrics(
                    live_predictions["proxy_actual"], live_predictions["prediction"]
                ),
                "proxy_vs_published": _metrics(
                    live_predictions["actual"], live_predictions["proxy_actual"]
                ),
                "correlation_proxy_vs_published": float(
                    live_predictions["actual"].corr(
                        live_predictions["proxy_actual"]
                    )
                ),
            },
        }

    historical_common = historical[CARBON].notna() & historical_published[CARBON].notna()
    target_gap = _metrics(
        historical_published.loc[historical_common, CARBON],
        historical.loc[historical_common, CARBON],
    )
    target_gap["correlation"] = float(
        historical_published.loc[historical_common, CARBON].corr(
            historical.loc[historical_common, CARBON]
        )
    )
    report = {
        "protocol": {
            "target": "RTE provisional production proxy from the eight domestic generation aggregates",
            "formula_gco2_kwh": "(986*coal + 777*fuel_oil + 429*gas + 494*bioenergy) / total_generation",
            "component_train_before_dev": "2026-01-01T00:00:00Z",
            "scale_calibration": "January 2026 only",
            "candidate_internal_validation": "February 2026",
            "retrospective_confirmation": (
                ["2026-03-01", args.retrospective_end] if candidate_wins else None
            ),
            "live_diagnostic": (
                [args.live_start, args.live_end] if live_report is not None else None
            ),
            "day_ahead_visibility": "masked after the origin's local delivery day",
            "scientific_status": (
                "The raw-candidate menu and warm-season coal persistence were "
                "added after the first March-April/live diagnostic; their "
                "reported results are retrospective/exploratory, not a fresh holdout."
            ),
        },
        "historical_consolidated_vs_operational_proxy": target_gap,
        "selection": {
            "candidate_wins_vs_scaled_d1": bool(candidate_wins),
            "selected": selected,
            "all_candidates": candidates,
        },
        "dev_jan_feb": dev_report,
        "retrospective_mar_apr": retrospective_report,
        "live_jun_jul": live_report,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    dev.to_parquet(output_dir / "dev_predictions.parquet", index=False)
    if not retrospective.empty:
        retrospective.to_parquet(
            output_dir / "retrospective_predictions.parquet", index=False
        )
    if not live_predictions.empty:
        live_predictions.to_parquet(
            output_dir / "live_diagnostic_predictions.parquet", index=False
        )
    print(json.dumps(report, indent=2))
    return report


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
        "--output-dir", default="runs/daily_refit_2026/realtime_proxy"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
