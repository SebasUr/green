"""Development-only selection of cross-border day-ahead price features.

This module deliberately stops at the end of February 2026.  It does not load
March-April daily-refit checkpoints and never constructs holdout predictions.
The winning feature subset can subsequently be evaluated once with
``regime_moe_evaluation``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from green_observatory.carbon.adaptive_ensemble import causal_block_scaled_expert
from green_observatory.carbon.annual_evaluation import _forecast_frame, _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import (
    DirectRegimeMoE,
    RegimeMoEFeatureBuilder,
    select_mape_scale,
)
from green_observatory.carbon.regime_moe_evaluation import (
    _complete_origins,
    _mape,
    _select_level_shape_blend,
)
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_odre import OdreCarbonProvider


SUBSETS = {
    "ch_raw_spread": (
        "day_ahead_price_ch_eur_mwh",
        "day_ahead_price_spread_fr_minus_ch_eur_mwh",
    ),
    "es_raw_spread": (
        "day_ahead_price_es_eur_mwh",
        "day_ahead_price_spread_fr_minus_es_eur_mwh",
    ),
    "be_de_raw_spreads": (
        "day_ahead_price_be_eur_mwh",
        "day_ahead_price_de_lu_eur_mwh",
        "day_ahead_price_spread_fr_minus_be_eur_mwh",
        "day_ahead_price_spread_fr_minus_de_lu_eur_mwh",
    ),
    "all_spreads": (
        "day_ahead_price_spread_fr_minus_be_eur_mwh",
        "day_ahead_price_spread_fr_minus_de_lu_eur_mwh",
        "day_ahead_price_spread_fr_minus_es_eur_mwh",
        "day_ahead_price_spread_fr_minus_ch_eur_mwh",
    ),
}


def _development_checkpoints(directory: str, end: pd.Timestamp) -> pd.DataFrame:
    paths = []
    for path in sorted(Path(directory).glob("*.parquet")):
        try:
            origin = _utc(path.stem)
        except (TypeError, ValueError):
            continue
        if origin <= end:
            paths.append(path)
    if not paths:
        raise FileNotFoundError("no development checkpoints found")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    if (frame["origin"] > end).any():
        raise AssertionError("development selector loaded a post-cutoff origin")
    return frame.drop_duplicates(["model", "origin", "horizon"], keep="last")


def run(args: argparse.Namespace) -> dict:
    dev_end = _utc(args.dev_end)
    exclusive_data_end = dev_end + pd.Timedelta(days=1)
    full = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    base_forecasts = _forecast_frame([args.mix_forecast, args.france_price])
    base_forecasts = base_forecasts.loc[base_forecasts.index < exclusive_data_end]
    neighbours = pd.read_parquet(args.neighbour_prices)
    if neighbours.index.tz is None:
        neighbours.index = neighbours.index.tz_localize("UTC")
    else:
        neighbours.index = neighbours.index.tz_convert("UTC")
    neighbours = neighbours.loc[neighbours.index < exclusive_data_end]

    missing = sorted({column for values in SUBSETS.values() for column in values} - set(neighbours))
    if missing:
        raise ValueError(f"neighbour-price snapshot is missing columns: {missing}")

    availability = RteAvailabilityFeatureStore.from_parquet(args.rte_unavailability)
    feature_origins = pd.date_range(_utc(args.feature_start), dev_end, freq="1D")
    scale_train_end = _utc(args.scale_train_end)
    final_train_end = _utc(args.eval_start)

    checkpoints = _development_checkpoints(args.daily_predictions, dev_end)
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

    results = []
    for name, columns in SUBSETS.items():
        print(f"development candidate {name}: {', '.join(columns)}", flush=True)
        forecasts = base_forecasts.join(neighbours[list(columns)], how="outer")
        builder = RegimeMoEFeatureBuilder(
            forecasts,
            availability_store=availability,
            availability_feature_mode="all",
            include_curve_summaries=True,
        )
        x, meta = builder.build(full, feature_origins, supervised=True)
        tuning_train = meta["target_time"] < scale_train_end
        calibration = (meta["origin"] >= scale_train_end) & (
            meta["origin"] < final_train_end
        )
        final_train = meta["target_time"] < final_train_end
        development = (meta["origin"] >= final_train_end) & (
            meta["origin"] <= dev_end
        )

        tuning = DirectRegimeMoE().fit(
            x.loc[tuning_train].reset_index(drop=True),
            meta.loc[tuning_train].reset_index(drop=True),
        )
        raw = tuning.predict_matrix(x.loc[calibration].reset_index(drop=True))
        scale, scale_mape = select_mape_scale(
            meta.loc[calibration, "actual"].to_numpy(),
            raw["prediction"].to_numpy(),
        )
        model = DirectRegimeMoE(point_scale=scale).fit(
            x.loc[final_train].reset_index(drop=True),
            meta.loc[final_train].reset_index(drop=True),
        )
        moe = model.predict(
            x.loc[development].reset_index(drop=True),
            meta.loc[development].reset_index(drop=True),
        )
        moe["actual"] = meta.loc[development, "actual"].to_numpy()
        common = _complete_origins(moe) & _complete_origins(level_shape)
        moe = moe[moe["origin"].isin(common)].copy()
        shape = level_shape[level_shape["origin"].isin(common)].copy()
        weight, blend, candidates = _select_level_shape_blend(
            moe, shape, exclusive_data_end
        )
        result = {
            "subset": name,
            "columns": list(columns),
            "origins": len(common),
            "point_scale": scale,
            "scale_calibration_2025_mape": scale_mape,
            "direct_dev_mape": _mape(moe),
            "level_shape_blend_weight": weight,
            "blend_dev_mape": _mape(blend),
            "blend_candidates": candidates,
        }
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

    selected = min(results, key=lambda row: row["blend_dev_mape"])
    report = {
        "protocol": {
            "development_last_origin": str(dev_end),
            "post_cutoff_checkpoints_loaded": False,
            "holdout_predictions_constructed": False,
            "baseline_without_neighbours_dev_mape": args.baseline_dev_mape,
        },
        "results": results,
        "selected": selected,
        "beats_baseline": selected["blend_dev_mape"] < args.baseline_dev_mape,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet")
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument("--france-price", default="data/cache/day_ahead_price_fr_hourly.parquet")
    parser.add_argument(
        "--neighbour-prices",
        default="data/cache/day_ahead_price_neighbors_fr_hourly.parquet",
    )
    parser.add_argument(
        "--rte-unavailability",
        default="data/cache/rte_unavailability_messages.parquet",
    )
    parser.add_argument("--daily-predictions", required=True)
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--scale-train-end", default="2025-01-01")
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--dev-end", default="2026-02-28")
    parser.add_argument("--baseline-dev-mape", type=float, default=11.244722121243361)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

