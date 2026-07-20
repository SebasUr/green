"""Causal second-stage evaluation over daily-refit carbon checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.adaptive_ensemble import (
    DEFAULT_POINT_EXPERTS,
    causal_block_scaled_expert,
    causal_scaled_blend,
    causal_scaled_expert,
    rank_consensus,
)
from green_observatory.carbon.protocols import aggregate_metrics, regularize_hourly
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider
from green_observatory.windows.oracle import window_selection_metrics


def load_checkpoints(directory: str | Path) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet checkpoints in {directory}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame = frame.drop_duplicates(["model", "origin", "horizon"], keep="last")
    return frame


def persistence_baselines(
    predictions: pd.DataFrame, carbon: pd.DataFrame
) -> pd.DataFrame:
    template = (
        predictions[["origin", "horizon", "target_time", "actual"]]
        .drop_duplicates(["origin", "horizon"])
        .copy()
    )
    parts: list[pd.DataFrame] = []
    for days, name in ((1, "persistence_d1"), (7, "persistence_d7")):
        part = template.copy()
        lag_times = pd.DatetimeIndex(part["target_time"]) - pd.Timedelta(days=days)
        values = carbon[CARBON].reindex(lag_times).to_numpy(dtype=float)
        values = values.copy()
        values[lag_times >= pd.DatetimeIndex(part["origin"])] = np.nan
        part["prediction"] = values
        part["model"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True).dropna(subset=["prediction"])


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.round(6).to_json(orient="records"))


def _daily_mape(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ape"] = (
        (out["prediction"] - out["actual"]).abs()
        / out["actual"].abs().clip(lower=1e-9)
        * 100.0
    )
    return (
        out.groupby(["model", "origin"], as_index=False)["ape"]
        .mean()
        .rename(columns={"ape": "mape"})
    )


def _paired_block_ci(
    daily: pd.DataFrame,
    challenger: str,
    baseline: str,
    *,
    samples: int = 5000,
    block_days: int = 7,
) -> dict:
    wide = daily.pivot(index="origin", columns="model", values="mape").dropna(
        subset=[challenger, baseline]
    )
    differences = (wide[challenger] - wide[baseline]).to_numpy(dtype=float)
    if len(differences) < block_days:
        return {"mean_pp": float(differences.mean()), "ci95_pp": [np.nan, np.nan]}
    rng = np.random.default_rng(20260719)
    blocks = int(np.ceil(len(differences) / block_days))
    max_start = len(differences) - block_days
    means = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, max_start + 1, size=blocks)
        draw = np.concatenate(
            [differences[start : start + block_days] for start in starts]
        )[: len(differences)]
        means[sample] = draw.mean()
    return {
        "mean_pp": float(differences.mean()),
        "ci95_pp": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "days_challenger_wins_pct": 100.0 * float((differences < 0.0).mean()),
    }


def _period_metrics(
    predictions: pd.DataFrame, full: pd.DataFrame, start: pd.Timestamp | None
) -> dict:
    part = predictions if start is None else predictions.loc[predictions["origin"] >= start]
    return {
        "origins": int(part["origin"].nunique()),
        "aggregate": _records(aggregate_metrics(part)),
        "selection": _records(window_selection_metrics(part, full).reset_index()),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Metaevaluación causal de los modelos diarios",
        "",
        "La configuración del ensamble se fijó en enero-febrero y marzo-abril "
        "se reporta como holdout temporal. Cada escala y peso usa únicamente "
        "targets ya observados al origen correspondiente.",
        "",
        "## Holdout",
        "",
        "| Modelo | MAPE | WAPE | MAE | RMSE | Sesgo | Oráculo | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    point = {row["model"]: row for row in report["holdout"]["aggregate"]}
    selection = {
        row["strategy"]: row for row in report["holdout"]["selection"]
    }
    preferred = [
        "causal_level_shape_21d",
        "causal_scaled_blend_7d",
        "adaptive_signal_7d",
        "fossil_regime_recent_mapper",
        "fossil_regime_decision",
        "hybrid_h2_mapper_delta",
        "persistence_d1",
        "persistence_d7",
    ]
    for model in preferred:
        if model not in point or model not in selection:
            continue
        metric = point[model]
        decision = selection[model]
        lines.append(
            f"| `{model}` | {metric['mape']:.2f}% | {metric['wape']:.2f}% | "
            f"{metric['mae']:.2f} | {metric['rmse']:.2f} | {metric['bias']:+.2f} | "
            f"{decision['pct_oracle_potential']:.1f}% | {decision['mean_regret']:.2f} |"
        )
    paired = report["paired_selected_vs_adaptive"]
    lines.extend(
        [
            "",
            "## Comparación pareada",
            "",
            f"El calibrador nivel-forma cambia el MAPE diario frente al selector duro en "
            f"{paired['mean_pp']:+.2f} puntos; IC 95% por bloques: "
            f"[{paired['ci95_pp'][0]:+.2f}, {paired['ci95_pp'][1]:+.2f}].",
            "",
            "La segunda etapa es modular: no altera ni reentrena los "
            "modelos base y puede retirarse sin afectar los demás experimentos.",
            "",
            "## Cabeza de decisión",
            "",
            "La señal de menor MAPE no es el mejor selector de ventanas. El "
            "`rank_consensus` se conserva como salida independiente para ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    base = load_checkpoints(args.checkpoints)
    full = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    adaptive = causal_scaled_expert(
        base,
        lookback_days=7,
        candidates=DEFAULT_POINT_EXPERTS,
        default_expert="fossil_regime_recent_mapper",
        name="adaptive_signal_7d",
    )
    blend = causal_scaled_blend(
        base,
        lookback_days=args.lookback_days,
        candidates=DEFAULT_POINT_EXPERTS,
        temperature=args.temperature,
        top_k=args.top_k,
        name=f"causal_scaled_blend_{args.lookback_days}d",
    )
    level_shape = causal_block_scaled_expert(
        base,
        lookback_days=21,
        half_life_days=3.0,
        blocks=((1, 6), (7, 16), (17, 21), (22, 24)),
        block_weight=0.25,
        shape_expert="hybrid_h2_mapper_delta",
        shape_weight=0.10,
        name="causal_level_shape_21d",
    )
    ranks = rank_consensus(base, name="rank_consensus")
    persistence = persistence_baselines(base, full)
    predictions = pd.concat(
        [base, adaptive, blend, level_shape, ranks, persistence], ignore_index=True
    )
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    holdout_start = validation_end + pd.Timedelta(days=1)
    report = {
        "protocol": "causal_meta_ensemble_with_fixed_temporal_holdout",
        "configuration": {
            "selection_period_end": str(validation_end),
            "holdout_start": str(holdout_start),
            "lookback_days": args.lookback_days,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
        "full": _period_metrics(predictions, full, None),
        "holdout": _period_metrics(predictions, full, holdout_start),
    }
    holdout = predictions.loc[predictions["origin"] >= holdout_start]
    report["paired_selected_vs_adaptive"] = _paired_block_ci(
        _daily_mape(holdout),
        "causal_level_shape_21d",
        "adaptive_signal_7d",
    )
    report["paired_blend_vs_adaptive"] = _paired_block_ci(
        _daily_mape(holdout),
        f"causal_scaled_blend_{args.lookback_days}d",
        "adaptive_signal_7d",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    predictions.to_parquet(output.with_suffix(".predictions.parquet"), index=False)
    print(_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", required=True, help="Directory of daily parquet files."
    )
    parser.add_argument(
        "--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet"
    )
    parser.add_argument("--validation-end", default="2026-02-28")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.005)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
