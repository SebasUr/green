"""Aggregate paired daily-refit checkpoints into a reproducible report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from green_observatory.carbon.protocols import aggregate_metrics, regularize_hourly
from green_observatory.providers.carbon_odre import OdreCarbonProvider
from green_observatory.windows.oracle import window_selection_metrics


MODEL = "fossil_regime_recent_mapper"


def _load(directory: str, label: str) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found in {directory}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame = frame.loc[frame["model"] == MODEL].copy()
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame["model"] = label
    return frame.drop_duplicates(["origin", "horizon"], keep="last")


def _complete_origins(frame: pd.DataFrame) -> set[pd.Timestamp]:
    complete: set[pd.Timestamp] = set()
    for origin, group in frame.groupby("origin"):
        finite = np.isfinite(group[["prediction", "actual"]].to_numpy()).all()
        horizons = set(group["horizon"].astype(int))
        if len(group) == 24 and horizons == set(range(1, 25)) and finite:
            complete.add(origin)
    return complete


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.round(6).to_json(orient="records"))


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, origin), group in frame.groupby(["model", "origin"]):
        group = group.sort_values("horizon")
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        i_model = int(np.argmin(prediction))
        i_oracle = int(np.argmin(actual))
        rows.append(
            {
                "model": model,
                "origin": origin,
                "mape": 100.0 * np.mean(np.abs(prediction - actual) / np.abs(actual)),
                "mae": np.mean(np.abs(prediction - actual)),
                "regret": actual[i_model] - actual[i_oracle],
                "top1": float(actual[i_model] <= actual[i_oracle] + 1e-9),
                "spearman": spearmanr(prediction, actual).statistic,
            }
        )
    return pd.DataFrame(rows)


def _paired_summary(daily: pd.DataFrame) -> dict:
    wide_mape = daily.pivot(index="origin", columns="model", values="mape")
    wide_regret = daily.pivot(index="origin", columns="model", values="regret")
    all_mape = wide_mape["all_history"]
    two_mape = wide_mape["trailing_2y"]
    all_regret = wide_regret["all_history"]
    two_regret = wide_regret["trailing_2y"]
    mape_difference = (two_mape - all_mape).to_numpy()
    regret_difference = (two_regret - all_regret).to_numpy()
    return {
        "mape_difference_2y_minus_all_pp": float((two_mape - all_mape).mean()),
        "mape_difference_95pct_block_ci": _block_mean_ci(mape_difference),
        "days_2y_lower_mape_pct": 100.0 * float((two_mape < all_mape).mean()),
        "regret_difference_2y_minus_all": float((two_regret - all_regret).mean()),
        "regret_difference_95pct_block_ci": _block_mean_ci(regret_difference),
        "days_2y_lower_regret_pct": 100.0 * float((two_regret < all_regret).mean()),
        "days_equal_regret_pct": 100.0 * float((two_regret == all_regret).mean()),
    }


def _block_mean_ci(
    values: np.ndarray, *, block_days: int = 7, samples: int = 5000
) -> list[float]:
    """Moving-block bootstrap CI that preserves short serial dependence."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260717)
    blocks_needed = int(np.ceil(len(values) / block_days))
    max_start = len(values) - block_days
    means = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        draw = np.concatenate([values[start : start + block_days] for start in starts])
        means[sample] = draw[: len(values)].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _markdown(report: dict) -> str:
    lines = [
        "# Evaluación daily-refit 2026",
        "",
        "Cada día se entrena con datos estrictamente anteriores al origen y se "
        "pronostican las 24 horas siguientes. La comparación usa sólo días completos "
        "comunes a ambas ventanas.",
        "",
        f"- Periodo: {report['period']['first_origin']} a {report['period']['last_origin']}",
        f"- Días válidos: {report['period']['complete_paired_days']}",
        f"- Pronósticos por estrategia: {report['period']['observations_per_strategy']}",
        "",
        "## Resultado global",
        "",
        "| Ventana de entrenamiento | MAPE | WAPE | MAE | RMSE | Oracle potencial | Regret | Spearman | Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = {row["model"]: row for row in report["aggregate_metrics"]}
    selection = {row["strategy"]: row for row in report["window_selection"]}
    for model, label in (("all_history", "Todo el historial"), ("trailing_2y", "Últimos 2 años")):
        m = metrics[model]
        s = selection[model]
        lines.append(
            f"| {label} | {m['mape']:.2f}% | {m['wape']:.2f}% | {m['mae']:.2f} | "
            f"{m['rmse']:.2f} | {s['pct_oracle_potential']:.1f}% | "
            f"{s['mean_regret']:.2f} | {s['spearman']:.3f} | {100*s['top1_accuracy']:.1f}% |"
        )
    paired = report["paired_comparison"]
    lines.extend(
        [
            "",
            "## Comparación pareada por día",
            "",
            f"- Diferencia MAPE (2 años − todo): {paired['mape_difference_2y_minus_all_pp']:+.2f} puntos.",
            f"- IC 95% por bloques de 7 días para esa diferencia: "
            f"[{paired['mape_difference_95pct_block_ci'][0]:+.2f}, "
            f"{paired['mape_difference_95pct_block_ci'][1]:+.2f}] puntos.",
            f"- Dos años gana en MAPE en {paired['days_2y_lower_mape_pct']:.1f}% de los días.",
            f"- Diferencia de regret (2 años − todo): {paired['regret_difference_2y_minus_all']:+.2f} gCO₂/kWh.",
            f"- IC 95% por bloques para esa diferencia: "
            f"[{paired['regret_difference_95pct_block_ci'][0]:+.2f}, "
            f"{paired['regret_difference_95pct_block_ci'][1]:+.2f}] gCO₂/kWh.",
            f"- Dos años gana en regret en {paired['days_2y_lower_regret_pct']:.1f}% de los días; "
            f"empatan en {paired['days_equal_regret_pct']:.1f}%.",
            "",
            "## Por mes",
            "",
            "| Mes | Ventana | MAPE | MAE | Oracle potencial | Regret |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for month in report["monthly"]:
        monthly_metrics = {row["model"]: row for row in month["aggregate_metrics"]}
        monthly_selection = {row["strategy"]: row for row in month["window_selection"]}
        for model, label in (("all_history", "Todo"), ("trailing_2y", "2 años")):
            m = monthly_metrics[model]
            s = monthly_selection[model]
            lines.append(
                f"| {month['month']} | {label} | {m['mape']:.2f}% | {m['mae']:.2f} | "
                f"{s['pct_oracle_potential']:.1f}% | {s['mean_regret']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    all_history = _load(args.all_history, "all_history")
    trailing_2y = _load(args.trailing_2y, "trailing_2y")
    common = _complete_origins(all_history) & _complete_origins(trailing_2y)
    if not common:
        raise ValueError("no complete paired origins")
    predictions = pd.concat(
        [
            all_history.loc[all_history["origin"].isin(common)],
            trailing_2y.loc[trailing_2y["origin"].isin(common)],
        ],
        ignore_index=True,
    )
    full = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    aggregate = aggregate_metrics(predictions)
    selection = window_selection_metrics(predictions, full).reset_index()
    daily = _daily(predictions)

    monthly: list[dict] = []
    for month, part in predictions.groupby(predictions["origin"].dt.strftime("%Y-%m")):
        monthly.append(
            {
                "month": month,
                "days": int(part["origin"].nunique()),
                "aggregate_metrics": _records(aggregate_metrics(part)),
                "window_selection": _records(
                    window_selection_metrics(part, full).reset_index()
                ),
            }
        )

    fit = (
        predictions.groupby(["model", "origin"])["fit_seconds"]
        .first()
        .groupby("model")
        .agg(["mean", "median", "max"])
        .reset_index()
    )
    daily_summary = daily.groupby("model")[[
        "mape", "mae", "regret", "spearman", "top1"
    ]].agg(["mean", "median", "std"])
    daily_summary.columns = ["_".join(column) for column in daily_summary.columns]
    daily_summary = daily_summary.reset_index()
    report = {
        "protocol": "daily_expanding_refit_vs_trailing_2y_refit",
        "target_model": MODEL,
        "period": {
            "first_origin": str(min(common)),
            "last_origin": str(max(common)),
            "complete_paired_days": len(common),
            "observations_per_strategy": 24 * len(common),
        },
        "aggregate_metrics": _records(aggregate),
        "window_selection": _records(selection),
        "paired_comparison": _paired_summary(daily),
        "daily_metric_summary": _records(daily_summary),
        "fit_seconds": _records(fit),
        "monthly": monthly,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    predictions.to_parquet(output.with_suffix(".predictions.parquet"), index=False)
    daily.to_parquet(output.with_suffix(".daily.parquet"), index=False)
    print(_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-history", required=True)
    parser.add_argument("--trailing-2y", required=True)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_enriched.parquet")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
