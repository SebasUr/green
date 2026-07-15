"""Reusable loaders and comparisons for Experiment 01 captures.

The notebook imports this module, but the functions are intentionally usable
from scripts and tests as well.  Runs without a complete summary/result remain
visible in the inventory and are excluded only from scientific comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


POWER_METRICS = {
    "kepler_node_cpu_watts": "Node total",
    "kepler_node_cpu_active_watts": "Node active",
    "kepler_node_cpu_idle_watts": "Node idle",
    "workload_pod_watts": "Monte Carlo pod",
}


def find_experiment_root(start: Path | str | None = None) -> Path:
    """Find the experiment root from the notebook, repo root, or analysis dir."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        direct = candidate / "scripts" / "run_monte_carlo.py"
        nested = candidate / "kubernetes" / "experiment-01" / "scripts" / "run_monte_carlo.py"
        if direct.is_file():
            return candidate
        if nested.is_file():
            return candidate / "kubernetes" / "experiment-01"
    raise FileNotFoundError("no se encontró kubernetes/experiment-01")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nested(data: dict[str, Any], path: str, default: Any = np.nan) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _messages(value: Any) -> str:
    return " | ".join(str(item) for item in (value or []))


def build_inventory(root: Path | str) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for kind, parent, result_name, valid_key in (
        ("idle", root / "idle", "summary.json", "valid_idle_baseline"),
        ("trial", root / "trials", "result.json", "valid_trial"),
    ):
        if not parent.is_dir():
            continue
        for run_dir in sorted(path for path in parent.iterdir() if path.is_dir()):
            status = read_json(run_dir / "status.json")
            result = read_json(run_dir / result_name)
            metadata = result.get("metadata", {})
            rows.append(
                {
                    "run_id": run_dir.name,
                    "kind": kind,
                    "status": status.get("status", "unknown"),
                    "complete": status.get("status") == "complete",
                    "valid": result.get(valid_key, False) is True,
                    "policy": result.get("policy", "idle" if kind == "idle" else None),
                    "started_at": metadata.get(
                        "started_at", metadata.get("submitted_at")
                    ),
                    "duration_s": metadata.get(
                        "actual_duration_seconds",
                        nested(result, "execution.runtime_seconds"),
                    ),
                    "blocking_conditions": _messages(
                        result.get("blocking_conditions")
                    ),
                    "warnings": _messages(result.get("collection_warnings")),
                    "path": str(run_dir.resolve()),
                    "has_metrics_csv": (run_dir / "metrics.csv").is_file(),
                    "metrics_size_mb": (
                        (run_dir / "metrics.csv").stat().st_size / 1_000_000
                        if (run_dir / "metrics.csv").is_file()
                        else 0.0
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
    return frame.sort_values(["kind", "started_at", "run_id"], na_position="last").reset_index(
        drop=True
    )


def load_idle_summary(root: Path | str) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "idle").glob("*/summary.json")):
        data = read_json(path)
        metrics = data.get("primary_metrics", {})
        rows.append(
            {
                "run_id": path.parent.name,
                "valid": data.get("valid_idle_baseline", False) is True,
                "started_at": nested(data, "metadata.started_at", None),
                "duration_s": nested(data, "metadata.actual_duration_seconds"),
                "sample_count": nested(data, "primary_metrics.total_watts.sample_count"),
                "coverage": nested(data, "primary_metrics.total_watts.coverage_ratio"),
                "total_w_mean": nested(data, "primary_metrics.total_watts.mean"),
                "total_w_median": nested(data, "primary_metrics.total_watts.median"),
                "total_w_p95": nested(data, "primary_metrics.total_watts.p95"),
                "active_w_mean": nested(data, "primary_metrics.active_watts.mean"),
                "active_w_median": nested(data, "primary_metrics.active_watts.median"),
                "idle_w_mean": nested(data, "primary_metrics.idle_watts.mean"),
                "idle_w_median": nested(data, "primary_metrics.idle_watts.median"),
                "cpu_ratio_mean": nested(data, "primary_metrics.cpu_usage_ratio.mean"),
                "cpu_ratio_median": nested(data, "primary_metrics.cpu_usage_ratio.median"),
                "observed_energy_j": nested(
                    data, "primary_metrics.total_energy.observed_delta_joules"
                ),
                "counter_resets": nested(data, "primary_metrics.total_energy.counter_resets"),
                "blocking_conditions": _messages(data.get("blocking_conditions")),
                "warnings": _messages(data.get("collection_warnings")),
                "path": str(path.parent.resolve()),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
    return frame


def _energy_delta(data: dict[str, Any], name: str) -> Any:
    return nested(data, f"energy.{name}.observed_delta_joules")


def load_trial_summary(root: Path | str) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "trials").glob("*/result.json")):
        data = read_json(path)
        rows.append(
            {
                "run_id": data.get("trial_id", path.parent.name),
                "policy": data.get("policy"),
                "valid": data.get("valid_trial", False) is True,
                "started_at": nested(data, "execution.workload_started_at", None),
                "runtime_s": nested(data, "execution.runtime_seconds"),
                "workers": nested(data, "parameters.workers"),
                "samples_per_worker": nested(data, "parameters.samples_per_worker"),
                "total_samples": nested(data, "parameters.total_samples"),
                "base_seed": nested(data, "parameters.base_seed"),
                "pi_estimate": nested(data, "execution.pi_estimate"),
                "scientific_ok": nested(
                    data, "execution.scientific_output_matches_parameters", False
                ),
                "pod_energy_j": nested(data, "energy.pod_cpu_energy_joules"),
                "pod_energy_kwh": nested(data, "energy.pod_cpu_energy_kwh"),
                "pod_avg_w": nested(data, "energy.average_attributed_pod_power_watts"),
                "pod_w_mean": nested(data, "energy.pod_watts.mean"),
                "pod_w_median": nested(data, "energy.pod_watts.median"),
                "node_total_energy_j": _energy_delta(data, "node_total_energy"),
                "node_active_energy_j": _energy_delta(data, "node_active_energy"),
                "node_idle_energy_j": _energy_delta(data, "node_idle_energy"),
                "node_total_w_mean": nested(data, "energy.node_total_watts.mean"),
                "node_total_w_median": nested(data, "energy.node_total_watts.median"),
                "node_active_w_mean": nested(data, "energy.node_active_watts.mean"),
                "node_active_w_median": nested(data, "energy.node_active_watts.median"),
                "cpu_ratio_mean": nested(data, "energy.node_cpu_usage_ratio.mean"),
                "cpu_ratio_median": nested(data, "energy.node_cpu_usage_ratio.median"),
                "node_counter_resets": nested(
                    data, "energy.node_total_energy.counter_resets"
                ),
                "recommended_samples": nested(
                    data,
                    "execution.recommended_samples_per_worker_for_target_runtime",
                ),
                "image_id": nested(data, "execution.image.image_id", None),
                "blocking_conditions": _messages(data.get("blocking_conditions")),
                "warnings": _messages(data.get("collection_warnings")),
                "path": str(path.parent.resolve()),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
        frame["energy_per_billion_samples_j"] = (
            frame["pod_energy_j"] / frame["total_samples"] * 1e9
        )
        frame["samples_per_second"] = frame["total_samples"] / frame["runtime_s"]
        frame["attribution_share_active"] = (
            frame["pod_energy_j"] / frame["node_active_energy_j"]
        )
    return frame


def build_idle_trial_comparison(
    idle: pd.DataFrame, trials: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare every valid trial with the median of valid idle baselines."""
    valid_idle = idle[idle["valid"]].copy()
    valid_trials = trials[trials["valid"]].copy()
    if valid_idle.empty or valid_trials.empty:
        return pd.DataFrame(), {}
    reference = {
        column: float(valid_idle[column].median())
        for column in (
            "total_w_mean",
            "total_w_median",
            "active_w_mean",
            "active_w_median",
            "idle_w_mean",
            "idle_w_median",
            "cpu_ratio_mean",
            "cpu_ratio_median",
        )
    }
    compared = valid_trials[
        [
            "run_id",
            "policy",
            "runtime_s",
            "total_samples",
            "pod_energy_j",
            "pod_avg_w",
            "node_total_w_mean",
            "node_total_w_median",
            "node_active_w_mean",
            "cpu_ratio_mean",
            "energy_per_billion_samples_j",
        ]
    ].copy()
    compared["idle_reference_total_w"] = reference["total_w_mean"]
    compared["total_power_delta_w"] = (
        compared["node_total_w_mean"] - reference["total_w_mean"]
    )
    compared["total_power_delta_pct"] = (
        100 * compared["total_power_delta_w"] / reference["total_w_mean"]
    )
    compared["active_power_delta_w"] = (
        compared["node_active_w_mean"] - reference["active_w_mean"]
    )
    compared["idle_energy_for_runtime_j"] = (
        reference["total_w_mean"] * compared["runtime_s"]
    )
    compared["estimated_incremental_node_energy_j"] = (
        compared["total_power_delta_w"] * compared["runtime_s"]
    )
    compared["pod_vs_incremental_energy_ratio"] = (
        compared["pod_energy_j"] / compared["estimated_incremental_node_energy_j"]
    )
    return compared, reference


def load_metric_inventory(root: Path | str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    root = Path(root)
    for kind, parent in (("idle", root / "idle"), ("trial", root / "trials")):
        for path in sorted(parent.glob("*/series-summary.csv")):
            frame = pd.read_csv(path, usecols=["query", "metric", "sample_count"])
            frame.insert(0, "run_id", path.parent.name)
            frame.insert(1, "kind", kind)
            rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["run_id", "kind", "query", "metric", "series", "samples"])
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby(["run_id", "kind", "query", "metric"], dropna=False)
        .agg(series=("sample_count", "size"), samples=("sample_count", "sum"))
        .reset_index()
        .sort_values(["run_id", "query", "metric"])
    )


def load_metrics(
    run_path: Path | str,
    metrics: Iterable[str] | None = None,
    queries: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load selected rows from a run's long metrics.csv."""
    path = Path(run_path) / "metrics.csv"
    usecols = [
        "query",
        "metric",
        "timestamp_utc",
        "timestamp_unix",
        "value",
        "zone",
        "path",
        "namespace",
        "pod_namespace",
        "pod",
        "pod_name",
        "container",
        "device",
        "mode",
        "cpu",
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    if metrics is not None:
        frame = frame[frame["metric"].isin(list(metrics))]
    if queries is not None:
        frame = frame[frame["query"].isin(list(queries))]
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_power_timeseries(
    run_path: Path | str,
    kind: str,
    zone: str = "package",
) -> pd.DataFrame:
    run_path = Path(run_path)
    metrics = load_metrics(run_path, metrics=POWER_METRICS)
    metrics = metrics[(metrics["zone"].fillna(zone) == zone) | metrics["zone"].isna()].copy()
    if kind == "trial":
        result = read_json(run_path / "result.json")
        start = pd.to_datetime(nested(result, "execution.workload_started_at"), utc=True)
        end = pd.to_datetime(nested(result, "execution.workload_finished_at"), utc=True)
        metrics = metrics[(metrics["timestamp"] >= start) & (metrics["timestamp"] <= end)]
    elif metrics.empty:
        return metrics
    else:
        start = metrics["timestamp"].min()
    metrics["elapsed_s"] = (metrics["timestamp"] - start).dt.total_seconds()
    metrics["series"] = metrics["metric"].map(POWER_METRICS).fillna(metrics["metric"])
    return metrics


def integrate_power(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Integrate each power series using observed timestamps (trapezoidal rule)."""
    rows: list[dict[str, Any]] = []
    for name, group in timeseries.groupby("series"):
        group = group.dropna(subset=["elapsed_s", "value"]).sort_values("elapsed_s")
        rows.append(
            {
                "series": name,
                "samples": len(group),
                "observed_span_s": (
                    group["elapsed_s"].iloc[-1] - group["elapsed_s"].iloc[0]
                    if len(group) > 1
                    else 0.0
                ),
                "integrated_energy_j": (
                    float(np.trapezoid(group["value"], x=group["elapsed_s"]))
                    if len(group) > 1
                    else np.nan
                ),
                "mean_w": group["value"].mean(),
                "median_w": group["value"].median(),
                "p95_w": group["value"].quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def repeatability_summary(trials: pd.DataFrame) -> pd.DataFrame:
    valid = trials[trials["valid"]].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby(
        ["workers", "samples_per_worker", "base_seed", "image_id"], dropna=False
    )
    return grouped.agg(
        trials=("run_id", "count"),
        runtime_mean_s=("runtime_s", "mean"),
        runtime_std_s=("runtime_s", "std"),
        pod_energy_mean_j=("pod_energy_j", "mean"),
        pod_energy_std_j=("pod_energy_j", "std"),
    ).reset_index().assign(
        runtime_cv_pct=lambda frame: 100
        * frame["runtime_std_s"]
        / frame["runtime_mean_s"],
        pod_energy_cv_pct=lambda frame: 100
        * frame["pod_energy_std_j"]
        / frame["pod_energy_mean_j"],
    )


def paired_policy_comparison(
    trials: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Compare explicit run-now/green-window pairs supplied by the notebook user."""
    required = {"pair_id", "run_now", "green_window"}
    if pairs.empty:
        return pd.DataFrame()
    if not required.issubset(pairs.columns):
        raise ValueError(f"pairs requiere columnas {sorted(required)}")
    indexed = trials.set_index("run_id")
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        now = indexed.loc[pair.run_now]
        green = indexed.loc[pair.green_window]
        rows.append(
            {
                "pair_id": pair.pair_id,
                "run_now": pair.run_now,
                "green_window": pair.green_window,
                "energy_run_now_j": now["pod_energy_j"],
                "energy_green_j": green["pod_energy_j"],
                "energy_change_pct": 100
                * (green["pod_energy_j"] - now["pod_energy_j"])
                / now["pod_energy_j"],
                "runtime_run_now_s": now["runtime_s"],
                "runtime_green_s": green["runtime_s"],
                "runtime_change_pct": 100
                * (green["runtime_s"] - now["runtime_s"])
                / now["runtime_s"],
            }
        )
    return pd.DataFrame(rows)
