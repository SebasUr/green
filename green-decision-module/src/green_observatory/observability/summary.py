"""Flatten JobCarbonReport JSON files for notebooks and experiment comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from green_observatory.observability.models import JobCarbonReport


LABEL_PREFIX = "sustainability.cern.ch/"
COMMON_LABELS = ("workload", "policy", "scheduler", "experiment", "trial")


def summarize_reports(
    directory: Path | str,
    *,
    include_provisional: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            report = JobCarbonReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not include_provisional and not report.quality.final:
            continue
        labels = report.job.labels
        row = {
            "report_path": str(path.resolve()),
            "job_uid": report.job.uid,
            "namespace": report.job.namespace,
            "job_name": report.job.name,
            "outcome": report.job.outcome,
            "started_at": report.execution_started_at,
            "finished_at": report.execution_finished_at,
            "duration_seconds": report.duration_seconds,
            "pod_count": len(report.pod_executions),
            "energy_joules": report.energy.total_joules,
            "energy_kwh": report.energy.total_kwh,
            "average_power_watts": report.energy.average_power_watts,
            "weighted_intensity_gco2eq_per_kwh": (
                report.carbon.energy_weighted_intensity_gco2eq_per_kwh
            ),
            "emissions_gco2eq": report.carbon.emissions_gco2eq,
            "valid": report.quality.valid,
            "final": report.quality.final,
            "energy_coverage_ratio": report.quality.energy_coverage_ratio,
            "carbon_energy_coverage_ratio": report.quality.carbon_energy_coverage_ratio,
            "counter_resets": report.quality.counter_resets,
            "warnings": " | ".join(report.quality.warnings),
            "labels_json": json.dumps(labels, sort_keys=True, ensure_ascii=False),
        }
        for name in COMMON_LABELS:
            row[name] = labels.get(f"{LABEL_PREFIX}{name}")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
        frame["finished_at"] = pd.to_datetime(frame["finished_at"], utc=True)
        frame = frame.sort_values(["started_at", "namespace", "job_name"]).reset_index(drop=True)
    return frame
