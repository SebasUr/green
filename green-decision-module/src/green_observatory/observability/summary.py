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
        node = report.node_context
        isolation = report.isolation
        first_output = report.workload_outputs[0] if report.workload_outputs else None
        first_container = next(
            (
                container
                for pod in (report.provenance.pods if report.provenance else [])
                for container in pod.containers
            ),
            None,
        )
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
            # Reproducibility dimensions (schema 1.1): are two runs comparable,
            # and did they produce identical scientific output?
            "clean_node": isolation.clean_node if isolation else None,
            "co_tenant_energy_share": isolation.co_tenant_energy_share if isolation else None,
            "kepler_up_ratio": isolation.kepler_up_ratio if isolation else None,
            "node_name": node.node_name if node else None,
            "node_active_energy_joules": node.active_energy_joules if node else None,
            "job_share_of_active_energy": node.job_share_of_active_energy if node else None,
            "cpu_utilization_mean": node.cpu_utilization_mean if node else None,
            "image": first_container.image if first_container else None,
            "image_id": first_container.image_id if first_container else None,
            "stdout_sha256": first_output.stdout_sha256 if first_output else None,
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
