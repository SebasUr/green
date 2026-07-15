"""Build an auditable energy/carbon JSON report for one terminal Job."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd

from green_observatory.observability.accounting import account_carbon, measure_counter
from green_observatory.observability.cluster import job_outcome, pod_execution
from green_observatory.observability.models import (
    CarbonAccounting,
    EnergyAccounting,
    JobCarbonReport,
    JobIdentity,
    MeasurementQuality,
    PodEnergyMeasurement,
    PodExecution,
    utc_now,
)
from green_observatory.observability.prometheus import (
    PrometheusClient,
    kepler_pod_counter_query,
)


class CarbonSource(Protocol):
    def load(self, start: float, end: float) -> pd.Series: ...


def _utc_from_epoch(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, tz=timezone.utc) if value is not None else None


class JobReporter:
    def __init__(
        self,
        prometheus: PrometheusClient,
        carbon_source: CarbonSource,
        *,
        zone: str = "package",
        step_seconds: int = 10,
        boundary_buffer_seconds: int = 60,
        min_energy_coverage: float = 0.90,
        min_carbon_coverage: float = 0.95,
    ) -> None:
        self.prometheus = prometheus
        self.carbon_source = carbon_source
        self.zone = zone
        self.step_seconds = step_seconds
        self.boundary_buffer_seconds = boundary_buffer_seconds
        self.min_energy_coverage = min_energy_coverage
        self.min_carbon_coverage = min_carbon_coverage

    def build(self, job: dict[str, Any], pods: list[dict[str, Any]]) -> JobCarbonReport:
        outcome = job_outcome(job)
        if outcome is None:
            raise ValueError("Job is not terminal")
        executions = [item for pod in pods if (item := pod_execution(pod)) is not None]
        if not executions:
            raise ValueError("Job has no terminal pods with execution timestamps")
        start = min(item.started_at.timestamp() for item in executions)
        end = max(item.finished_at.timestamp() for item in executions)
        uid_pattern = "|".join(item.uid for item in executions)
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:|" for char in uid_pattern):
            raise ValueError("pod UID contains characters unsafe for a Prometheus regex")
        query = kepler_pod_counter_query(uid_pattern, self.zone)
        matrix = self.prometheus.query_range(
            query,
            start - self.boundary_buffer_seconds,
            end + self.boundary_buffer_seconds,
            self.step_seconds,
        )
        by_uid = {series.get("metric", {}).get("pod_id"): series for series in matrix}

        measurements: dict[str, Any] = {}
        pod_energy: list[PodEnergyMeasurement] = []
        warnings: list[str] = []
        for execution in executions:
            series = by_uid.get(execution.uid)
            samples = series.get("values", []) if series else []
            measurement = measure_counter(
                [(float(ts), float(value)) for ts, value in samples],
                execution.started_at.timestamp(),
                execution.finished_at.timestamp(),
                boundary_tolerance_seconds=max(90, self.step_seconds * 4),
                assume_created_at_start=True,
            )
            measurements[execution.uid] = measurement
            item_warnings = list(measurement.warnings)
            if series is None:
                item_warnings.append("Prometheus returned no Kepler series for this pod UID")
            warnings.extend(f"{execution.name}: {message}" for message in item_warnings)
            pod_energy.append(
                PodEnergyMeasurement(
                    pod_uid=execution.uid,
                    pod_name=execution.name,
                    node_name=execution.node_name,
                    energy_joules=measurement.energy_joules,
                    energy_kwh=measurement.energy_joules / 3_600_000,
                    sample_count=measurement.sample_count,
                    counter_resets=measurement.counter_resets,
                    coverage_ratio=measurement.coverage_ratio,
                    first_sample_at=_utc_from_epoch(measurement.first_sample),
                    last_sample_at=_utc_from_epoch(measurement.last_sample),
                    warnings=item_warnings,
                )
            )

        total_joules = sum(item.energy_joules for item in pod_energy)
        total_pod_seconds = sum(item.duration_seconds for item in executions)
        energy_coverage = (
            sum(
                execution.duration_seconds * measurements[execution.uid].coverage_ratio
                for execution in executions
            )
            / total_pod_seconds
            if total_pod_seconds > 0
            else 0.0
        )
        all_increments = [
            increment
            for execution in executions
            for increment in measurements[execution.uid].increments
        ]
        carbon_series = self.carbon_source.load(start, end)
        carbon_measurement = account_carbon(all_increments, carbon_series)
        warnings.extend(carbon_measurement.warnings)

        series_found = sum(uid in by_uid for uid in measurements)
        sample_count = sum(item.sample_count for item in pod_energy)
        resets = sum(item.counter_resets for item in pod_energy)
        valid = (
            total_joules > 0
            and series_found == len(executions)
            and energy_coverage >= self.min_energy_coverage
            and carbon_measurement.energy_coverage_ratio >= self.min_carbon_coverage
            and carbon_measurement.emissions_gco2eq is not None
        )
        final = valid
        if energy_coverage < self.min_energy_coverage:
            warnings.append(
                f"energy coverage {energy_coverage:.3f} is below {self.min_energy_coverage:.3f}"
            )
        if carbon_measurement.energy_coverage_ratio < self.min_carbon_coverage:
            warnings.append(
                "carbon-weighted energy coverage "
                f"{carbon_measurement.energy_coverage_ratio:.3f} is below "
                f"{self.min_carbon_coverage:.3f}; RTE may not have published the final interval"
            )

        metadata = job["metadata"]
        safe_annotations = {
            key: value
            for key, value in (metadata.get("annotations", {}) or {}).items()
            if key.startswith("sustainability.cern.ch/")
            or key.startswith("green-observatory.io/")
        }
        wall_duration = max(0.0, end - start)
        return JobCarbonReport(
            generated_at=utc_now(),
            job=JobIdentity(
                uid=metadata["uid"],
                namespace=metadata["namespace"],
                name=metadata["name"],
                labels=metadata.get("labels", {}) or {},
                annotations=safe_annotations,
                outcome=outcome,
            ),
            execution_started_at=_utc_from_epoch(start),
            execution_finished_at=_utc_from_epoch(end),
            duration_seconds=wall_duration,
            pod_executions=executions,
            energy=EnergyAccounting(
                zone=self.zone,
                total_joules=total_joules,
                total_kwh=total_joules / 3_600_000,
                average_power_watts=total_joules / wall_duration if wall_duration else None,
                pods=pod_energy,
            ),
            carbon=CarbonAccounting(
                energy_weighted_intensity_gco2eq_per_kwh=(
                    carbon_measurement.weighted_intensity_gco2eq_per_kwh
                ),
                emissions_gco2eq=carbon_measurement.emissions_gco2eq,
                accounted_energy_joules=carbon_measurement.accounted_energy_joules,
                interval_count=carbon_measurement.interval_count,
                first_carbon_at=_utc_from_epoch(carbon_measurement.first_carbon_at),
                last_carbon_at=_utc_from_epoch(carbon_measurement.last_carbon_at),
            ),
            quality=MeasurementQuality(
                valid=valid,
                final=final,
                energy_coverage_ratio=energy_coverage,
                carbon_energy_coverage_ratio=carbon_measurement.energy_coverage_ratio,
                prometheus_series_expected=len(executions),
                prometheus_series_found=series_found,
                sample_count=sample_count,
                counter_resets=resets,
                carbon_points=len(carbon_series),
                missing_carbon_intervals=carbon_measurement.missing_intervals,
                warnings=sorted(set(warnings)),
            ),
        )
