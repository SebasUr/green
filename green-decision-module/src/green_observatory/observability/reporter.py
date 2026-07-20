"""Build an auditable energy/carbon JSON report for one terminal Job."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd

from green_observatory.observability.accounting import account_carbon, measure_counter
from green_observatory.observability.cluster import job_outcome, job_provenance, pod_execution
from green_observatory.observability.models import (
    CarbonAccounting,
    CoTenantPod,
    EnergyAccounting,
    EnergyIntervalRecord,
    JobCarbonReport,
    JobIdentity,
    MeasurementQuality,
    NodeContext,
    NodeIsolation,
    PodEnergyMeasurement,
    PodExecution,
    WorkloadOutput,
    utc_now,
)
from green_observatory.observability.prometheus import (
    PrometheusClient,
    kepler_node_counter_query,
    kepler_node_cpu_ratio_query,
    kepler_node_pods_counter_query,
    kepler_pod_counter_query,
    kepler_up_ratio_query,
    node_container_restarts_query,
)


class CarbonSource(Protocol):
    def load(self, start: float, end: float) -> pd.Series: ...


def _utc_from_epoch(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, tz=timezone.utc) if value is not None else None


def _samples(series: dict[str, Any]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for timestamp, value in series.get("values", []) or []:
        try:
            out.append((float(timestamp), float(value)))
        except (TypeError, ValueError):
            continue
    return out


def _scalar(result: list[dict[str, Any]]) -> float | None:
    """First finite value of an instant-query result, if any."""
    for series in result:
        value = series.get("value") or [None, None]
        try:
            return float(value[1])
        except (TypeError, ValueError, IndexError):
            continue
    return None


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
        kubectl: Any | None = None,
        collect_context: bool = True,
        capture_logs: bool = True,
        include_intervals: bool = False,
        max_log_bytes: int = 64 * 1024,
        max_cotenant_energy_share: float = 0.05,
        min_kepler_up_ratio: float = 0.99,
    ) -> None:
        self.prometheus = prometheus
        self.carbon_source = carbon_source
        self.zone = zone
        self.step_seconds = step_seconds
        self.boundary_buffer_seconds = boundary_buffer_seconds
        self.min_energy_coverage = min_energy_coverage
        self.min_carbon_coverage = min_carbon_coverage
        #: Optional read-only kubectl adapter; without it logs are skipped.
        self.kubectl = kubectl
        self.collect_context = collect_context
        self.capture_logs = capture_logs
        self.include_intervals = include_intervals
        self.max_log_bytes = max_log_bytes
        self.max_cotenant_energy_share = max_cotenant_energy_share
        self.min_kepler_up_ratio = min_kepler_up_ratio

    # ------------------------------------------------------------------ #
    # Reproducibility enrichment. Best-effort by design: any failure adds a
    # warning and leaves the block empty rather than losing the accounting.
    # ------------------------------------------------------------------ #
    def _measure_node_counter(
        self, metric: str, node: str, start: float, end: float
    ) -> tuple[float | None, int]:
        try:
            matrix = self.prometheus.query_range(
                kepler_node_counter_query(metric, node, self.zone),
                start - self.boundary_buffer_seconds,
                end + self.boundary_buffer_seconds,
                self.step_seconds,
            )
        except Exception:
            return None, 0
        if not matrix:
            return None, 0
        measured = measure_counter(
            _samples(matrix[0]),
            start,
            end,
            boundary_tolerance_seconds=max(90, self.step_seconds * 4),
        )
        return measured.energy_joules, measured.counter_resets

    def _build_node_context(
        self, node: str | None, start: float, end: float, job_joules: float
    ) -> NodeContext | None:
        if not node:
            return None
        warnings: list[str] = []
        total, total_resets = self._measure_node_counter(
            "kepler_node_cpu_joules_total", node, start, end
        )
        active, active_resets = self._measure_node_counter(
            "kepler_node_cpu_active_joules_total", node, start, end
        )
        idle, idle_resets = self._measure_node_counter(
            "kepler_node_cpu_idle_joules_total", node, start, end
        )
        if total is None and active is None:
            warnings.append("no Kepler node-level counters found for this node/zone")

        ratio_mean = ratio_max = None
        try:
            matrix = self.prometheus.query_range(
                kepler_node_cpu_ratio_query(node), start, end, self.step_seconds
            )
            values = [value for series in matrix for _, value in _samples(series)]
            if values:
                ratio_mean = statistics.fmean(values)
                ratio_max = max(values)
        except Exception:
            warnings.append("could not read kepler_node_cpu_usage_ratio")

        share = job_joules / active if active and active > 0 else None
        if share is not None and share > 1:
            warnings.append(
                "attributed Job energy exceeds node active energy; "
                "check the zone or concurrent measurement windows"
            )
        return NodeContext(
            node_name=node,
            zone=self.zone,
            total_energy_joules=total,
            active_energy_joules=active,
            idle_energy_joules=idle,
            cpu_utilization_mean=min(1.0, ratio_mean) if ratio_mean is not None else None,
            cpu_utilization_max=min(1.0, ratio_max) if ratio_max is not None else None,
            job_share_of_active_energy=share,
            counter_resets=total_resets + active_resets + idle_resets,
            warnings=warnings,
        )

    def _build_isolation(
        self,
        node: str | None,
        start: float,
        end: float,
        job_pod_uids: set[str],
        job_joules: float,
    ) -> NodeIsolation | None:
        if not node:
            return None
        warnings: list[str] = []
        window = int(max(1.0, end - start))

        co_tenants: list[CoTenantPod] = []
        co_tenant_joules: float | None = None
        try:
            matrix = self.prometheus.query_range(
                kepler_node_pods_counter_query(node, self.zone),
                start - self.boundary_buffer_seconds,
                end + self.boundary_buffer_seconds,
                self.step_seconds,
            )
            total = 0.0
            for series in matrix:
                metric = series.get("metric", {})
                if metric.get("pod_id") in job_pod_uids:
                    continue
                measured = measure_counter(
                    _samples(series),
                    start,
                    end,
                    boundary_tolerance_seconds=max(90, self.step_seconds * 4),
                )
                if measured.energy_joules <= 0:
                    continue
                total += measured.energy_joules
                co_tenants.append(
                    CoTenantPod(
                        pod_namespace=metric.get("pod_namespace"),
                        pod_name=metric.get("pod_name"),
                        energy_joules=measured.energy_joules,
                    )
                )
            co_tenant_joules = total
            co_tenants.sort(key=lambda item: item.energy_joules, reverse=True)
        except Exception:
            warnings.append("could not enumerate co-tenant pods from Kepler")

        share = None
        if co_tenant_joules is not None:
            denominator = job_joules + co_tenant_joules
            share = co_tenant_joules / denominator if denominator > 0 else 0.0

        restarts = None
        try:
            restarts_value = _scalar(
                self.prometheus.query(node_container_restarts_query(node, window), end)
            )
            restarts = int(round(restarts_value)) if restarts_value is not None else None
        except Exception:
            warnings.append("could not read container restarts (kube-state-metrics)")

        kepler_up = None
        try:
            kepler_up = _scalar(self.prometheus.query(kepler_up_ratio_query(window), end))
        except Exception:
            warnings.append("could not read the Kepler scrape-up ratio")

        clean = True
        if share is None:
            clean = False
            warnings.append("co-tenant energy is unknown, so isolation cannot be confirmed")
        elif share > self.max_cotenant_energy_share:
            clean = False
            warnings.append(
                f"co-tenant pods account for {share:.1%} of attributed energy "
                f"(limit {self.max_cotenant_energy_share:.1%})"
            )
        if restarts:
            clean = False
            warnings.append(f"{restarts} container restart(s) on the node during the window")
        if kepler_up is not None and kepler_up < self.min_kepler_up_ratio:
            clean = False
            warnings.append(
                f"Kepler was scrapeable only {kepler_up:.1%} of the window "
                f"(limit {self.min_kepler_up_ratio:.1%})"
            )
        return NodeIsolation(
            clean_node=clean,
            co_tenant_pods=co_tenants,
            co_tenant_energy_joules=co_tenant_joules,
            co_tenant_energy_share=min(1.0, share) if share is not None else None,
            container_restarts=restarts,
            kepler_up_ratio=min(1.0, kepler_up) if kepler_up is not None else None,
            warnings=warnings,
        )

    def _build_workload_outputs(
        self, namespace: str, executions: list[PodExecution]
    ) -> list[WorkloadOutput]:
        if not (self.capture_logs and self.kubectl):
            return []
        outputs: list[WorkloadOutput] = []
        for execution in executions:
            warnings: list[str] = []
            stdout: str | None = None
            try:
                stdout = self.kubectl.pod_logs(namespace, execution.name)
            except Exception as exc:
                warnings.append(f"could not read pod logs: {exc}")
            if stdout is None:
                outputs.append(
                    WorkloadOutput(
                        pod_uid=execution.uid, pod_name=execution.name, warnings=warnings
                    )
                )
                continue
            raw = stdout.encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            truncated = len(raw) > self.max_log_bytes
            body = raw[: self.max_log_bytes].decode("utf-8", errors="replace") if raw else ""
            parsed = None
            if not truncated:
                try:
                    candidate = json.loads(stdout)
                    parsed = candidate if isinstance(candidate, dict) else None
                except ValueError:
                    parsed = None
            outputs.append(
                WorkloadOutput(
                    pod_uid=execution.uid,
                    pod_name=execution.name,
                    stdout_sha256=digest,
                    stdout_bytes=len(raw),
                    truncated=truncated,
                    stdout=body,
                    parsed_json=parsed,
                    warnings=warnings,
                )
            )
        return outputs

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

        # --- reproducibility enrichment (never fails the accounting) ---------
        node = next(
            (item.node_name for item in executions if item.node_name), None
        )
        node_context = isolation = None
        if self.collect_context:
            node_context = self._build_node_context(node, start, end, total_joules)
            isolation = self._build_isolation(
                node, start, end, {item.uid for item in executions}, total_joules
            )
            for block in (node_context, isolation):
                warnings.extend(block.warnings if block else [])
        if isolation and not isolation.clean_node:
            warnings.append(
                "node was not isolated for the whole window; see isolation.warnings"
            )
        workload_outputs = self._build_workload_outputs(metadata["namespace"], executions)
        warnings.extend(
            f"{item.pod_name}: {message}"
            for item in workload_outputs
            for message in item.warnings
        )
        intervals = None
        if self.include_intervals:
            intervals = [
                EnergyIntervalRecord(
                    start=_utc_from_epoch(item.start),
                    end=_utc_from_epoch(item.end),
                    joules=item.joules,
                    carbon_intensity_gco2eq_per_kwh=item.intensity_gco2eq_per_kwh,
                    emissions_gco2eq=item.emissions_gco2eq,
                )
                for item in carbon_measurement.intervals
            ]

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
            provenance=job_provenance(job, pods) if self.collect_context else None,
            node_context=node_context,
            isolation=isolation,
            workload_outputs=workload_outputs,
            energy_intervals=intervals,
        )
