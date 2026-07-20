"""Typed output contract for per-Job energy and carbon reports."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from green_observatory.models import CarbonBasis, UtcDatetime


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PodExecution(_ReportModel):
    uid: str
    name: str
    phase: str | None = None
    node_name: str | None = None
    started_at: UtcDatetime
    finished_at: UtcDatetime
    duration_seconds: float = Field(..., ge=0)
    succeeded: bool = False


class JobIdentity(_ReportModel):
    uid: str
    namespace: str
    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    outcome: str


class PodEnergyMeasurement(_ReportModel):
    pod_uid: str
    pod_name: str
    node_name: str | None = None
    energy_joules: float = Field(..., ge=0)
    energy_kwh: float = Field(..., ge=0)
    sample_count: int = Field(..., ge=0)
    counter_resets: int = Field(..., ge=0)
    coverage_ratio: float = Field(..., ge=0, le=1)
    first_sample_at: UtcDatetime | None = None
    last_sample_at: UtcDatetime | None = None
    warnings: list[str] = Field(default_factory=list)


class EnergyAccounting(_ReportModel):
    metric: str = "kepler_pod_cpu_joules_total"
    zone: str = "package"
    scope: str = "operational_cpu_energy_attributed_by_kepler"
    total_joules: float = Field(..., ge=0)
    total_kwh: float = Field(..., ge=0)
    average_power_watts: float | None = Field(None, ge=0)
    pods: list[PodEnergyMeasurement]


class CarbonAccounting(_ReportModel):
    source: str = "rte:eco2mix"
    basis: CarbonBasis = CarbonBasis.production
    accounting_scope: str = "operational_cpu_no_pue"
    energy_weighted_intensity_gco2eq_per_kwh: float | None = Field(None, ge=0)
    emissions_gco2eq: float | None = Field(None, ge=0)
    accounted_energy_joules: float = Field(..., ge=0)
    interval_count: int = Field(..., ge=0)
    first_carbon_at: UtcDatetime | None = None
    last_carbon_at: UtcDatetime | None = None


class MeasurementQuality(_ReportModel):
    valid: bool
    final: bool
    energy_coverage_ratio: float = Field(..., ge=0, le=1)
    carbon_energy_coverage_ratio: float = Field(..., ge=0, le=1)
    prometheus_series_expected: int = Field(..., ge=0)
    prometheus_series_found: int = Field(..., ge=0)
    sample_count: int = Field(..., ge=0)
    counter_resets: int = Field(..., ge=0)
    carbon_points: int = Field(..., ge=0)
    missing_carbon_intervals: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reproducibility enrichment (schema 1.1). Every block is optional: it is filled
# on a best-effort basis and degrades to ``None`` with a warning rather than
# failing the report, so accounting never depends on it.
# --------------------------------------------------------------------------- #
class ContainerProvenance(_ReportModel):
    """Exactly what ran, as resolved by the kubelet."""

    name: str
    image: str | None = None
    #: Digest-pinned identity actually pulled (``status.containerStatuses[].imageID``).
    image_id: str | None = None
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cpu_request: str | None = None
    cpu_limit: str | None = None
    memory_request: str | None = None
    memory_limit: str | None = None


class PodProvenance(_ReportModel):
    pod_uid: str
    pod_name: str
    node_name: str | None = None
    node_selector: dict[str, str] = Field(default_factory=dict)
    containers: list[ContainerProvenance] = Field(default_factory=list)


class JobProvenance(_ReportModel):
    backoff_limit: int | None = None
    completions: int | None = None
    parallelism: int | None = None
    pods: list[PodProvenance] = Field(default_factory=list)


class NodeContext(_ReportModel):
    """Node-level energy during the window.

    Context, **not attribution**: if anything else ran on the node these numbers
    include it. Use ``energy.total_joules`` for what the Job actually caused.
    """

    node_name: str | None = None
    zone: str = "package"
    total_energy_joules: float | None = Field(None, ge=0)
    active_energy_joules: float | None = Field(None, ge=0)
    idle_energy_joules: float | None = Field(None, ge=0)
    cpu_utilization_mean: float | None = Field(None, ge=0, le=1)
    cpu_utilization_max: float | None = Field(None, ge=0, le=1)
    #: Job energy / node active energy over the window.
    job_share_of_active_energy: float | None = Field(None, ge=0)
    counter_resets: int = Field(0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class CoTenantPod(_ReportModel):
    pod_namespace: str | None = None
    pod_name: str | None = None
    energy_joules: float = Field(..., ge=0)


class NodeIsolation(_ReportModel):
    """Post-flight: was the node clean for the *whole* window?

    Stronger than a pre-flight, which only proves the node was clean at t0.
    Reconstructed from Prometheus, so it covers every interval of the run.
    """

    clean_node: bool
    co_tenant_pods: list[CoTenantPod] = Field(default_factory=list)
    co_tenant_energy_joules: float | None = Field(None, ge=0)
    #: co-tenant energy / (job + co-tenant) energy over the window.
    co_tenant_energy_share: float | None = Field(None, ge=0, le=1)
    container_restarts: int | None = Field(None, ge=0)
    kepler_up_ratio: float | None = Field(None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class WorkloadOutput(_ReportModel):
    """Container stdout, for scientific-output equality across runs.

    ``stdout_sha256`` is the generic replacement for workload-specific parsing:
    two deterministic runs with the same inputs must produce the same hash.
    """

    pod_uid: str
    pod_name: str
    stdout_sha256: str | None = None
    stdout_bytes: int = Field(0, ge=0)
    truncated: bool = False
    stdout: str | None = None
    #: Populated when stdout parses as a JSON object.
    parsed_json: dict | None = None
    warnings: list[str] = Field(default_factory=list)


class EnergyIntervalRecord(_ReportModel):
    """One scrape interval: the audit trail behind the totals."""

    start: UtcDatetime
    end: UtcDatetime
    joules: float = Field(..., ge=0)
    carbon_intensity_gco2eq_per_kwh: float | None = Field(None, ge=0)
    emissions_gco2eq: float | None = Field(None, ge=0)


class JobCarbonReport(_ReportModel):
    schema_version: str = "1.1"
    generated_at: UtcDatetime
    job: JobIdentity
    execution_started_at: UtcDatetime
    execution_finished_at: UtcDatetime
    duration_seconds: float = Field(..., ge=0)
    pod_executions: list[PodExecution]
    energy: EnergyAccounting
    carbon: CarbonAccounting
    quality: MeasurementQuality
    # Optional reproducibility blocks (schema 1.1); absent when not collected.
    provenance: JobProvenance | None = None
    node_context: NodeContext | None = None
    isolation: NodeIsolation | None = None
    workload_outputs: list[WorkloadOutput] = Field(default_factory=list)
    #: Per-scrape-interval audit trail; only when ``--include-intervals``.
    energy_intervals: list[EnergyIntervalRecord] | None = None
    formula: str = (
        "sum(delta_kepler_joules_interval / 3600000 "
        "* rte_carbon_intensity_gco2eq_per_kwh_interval)"
    )
    notes: list[str] = Field(
        default_factory=lambda: [
            "No PUE is applied.",
            "This is operational CPU energy attributed to pods by Kepler, not total node energy.",
            "All timestamps are UTC.",
            "node_context is node-level context, not attribution to this Job.",
        ]
    )


def utc_now() -> datetime:
    """Return an aware UTC timestamp without importing pandas."""
    from datetime import timezone

    return datetime.now(timezone.utc)
