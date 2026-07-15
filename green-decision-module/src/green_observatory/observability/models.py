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


class JobCarbonReport(_ReportModel):
    schema_version: str = "1.0"
    generated_at: UtcDatetime
    job: JobIdentity
    execution_started_at: UtcDatetime
    execution_finished_at: UtcDatetime
    duration_seconds: float = Field(..., ge=0)
    pod_executions: list[PodExecution]
    energy: EnergyAccounting
    carbon: CarbonAccounting
    quality: MeasurementQuality
    formula: str = (
        "sum(delta_kepler_joules_interval / 3600000 "
        "* rte_carbon_intensity_gco2eq_per_kwh_interval)"
    )
    notes: list[str] = Field(
        default_factory=lambda: [
            "No PUE is applied.",
            "This is operational CPU energy attributed to pods by Kepler, not total node energy.",
            "All timestamps are UTC.",
        ]
    )


def utc_now() -> datetime:
    """Return an aware UTC timestamp without importing pandas."""
    from datetime import timezone

    return datetime.now(timezone.utc)
