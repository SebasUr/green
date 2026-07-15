"""Typed data contracts for the Green Window Observatory (V1.0).

These ``pydantic`` v2 models are the *boundary* contracts used by the CLI, the
(future) API and the exporters. Heavy time-series work happens on ``pandas``
DataFrames inside the ``carbon`` / ``windows`` modules; these models describe
single records, forecasts, window descriptions, simulation results and data
quality reports so that everything that crosses a module boundary is typed,
validated and JSON-serializable.

Locked V1.0 conventions (see ``IMPLEMENTATION_PLAN_V1.md`` and the project
decision log):

* **Carbon ground truth** is RTE ``taux_co2`` (gCO2/kWh), a *production-based*
  intensity for the French bidding zone. It is stored raw in
  ``carbon_intensity_gco2_kwh`` and is never overwritten by a normalized score.
* **``green_score``** is normalized to ``[0, 1]`` where **higher means greener**
  (i.e. lower carbon). ``1.0`` is the cleanest and ``0.0`` the dirtiest within
  the chosen normalization reference. This resolves the contradiction between
  the two source documents in favour of the ``window_score`` convention.
* **Time** is always timezone-aware and normalized to UTC on the way in.
  Calendar features for climatology are later derived in ``Europe/Paris`` local
  time, but every stored instant stays in UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Timezone handling
# --------------------------------------------------------------------------- #


def _to_utc(value: datetime) -> datetime:
    """Require an aware datetime and normalize it to UTC.

    Naive datetimes are rejected on purpose: silently assuming a timezone is a
    classic source of off-by-one-hour bugs in carbon/grid data.
    """
    if value.tzinfo is None:
        raise ValueError(
            "datetime must be timezone-aware; attach UTC or a real offset "
            "before constructing the model (naive datetimes are rejected)."
        )
    return value.astimezone(timezone.utc)


#: A ``datetime`` that is validated to be timezone-aware and coerced to UTC.
UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class CarbonBasis(str, Enum):
    """Accounting basis for a carbon-intensity value."""

    production = "production"  # French generation mix only (RTE taux_co2)
    consumption = "consumption"  # imports/exports adjusted (Electricity Maps style)


class ModelName(str, Enum):
    """Forecasting strategies on the V1.0 baseline ladder."""

    persistence = "persistence"
    climatology = "climatology"
    corrected_climatology = "corrected_climatology"
    sarimax = "sarimax"
    lstm = "lstm"
    project_model = "project_model"
    electricity_maps = "electricity_maps"
    oracle = "oracle"


class WindowType(str, Enum):
    """Kinds of green window the observatory can emit."""

    low_carbon_window = "low_carbon_window"
    predicted_low_carbon_window = "predicted_low_carbon_window"
    low_facility_pressure_window = "low_facility_pressure_window"  # M3+
    combined_green_window = "combined_green_window"  # M5+
    oracle_window = "oracle_window"


class PolicyName(str, Enum):
    """When-to-run policies compared in the simulation layer (M6+)."""

    baseline = "baseline"
    carbon_only = "carbon_only"
    facility_only = "facility_only"
    combined_green_window = "combined_green_window"
    oracle = "oracle"


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #


class _ObsBase(BaseModel):
    """Strict base: unknown fields are rejected to catch typos early."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Electricity generation mix
# --------------------------------------------------------------------------- #


class GenerationMix(_ObsBase):
    """Instantaneous French electricity system state (MW), from eCO2mix.

    All fields are optional because near-real-time points can arrive before RTE
    consolidates them. ``physical_exchange_mw`` is the net physical exchange
    with neighbours; the sign convention (``ech_physiques``) is validated and
    documented at ingest in the ODRE provider rather than assumed here.
    """

    consumption_mw: float | None = None
    nuclear_mw: float | None = None
    gas_mw: float | None = None
    coal_mw: float | None = None
    fuel_oil_mw: float | None = None
    wind_mw: float | None = None
    solar_mw: float | None = None
    hydro_mw: float | None = None
    bioenergy_mw: float | None = None
    pumped_storage_mw: float | None = None
    physical_exchange_mw: float | None = None

    def low_carbon_generation_mw(self) -> float | None:
        """Nuclear + wind + solar + hydro (bioenergy deliberately excluded).

        Returns ``None`` if any component is missing, so callers never silently
        treat a partial sum as a complete one.
        """
        parts = [self.nuclear_mw, self.wind_mw, self.solar_mw, self.hydro_mw]
        if any(p is None for p in parts):
            return None
        return float(sum(p for p in parts))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Carbon signal and forecast
# --------------------------------------------------------------------------- #


class CarbonSignal(_ObsBase):
    """A single carbon-intensity observation for a zone at an instant.

    ``carbon_intensity_gco2_kwh`` is the physical ground truth (RTE
    ``taux_co2``). ``green_score`` is an optional derived, normalized view where
    higher is greener; it is ``None`` until a normalization reference is chosen.
    """

    timestamp: UtcDatetime
    zone: str = "FR"
    carbon_intensity_gco2_kwh: float = Field(..., ge=0)
    basis: CarbonBasis = CarbonBasis.production
    green_score: float | None = Field(None, ge=0, le=1)
    resolution_minutes: int = Field(60, gt=0)
    is_consolidated: bool = True
    source: str = "odre:eco2mix"
    mix: GenerationMix | None = None


class CarbonForecast(_ObsBase):
    """A forecast of carbon intensity for ``target_time``, made at ``issued_at``.

    ``issued_at`` is the *as-of* time: the forecast may only use information
    available up to that instant. Storing it makes every forecast auditable for
    look-ahead leakage, which is central to the evaluation protocol.
    """

    issued_at: UtcDatetime
    target_time: UtcDatetime
    horizon_hours: float = Field(..., ge=0)
    zone: str = "FR"
    model: ModelName
    basis: CarbonBasis = CarbonBasis.production

    predicted_carbon_intensity_gco2_kwh: float = Field(..., ge=0)
    predicted_green_score: float | None = Field(None, ge=0, le=1)
    lower_gco2_kwh: float | None = Field(None, ge=0)
    upper_gco2_kwh: float | None = Field(None, ge=0)
    confidence: float | None = Field(None, ge=0, le=1)


# --------------------------------------------------------------------------- #
# Facility signal (schema only in V1.0 - CDC implementation deferred to M3+)
# --------------------------------------------------------------------------- #


class FacilitySignal(_ObsBase):
    """CERN CDC facility context at an instant.

    Defined now so the combined-window and simulation schemas are stable, but
    **not populated in Milestones 0-2**. The CDC provider, cleaning rules and
    facility scoring arrive in Milestones 3-4.
    """

    timestamp: UtcDatetime
    pod_load_kw: float | None = None
    representative_temperature_c: float | None = None
    representative_humidity_pct: float | None = Field(None, ge=0, le=100)
    pue: float | None = Field(None, ge=1)
    pue_proxy: float | None = None
    cooling_pressure_proxy: float | None = None
    wue_delta: float | None = None
    facility_score: float | None = Field(None, ge=0, le=1)
    source: str = "cdc:csv"


# --------------------------------------------------------------------------- #
# Green windows
# --------------------------------------------------------------------------- #


class GreenWindow(_ObsBase):
    """A contiguous time window scored for how green it is to run.

    Score convention (locked): every ``*_score`` is a **green-score in [0, 1]
    where higher is greener**. To avoid any ambiguity with the physical signal,
    the raw mean intensity is also carried in
    ``mean_carbon_intensity_gco2_kwh``.
    """

    start: UtcDatetime
    end: UtcDatetime
    zone: str = "FR"
    window_type: WindowType

    carbon_score: float | None = Field(None, ge=0, le=1)
    facility_score: float | None = Field(None, ge=0, le=1)  # None until M4+
    combined_score: float | None = Field(None, ge=0, le=1)  # None until M5+
    mean_carbon_intensity_gco2_kwh: float | None = Field(None, ge=0)

    confidence: float | None = Field(None, ge=0, le=1)
    rank: int | None = Field(None, ge=1)
    source_model: ModelName | None = None
    issued_at: UtcDatetime | None = None
    reasons: list[str] = Field(default_factory=list)

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0


# --------------------------------------------------------------------------- #
# Simulation result (schema only in V1.0 - engine arrives in M6)
# --------------------------------------------------------------------------- #


class SimulationResult(_ObsBase):
    """Outcome of running one synthetic workload under one when-to-run policy.

    Defined now for schema stability; the replay engine is Milestone 6.
    """

    workload_id: str
    policy: PolicyName
    arrival_time: UtcDatetime
    deadline: UtcDatetime | None = None

    selected_start_time: UtcDatetime
    selected_window_type: WindowType | None = None

    carbon_score_at_start: float | None = Field(None, ge=0, le=1)
    facility_score_at_start: float | None = Field(None, ge=0, le=1)
    combined_green_score_at_start: float | None = Field(None, ge=0, le=1)
    mean_carbon_intensity_at_start_gco2_kwh: float | None = Field(None, ge=0)

    delay_minutes: float = Field(0.0, ge=0)
    deadline_violation: bool = False
    oracle_regret: float | None = None
    pct_oracle_potential_captured: float | None = None
    estimated_green_score_improvement: float | None = None
    estimated_co2_proxy_g: float | None = None


# --------------------------------------------------------------------------- #
# Data quality reporting
# --------------------------------------------------------------------------- #


class SeriesQuality(_ObsBase):
    """Per-series quality summary (mainly for CDC series in M3, generic now)."""

    name: str
    n: int = Field(..., ge=0)
    missing_fraction: float = Field(0.0, ge=0, le=1)
    zero_fraction: float = Field(0.0, ge=0, le=1)
    min: float | None = None
    max: float | None = None
    flagged: bool = False
    flags: list[str] = Field(default_factory=list)


class DataQualityReport(_ObsBase):
    """Dataset-level quality report shared by the carbon and facility layers."""

    dataset: str
    generated_at: UtcDatetime
    time_range_start: UtcDatetime | None = None
    time_range_end: UtcDatetime | None = None
    expected_resolution_minutes: int | None = Field(None, gt=0)

    n_rows: int = Field(0, ge=0)
    n_series: int | None = Field(None, ge=0)
    missing_fraction: float | None = Field(None, ge=0, le=1)
    duplicate_timestamps: int | None = Field(None, ge=0)
    gap_count: int | None = Field(None, ge=0)
    largest_gap_hours: float | None = Field(None, ge=0)
    out_of_range_count: int | None = Field(None, ge=0)

    series_reports: list[SeriesQuality] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    passed: bool = True


__all__ = [
    "UtcDatetime",
    "CarbonBasis",
    "ModelName",
    "WindowType",
    "PolicyName",
    "GenerationMix",
    "CarbonSignal",
    "CarbonForecast",
    "FacilitySignal",
    "GreenWindow",
    "SimulationResult",
    "SeriesQuality",
    "DataQualityReport",
]
