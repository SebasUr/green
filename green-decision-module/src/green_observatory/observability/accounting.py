"""Pure interval accounting for Kepler counters and RTE carbon intensity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class EnergyIncrement:
    """Energy observed between two adjacent counter samples."""

    start: float
    end: float
    joules: float


@dataclass
class CounterMeasurement:
    """Reset-aware counter delta plus the intervals used to derive it."""

    energy_joules: float
    increments: list[EnergyIncrement]
    sample_count: int
    counter_resets: int
    coverage_ratio: float
    first_sample: float | None
    last_sample: float | None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CarbonInterval:
    """One energy increment joined to the carbon intensity in force at its midpoint."""

    start: float
    end: float
    joules: float
    intensity_gco2eq_per_kwh: float | None
    emissions_gco2eq: float | None


@dataclass
class CarbonMeasurement:
    """Carbon attributed to energy increments."""

    emissions_gco2eq: float | None
    weighted_intensity_gco2eq_per_kwh: float | None
    accounted_energy_joules: float
    energy_coverage_ratio: float
    interval_count: int
    missing_intervals: int
    first_carbon_at: float | None
    last_carbon_at: float | None
    warnings: list[str] = field(default_factory=list)
    #: Per-interval audit trail behind the totals (one entry per energy increment).
    intervals: list[CarbonInterval] = field(default_factory=list)


def _finite_samples(samples: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = {
        float(timestamp): float(value)
        for timestamp, value in samples
        if math.isfinite(float(timestamp)) and math.isfinite(float(value))
    }
    return sorted(clean.items())


def measure_counter(
    samples: list[tuple[float, float]],
    start: float,
    end: float,
    *,
    boundary_tolerance_seconds: float = 90.0,
    assume_created_at_start: bool = False,
) -> CounterMeasurement:
    """Measure a reset-aware counter over ``[start, end]``.

    The preferred anchors are the last sample at/before ``start`` and the first
    sample at/after ``end``. If either is unavailable, the closest interior
    sample is used and the reduced temporal coverage is reported explicitly.
    """
    if end <= start:
        raise ValueError("end must be after start")
    clean = _finite_samples(samples)
    if len(clean) < 2:
        return CounterMeasurement(
            energy_joules=0.0,
            increments=[],
            sample_count=len(clean),
            counter_resets=0,
            coverage_ratio=0.0,
            first_sample=clean[0][0] if clean else None,
            last_sample=clean[-1][0] if clean else None,
            warnings=["fewer than two finite Kepler counter samples"],
        )

    before = [sample for sample in clean if sample[0] <= start]
    after_start = [sample for sample in clean if sample[0] > start]
    after_end = [sample for sample in clean if sample[0] >= end]
    before_end = [sample for sample in clean if sample[0] < end]
    warnings: list[str] = []

    if before and start - before[-1][0] <= boundary_tolerance_seconds:
        first = before[-1]
    elif after_start and after_start[0][0] - start <= boundary_tolerance_seconds:
        if assume_created_at_start:
            first = (start, 0.0)
            clean.append(first)
            clean.sort()
            warnings.append("counter assumed to start at zero when the pod was created")
        else:
            first = after_start[0]
            warnings.append("no Kepler sample at/before pod start; initial energy is uncovered")
    else:
        first = min(clean, key=lambda item: abs(item[0] - start))
        warnings.append("Kepler start boundary exceeds tolerance")

    if after_end and after_end[0][0] - end <= boundary_tolerance_seconds:
        last = after_end[0]
    elif before_end and end - before_end[-1][0] <= boundary_tolerance_seconds:
        last = before_end[-1]
        warnings.append("no Kepler sample at/after pod finish; tail energy is uncovered")
    else:
        last = min(clean, key=lambda item: abs(item[0] - end))
        warnings.append("Kepler finish boundary exceeds tolerance")

    if last[0] <= first[0]:
        return CounterMeasurement(
            energy_joules=0.0,
            increments=[],
            sample_count=0,
            counter_resets=0,
            coverage_ratio=0.0,
            first_sample=first[0],
            last_sample=last[0],
            warnings=[*warnings, "Kepler sample window does not overlap pod execution"],
        )

    selected = [sample for sample in clean if first[0] <= sample[0] <= last[0]]
    increments: list[EnergyIncrement] = []
    resets = 0
    for previous, current in zip(selected, selected[1:]):
        if current[1] >= previous[1]:
            delta = current[1] - previous[1]
        else:
            delta = max(0.0, current[1])
            resets += 1
        overlap_start = max(start, previous[0])
        overlap_end = min(end, current[0])
        if overlap_end <= overlap_start:
            continue
        # Counters are observed only at scrape instants. Linear interpolation
        # avoids charging the Job for the complete boundary scrape intervals.
        interval_seconds = current[0] - previous[0]
        fraction = (overlap_end - overlap_start) / interval_seconds
        increments.append(EnergyIncrement(overlap_start, overlap_end, delta * fraction))

    covered_start = max(start, first[0])
    covered_end = min(end, last[0])
    coverage = max(0.0, covered_end - covered_start) / (end - start)
    return CounterMeasurement(
        energy_joules=sum(item.joules for item in increments),
        increments=increments,
        sample_count=len(selected),
        counter_resets=resets,
        coverage_ratio=min(1.0, coverage),
        first_sample=selected[0][0],
        last_sample=selected[-1][0],
        warnings=warnings,
    )


def _carbon_points(carbon: pd.Series) -> list[tuple[float, float]]:
    if carbon.empty:
        return []
    index = pd.to_datetime(carbon.index, utc=True, errors="coerce")
    points: list[tuple[float, float]] = []
    for timestamp, value in zip(index, carbon.to_numpy()):
        if pd.isna(timestamp) or pd.isna(value):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric >= 0:
            points.append((timestamp.timestamp(), numeric))
    return sorted(dict(points).items())


def account_carbon(
    increments: list[EnergyIncrement],
    carbon: pd.Series,
    *,
    max_carbon_age_seconds: float = 2100.0,
) -> CarbonMeasurement:
    """Join energy increments to the latest valid RTE point at each midpoint."""
    points = _carbon_points(carbon)
    total_energy = sum(item.joules for item in increments)
    if not points:
        return CarbonMeasurement(
            emissions_gco2eq=None,
            weighted_intensity_gco2eq_per_kwh=None,
            accounted_energy_joules=0.0,
            energy_coverage_ratio=0.0,
            interval_count=0,
            missing_intervals=len(increments),
            first_carbon_at=None,
            last_carbon_at=None,
            warnings=["no RTE carbon-intensity points cover the execution"],
            intervals=[
                CarbonInterval(item.start, item.end, item.joules, None, None)
                for item in increments
            ],
        )

    point_index = 0
    emissions = 0.0
    accounted_energy = 0.0
    accounted_intervals = 0
    missing = 0
    used_timestamps: list[float] = []
    detail: list[CarbonInterval] = []
    for increment in increments:
        midpoint = increment.start + (increment.end - increment.start) / 2
        while point_index + 1 < len(points) and points[point_index + 1][0] <= midpoint:
            point_index += 1
        carbon_at, intensity = points[point_index]
        if carbon_at > midpoint or midpoint - carbon_at > max_carbon_age_seconds:
            missing += 1
            detail.append(
                CarbonInterval(increment.start, increment.end, increment.joules, None, None)
            )
            continue
        interval_emissions = increment.joules / 3_600_000 * intensity
        emissions += interval_emissions
        accounted_energy += increment.joules
        accounted_intervals += 1
        used_timestamps.append(carbon_at)
        detail.append(
            CarbonInterval(
                increment.start, increment.end, increment.joules, intensity, interval_emissions
            )
        )

    coverage = 1.0 if total_energy == 0 and not missing else (
        accounted_energy / total_energy if total_energy > 0 else 0.0
    )
    weighted = emissions * 3_600_000 / accounted_energy if accounted_energy > 0 else None
    warnings = []
    if missing:
        warnings.append(f"{missing} energy intervals have no sufficiently recent RTE value")
    return CarbonMeasurement(
        emissions_gco2eq=emissions if accounted_energy > 0 else None,
        weighted_intensity_gco2eq_per_kwh=weighted,
        accounted_energy_joules=accounted_energy,
        energy_coverage_ratio=min(1.0, max(0.0, coverage)),
        interval_count=accounted_intervals,
        missing_intervals=missing,
        first_carbon_at=min(used_timestamps) if used_timestamps else None,
        last_carbon_at=max(used_timestamps) if used_timestamps else None,
        warnings=warnings,
        intervals=detail,
    )
