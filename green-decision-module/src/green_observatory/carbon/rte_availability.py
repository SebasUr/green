"""Leakage-safe RTE generation-unavailability features.

RTE publishes versioned messages.  For every forecast origin this store applies
only messages whose ``publication_date <= origin``, retains the latest version
of each event identifier, and evaluates its capacity intervals at the target
hours.  Later corrections and cancellations therefore cannot leak backwards.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FUEL_GROUPS = {
    "NUCLEAR": "nuclear",
    "FOSSIL_GAS": "gas",
    "FOSSIL_HARD_COAL": "coal",
    "FOSSIL_OIL": "oil",
    "ENERGY_STORAGE": "storage",
}
FEATURE_SOURCES = ("nuclear", "gas", "coal", "oil", "hydro", "storage")


@dataclass(frozen=True)
class _Message:
    identifier: str
    message_id: str
    version: int
    publication_ns: int
    status: str
    unavailability_type: str
    fuel_group: str | None
    interval_start_ns: np.ndarray
    interval_end_ns: np.ndarray
    unavailable_mw: np.ndarray
    max_end_ns: int


def _fuel_group(value: Any) -> str | None:
    fuel = str(value or "").upper()
    if fuel.startswith("HYDRO_"):
        return "hydro"
    return FUEL_GROUPS.get(fuel)


class RteAvailabilityFeatureStore:
    """Replay versioned RTE outage messages as origin-safe target features."""

    def __init__(self, intervals: pd.DataFrame) -> None:
        required = {
            "identifier",
            "message_id",
            "version",
            "publication_date",
            "event_status",
            "unavailability_type",
            "fuel_type",
            "interval_start",
            "interval_end",
            "unavailable_capacity_mw",
        }
        missing = sorted(required - set(intervals.columns))
        if missing:
            raise ValueError(f"RTE unavailability snapshot is missing: {missing}")
        frame = intervals.copy()
        for column in ("publication_date", "interval_start", "interval_end"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        frame["unavailable_capacity_mw"] = pd.to_numeric(
            frame["unavailable_capacity_mw"], errors="coerce"
        )
        frame = frame.dropna(
            subset=[
                "identifier",
                "message_id",
                "publication_date",
                "interval_start",
                "interval_end",
                "unavailable_capacity_mw",
            ]
        )
        messages: list[_Message] = []
        for message_id, group in frame.groupby("message_id", sort=False):
            first = group.iloc[0]
            # ``DatetimeIndex.asi8`` is always nanoseconds.  Plain
            # ``astype(int64)`` can retain a microsecond dtype under pandas 3,
            # which would make interval comparisons silently fail.
            starts = pd.DatetimeIndex(group["interval_start"]).as_unit("ns").asi8
            ends = pd.DatetimeIndex(group["interval_end"]).as_unit("ns").asi8
            messages.append(
                _Message(
                    identifier=str(first["identifier"]),
                    message_id=str(message_id),
                    version=int(first["version"]),
                    publication_ns=int(first["publication_date"].value),
                    status=str(first["event_status"]).upper(),
                    unavailability_type=str(first["unavailability_type"]).upper(),
                    fuel_group=_fuel_group(first["fuel_type"]),
                    interval_start_ns=starts,
                    interval_end_ns=ends,
                    unavailable_mw=group["unavailable_capacity_mw"].to_numpy(
                        dtype=float
                    ),
                    max_end_ns=int(ends.max()),
                )
            )
        self.messages_ = sorted(
            messages,
            key=lambda message: (
                message.publication_ns,
                message.identifier,
                message.version,
            ),
        )
        self.publication_start_ = (
            pd.Timestamp(self.messages_[0].publication_ns, tz="UTC")
            if self.messages_
            else None
        )
        self.publication_end_ = (
            pd.Timestamp(self.messages_[-1].publication_ns, tz="UTC")
            if self.messages_
            else None
        )

    @classmethod
    def from_parquet(cls, path) -> RteAvailabilityFeatureStore:
        return cls(pd.read_parquet(path))

    @staticmethod
    def _empty_arrays(n_origins: int, n_horizons: int) -> dict[str, np.ndarray]:
        arrays = {
            f"rte_tgt_{source}_unavailable_mw": np.zeros(
                (n_origins, n_horizons), dtype=float
            )
            for source in FEATURE_SOURCES
        }
        arrays["rte_tgt_total_unavailable_mw"] = np.zeros(
            (n_origins, n_horizons), dtype=float
        )
        arrays["rte_tgt_nuclear_planned_unavailable_mw"] = np.zeros(
            (n_origins, n_horizons), dtype=float
        )
        arrays["rte_tgt_nuclear_unplanned_unavailable_mw"] = np.zeros(
            (n_origins, n_horizons), dtype=float
        )
        arrays["rte_tgt_nuclear_outage_count"] = np.zeros(
            (n_origins, n_horizons), dtype=float
        )
        return arrays

    def features_by_horizon(
        self,
        origins: pd.DatetimeIndex,
        horizons: tuple[int, ...] | list[int],
    ) -> dict[int, pd.DataFrame]:
        origins = pd.DatetimeIndex(origins)
        if origins.tz is None:
            origins = origins.tz_localize("UTC")
        else:
            origins = origins.tz_convert("UTC")
        horizons = tuple(int(horizon) for horizon in horizons)
        n_origins, n_horizons = len(origins), len(horizons)
        arrays = self._empty_arrays(n_origins, n_horizons)
        origin_unavailable = {
            source: np.zeros(n_origins, dtype=float) for source in FEATURE_SOURCES
        }
        if not self.messages_ or not len(origins):
            return self._frames(origins, horizons, arrays)

        origin_ns_values = origins.as_unit("ns").asi8
        order = np.argsort(origin_ns_values)
        latest: dict[str, _Message] = {}
        expiry_heap: list[tuple[int, str, str]] = []
        message_position = 0
        horizon_ns = np.asarray(horizons, dtype=np.int64) * 3_600_000_000_000

        for origin_position in order:
            origin_ns = int(origin_ns_values[origin_position])
            while (
                message_position < len(self.messages_)
                and self.messages_[message_position].publication_ns <= origin_ns
            ):
                message = self.messages_[message_position]
                current = latest.get(message.identifier)
                if current is None or (
                    message.publication_ns,
                    message.version,
                ) >= (current.publication_ns, current.version):
                    latest[message.identifier] = message
                    heapq.heappush(
                        expiry_heap,
                        (message.max_end_ns, message.identifier, message.message_id),
                    )
                message_position += 1

            minimum_target_ns = origin_ns + int(horizon_ns.min())
            while expiry_heap and expiry_heap[0][0] <= minimum_target_ns:
                _, identifier, message_id = heapq.heappop(expiry_heap)
                current = latest.get(identifier)
                if current is not None and current.message_id == message_id:
                    latest.pop(identifier, None)

            targets = origin_ns + horizon_ns
            for message in latest.values():
                if message.status != "ACTIVE" or message.fuel_group is None:
                    continue
                source_values = arrays[
                    f"rte_tgt_{message.fuel_group}_unavailable_mw"
                ]
                for start_ns, end_ns, unavailable_mw in zip(
                    message.interval_start_ns,
                    message.interval_end_ns,
                    message.unavailable_mw,
                ):
                    if start_ns <= origin_ns < end_ns:
                        origin_unavailable[message.fuel_group][origin_position] += (
                            unavailable_mw
                        )
                    mask = (targets >= start_ns) & (targets < end_ns)
                    if not mask.any():
                        continue
                    source_values[origin_position, mask] += unavailable_mw
                    arrays["rte_tgt_total_unavailable_mw"][origin_position, mask] += (
                        unavailable_mw
                    )
                    if message.fuel_group == "nuclear":
                        arrays["rte_tgt_nuclear_outage_count"][origin_position, mask] += 1
                        type_key = (
                            "rte_tgt_nuclear_planned_unavailable_mw"
                            if "PLANNED" in message.unavailability_type
                            else "rte_tgt_nuclear_unplanned_unavailable_mw"
                        )
                        arrays[type_key][origin_position, mask] += unavailable_mw

        for source in FEATURE_SOURCES:
            unavailable = arrays[f"rte_tgt_{source}_unavailable_mw"]
            arrays[f"rte_tgt_{source}_unavailable_delta_mw"] = (
                unavailable - origin_unavailable[source][:, None]
            )
        return self._frames(origins, horizons, arrays)

    @staticmethod
    def _frames(
        origins: pd.DatetimeIndex,
        horizons: tuple[int, ...],
        arrays: dict[str, np.ndarray],
    ) -> dict[int, pd.DataFrame]:
        return {
            horizon: pd.DataFrame(
                {name: values[:, position] for name, values in arrays.items()},
                index=origins,
            )
            for position, horizon in enumerate(horizons)
        }


__all__ = ["RteAvailabilityFeatureStore"]
