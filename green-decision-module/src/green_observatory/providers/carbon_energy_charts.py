"""Energy-Charts realised French generation used to bridge an RTE cache gap.

This provider parses the public ``/public_power`` response.  It is intentionally
kept separate from :mod:`carbon_odre`: Energy-Charts and RTE do not guarantee
the same publication vintage or revision policy.  A snapshot downloaded after
the evaluation period is timestamp-causal but **not vintage-causal**.  The
resulting frame is therefore suitable only for an explicitly enabled
retrospective state/lag experiment; it must not silently become an RTE carbon
label or a prospectively clean holdout feature.

Energy-Charts publishes quarter-hourly average power.  Timestamps are UTC
hour-start labels, so four quarters are averaged into the matching hourly bin.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.providers.carbon_base import (
    CARBON,
    TIMESTAMP,
    ensure_canonical,
    to_utc_timestamp,
)


SERIES_MAP: dict[str, str] = {
    "Load": "consumption_mw",
    "Nuclear": "nuclear_mw",
    "Fossil gas": "gas_mw",
    "Fossil hard coal": "coal_mw",
    "Fossil brown coal / lignite": "coal_mw",
    "Fossil oil": "fuel_oil_mw",
    "Wind onshore": "wind_onshore_mw",
    "Wind offshore": "wind_offshore_mw",
    "Solar": "solar_mw",
    "Hydro Run-of-River": "hydro_run_of_river_mw",
    "Hydro water reservoir": "hydro_reservoir_mw",
    "Hydro pumped storage": "hydro_pumped_turbining_mw",
    "Hydro pumped storage consumption": "pumped_storage_mw",
    "Biomass": "bioenergy_biomass_mw",
    "Waste": "bioenergy_waste_mw",
}

# Deliberately absent from ``SERIES_MAP``: ``Cross border electricity
# trading`` is a commercial schedule, not RTE ``ech_physiques``.  The
# canonical ``physical_exchange_mw`` therefore remains NaN in this bridge.

REQUIRED_SERIES = (
    "Load",
    "Nuclear",
    "Fossil gas",
    "Fossil oil",
    "Wind onshore",
    "Wind offshore",
    "Solar",
    "Hydro Run-of-River",
    "Hydro water reservoir",
    "Hydro pumped storage",
    "Biomass",
    "Waste",
)


def _utc_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(index)
    return out.tz_localize("UTC") if out.tz is None else out.tz_convert("UTC")


class EnergyChartsCarbonGapProvider:
    """Pure parser for realised French Energy-Charts production snapshots."""

    zone = "FR"

    @staticmethod
    def parse_quarter_hourly(payload: dict) -> pd.DataFrame:
        """Return source series at their original 15-minute UTC cadence."""

        timestamps = payload.get("unix_seconds", [])
        production_types = payload.get("production_types", [])
        if not isinstance(timestamps, list) or not isinstance(production_types, list):
            raise ValueError("invalid Energy-Charts public_power payload")
        index = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
        if index.isna().any():
            raise ValueError("Energy-Charts payload contains invalid timestamps")

        source: dict[str, np.ndarray] = {}
        for item in production_types:
            name = item.get("name")
            values = item.get("data", [])
            if len(values) != len(index):
                raise ValueError(
                    f"Energy-Charts series {name!r} has {len(values)} values "
                    f"for {len(index)} timestamps"
                )
            if name in SERIES_MAP:
                source[name] = pd.to_numeric(
                    pd.Series(values), errors="coerce"
                ).to_numpy(dtype=float)

        missing = sorted(set(REQUIRED_SERIES).difference(source))
        if missing:
            raise ValueError(f"Energy-Charts payload is missing series: {missing}")

        columns: dict[str, np.ndarray] = {}
        for source_name, canonical_name in SERIES_MAP.items():
            if source_name not in source:
                continue
            # France currently has no coal series in this endpoint.  If both
            # hard coal and lignite ever appear, add them into the aggregate.
            if canonical_name in columns:
                columns[canonical_name] = columns[canonical_name] + source[source_name]
            else:
                columns[canonical_name] = source[source_name]
        out = pd.DataFrame(columns, index=index).sort_index()
        out.index.name = TIMESTAMP
        if out.index.has_duplicates:
            out = out.groupby(level=0).last()
        return out

    @classmethod
    def parse_payloads(
        cls,
        payloads: Sequence[dict],
        *,
        biogas_history: pd.Series | pd.DataFrame,
        biogas_lookback_days: int = 28,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Parse, hourly-average and standardise one or more API payloads.

        ``biogas_history`` must be observations published strictly before the
        first Energy-Charts quarter.  A trailing median is frozen at that
        cutoff and used throughout the gap.  This deliberately simple
        imputation is causal and reflects that Energy-Charts exposes Biomass
        and Waste, but not RTE's separate Biogas category.
        """

        if not payloads:
            return ensure_canonical(pd.DataFrame(), require_carbon=False)
        quarters = pd.concat(
            [cls.parse_quarter_hourly(payload) for payload in payloads]
        ).sort_index()
        quarters = quarters.loc[~quarters.index.duplicated(keep="last")]
        if len(quarters) > 1:
            delta = quarters.index.to_series().diff().dropna()
            if not delta.eq(pd.Timedelta(minutes=15)).all():
                bad = delta[delta.ne(pd.Timedelta(minutes=15))].index[0]
                raise ValueError(f"Energy-Charts quarter-hour gap before {bad}")

        first = quarters.index.min()
        history = (
            biogas_history.iloc[:, 0]
            if isinstance(biogas_history, pd.DataFrame)
            else biogas_history
        ).copy()
        history.index = _utc_index(history.index)
        history = pd.to_numeric(history, errors="coerce").sort_index()
        causal = history.loc[history.index < first].dropna()
        if causal.empty:
            raise ValueError("biogas imputation requires observations before the gap")
        cutoff = first - pd.Timedelta(days=int(biogas_lookback_days))
        recent = causal.loc[causal.index >= cutoff]
        if recent.empty:
            recent = causal
        biogas_level = float(recent.median())

        counts = quarters.resample("1h").size()
        hourly = quarters.resample("1h").mean()
        # Do not manufacture partially observed hours: their average would not
        # be comparable with RTE's completed hourly bins.
        hourly.loc[counts < 4, :] = np.nan
        hourly["coal_mw"] = 0.0
        hourly["wind_mw"] = hourly[
            ["wind_onshore_mw", "wind_offshore_mw"]
        ].sum(axis=1, min_count=2)
        hourly["hydro_mw"] = hourly[
            [
                "hydro_run_of_river_mw",
                "hydro_reservoir_mw",
                "hydro_pumped_turbining_mw",
            ]
        ].sum(axis=1, min_count=3)
        hourly["bioenergy_biogas_mw"] = biogas_level
        hourly["bioenergy_mw"] = hourly[
            ["bioenergy_biomass_mw", "bioenergy_waste_mw", "bioenergy_biogas_mw"]
        ].sum(axis=1, min_count=3)
        # This cache carries physical observations only.  Carbon is derived by
        # the operational proxy layer, never presented as an RTE observation.
        hourly[CARBON] = np.nan

        if start is not None:
            hourly = hourly.loc[hourly.index >= to_utc_timestamp(start)]
        if end is not None:
            hourly = hourly.loc[hourly.index < to_utc_timestamp(end)]
        return ensure_canonical(hourly, require_carbon=False)

    @classmethod
    def parse_files(
        cls,
        paths: Sequence[str | Path],
        **kwargs,
    ) -> pd.DataFrame:
        payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
        return cls.parse_payloads(payloads, **kwargs)

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ensure_canonical(frame, require_carbon=False).to_parquet(path)

    @staticmethod
    def load_snapshot(path: str | Path) -> pd.DataFrame:
        return ensure_canonical(pd.read_parquet(path), require_carbon=False)


__all__ = [
    "EnergyChartsCarbonGapProvider",
    "REQUIRED_SERIES",
    "SERIES_MAP",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated hourly Energy-Charts physical-state bridge"
    )
    parser.add_argument("--json", nargs="+", required=True, dest="json_paths")
    parser.add_argument(
        "--biogas-history",
        default="data/cache/carbon_fr_hourly_detailed.parquet",
        help="Earlier RTE frame used only for the causal biogas level",
    )
    parser.add_argument(
        "--output",
        default="data/cache/carbon_fr_hourly_energy_charts_gap_2026.parquet",
    )
    parser.add_argument("--biogas-lookback-days", type=int, default=28)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    history = pd.read_parquet(args.biogas_history)
    if "bioenergy_biogas_mw" not in history:
        raise ValueError("biogas history has no bioenergy_biogas_mw column")
    frame = EnergyChartsCarbonGapProvider.parse_files(
        args.json_paths,
        biogas_history=history["bioenergy_biogas_mw"],
        biogas_lookback_days=args.biogas_lookback_days,
        start=args.start,
        end=args.end,
    )
    EnergyChartsCarbonGapProvider.save_snapshot(frame, args.output)
    print(
        f"saved {args.output}: {len(frame)} hourly rows, "
        f"{frame.index.min()}..{frame.index.max()}"
    )


if __name__ == "__main__":
    _main()
