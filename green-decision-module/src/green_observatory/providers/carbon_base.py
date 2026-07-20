"""Canonical carbon-frame schema and the provider interface.

A *carbon frame* is a tidy ``pandas`` DataFrame indexed by a timezone-aware UTC
``DatetimeIndex`` named ``timestamp``, with at least the column
``carbon_intensity_gco2_kwh`` and, optionally, the electricity generation mix.
Every provider returns frames in this canonical shape so the rest of the carbon
track (features, climatology, model, windows) stays provider-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

#: Index name for every carbon frame.
TIMESTAMP = "timestamp"

#: Ground-truth carbon-intensity column (RTE ``taux_co2``, gCO2/kWh).
CARBON = "carbon_intensity_gco2_kwh"

#: Electricity generation-mix columns (MW), in canonical order.
MIX_COLUMNS: list[str] = [
    "consumption_mw",
    "nuclear_mw",
    "gas_mw",
    "coal_mw",
    "fuel_oil_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "bioenergy_mw",
    "pumped_storage_mw",
    "physical_exchange_mw",
    # Optional eCO2mix detail.  These columns are not populated over the whole
    # history (for example onshore/offshore wind and batteries are recent), so
    # consumers must keep treating them as nullable features.
    "wind_onshore_mw",
    "wind_offshore_mw",
    "fuel_oil_turbine_mw",
    "fuel_oil_cogeneration_mw",
    "fuel_oil_other_mw",
    "gas_turbine_mw",
    "gas_cogeneration_mw",
    "gas_ccg_mw",
    "gas_other_mw",
    "hydro_run_of_river_mw",
    "hydro_reservoir_mw",
    "hydro_pumped_turbining_mw",
    "bioenergy_waste_mw",
    "bioenergy_biomass_mw",
    "bioenergy_biogas_mw",
    "commercial_exchange_gb_mw",
    "commercial_exchange_es_mw",
    "commercial_exchange_it_mw",
    "commercial_exchange_ch_mw",
    "commercial_exchange_de_be_mw",
    "battery_charging_mw",
    "battery_discharging_mw",
]

#: Full canonical column set (carbon first, then the mix).
CANONICAL_COLUMNS: list[str] = [CARBON, *MIX_COLUMNS]


@runtime_checkable
class CarbonProvider(Protocol):
    """Minimal interface every carbon provider satisfies."""

    zone: str

    def load_hourly(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return an hourly canonical carbon frame for ``[start, end)``."""
        ...


def to_utc_timestamp(value) -> pd.Timestamp:
    """Coerce a datetime-like to a tz-aware UTC ``pd.Timestamp``.

    Naive inputs are assumed to be UTC only as an explicit last resort; callers
    working with real grid data should always pass aware values.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def ensure_canonical(df: pd.DataFrame, *, require_carbon: bool = True) -> pd.DataFrame:
    """Validate and normalize a carbon frame in place-safe fashion.

    Guarantees on the returned frame:

    * index is a tz-aware UTC ``DatetimeIndex`` named ``timestamp``;
    * index is sorted ascending and free of duplicate timestamps (last wins);
    * all canonical columns exist (missing ones filled with ``NaN``) and are
      numeric, in canonical order.
    """
    out = df.copy()

    # Accept a 'timestamp' column instead of an index.
    if TIMESTAMP in out.columns:
        out = out.set_index(TIMESTAMP)

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = TIMESTAMP

    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    out = out.reindex(columns=CANONICAL_COLUMNS)
    for col in CANONICAL_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if require_carbon and out[CARBON].notna().sum() == 0:
        raise ValueError(
            f"carbon frame has no non-null '{CARBON}' values after normalization"
        )
    return out
