"""ODRE / eCO2mix carbon provider (primary V1.0 source).

Ground truth is RTE ``taux_co2`` (production-based gCO2/kWh), published on the
Opendatasoft **ODRE** portal. Two datasets are used:

* ``eco2mix-national-cons-def`` - consolidated + definitive, long history
  (``taux_co2`` populated from 2011-12-31 to the last consolidated day).
* ``eco2mix-national-tr`` - near-real-time, rolling ~weeks window.

Field notes verified against the live API (2026-07):

* ``date_heure`` is genuine UTC (ISO-8601 with ``+00:00``); solar peaks near
  local noon, confirming the instants are correct.
* ``taux_co2`` is published at **30-min cadence** (``:00`` and ``:30``); the
  ``:15``/``:45`` 15-min slots and the not-yet-consolidated tail arrive as
  ``null`` and are dropped.
* ``ech_physiques`` is the net physical exchange with neighbours; **negative =
  export** from France, positive = import (RTE convention).

Design: network I/O (``_fetch_export``) is isolated from the pure parsing /
standardization / resampling helpers so the latter are unit-tested on fixtures
without hitting the network.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
import pandas as pd

from green_observatory.providers.carbon_base import (
    CANONICAL_COLUMNS,
    CARBON,
    TIMESTAMP,
    ensure_canonical,
    to_utc_timestamp,
)

DEFAULT_BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1"
DEFAULT_HISTORY_DATASET = "eco2mix-national-cons-def"
DEFAULT_REALTIME_DATASET = "eco2mix-national-tr"

#: ODRE eCO2mix field -> canonical carbon-frame column.
FIELD_MAP: dict[str, str] = {
    "taux_co2": CARBON,
    "consommation": "consumption_mw",
    "nucleaire": "nuclear_mw",
    "gaz": "gas_mw",
    "charbon": "coal_mw",
    "fioul": "fuel_oil_mw",
    "eolien": "wind_mw",
    "solaire": "solar_mw",
    "hydraulique": "hydro_mw",
    "bioenergies": "bioenergy_mw",
    "pompage": "pumped_storage_mw",
    "ech_physiques": "physical_exchange_mw",
    "eolien_terrestre": "wind_onshore_mw",
    "eolien_offshore": "wind_offshore_mw",
    "fioul_tac": "fuel_oil_turbine_mw",
    "fioul_cogen": "fuel_oil_cogeneration_mw",
    "fioul_autres": "fuel_oil_other_mw",
    "gaz_tac": "gas_turbine_mw",
    "gaz_cogen": "gas_cogeneration_mw",
    "gaz_ccg": "gas_ccg_mw",
    "gaz_autres": "gas_other_mw",
    "hydraulique_fil_eau_eclusee": "hydro_run_of_river_mw",
    "hydraulique_lacs": "hydro_reservoir_mw",
    "hydraulique_step_turbinage": "hydro_pumped_turbining_mw",
    "bioenergies_dechets": "bioenergy_waste_mw",
    "bioenergies_biomasse": "bioenergy_biomass_mw",
    "bioenergies_biogaz": "bioenergy_biogas_mw",
    "ech_comm_angleterre": "commercial_exchange_gb_mw",
    "ech_comm_espagne": "commercial_exchange_es_mw",
    "ech_comm_italie": "commercial_exchange_it_mw",
    "ech_comm_suisse": "commercial_exchange_ch_mw",
    "ech_comm_allemagne_belgique": "commercial_exchange_de_be_mw",
    "stockage_batterie": "battery_charging_mw",
    "destockage_batterie": "battery_discharging_mw",
}
TIMESTAMP_FIELD = "date_heure"

# These fields currently exist only in the rolling real-time dataset.  Keeping
# them in the canonical schema is useful for live inference, but asking the
# consolidated export endpoint to select an unknown field makes the whole
# request fail.
REALTIME_ONLY_FIELDS = {
    "eolien_terrestre",
    "eolien_offshore",
    "stockage_batterie",
    "destockage_batterie",
}

#: Earliest instant with a populated ``taux_co2`` in the consolidated dataset.
EARLIEST_TAUX_CO2 = pd.Timestamp("2011-12-31T23:00:00Z")


def _year_bounds(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield ``[a, b)`` chunks split on calendar-year boundaries (UTC)."""
    cur = start
    while cur < end:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC"), end)
        yield cur, nxt
        cur = nxt


class OdreCarbonProvider:
    """Import + replay provider for ODRE eCO2mix carbon data."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        history_dataset: str = DEFAULT_HISTORY_DATASET,
        realtime_dataset: str = DEFAULT_REALTIME_DATASET,
        zone: str = "FR",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.history_dataset = history_dataset
        self.realtime_dataset = realtime_dataset
        self.zone = zone
        self.timeout = timeout
        self.max_retries = max_retries

    @classmethod
    def from_config(cls, cfg: dict) -> OdreCarbonProvider:
        prov = (cfg or {}).get("provider", {})
        return cls(
            base_url=prov.get("base_url", DEFAULT_BASE_URL),
            history_dataset=prov.get("history_dataset", DEFAULT_HISTORY_DATASET),
            realtime_dataset=prov.get("realtime_dataset", DEFAULT_REALTIME_DATASET),
            zone=cfg.get("zone", "FR"),
        )

    # ------------------------------------------------------------------ #
    # Pure parsing / standardization (no network - unit tested)
    # ------------------------------------------------------------------ #
    @staticmethod
    def standardize(raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw ODRE records to a canonical (sub-hourly) carbon frame."""
        if raw.empty:
            return ensure_canonical(
                pd.DataFrame(columns=[TIMESTAMP, *CANONICAL_COLUMNS]),
                require_carbon=False,
            )

        df = raw.rename(columns={**FIELD_MAP, TIMESTAMP_FIELD: TIMESTAMP})
        if TIMESTAMP not in df.columns:
            raise ValueError(
                f"raw ODRE frame is missing the timestamp field '{TIMESTAMP_FIELD}'"
            )
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True, errors="coerce")
        df = df.dropna(subset=[TIMESTAMP])
        return ensure_canonical(df, require_carbon=False)

    @staticmethod
    def parse_records(records: list[dict]) -> pd.DataFrame:
        """Standardize a list of raw ODRE record dicts."""
        return OdreCarbonProvider.standardize(pd.DataFrame.from_records(records))

    @staticmethod
    def to_hourly(df: pd.DataFrame, *, aggregation: str = "mean") -> pd.DataFrame:
        """Resample a sub-hourly canonical frame to hourly (UTC hour-beginning).

        ``taux_co2``'s two half-hourly points per hour are averaged; mix columns
        (instantaneous MW ~ average power) are averaged too, so the hourly MW is
        the mean power over the hour.
        """
        if df.empty:
            return df
        resampler = df.resample("1h")
        hourly = getattr(resampler, aggregation)()
        return ensure_canonical(hourly, require_carbon=False)

    # ------------------------------------------------------------------ #
    # Network I/O
    # ------------------------------------------------------------------ #
    def _fetch_export(
        self,
        dataset: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        only_populated: bool = True,
        client: httpx.Client | None = None,
    ) -> list[dict]:
        """Fetch all records in ``[start, end)`` via the export/json endpoint."""
        fields = list(FIELD_MAP)
        if dataset == self.history_dataset:
            fields = [field for field in fields if field not in REALTIME_ONLY_FIELDS]
        select = ",".join([TIMESTAMP_FIELD, *fields])
        s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        e = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        where = f"{TIMESTAMP_FIELD} >= '{s}' AND {TIMESTAMP_FIELD} < '{e}'"
        if only_populated:
            where = f"taux_co2 IS NOT NULL AND {where}"
        url = f"{self.base_url}/catalog/datasets/{dataset}/exports/json"
        params = {"select": select, "where": where, "timezone": "UTC"}

        owns_client = client is None
        client = client or httpx.Client(timeout=self.timeout)
        try:
            last_exc: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = client.get(url, params=params)
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.HTTPError, ValueError) as exc:  # network or JSON
                    last_exc = exc
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** attempt, 10))
            raise RuntimeError(
                f"ODRE export failed for {dataset} [{s}, {e}) after "
                f"{self.max_retries} attempts: {last_exc}"
            )
        finally:
            if owns_client:
                client.close()

    # ------------------------------------------------------------------ #
    # High-level import
    # ------------------------------------------------------------------ #
    def import_history(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        hourly: bool = True,
        drop_null_target: bool = True,
        client: httpx.Client | None = None,
        progress: bool = False,
    ) -> pd.DataFrame:
        """Import consolidated history for ``[start, end)`` as a canonical frame.

        Fetched year-by-year for resilience; sub-hourly rows without a populated
        ``taux_co2`` are dropped before optional hourly resampling.
        """
        start = max(to_utc_timestamp(start), EARLIEST_TAUX_CO2)
        end = to_utc_timestamp(end)
        if end <= start:
            raise ValueError(f"end ({end}) must be after start ({start})")

        owns_client = client is None
        client = client or httpx.Client(timeout=self.timeout)
        frames: list[pd.DataFrame] = []
        try:
            for a, b in _year_bounds(start, end):
                records = self._fetch_export(
                    self.history_dataset, a, b, only_populated=drop_null_target, client=client
                )
                if records:
                    frames.append(self.standardize(pd.DataFrame.from_records(records)))
                if progress:
                    print(f"  fetched {a.date()}..{b.date()}: {len(records)} rows")
        finally:
            if owns_client:
                client.close()

        if not frames:
            return ensure_canonical(
                pd.DataFrame(columns=[TIMESTAMP, *CANONICAL_COLUMNS]), require_carbon=False
            )

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if drop_null_target:
            df = df[df[CARBON].notna()]
        if hourly:
            df = self.to_hourly(df)
            if drop_null_target:
                df = df[df[CARBON].notna()]
        return ensure_canonical(df, require_carbon=drop_null_target)

    def import_realtime(
        self,
        *,
        hourly: bool = True,
        client: httpx.Client | None = None,
    ) -> pd.DataFrame:
        """Import the near-real-time rolling window (last populated points)."""
        end = pd.Timestamp.now(tz="UTC").ceil("h")
        start = end - pd.Timedelta(days=30)
        records = self._fetch_export(
            self.realtime_dataset, start, end, only_populated=True, client=client
        )
        df = self.standardize(pd.DataFrame.from_records(records))
        df = df[df[CARBON].notna()]
        if hourly:
            df = self.to_hourly(df)
            df = df[df[CARBON].notna()]
        return ensure_canonical(df, require_carbon=False)

    def import_consumption_forecast(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        hourly: bool = True,
        client: httpx.Client | None = None,
    ) -> pd.DataFrame:
        """Day-ahead consumption forecast (``prevision_j1``), indexed by target time.

        This is a genuine forecast published ~D-1, so at a decision time ``t0`` it
        is available for target hours up to ~24-32 h out (the feature layer gates
        it accordingly). Returned as a one-column frame ``consumption_forecast_mw``.
        """
        start = to_utc_timestamp(start)
        end = to_utc_timestamp(end)
        url = f"{self.base_url}/catalog/datasets/{self.history_dataset}/exports/json"
        owns = client is None
        client = client or httpx.Client(timeout=self.timeout)
        frames: list[pd.DataFrame] = []
        try:
            for a, b in _year_bounds(start, end):
                s = a.strftime("%Y-%m-%dT%H:%M:%SZ")
                e = b.strftime("%Y-%m-%dT%H:%M:%SZ")
                where = f"prevision_j1 IS NOT NULL AND date_heure >= '{s}' AND date_heure < '{e}'"
                resp = client.get(
                    url, params={"select": "date_heure,prevision_j1", "where": where,
                                 "timezone": "UTC"}
                )
                resp.raise_for_status()
                recs = resp.json()
                if recs:
                    frames.append(pd.DataFrame.from_records(recs))
        finally:
            if owns:
                client.close()
        if not frames:
            return pd.DataFrame(columns=["consumption_forecast_mw"])
        raw = pd.concat(frames)
        raw["date_heure"] = pd.to_datetime(raw["date_heure"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["date_heure"]).set_index("date_heure").sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        series = pd.to_numeric(raw["prevision_j1"], errors="coerce")
        if hourly:
            series = series.resample("1h").mean()
        out = series.dropna().rename("consumption_forecast_mw").to_frame()
        out.index.name = TIMESTAMP
        return out

    def load_hourly(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """CarbonProvider interface: hourly history for ``[start, end)``."""
        end = to_utc_timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
        start = to_utc_timestamp(start) if start is not None else end - pd.Timedelta(days=365 * 3)
        return self.import_history(start, end, hourly=True)

    # ------------------------------------------------------------------ #
    # Snapshot persistence (replay)
    # ------------------------------------------------------------------ #
    @staticmethod
    def save_snapshot(df: pd.DataFrame, path) -> None:
        """Persist a canonical frame to parquet for reproducible replay."""
        import pathlib

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ensure_canonical(df, require_carbon=False).to_parquet(path)

    @staticmethod
    def load_snapshot(path) -> pd.DataFrame:
        """Load a canonical frame previously saved with :meth:`save_snapshot`."""
        return ensure_canonical(pd.read_parquet(path), require_carbon=False)
