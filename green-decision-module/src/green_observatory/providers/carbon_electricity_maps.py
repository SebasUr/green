"""Electricity Maps provider - **optional** comparative forecast only.

Electricity Maps publishes a *consumption-based* carbon intensity (imports and
exports are attributed to the consuming zone), which is a different accounting
basis from RTE ``taux_co2`` (*production-based*). Absolute values are therefore
not directly comparable; the useful comparison is (a) as an external
professional **forecast** benchmark and (b) on green-window **ranking**.

This provider is never a core dependency: without ``ELECTRICITYMAPS_API_TOKEN``
it reports ``available() is False`` and every caller skips it gracefully.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
import pandas as pd

from green_observatory.models import CarbonBasis, CarbonSignal

DEFAULT_BASE_URL = "https://api.electricitymap.org/v3"
TOKEN_ENV_VARS = ("ELECTRICITYMAPS_API_TOKEN", "ELECTRICITY_MAPS_TOKEN")


def _token_from_env() -> str | None:
    for var in TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


class ElectricityMapsProvider:
    """Thin client for Electricity Maps carbon-intensity endpoints (zone FR)."""

    def __init__(
        self,
        token: str | None = None,
        *,
        zone: str = "FR",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.token = token or _token_from_env()
        self.zone = zone
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        """True only if an API token is configured."""
        return bool(self.token)

    def _get(self, path: str, params: dict, client: httpx.Client | None = None) -> dict:
        if not self.available():
            raise RuntimeError(
                "Electricity Maps token missing; set ELECTRICITYMAPS_API_TOKEN "
                "or check available() first."
            )
        owns = client is None
        client = client or httpx.Client(timeout=self.timeout)
        try:
            resp = client.get(
                f"{self.base_url}/{path}",
                params=params,
                headers={"auth-token": self.token},
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            if owns:
                client.close()

    def latest_signal(self, client: httpx.Client | None = None) -> CarbonSignal:
        data = self._get("carbon-intensity/latest", {"zone": self.zone}, client)
        return CarbonSignal(
            timestamp=pd.Timestamp(data["datetime"]).to_pydatetime(),
            zone=self.zone,
            carbon_intensity_gco2_kwh=float(data["carbonIntensity"]),
            basis=CarbonBasis.consumption,
            is_consolidated=not bool(data.get("isEstimated", False)),
            source="electricity_maps:latest",
        )

    def forecast_series(self, client: httpx.Client | None = None) -> pd.Series:
        """Hourly consumption-based carbon-intensity forecast (UTC index)."""
        data = self._get("carbon-intensity/forecast", {"zone": self.zone}, client)
        records = data.get("forecast", [])
        if not records:
            return pd.Series(dtype=float, name="electricity_maps")
        idx = pd.to_datetime([r["datetime"] for r in records], utc=True)
        vals = [float(r["carbonIntensity"]) for r in records]
        return pd.Series(vals, index=idx, name="electricity_maps").sort_index()

    def history_series(self, client: httpx.Client | None = None) -> pd.Series:
        """Last ~24h of actual/estimated consumption-based intensity (UTC index)."""
        data = self._get("carbon-intensity/history", {"zone": self.zone}, client)
        records = data.get("history", [])
        if not records:
            return pd.Series(dtype=float, name="electricity_maps")
        idx = pd.to_datetime([r["datetime"] for r in records], utc=True)
        vals = [float(r["carbonIntensity"]) for r in records]
        return pd.Series(vals, index=idx, name="electricity_maps").sort_index()

    def forecast_frame(self, client: httpx.Client | None = None) -> pd.DataFrame:
        """Forecast as a frame with ``issued_at`` for auditability."""
        s = self.forecast_series(client)
        issued = datetime.now(timezone.utc)
        return pd.DataFrame(
            {
                "issued_at": issued,
                "target_time": s.index,
                "prediction": s.to_numpy(),
                "horizon_hours": (s.index - pd.Timestamp(issued)).total_seconds() / 3600.0,
            }
        )
