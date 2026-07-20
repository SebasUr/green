"""Historical day-ahead generation forecasts from Energy-Charts.

The public API exposes French wind, solar, and load forecasts by target time.
Unlike weather reanalysis or realized generation, the ``day-ahead`` series is a
forecast vintage and is therefore suitable for leakage-free replay.  This
provider is deliberately optional: network failures or missing forecast types
must never affect the existing carbon baseline.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

from green_observatory.providers.carbon_base import TIMESTAMP

BASE_URL = "https://api.energy-charts.info"
SUPPORTED_TYPES = ("wind_onshore", "wind_offshore", "solar", "load")
COLUMN_NAMES = {
    "wind_onshore": "wind_onshore_day_ahead_forecast_mw",
    "wind_offshore": "wind_offshore_day_ahead_forecast_mw",
    "solar": "solar_day_ahead_forecast_mw",
    "load": "load_day_ahead_forecast_mw",
}


class EnergyChartsMixForecastProvider:
    """Fetch and cache public French day-ahead mix forecasts."""

    def __init__(
        self,
        *,
        country: str = "fr",
        base_url: str = BASE_URL,
        timeout: float = 360.0,
        max_retries: int = 2,
    ) -> None:
        self.country = country.lower()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    @staticmethod
    def parse(payload: dict, production_type: str | None = None) -> pd.DataFrame:
        """Parse one API response and aggregate its 15-minute values hourly."""
        kind = production_type or payload.get("production_type")
        if kind not in COLUMN_NAMES:
            raise ValueError(f"unsupported Energy-Charts production type: {kind!r}")
        timestamps = payload.get("unix_seconds", [])
        values = payload.get("forecast_values", [])
        if len(timestamps) != len(values):
            raise ValueError("Energy-Charts timestamps and values have different lengths")
        index = pd.to_datetime(timestamps, unit="s", utc=True)
        series = pd.Series(
            pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
            index=index,
            name=COLUMN_NAMES[kind],
            dtype="float64",
        )
        out = series.resample("1h").mean().to_frame()
        out.index.name = TIMESTAMP
        return out

    def _fetch_one(
        self,
        production_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        client: httpx.Client,
    ) -> pd.DataFrame:
        last: Exception | None = None
        params = {
            "country": self.country,
            "production_type": production_type,
            "forecast_type": "day-ahead",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.get(f"{self.base_url}/public_power_forecast", params=params)
                response.raise_for_status()
                payload = response.json()
                if payload.get("forecast_type") != "day-ahead":
                    raise ValueError("API did not return the requested day-ahead vintage")
                return self.parse(payload, production_type)
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(
            f"Energy-Charts forecast failed for {production_type} after "
            f"{self.max_retries} attempts: {last}"
        )

    def fetch(
        self,
        start,
        end,
        *,
        production_types: Sequence[str] = SUPPORTED_TYPES,
        client: httpx.Client | None = None,
        progress: bool = False,
    ) -> pd.DataFrame:
        """Return hourly target-time day-ahead forecasts for ``[start, end]``."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        unknown = sorted(set(production_types) - set(SUPPORTED_TYPES))
        if unknown:
            raise ValueError(f"unsupported Energy-Charts forecast types: {unknown}")
        if not production_types:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name=TIMESTAMP))

        owns_client = client is None
        client = client or httpx.Client(timeout=self.timeout)
        frames: list[pd.DataFrame] = []
        try:
            # The public endpoint is high-latency but the four forecast series
            # are independent. httpx.Client is thread-safe, so fetch them in
            # parallel and keep the total import close to one request latency.
            with ThreadPoolExecutor(max_workers=min(4, len(production_types))) as pool:
                futures = {
                    pool.submit(self._fetch_one, kind, start, end, client): kind
                    for kind in production_types
                }
                for future in as_completed(futures):
                    kind = futures[future]
                    frames.append(future.result())
                    if progress:
                        print(f"  Energy-Charts day-ahead {kind}")
        finally:
            if owns_client:
                client.close()
        if not frames:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name=TIMESTAMP))
        out = pd.concat(frames, axis=1).sort_index()
        return out.loc[(out.index >= start.floor("D")) & (out.index < end.ceil("D"))]

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path) -> None:
        import pathlib

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    @staticmethod
    def load_snapshot(path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        frame.index.name = TIMESTAMP
        return frame.sort_index()


__all__ = ["COLUMN_NAMES", "EnergyChartsMixForecastProvider", "SUPPORTED_TYPES"]
