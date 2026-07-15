"""Realized RTE carbon signal used by workload accounting."""

from __future__ import annotations

import time

import pandas as pd

from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


class OdreRealtimeCarbonSource:
    """Fetch and briefly cache the sub-hourly eCO2mix near-real-time series."""

    def __init__(
        self,
        provider: OdreCarbonProvider | None = None,
        cache_seconds: float = 300.0,
    ) -> None:
        self.provider = provider or OdreCarbonProvider()
        self.cache_seconds = cache_seconds
        self._fetched_at = 0.0
        self._frame = pd.DataFrame()

    def _refresh_realtime(self) -> None:
        if self._frame.empty or time.monotonic() - self._fetched_at >= self.cache_seconds:
            self._frame = self.provider.import_realtime(hourly=False)
            self._fetched_at = time.monotonic()

    def load(self, start: float, end: float) -> pd.Series:
        start_ts = pd.Timestamp(start, unit="s", tz="UTC") - pd.Timedelta(hours=2)
        end_ts = pd.Timestamp(end, unit="s", tz="UTC") + pd.Timedelta(minutes=30)
        now = pd.Timestamp.now(tz="UTC")
        if start_ts >= now - pd.Timedelta(days=30):
            self._refresh_realtime()
            frame = self._frame.loc[start_ts:end_ts]
        else:
            frame = self.provider.import_history(
                start_ts, end_ts, hourly=False, drop_null_target=True
            )
        if CARBON not in frame:
            return pd.Series(dtype=float, name=CARBON)
        return frame[CARBON].dropna().sort_index()


class SnapshotCarbonSource:
    """Offline source for replaying reports from a parquet or CSV snapshot."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self, start: float, end: float) -> pd.Series:
        if self.path.endswith(".parquet"):
            frame = pd.read_parquet(self.path)
        else:
            frame = pd.read_csv(self.path)
            timestamp = "timestamp" if "timestamp" in frame else frame.columns[0]
            frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
            frame = frame.set_index(timestamp)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("carbon snapshot must have a DatetimeIndex or timestamp column")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        start_ts = pd.Timestamp(start, unit="s", tz="UTC") - pd.Timedelta(hours=2)
        end_ts = pd.Timestamp(end, unit="s", tz="UTC") + pd.Timedelta(minutes=30)
        return frame.loc[start_ts:end_ts, CARBON].dropna().sort_index()
