"""Authenticated RTE system-forecast and unavailability snapshots.

The provider keeps credentials in memory only.  OAuth tokens are never written
to snapshots or logs.  Two public RTE APIs are normalized into flat, parquet-
safe tables:

* Generation Forecast v3 (D-3, D-2, D-1, intraday and current forecasts);
* Unavailability Additional Information v7 (versioned generation messages).

Publication and update timestamps are preserved so downstream feature builders
can replay exactly what was known at a historical forecast origin.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

DEFAULT_BASE_URL = "https://digital.iservices.rte-france.com"
TOKEN_PATH = "/token/oauth/"
UNAVAILABILITY_PATH = (
    "/open_api/unavailability_additional_information/v7/"
    "generation_unavailabilities"
)
GENERATION_FORECAST_PATH = "/open_api/generation_forecast/v3/forecasts"


def _utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _rte_timestamp(value) -> str:
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_dotenv_credentials(path: str | Path) -> tuple[str | None, str | None]:
    values: dict[str, str] = {}
    path = Path(path)
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in {"RTE_CLIENT_ID", "RTE_CLIENT_SECRET"}:
                continue
            values[key] = value.strip().strip("\"'")
    return values.get("RTE_CLIENT_ID"), values.get("RTE_CLIENT_SECRET")


@dataclass(frozen=True)
class RteSystemSnapshots:
    unavailability: pd.DataFrame
    generation_forecast: pd.DataFrame


class RteSystemForecastProvider:
    """OAuth client and pure normalizers for the subscribed public RTE APIs."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError("RTE OAuth client ID and secret are required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        dotenv_path: str | Path | None = ".env",
        **kwargs,
    ) -> RteSystemForecastProvider:
        client_id = os.getenv("RTE_CLIENT_ID")
        client_secret = os.getenv("RTE_CLIENT_SECRET")
        if (not client_id or not client_secret) and dotenv_path is not None:
            file_id, file_secret = _read_dotenv_credentials(dotenv_path)
            client_id = client_id or file_id
            client_secret = client_secret or file_secret
        if not client_id or not client_secret:
            raise ValueError(
                "set RTE_CLIENT_ID and RTE_CLIENT_SECRET in the environment "
                "or an ignored .env file"
            )
        return cls(client_id, client_secret, **kwargs)

    def _token(self, client: httpx.Client) -> str:
        if self._access_token and time.monotonic() < self._token_expires_at - 30.0:
            return self._access_token
        response = client.post(
            f"{self.base_url}{TOKEN_PATH}",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = str(payload["access_token"])
        self._token_expires_at = time.monotonic() + float(
            payload.get("expires_in", 300)
        )
        return self._access_token

    def _get(
        self,
        client: httpx.Client,
        path: str,
        params: dict[str, str],
    ) -> tuple[dict, int]:
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers={"Authorization": f"Bearer {self._token(client)}"},
                )
                if response.status_code == 401 and attempt < self.max_retries:
                    self._access_token = None
                    continue
                response.raise_for_status()
                return response.json(), response.status_code
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"RTE request failed for {path}: {last}")

    def _fetch_chunks(
        self,
        path: str,
        root_key: str,
        start,
        end,
        *,
        chunk_days: int,
        progress: bool,
        extra_params: dict[str, str] | None = None,
    ) -> list[dict]:
        start_ts, end_ts = _utc(start), _utc(end)
        if end_ts <= start_ts:
            raise ValueError("RTE snapshot end must be after start")
        records: list[dict] = []
        with httpx.Client(timeout=self.timeout) as client:
            cursor = start_ts
            while cursor < end_ts:
                chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
                records.extend(
                    self._fetch_interval(
                        client,
                        path,
                        root_key,
                        cursor,
                        chunk_end,
                        progress=progress,
                        extra_params=extra_params,
                    )
                )
                cursor = chunk_end
        return records

    def _fetch_interval(
        self,
        client: httpx.Client,
        path: str,
        root_key: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        progress: bool,
        extra_params: dict[str, str] | None = None,
    ) -> list[dict]:
        params = {
            "start_date": _rte_timestamp(start),
            "end_date": _rte_timestamp(end),
            **(extra_params or {}),
        }
        payload, status = self._get(
            client,
            path,
            params,
        )
        rows = payload.get(root_key, [])
        # RTE returns HTTP 206 and exactly 1,000 unavailability messages when a
        # period is truncated. Split recursively so no historical revisions are
        # silently lost.
        if status == 206 or (root_key == "generation_unavailabilities" and len(rows) >= 1000):
            if end - start <= pd.Timedelta(hours=1):
                raise RuntimeError("RTE interval remains truncated at one hour")
            middle = start + (end - start) / 2
            return self._fetch_interval(
                client,
                path,
                root_key,
                start,
                middle,
                progress=progress,
                extra_params=extra_params,
            ) + self._fetch_interval(
                client,
                path,
                root_key,
                middle,
                end,
                progress=progress,
                extra_params=extra_params,
            )
        if progress:
            print(f"  RTE {root_key}: {start.date()}..{end.date()} ({len(rows)})")
        return rows

    @staticmethod
    def normalize_unavailability(records: list[dict]) -> pd.DataFrame:
        rows: list[dict] = []
        for message in records:
            base = {key: value for key, value in message.items() if key != "values"}
            for interval in message.get("values", []) or [{}]:
                rows.append(
                    {
                        **base,
                        "interval_start": interval.get("start_date"),
                        "interval_end": interval.get("end_date"),
                        "available_capacity_mw": interval.get("available_capacity"),
                        "unavailable_capacity_mw": interval.get("unavailable_capacity"),
                    }
                )
        frame = pd.DataFrame(rows)
        date_columns = (
            "creation_date",
            "publication_date",
            "start_date",
            "end_date",
            "interval_start",
            "interval_end",
        )
        for column in date_columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        numeric_columns = (
            "version",
            "affected_asset_or_unit_installed_capacity",
            "available_capacity_mw",
            "unavailable_capacity_mw",
        )
        for column in numeric_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame.empty:
            frame = frame.drop_duplicates(
                subset=["message_id", "interval_start", "interval_end"], keep="last"
            ).sort_values(["publication_date", "message_id", "interval_start"])
        return frame.reset_index(drop=True)

    @staticmethod
    def normalize_generation_forecast(records: list[dict]) -> pd.DataFrame:
        rows: list[dict] = []
        for forecast in records:
            base = {
                "forecast_start": forecast.get("start_date"),
                "forecast_end": forecast.get("end_date"),
                "forecast_type": forecast.get("type"),
                "production_type": forecast.get("production_type"),
                "sub_type": forecast.get("sub_type"),
            }
            for value in forecast.get("values", []):
                rows.append(
                    {
                        **base,
                        "target_start": value.get("start_date"),
                        "target_end": value.get("end_date"),
                        "updated_date": value.get("updated_date"),
                        "value_mw": value.get("value"),
                    }
                )
        frame = pd.DataFrame(rows)
        for column in (
            "forecast_start",
            "forecast_end",
            "target_start",
            "target_end",
            "updated_date",
        ):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if "value_mw" in frame:
            frame["value_mw"] = pd.to_numeric(frame["value_mw"], errors="coerce")
        if not frame.empty:
            frame = frame.drop_duplicates(
                subset=[
                    "forecast_type",
                    "production_type",
                    "sub_type",
                    "target_start",
                    "updated_date",
                ],
                keep="last",
            ).sort_values(
                ["target_start", "forecast_type", "production_type", "updated_date"]
            )
        return frame.reset_index(drop=True)

    def fetch_unavailability(
        self, start, end, *, chunk_days: int = 7, progress: bool = False
    ) -> pd.DataFrame:
        records = self._fetch_chunks(
            UNAVAILABILITY_PATH,
            "generation_unavailabilities",
            start,
            end,
            chunk_days=chunk_days,
            progress=progress,
        )
        return self.normalize_unavailability(records)

    def fetch_generation_forecast(
        self,
        start,
        end,
        *,
        forecast_type: str | None = "D-1",
        chunk_days: int = 7,
        progress: bool = False,
    ) -> pd.DataFrame:
        records = self._fetch_chunks(
            GENERATION_FORECAST_PATH,
            "forecasts",
            start,
            end,
            chunk_days=chunk_days,
            progress=progress,
            extra_params={"type": forecast_type} if forecast_type else None,
        )
        return self.normalize_generation_forecast(records)

    def fetch(
        self,
        start,
        end,
        *,
        unavailability_chunk_days: int = 7,
        forecast_chunk_days: int = 7,
        progress: bool = False,
    ) -> RteSystemSnapshots:
        return RteSystemSnapshots(
            unavailability=self.fetch_unavailability(
                start,
                end,
                chunk_days=unavailability_chunk_days,
                progress=progress,
            ),
            generation_forecast=self.fetch_generation_forecast(
                start,
                end,
                chunk_days=forecast_chunk_days,
                progress=progress,
            ),
        )

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    @staticmethod
    def load_snapshot(path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)


__all__ = [
    "GENERATION_FORECAST_PATH",
    "RteSystemForecastProvider",
    "RteSystemSnapshots",
    "UNAVAILABILITY_PATH",
]
