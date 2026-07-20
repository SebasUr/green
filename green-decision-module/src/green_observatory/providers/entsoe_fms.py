"""ENTSO-E Transparency **File Library (FMS)** provider.

Official no-approval route to Transparency Platform bulk data: a normal TP
account authenticates against Keycloak (``client_id=tp-fms-public``,
password grant) and downloads the monthly CSV exports under ``/TP_export``.
Unlike the classic RESTful API, no manually granted security token is
needed.

Verified against the live service (2026-07):

* ``POST /listFolder`` with ``{"path": "/TP_export/<dataset>/",
  "pageInfo": ...}`` returns ``contentItemList`` items carrying ``fileId``.
* ``POST /downloadFileContent`` wants ``{"topLevelFolder": "TP_export",
  "fileIdList": [...]}`` and streams the (decompressed) file back.
* Monthly files are tab-separated CSV with an ``UpdateTime(UTC)`` column —
  only the **latest** version of each value is kept, so causal use must
  drop rows whose update time is not strictly before the Paris midnight
  that starts the delivery day (same rule as the RTE exchange programs).

Credentials come from ``.env``: ``ENTSOE_EMAIL`` / ``ENTSOE_PASSWORD``.
They are read into memory only; nothing is logged.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import httpx
import pandas as pd

KEYCLOAK_TOKEN_URL = "https://keycloak.tp.entsoe.eu/realms/tp/protocol/openid-connect/token"
FMS_BASE_URL = "https://fms.tp.entsoe.eu"
TOP_LEVEL_FOLDER = "TP_export"
PARIS_TZ = "Europe/Paris"

A71_DATASET = "DayAheadAggregatedGeneration_14.1.C_r3"


def _read_env(dotenv_path: str | Path) -> tuple[str, str]:
    email = os.getenv("ENTSOE_EMAIL")
    password = os.getenv("ENTSOE_PASSWORD")
    if (not email or not password) and Path(dotenv_path).exists():
        for line in Path(dotenv_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() == "ENTSOE_EMAIL":
                email = email or value.strip()
            elif key.strip() == "ENTSOE_PASSWORD":
                password = password or value.strip()
    if not email or not password:
        raise RuntimeError(
            "set ENTSOE_EMAIL and ENTSOE_PASSWORD in the environment or .env"
        )
    return email, password


class EntsoeFmsProvider:
    """Minimal FMS client: list a dataset folder, download monthly files."""

    def __init__(self, email: str, password: str, *, timeout: float = 180.0) -> None:
        self._email = email
        self._password = password
        self.timeout = timeout
        self._token: str | None = None
        self._token_expires_at = 0.0

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env", **kwargs) -> EntsoeFmsProvider:
        return cls(*_read_env(dotenv_path), **kwargs)

    def _bearer(self, client: httpx.Client) -> str:
        if self._token and time.monotonic() < self._token_expires_at - 30.0:
            return self._token
        response = client.post(
            KEYCLOAK_TOKEN_URL,
            data={
                "client_id": "tp-fms-public",
                "grant_type": "password",
                "username": self._email,
                "password": self._password,
            },
        )
        response.raise_for_status()
        payload = response.json()
        self._token = str(payload["access_token"])
        self._token_expires_at = time.monotonic() + float(payload.get("expires_in", 300))
        return self._token

    def list_dataset(self, dataset: str, *, page_size: int = 500) -> pd.DataFrame:
        """All files of one dataset folder as (name, file_id, size) rows."""
        rows: list[dict] = []
        with httpx.Client(timeout=self.timeout) as client:
            page = 0
            while True:
                response = client.post(
                    f"{FMS_BASE_URL}/listFolder",
                    headers={
                        "Authorization": f"Bearer {self._bearer(client)}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "path": f"/{TOP_LEVEL_FOLDER}/{dataset}/",
                        "pageInfo": {"pageIndex": page, "pageSize": page_size},
                    },
                )
                response.raise_for_status()
                items = response.json().get("contentItemList", [])
                rows.extend(
                    {
                        "name": item.get("name"),
                        "file_id": item.get("fileId"),
                        "size": item.get("size"),
                        "last_updated": item.get("lastUpdatedTimestamp"),
                    }
                    for item in items
                )
                if len(items) < page_size:
                    break
                page += 1
        return pd.DataFrame(rows)

    def download_file(self, file_id: str, *, client: httpx.Client | None = None) -> bytes:
        owns = client is None
        client = client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{FMS_BASE_URL}/downloadFileContent",
                headers={
                    "Authorization": f"Bearer {self._bearer(client)}",
                    "Content-Type": "application/json",
                },
                json={"topLevelFolder": TOP_LEVEL_FOLDER, "fileIdList": [file_id]},
            )
            response.raise_for_status()
            return response.content
        finally:
            if owns:
                client.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_export_csv(raw: bytes) -> pd.DataFrame:
        """Parse a tab-separated TP monthly export into a raw DataFrame."""
        return pd.read_csv(io.BytesIO(raw), sep="\t")

    def fetch_a71_hourly(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        area_map_code: str = "FR",
        progress: bool = False,
    ) -> pd.DataFrame:
        """Hourly day-ahead total generation forecast for one bidding zone.

        Returns columns ``generation_forecast_mw`` and ``update_time`` on an
        hourly UTC index (15-min values averaged; latest published version).
        """
        listing = self.list_dataset(A71_DATASET)
        months = pd.period_range(
            pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M")
        )
        wanted = {f"{p.year}_{p.month:02d}_{A71_DATASET}.csv" for p in months}
        listing = listing[listing["name"].isin(wanted)].sort_values("name")
        frames: list[pd.DataFrame] = []
        with httpx.Client(timeout=self.timeout) as client:
            for row in listing.itertuples():
                raw = self.download_file(row.file_id, client=client)
                df = self.parse_export_csv(raw)
                df = df[df["AreaMapCode"] == area_map_code]
                if progress:
                    print(f"  {row.name}: {len(df)} rows for {area_map_code}", flush=True)
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["generation_forecast_mw", "update_time"])
        df = pd.concat(frames, ignore_index=True)
        df["datetime"] = pd.to_datetime(df["DateTime(UTC)"], utc=True)
        df["update_time"] = pd.to_datetime(df["UpdateTime(UTC)"], utc=True)
        hourly = (
            df.set_index("datetime")
            .groupby(pd.Grouper(freq="1h"))
            .agg(
                generation_forecast_mw=("GenerationForecast[MW]", "mean"),
                update_time=("update_time", "max"),
            )
            .dropna(subset=["generation_forecast_mw"])
        )
        return hourly.sort_index()


#: The 2025 TP platform migration re-stamped historical ``UpdateTime`` with
#: the re-export date, destroying vintage stamps before roughly this date.
#: Values before it are still genuine D-1 forecasts (their MAE vs realized
#: generation is 1.7-4.4 GW every year — backfilled actuals would show ~0),
#: so they are trusted as-is; from this date on the strict guard applies.
A71_TRUSTED_STAMPS_FROM = pd.Timestamp("2025-10-01", tz="UTC")


def a71_day_ahead_features(
    hourly: pd.DataFrame,
    *,
    trusted_stamps_from: pd.Timestamp = A71_TRUSTED_STAMPS_FROM,
) -> pd.DataFrame:
    """Causal feature columns from the A71 hourly snapshot.

    Where update stamps are meaningful (>= ``trusted_stamps_from``), values
    whose latest ``update_time`` is not strictly before the Paris midnight
    starting their delivery day are dropped (revised after the information
    horizon; the pre-revision value is lost). Column names carry
    ``day_ahead`` so the feature builder masks target hours on the next
    local delivery day.
    """
    index = pd.DatetimeIndex(hourly.index).as_unit("ns")
    paris_day_start = index.tz_convert(PARIS_TZ).normalize().tz_convert("UTC")
    known_before_day = (
        pd.DatetimeIndex(hourly["update_time"]).as_unit("ns") < paris_day_start
    )
    keep = known_before_day | (index < trusted_stamps_from)
    out = pd.DataFrame(index=index)
    total = hourly["generation_forecast_mw"].where(keep)
    out["entsoe_total_generation_day_ahead_mw"] = total
    out["entsoe_total_generation_delta_day_ahead_mw"] = total - total.shift(
        freq="24h"
    ).reindex(index)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--area", default="FR")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = EntsoeFmsProvider.from_env(dotenv_path=args.dotenv)
    hourly = provider.fetch_a71_hourly(
        args.start, args.end, area_map_code=args.area, progress=True
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.output)
    print(
        f"saved {len(hourly)} hourly rows -> {args.output}  "
        f"[{hourly.index.min()}..{hourly.index.max()}]"
    )


if __name__ == "__main__":
    main()
