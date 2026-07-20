"""Causal archived D-1 temperature forecasts for mainland France.

Open-Meteo's Previous Runs API exposes a value indexed by valid time that was
predicted a fixed number of days earlier.  For a dense 1..24 hour forecast,
``temperature_2m_previous_day1`` was issued no later than the forecast origin,
so it can be replayed without substituting realised weather or reanalysis.

The national feature is an equal-weighted mean over the same representative
mainland points used by :mod:`weather_openmeteo`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd

from green_observatory.providers.weather_openmeteo import FRANCE_POINTS


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SOURCE_VARIABLE = "temperature_2m_previous_day1"
TEMPERATURE_D1_COLUMN = "temperature_2m_previous_day1_c"


class PreviousRunTemperatureProvider:
    """Fetch a publication-safe national D-1 temperature signal."""

    def __init__(
        self,
        points: list[tuple[float, float]] = FRANCE_POINTS,
        *,
        model: str = "gfs_seamless",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.points = list(points)
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    def _params(self, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, str]:
        return {
            "latitude": ",".join(str(point[0]) for point in self.points),
            "longitude": ",".join(str(point[1]) for point in self.points),
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": SOURCE_VARIABLE,
            "models": self.model,
            "timezone": "UTC",
        }

    @staticmethod
    def parse(payload: object) -> pd.DataFrame:
        locations = payload if isinstance(payload, list) else [payload]
        series: list[pd.Series] = []
        for location in locations:
            if not isinstance(location, dict):
                continue
            hourly = location.get("hourly", {})
            index = pd.to_datetime(hourly.get("time", []), utc=True)
            series.append(
                pd.Series(
                    pd.to_numeric(hourly.get(SOURCE_VARIABLE, []), errors="coerce"),
                    index=index,
                    dtype="float64",
                )
            )
        if not series:
            return pd.DataFrame(columns=[TEMPERATURE_D1_COLUMN])
        temperature = pd.concat(series, axis=1).mean(axis=1, skipna=True)
        out = temperature.rename(TEMPERATURE_D1_COLUMN).to_frame().sort_index()
        out.index.name = "target_time"
        return out

    def fetch(self, start: str, end: str, *, chunk_days: int = 366) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).tz_localize(None).normalize()
        end_ts = pd.Timestamp(end).tz_localize(None).normalize()
        if end_ts < start_ts:
            raise ValueError("temperature end must not precede start")
        frames: list[pd.DataFrame] = []
        with httpx.Client(timeout=self.timeout) as client:
            cursor = start_ts
            while cursor <= end_ts:
                chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
                last_error: Exception | None = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = client.get(
                            PREVIOUS_RUNS_URL,
                            params=self._params(cursor, chunk_end),
                        )
                        response.raise_for_status()
                        frame = self.parse(response.json())
                        frames.append(frame)
                        print(
                            "Open-Meteo previous-run temperature: "
                            f"{cursor.date()}..{chunk_end.date()} rows={len(frame)}"
                        )
                        break
                    except (httpx.HTTPError, ValueError) as exc:
                        last_error = exc
                        if attempt < self.max_retries:
                            time.sleep(min(2**attempt, 10))
                else:
                    raise RuntimeError(
                        "Open-Meteo previous-runs request failed after "
                        f"{self.max_retries} attempts: {last_error}"
                    )
                cursor = chunk_end + pd.Timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=[TEMPERATURE_D1_COLUMN])
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")]

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--model", default="gfs_seamless")
    parser.add_argument("--chunk-days", type=int, default=366)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = PreviousRunTemperatureProvider(model=args.model)
    frame = provider.fetch(args.start, args.end, chunk_days=args.chunk_days)
    provider.save_snapshot(frame, args.output)
    coverage = float(frame[TEMPERATURE_D1_COLUMN].notna().mean()) if len(frame) else 0.0
    print(
        f"saved {len(frame)} hourly D-1 temperatures -> {args.output}; "
        f"coverage={coverage:.3f}"
    )


if __name__ == "__main__":
    main()


__all__ = ["PreviousRunTemperatureProvider", "TEMPERATURE_D1_COLUMN"]
