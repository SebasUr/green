"""Open-Meteo weather provider (wind + solar) - free, no API key.

Wind speed at 100 m and shortwave solar radiation are the physical drivers of
French low-carbon hours: wind is the big day-to-day swing, solar is the
daytime-deterministic part. Both are what professional forecasters (RTE,
Electricity Maps) use to rank green windows 24-48 h out, and both are obtainable
in real time from Open-Meteo without any key.

Two modes:

* ``fetch_archive`` - ERA5 reanalysis (``archive-api``), 2021-> . Used for the
  backtest. It is a *near-actual* proxy for a 24-48 h weather forecast; real
  forecasts carry some error, but at these ranges they are very accurate, so the
  backtest is mildly optimistic, not fantasy.
* ``fetch_forecast`` - live forecast (``api``), the next days. Used at
  deployment time, so the same feature is genuinely available when predicting.

The national signal is the mean over representative France points.
"""

from __future__ import annotations

import time

import httpx
import numpy as np
import pandas as pd

from green_observatory.providers.carbon_base import TIMESTAMP

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: Default: representative France points, equal-weighted national mean.
FRANCE_POINTS: list[tuple[float, float]] = [
    (50.5, 2.6),   # Hauts-de-France
    (48.7, 4.0),   # Grand Est
    (47.2, -1.5),  # Pays de la Loire (west coast)
    (45.0, 1.4),   # Nouvelle-Aquitaine / center-south
    (43.6, 3.9),   # Occitanie / Mediterranean
    (48.1, -0.6),  # center-west
]

#: Optional denser point set with per-point installed wind capacity (GW), for a
#: capacity-weighted national wind signal:
#:     WeatherProvider(FRANCE_POINTS_CAPACITY, weights=FRANCE_WIND_CAPACITY_WEIGHTS)
#: In the reference backtest this performed on par with the default 6-point mean.
FRANCE_POINTS_CAPACITY: list[tuple[float, float]] = [
    (50.3, 2.8), (50.2, 1.5), (49.2, 0.2), (48.9, 4.6), (47.3, 5.0), (47.9, 1.9),
    (48.2, -3.0), (47.4, -1.0), (47.1, -2.5), (45.8, -0.8), (44.0, 2.4), (43.4, 2.8),
    (43.9, 4.8), (45.5, 3.2),
]
FRANCE_WIND_CAPACITY_WEIGHTS: list[float] = [
    5.5, 1.0, 1.0, 4.3, 0.8, 1.5, 1.3, 1.1, 0.5, 1.4, 0.9, 1.9, 0.6, 0.5,
]

WIND = "wind_speed_100m"
SOLAR = "shortwave_radiation"
WIND_COL = "wind_speed_100m"
SOLAR_COL = "solar_radiation"
WEATHER_COLUMNS = [WIND_COL, SOLAR_COL]


class WeatherProvider:
    """Wind/solar weather for France from Open-Meteo (national mean, hourly)."""

    def __init__(
        self,
        points: list[tuple[float, float]] = FRANCE_POINTS,
        *,
        weights: list[float] | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.points = points
        self.weights = weights if (weights is None or len(weights) == len(points)) else None
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Pure parsing (no network)
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse(payload, weights: list[float] | None = None) -> pd.DataFrame:
        """Aggregate per-location hourly series into one national frame.

        Wind is a capacity-weighted mean (NaN-aware) so the signal reflects where
        the fleet actually is; solar is a simple mean (it is spatially uniform).
        """
        locs = payload if isinstance(payload, list) else [payload]
        wind_cols, solar_cols = [], []
        for loc in locs:
            h = loc.get("hourly", {})
            idx = pd.to_datetime(h.get("time", []), utc=True)
            wind_cols.append(pd.Series(h.get(WIND, []), index=idx, dtype="float64"))
            solar_cols.append(pd.Series(h.get(SOLAR, []), index=idx, dtype="float64"))
        w_frame = pd.concat(wind_cols, axis=1)
        s_frame = pd.concat(solar_cols, axis=1)

        if weights is not None and len(weights) == w_frame.shape[1]:
            w = np.asarray(weights, dtype=float)
            vals = w_frame.to_numpy()
            mask = ~np.isnan(vals)
            num = np.where(mask, vals, 0.0) @ w
            den = mask @ w
            wind = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        else:
            wind = w_frame.mean(axis=1).to_numpy()

        out = pd.DataFrame(
            {WIND_COL: wind, SOLAR_COL: s_frame.mean(axis=1).to_numpy()}, index=w_frame.index
        )
        out.index.name = TIMESTAMP
        return out.sort_index()

    # ------------------------------------------------------------------ #
    # Network
    # ------------------------------------------------------------------ #
    def _latlon(self) -> dict:
        return {
            "latitude": ",".join(str(p[0]) for p in self.points),
            "longitude": ",".join(str(p[1]) for p in self.points),
        }

    def _get(self, url: str, params: dict, client: httpx.Client) -> object:
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Open-Meteo request failed after {self.max_retries} tries: {last}")

    def fetch_archive(self, start, end, *, progress: bool = False) -> pd.DataFrame:
        """ERA5 hourly wind/solar (national mean) for ``[start, end]``, by year."""
        start = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
        end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
        frames = []
        with httpx.Client(timeout=self.timeout) as client:
            cur = start
            while cur <= end:
                chunk_end = min(pd.Timestamp(year=cur.year, month=12, day=31), end)
                params = {
                    **self._latlon(),
                    "start_date": cur.strftime("%Y-%m-%d"),
                    "end_date": chunk_end.strftime("%Y-%m-%d"),
                    "hourly": f"{WIND},{SOLAR}",
                    "timezone": "UTC",
                }
                payload = self._get(ARCHIVE_URL, params, client)
                frames.append(self.parse(payload, self.weights))
                if progress:
                    print(f"  weather {cur.date()}..{chunk_end.date()}")
                cur = chunk_end + pd.Timedelta(days=1)
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()

    def fetch_forecast(self, *, days: int = 3, past_days: int = 2) -> pd.DataFrame:
        """Live wind/solar forecast (national mean) for the next ``days`` days."""
        params = {
            **self._latlon(),
            "hourly": f"{WIND},{SOLAR}",
            "forecast_days": days,
            "past_days": past_days,
            "timezone": "UTC",
        }
        with httpx.Client(timeout=self.timeout) as client:
            return self.parse(self._get(FORECAST_URL, params, client), self.weights)

    # ------------------------------------------------------------------ #
    @staticmethod
    def save_snapshot(df: pd.DataFrame, path) -> None:
        import pathlib

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    @staticmethod
    def load_snapshot(path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index.name = TIMESTAMP
        return df
