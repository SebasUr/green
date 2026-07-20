"""Origin-safe target features from RTE D-1 generation forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTION_COLUMNS = {
    "WIND_ONSHORE": "rte_tgt_wind_onshore_d1_mw",
    "WIND_OFFSHORE": "rte_tgt_wind_offshore_d1_mw",
    "SOLAR": "rte_tgt_solar_d1_mw",
}


class RteGenerationForecastFeatureStore:
    """Select the latest D-1 value published no later than each origin."""

    def __init__(
        self,
        forecasts: pd.DataFrame,
        *,
        production_types: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        required = {
            "forecast_type",
            "production_type",
            "target_start",
            "updated_date",
            "value_mw",
        }
        missing = sorted(required - set(forecasts.columns))
        if missing:
            raise ValueError(f"RTE generation forecast is missing: {missing}")
        selected = set(production_types or PRODUCTION_COLUMNS)
        unknown = sorted(selected - set(PRODUCTION_COLUMNS))
        if unknown:
            raise ValueError(f"unsupported RTE D-1 production types: {unknown}")
        self.production_types_ = tuple(
            production_type
            for production_type in PRODUCTION_COLUMNS
            if production_type in selected
        )
        frame = forecasts.copy()
        frame = frame[
            frame["forecast_type"].eq("D-1")
            & frame["production_type"].isin(selected)
        ]
        for column in ("target_start", "updated_date"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        frame["value_mw"] = pd.to_numeric(frame["value_mw"], errors="coerce")
        frame = frame.dropna(
            subset=["target_start", "updated_date", "value_mw"]
        )
        # Updates after their target are revisions, not usable forecasts.
        frame = frame[frame["updated_date"] <= frame["target_start"]]
        self.series_: dict[str, pd.DataFrame] = {}
        for production_type in self.production_types_:
            feature_name = PRODUCTION_COLUMNS[production_type]
            part = frame.loc[
                frame["production_type"].eq(production_type),
                ["target_start", "updated_date", "value_mw"],
            ].sort_values(["target_start", "updated_date"])
            self.series_[feature_name] = part.reset_index(drop=True)

    @classmethod
    def from_parquet(
        cls,
        path,
        *,
        production_types: tuple[str, ...] | list[str] | None = None,
    ) -> RteGenerationForecastFeatureStore:
        return cls(
            pd.read_parquet(path), production_types=production_types
        )

    def features_by_horizon(
        self,
        origins: pd.DatetimeIndex,
        horizons: tuple[int, ...] | list[int],
    ) -> dict[int, pd.DataFrame]:
        origins = pd.DatetimeIndex(origins)
        if origins.tz is None:
            origins = origins.tz_localize("UTC")
        else:
            origins = origins.tz_convert("UTC")
        horizons = tuple(int(horizon) for horizon in horizons)
        n_origins = len(origins)
        query_parts: list[pd.DataFrame] = []
        for position, horizon in enumerate(horizons):
            query_parts.append(
                pd.DataFrame(
                    {
                        "row": np.arange(n_origins, dtype=int),
                        "horizon_position": position,
                        "origin": origins,
                        "target_start": origins + pd.Timedelta(hours=horizon),
                    }
                )
            )
        queries = pd.concat(query_parts, ignore_index=True)
        arrays = {
            PRODUCTION_COLUMNS[production_type]: np.full(
                (n_origins, len(horizons)), np.nan
            )
            for production_type in self.production_types_
        }
        for feature_name, values in self.series_.items():
            if values.empty:
                continue
            matched = queries.merge(values, on="target_start", how="left")
            matched = matched[
                matched["updated_date"].notna()
                & (matched["updated_date"] <= matched["origin"])
            ]
            if matched.empty:
                continue
            latest = (
                matched.sort_values("updated_date")
                .drop_duplicates(["row", "horizon_position"], keep="last")
            )
            arrays[feature_name][
                latest["row"].to_numpy(dtype=int),
                latest["horizon_position"].to_numpy(dtype=int),
            ] = latest["value_mw"].to_numpy(dtype=float)

        wind_names = [
            PRODUCTION_COLUMNS[name]
            for name in ("WIND_ONSHORE", "WIND_OFFSHORE")
            if name in self.production_types_
        ]
        if wind_names:
            wind_parts = np.stack([arrays[name] for name in wind_names])
            with np.errstate(invalid="ignore"):
                wind = np.nansum(wind_parts, axis=0)
            wind[np.isnan(wind_parts).all(axis=0)] = np.nan
            arrays["rte_tgt_wind_d1_mw"] = wind
        if set(self.production_types_) == set(PRODUCTION_COLUMNS):
            arrays["rte_tgt_variable_renewables_d1_mw"] = (
                arrays["rte_tgt_wind_d1_mw"]
                + arrays["rte_tgt_solar_d1_mw"]
            )
        return {
            horizon: pd.DataFrame(
                {name: values[:, position] for name, values in arrays.items()},
                index=origins,
            )
            for position, horizon in enumerate(horizons)
        }


__all__ = ["RteGenerationForecastFeatureStore"]
