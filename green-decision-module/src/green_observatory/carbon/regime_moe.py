"""Direct mixture-of-experts for French day-ahead carbon intensity.

This experiment is deliberately isolated from :mod:`fossil_regime`.  It uses
the same physically meaningful operating regimes, but predicts carbon directly
inside each regime instead of first forecasting every emitting-generation
share.  A probabilistic classifier blends the three specialists:

``prediction = P(base)*expert_base + P(ccg)*expert_ccg + P(peak)*expert_peak``.

All features are available at the forecast origin.  RTE observations are
hour-start labelled averages, so the origin state comes from the last fully
closed hour.  ``tgtlag`` features refer to the same target hour one or seven
days earlier; a lag landing on the origin itself is masked because that hourly
average is not complete yet.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.protocols import fossil_regime_labels
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON


DEFAULT_HORIZONS = tuple(range(1, 25))
DEFAULT_STATE_COLUMNS = (
    CARBON,
    "consumption_mw",
    "nuclear_mw",
    "gas_mw",
    "coal_mw",
    "fuel_oil_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "physical_exchange_mw",
    "gas_turbine_mw",
    "gas_cogeneration_mw",
    "gas_ccg_mw",
    "gas_other_mw",
)

# Optional high-granularity RTE state.  Keeping this separate preserves every
# published run that used ``DEFAULT_STATE_COLUMNS`` while allowing the direct
# MoE to exploit the newly cached plant/fuel and border detail explicitly.
DETAILED_STATE_COLUMNS = (
    "bioenergy_mw",
    "pumped_storage_mw",
    "fuel_oil_turbine_mw",
    "fuel_oil_cogeneration_mw",
    "fuel_oil_other_mw",
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
)

# Keep the opt-in multi-scale state compact.  These columns describe the
# carbon level, demand, the dominant French baseload source, the marginal
# fossil source, variable renewables, hydro flexibility and interconnection.
# With the defaults below this adds 8 x 6 x 3 = 144 features, rather than
# blindly expanding every fuel subtype in ``DEFAULT_STATE_COLUMNS``.
DEFAULT_MULTISCALE_STATE_COLUMNS = (
    CARBON,
    "consumption_mw",
    "nuclear_mw",
    "gas_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "physical_exchange_mw",
)
DEFAULT_MULTISCALE_WINDOWS_HOURS = (3, 6, 12, 24, 48, 168)


def _utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


class RegimeMoEFeatureBuilder:
    """Build a stacked, leakage-safe matrix for daily dense forecasts."""

    def __init__(
        self,
        forecast_frame: pd.DataFrame,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        local_tz: str = "Europe/Paris",
        state_columns: Sequence[str] = DEFAULT_STATE_COLUMNS,
        target_lags_hours: Sequence[int] = (24, 168),
        hourly_state_lag_hours: int = 1,
        availability_store: RteAvailabilityFeatureStore | None = None,
        rte_forecast_store: RteGenerationForecastFeatureStore | None = None,
        availability_feature_mode: str = "all",
        include_curve_summaries: bool = False,
        include_detailed_state: bool = False,
        include_multiscale_state: bool = False,
        multiscale_state_columns: Sequence[str] = DEFAULT_MULTISCALE_STATE_COLUMNS,
        multiscale_windows_hours: Sequence[int] = DEFAULT_MULTISCALE_WINDOWS_HOURS,
    ) -> None:
        forecasts = forecast_frame.copy()
        forecasts.index = _utc(forecasts.index)
        self.forecast_frame = forecasts.sort_index()
        self.horizons = tuple(int(value) for value in horizons)
        self.local_tz = local_tz
        base_state_columns = tuple(state_columns)
        self.state_columns = (
            tuple(dict.fromkeys((*base_state_columns, *DETAILED_STATE_COLUMNS)))
            if include_detailed_state
            else base_state_columns
        )
        self.include_detailed_state = bool(include_detailed_state)
        self.target_lags_hours = tuple(int(value) for value in target_lags_hours)
        self.hourly_state_lag_hours = int(hourly_state_lag_hours)
        if self.hourly_state_lag_hours < 1:
            raise ValueError("hourly_state_lag_hours must be at least 1")
        if availability_feature_mode not in {"all", "delta"}:
            raise ValueError("availability_feature_mode must be 'all' or 'delta'")
        self.availability_store = availability_store
        self.rte_forecast_store = rte_forecast_store
        self.availability_feature_mode = availability_feature_mode
        self.include_curve_summaries = bool(include_curve_summaries)
        self.include_multiscale_state = bool(include_multiscale_state)
        self.multiscale_state_columns = tuple(multiscale_state_columns)
        self.multiscale_windows_hours = tuple(
            int(value) for value in multiscale_windows_hours
        )
        if any(value < 2 for value in self.multiscale_windows_hours):
            raise ValueError("multiscale windows must be at least 2 hours")

    def build(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        *,
        supervised: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        origins = _utc(origins)
        repeated_origins = pd.DatetimeIndex(
            np.repeat(origins.as_unit("ns").asi8, len(self.horizons)), tz="UTC"
        )
        horizons = np.tile(np.asarray(self.horizons, dtype=int), len(origins))
        targets = repeated_origins + pd.to_timedelta(horizons, unit="h")
        local_targets = targets.tz_convert(self.local_tz)
        local_origins = repeated_origins.tz_convert(self.local_tz)
        # At 00:00 UTC the auction/forecast for the current local delivery day
        # is already available, but the next local day's product is not.  The
        # dense 24-hour horizon crosses that boundary for h22+ in summer and
        # h23+ in winter.  Historical delivery-indexed snapshots must therefore
        # be hidden there or they would reveal information published later.
        unavailable_day_ahead = (
            local_targets.normalize() > local_origins.normalize()
        )

        x = pd.DataFrame(index=pd.RangeIndex(len(targets)))
        x["horizon"] = horizons
        x["horizon_sin"] = np.sin(2.0 * np.pi * horizons / 24.0)
        x["horizon_cos"] = np.cos(2.0 * np.pi * horizons / 24.0)
        x["target_hour"] = local_targets.hour
        x["target_hour_sin"] = np.sin(2.0 * np.pi * local_targets.hour / 24.0)
        x["target_hour_cos"] = np.cos(2.0 * np.pi * local_targets.hour / 24.0)
        x["target_day_of_week"] = local_targets.dayofweek
        x["target_month"] = local_targets.month
        x["target_is_weekend"] = (local_targets.dayofweek >= 5).astype(int)
        x["target_doy_sin"] = np.sin(
            2.0 * np.pi * local_targets.dayofyear / 365.25
        )
        x["target_doy_cos"] = np.cos(
            2.0 * np.pi * local_targets.dayofyear / 365.25
        )

        target_forecasts = self.forecast_frame.reindex(targets)
        for column in target_forecasts:
            values = pd.to_numeric(
                target_forecasts[column], errors="coerce"
            ).to_numpy()
            if "day_ahead" in column:
                values = values.copy()
                values[unavailable_day_ahead] = np.nan
            x[f"fc_{column}"] = values
        self._forecast_interactions(x)
        self._rte_generation_forecast_features(x, origins)
        if self.include_curve_summaries:
            self._curve_summary_features(x, len(origins))
        self._availability_features(x, origins)

        present_state = [column for column in self.state_columns if column in frame]
        # ``to_hourly`` labels an average over [t, t+1h) at t.  At origin t,
        # the row labelled t contains a future half-hour observation.  Use the
        # last completely closed hourly bin instead.
        state_times = repeated_origins - pd.Timedelta(
            hours=self.hourly_state_lag_hours
        )
        origin_state = frame[present_state].reindex(state_times)
        for column in present_state:
            x[f"origin_{column}"] = pd.to_numeric(
                origin_state[column], errors="coerce"
            ).to_numpy()
        if self.include_multiscale_state:
            x = pd.concat([x, self._multiscale_state_features(frame, origins)], axis=1)

        # For h<=24, target-24h is never after the origin.  These aligned lags
        # preserve yesterday's intraday shape, unlike one origin lag broadcast
        # across all 24 horizons.
        for lag in self.target_lags_hours:
            lag_times = targets - pd.Timedelta(hours=lag)
            lagged = frame[present_state].reindex(lag_times)
            for column in present_state:
                values = pd.to_numeric(lagged[column], errors="coerce").to_numpy(
                    dtype=float
                )
                # For h=24, the nominal D-1 timestamp equals the issue origin;
                # its hour-start-labelled average is not known yet.
                values = values.copy()
                values[lag_times >= repeated_origins] = np.nan
                x[f"tgtlag{lag}_{column}"] = values

        meta = pd.DataFrame(
            {
                "origin": repeated_origins,
                "horizon": horizons,
                "target_time": targets,
            }
        )
        if supervised:
            meta["actual"] = pd.to_numeric(
                frame[CARBON].reindex(targets), errors="coerce"
            ).to_numpy()
            labels = fossil_regime_labels(
                frame, ccg_threshold_mw=500.0, peak_threshold_mw=2500.0
            )
            meta["regime"] = labels.reindex(targets).to_numpy()
            valid = meta[["actual", "regime"]].notna().all(axis=1)
            x = x.loc[valid].reset_index(drop=True)
            meta = meta.loc[valid].reset_index(drop=True)
        return x.replace([np.inf, -np.inf], np.nan), meta

    def _multiscale_state_features(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Attach causal historical level, trend and volatility summaries.

        Every statistic ends at the forecast origin.  The trend is the
        difference between the mean of the recent and preceding half-window,
        divided by their centre-to-centre distance.  It is more robust than an
        endpoint difference while preserving useful units per hour.

        Values are calculated once per daily origin and then repeated for its
        24 target horizons.  No target-hour observation participates in these
        features.
        """
        columns = [
            column
            for column in self.multiscale_state_columns
            if column in frame.columns
        ]
        if not columns:
            return pd.DataFrame(
                index=pd.RangeIndex(len(origins) * len(self.horizons))
            )

        history = frame[columns].copy()
        history.index = _utc(history.index)
        history = history.sort_index()
        complete_times = origins - pd.Timedelta(hours=self.hourly_state_lag_hours)
        features: dict[str, np.ndarray] = {}
        for column in columns:
            series = pd.to_numeric(history[column], errors="coerce")
            for window in self.multiscale_windows_hours:
                minimum = max(2, int(np.ceil(window / 2.0)))
                rolling = series.rolling(
                    f"{window}h", min_periods=minimum
                )
                level = rolling.mean()
                volatility = rolling.std(ddof=0)

                previous_width = window // 2
                recent_width = window - previous_width
                recent = series.rolling(
                    f"{recent_width}h",
                    min_periods=max(1, int(np.ceil(recent_width / 2.0))),
                ).mean()
                previous = series.shift(recent_width).rolling(
                    f"{previous_width}h",
                    min_periods=max(1, int(np.ceil(previous_width / 2.0))),
                ).mean()
                centre_distance = max(window / 2.0, 1.0)
                trend = (recent - previous) / centre_distance

                prefix = f"ms_{column}_w{window}h"
                features[f"{prefix}_level"] = np.repeat(
                    level.reindex(complete_times).to_numpy(dtype=float),
                    len(self.horizons),
                )
                features[f"{prefix}_trend_per_h"] = np.repeat(
                    trend.reindex(complete_times).to_numpy(dtype=float),
                    len(self.horizons),
                )
                features[f"{prefix}_volatility"] = np.repeat(
                    volatility.reindex(complete_times).to_numpy(dtype=float),
                    len(self.horizons),
                )
        return pd.DataFrame(
            features, index=pd.RangeIndex(len(origins) * len(self.horizons))
        )

    def _availability_features(
        self, x: pd.DataFrame, origins: pd.DatetimeIndex
    ) -> None:
        """Attach origin-versioned outage values in the stacked row order."""
        if self.availability_store is None:
            return
        by_horizon = self.availability_store.features_by_horizon(
            origins, list(self.horizons)
        )
        if not by_horizon:
            return
        columns = list(by_horizon[self.horizons[0]].columns)
        if self.availability_feature_mode == "delta":
            columns = [name for name in columns if "_delta_" in name]
        for column in columns:
            matrix = np.column_stack(
                [
                    pd.to_numeric(by_horizon[horizon][column], errors="coerce")
                    .reindex(origins)
                    .to_numpy()
                    for horizon in self.horizons
                ]
            )
            x[column] = matrix.reshape(-1)

    def _rte_generation_forecast_features(
        self, x: pd.DataFrame, origins: pd.DatetimeIndex
    ) -> None:
        """Attach publication-safe RTE D-1 wind/solar target forecasts.

        The feature store performs the important vintage selection: for every
        ``(origin, target)`` pair it returns only the latest RTE update whose
        publication timestamp is no later than the issue origin.  Keeping the
        store optional makes this an isolated ablation and leaves all existing
        model matrices byte-for-byte unchanged when it is absent.

        Energy-Charts load remains the denominator/backbone.  Two residual-load
        views are useful here: an all-RTE VRE curve, and a hybrid which retains
        the historically stronger Energy-Charts wind forecast while replacing
        only solar with RTE D-1.
        """
        if self.rte_forecast_store is None:
            return
        by_horizon = self.rte_forecast_store.features_by_horizon(
            origins, list(self.horizons)
        )
        if not by_horizon:
            return
        first = by_horizon.get(self.horizons[0])
        if first is None:
            return
        for column in first.columns:
            matrix = np.column_stack(
                [
                    pd.to_numeric(by_horizon[horizon][column], errors="coerce")
                    .reindex(origins)
                    .to_numpy(dtype=float)
                    for horizon in self.horizons
                ]
            )
            x[column] = matrix.reshape(-1)

        load = next(
            (
                x[name]
                for name in (
                    "fc_load_day_ahead_forecast_mw",
                    "fc_consumption_forecast_mw",
                )
                if name in x
            ),
            None,
        )
        rte_vre = x.get("rte_tgt_variable_renewables_d1_mw")
        if load is not None and rte_vre is not None:
            x["rte_tgt_residual_load_d1_mw"] = load - rte_vre
            x["rte_tgt_variable_renewables_d1_share"] = (
                rte_vre / load.where(load > 0.0)
            )

        rte_solar = x.get("rte_tgt_solar_d1_mw")
        ec_wind = x.get("fc_wind_day_ahead_mw")
        if rte_solar is not None and ec_wind is not None:
            hybrid_vre = ec_wind + rte_solar
            x["rte_hybrid_tgt_variable_renewables_d1_mw"] = hybrid_vre
            if load is not None:
                x["rte_hybrid_tgt_residual_load_d1_mw"] = load - hybrid_vre
                x["rte_hybrid_tgt_variable_renewables_d1_share"] = (
                    hybrid_vre / load.where(load > 0.0)
                )

        # Explicit disagreement features let a tree trust the better source by
        # season/horizon instead of forcing a global replacement.
        ec_solar = x.get("fc_solar_day_ahead_forecast_mw")
        if rte_solar is not None and ec_solar is not None:
            x["rte_minus_ec_solar_day_ahead_mw"] = rte_solar - ec_solar
        rte_wind = x.get("rte_tgt_wind_d1_mw")
        if rte_wind is not None and ec_wind is not None:
            x["rte_minus_ec_wind_day_ahead_mw"] = rte_wind - ec_wind

    @staticmethod
    def _forecast_interactions(x: pd.DataFrame) -> None:
        load_names = (
            "fc_load_day_ahead_forecast_mw",
            "fc_consumption_forecast_mw",
        )
        load = next((x[name] for name in load_names if name in x), None)
        wind_columns = [
            name
            for name in (
                "fc_wind_onshore_day_ahead_forecast_mw",
                "fc_wind_offshore_day_ahead_forecast_mw",
            )
            if name in x
        ]
        solar = x.get("fc_solar_day_ahead_forecast_mw")
        if not wind_columns:
            return
        wind = x[wind_columns].sum(axis=1, min_count=1)
        x["fc_wind_day_ahead_mw"] = wind
        if solar is None:
            return
        vre = wind + solar
        x["fc_variable_renewables_mw"] = vre
        if load is not None:
            x["fc_residual_load_mw"] = load - vre
            x["fc_variable_renewables_share"] = vre / load.where(load > 0.0)

    def _curve_summary_features(self, x: pd.DataFrame, n_origins: int) -> None:
        """Summarize each known day-ahead curve and make its ramps explicit.

        Trees can in principle infer these relationships, but spelling them out
        makes the daily level and the evening thermal ramp much easier to learn
        with the intentionally small experts used here.
        """
        preferred = [
            "fc_load_day_ahead_forecast_mw",
            "fc_consumption_forecast_mw",
            "fc_solar_day_ahead_forecast_mw",
            "fc_wind_day_ahead_mw",
            "fc_residual_load_mw",
            "fc_day_ahead_price_eur_mwh",
            "rte_tgt_solar_d1_mw",
            "rte_tgt_wind_d1_mw",
            "rte_tgt_variable_renewables_d1_mw",
            "rte_tgt_residual_load_d1_mw",
            "rte_hybrid_tgt_variable_renewables_d1_mw",
            "rte_hybrid_tgt_residual_load_d1_mw",
        ]
        # Neighbour prices and France-minus-neighbour spreads describe the
        # same known auction curve as the French price.  Include them
        # generically so adding a new bidding zone does not require changing
        # the carbon model again.
        preferred.extend(
            sorted(
                column
                for column in x
                if column.startswith("fc_day_ahead_price")
                and column not in preferred
            )
        )
        n_horizons = len(self.horizons)
        if len(x) != n_origins * n_horizons:
            raise ValueError("forecast matrix does not match origins x horizons")
        for column in preferred:
            if column not in x:
                continue
            matrix = x[column].to_numpy(dtype=float).reshape(n_origins, n_horizons)
            finite = np.isfinite(matrix)
            count = finite.sum(axis=1)
            total = np.nansum(matrix, axis=1)
            mean = np.divide(
                total,
                count,
                out=np.full(n_origins, np.nan, dtype=float),
                where=count > 0,
            )
            minimum = np.min(np.where(finite, matrix, np.inf), axis=1)
            maximum = np.max(np.where(finite, matrix, -np.inf), axis=1)
            minimum[~np.isfinite(minimum)] = np.nan
            maximum[~np.isfinite(maximum)] = np.nan
            ramp = np.diff(matrix, axis=1, prepend=matrix[:, :1])
            x[f"{column}_ramp_1h"] = ramp.reshape(-1)
            for statistic, values in (
                ("day_mean", mean),
                ("day_min", minimum),
                ("day_max", maximum),
                ("day_range", maximum - minimum),
            ):
                x[f"{column}_{statistic}"] = np.repeat(values, n_horizons)

            # France's hardest errors cluster around the evening fossil ramp.
            evening = [
                index
                for index, horizon in enumerate(self.horizons)
                if 17 <= horizon <= 21
            ]
            if evening:
                evening_matrix = matrix[:, evening]
                evening_finite = np.isfinite(evening_matrix)
                evening_maximum = np.max(
                    np.where(evening_finite, evening_matrix, -np.inf), axis=1
                )
                evening_maximum[~np.isfinite(evening_maximum)] = np.nan
                x[f"{column}_h17_21_ramp"] = np.repeat(
                    evening_matrix[:, -1] - evening_matrix[:, 0], n_horizons
                )
                x[f"{column}_h17_21_max"] = np.repeat(
                    evening_maximum, n_horizons
                )


class DirectRegimeMoE:
    """LightGBM probabilistic regime classifier plus three direct experts."""

    def __init__(
        self,
        *,
        point_scale: float = 1.0,
        inverse_level_floor: float = 8.0,
        classifier_params: dict | None = None,
        expert_params: dict | None = None,
        random_state: int = 42,
    ) -> None:
        self.point_scale = float(point_scale)
        self.inverse_level_floor = float(inverse_level_floor)
        self.classifier_params = classifier_params or {}
        self.expert_params = expert_params or {}
        self.random_state = int(random_state)
        self.feature_names_: list[str] = []
        self.classifier_: object | None = None
        self.experts_: dict[int, object] = {}

    @staticmethod
    def _lightgbm():
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "DirectRegimeMoE requires the optional 'ensemble' dependencies "
                "(pip install -e '.[ensemble]')."
            ) from exc
        return LGBMClassifier, LGBMRegressor

    def fit(self, x: pd.DataFrame, meta: pd.DataFrame) -> DirectRegimeMoE:
        LGBMClassifier, LGBMRegressor = self._lightgbm()
        self.feature_names_ = list(x.columns)
        regime = meta["regime"].astype(int).to_numpy()
        actual = meta["actual"].to_numpy(dtype=float)
        if set(np.unique(regime)) != {0, 1, 2}:
            raise ValueError("DirectRegimeMoE training data must contain all 3 regimes")

        classifier_defaults = {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 30,
            "reg_lambda": 5.0,
            "verbosity": -1,
            "n_jobs": 1,
            "class_weight": "balanced",
        }
        classifier_defaults.update(self.classifier_params)
        self.classifier_ = LGBMClassifier(
            random_state=self.random_state, **classifier_defaults
        )
        self.classifier_.fit(x, regime)

        expert_defaults = {
            "objective": "regression_l1",
            "n_estimators": 250,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 20,
            "reg_lambda": 5.0,
            "verbosity": -1,
            "n_jobs": 1,
        }
        expert_defaults.update(self.expert_params)
        self.experts_ = {}
        for label in (0, 1, 2):
            mask = regime == label
            expert = LGBMRegressor(
                random_state=self.random_state + 10 + label,
                **expert_defaults,
            )
            expert.fit(
                x.loc[mask],
                actual[mask],
                sample_weight=1.0
                / np.clip(actual[mask], self.inverse_level_floor, None),
            )
            self.experts_[label] = expert
        return self

    def predict_matrix(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.classifier_ is None or len(self.experts_) != 3:
            raise RuntimeError("DirectRegimeMoE.predict_matrix called before fit")
        x = x.reindex(columns=self.feature_names_)
        raw_probability = self.classifier_.predict_proba(x)
        probability = np.zeros((len(x), 3), dtype=float)
        for position, label in enumerate(self.classifier_.classes_):
            probability[:, int(label)] = raw_probability[:, position]
        expert_prediction = np.column_stack(
            [self.experts_[label].predict(x) for label in (0, 1, 2)]
        )
        prediction = self.point_scale * np.sum(
            probability * expert_prediction, axis=1
        )
        out = pd.DataFrame(
            {
                "prediction": np.clip(prediction, 0.0, None),
                "prob_baseload": probability[:, 0],
                "prob_ccg": probability[:, 1],
                "prob_peak": probability[:, 2],
                "regime_prediction": np.argmax(probability, axis=1),
            },
            index=x.index,
        )
        for label, name in ((0, "baseload"), (1, "ccg"), (2, "peak")):
            out[f"expert_{name}"] = expert_prediction[:, label]
        return out

    def predict(self, x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        out = meta.reset_index(drop=True).copy()
        values = self.predict_matrix(x.reset_index(drop=True))
        for column in values:
            out[column] = values[column].to_numpy()
        out["model"] = "direct_regime_moe"
        return out


def select_mape_scale(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    grid: Sequence[float] = tuple(np.arange(0.70, 1.301, 0.01)),
) -> tuple[float, float]:
    """Return ``(scale, MAPE_percent)`` selected on a prior calibration set."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scales = np.asarray(grid, dtype=float)
    losses = np.mean(
        np.abs(scales[:, None] * prediction[None, :] - actual[None, :])
        / np.clip(np.abs(actual)[None, :], 1e-9, None),
        axis=1,
    )
    position = int(np.argmin(losses))
    return float(scales[position]), float(100.0 * losses[position])


__all__ = [
    "DETAILED_STATE_COLUMNS",
    "DEFAULT_HORIZONS",
    "DEFAULT_MULTISCALE_STATE_COLUMNS",
    "DEFAULT_MULTISCALE_WINDOWS_HOURS",
    "DEFAULT_STATE_COLUMNS",
    "DirectRegimeMoE",
    "RegimeMoEFeatureBuilder",
    "select_mape_scale",
]
