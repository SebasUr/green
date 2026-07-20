"""``greenctl`` - command line interface for the Green Window Observatory.

Carbon track (Milestones 0-2):

    greenctl carbon import   --output data/cache/carbon_fr_hourly.parquet
    greenctl carbon train    --config carbon_model --test-start 2026-02-01
    greenctl carbon forecast --horizon-hours 48
    greenctl carbon compare  --test-start 2026-02-01
    greenctl windows analyze --horizon-hours 48
    greenctl jobs observe    --selector sustainability.cern.ch/track=true
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from green_observatory import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Green Window Observatory (V1.0) - carbon when-to-run intelligence.",
)
carbon_app = typer.Typer(no_args_is_help=True, help="Carbon signal: import, train, forecast, compare.")
windows_app = typer.Typer(no_args_is_help=True, help="Green windows: compute low-carbon windows.")
jobs_app = typer.Typer(
    no_args_is_help=True,
    help="Observe labelled Kubernetes Jobs and account Kepler/RTE emissions.",
)
app.add_typer(carbon_app, name="carbon")
app.add_typer(windows_app, name="windows")
app.add_typer(jobs_app, name="jobs")

DEFAULT_SNAPSHOT = "data/cache/carbon_fr_hourly.parquet"
DEFAULT_ENRICHED_SNAPSHOT = "data/cache/carbon_fr_hourly_enriched.parquet"
DEFAULT_MODEL = "models/project_carbon_hgb.joblib"
DEFAULT_PHYSICAL_MODEL = "models/physical_carbon_hgb.joblib"
DEFAULT_RANKING_MODEL = "models/ranking_carbon_hgb.joblib"
DEFAULT_ENSEMBLE_CI_MODEL = "models/ensemble_ci.joblib"
DEFAULT_FOSSIL_REGIME_MODEL = "models/fossil_regime_france24.joblib"
DEFAULT_WEATHER = "data/cache/weather_fr_hourly.parquet"
DEFAULT_CONSUMPTION = "data/cache/consumption_forecast_fr_hourly.parquet"
DEFAULT_MIX_FORECAST = "data/cache/mix_day_ahead_fr_hourly.parquet"
DEFAULT_SYSTEM_FORECAST = "data/cache/system_day_ahead_fr_hourly.parquet"
DEFAULT_RTE_UNAVAILABILITY = "data/cache/rte_unavailability_messages.parquet"
DEFAULT_RTE_GENERATION_FORECAST = "data/cache/rte_generation_forecast.parquet"


def _forecast_frame(*paths: str | None):
    """Join available forecast-feature snapshots (wind/solar + consumption) or None."""
    parts = []
    for path in paths:
        if path and Path(path).exists():
            frame = pd.read_parquet(path)
            if frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            parts.append(frame)
    if not parts:
        return None
    out = parts[0]
    for extra in parts[1:]:
        out = out.join(extra, how="outer")
    return out.sort_index()


def _utc(ts: str | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _dump_json(obj, output: str | None, label: str) -> None:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(obj, indent=2, default=str))
        typer.echo(f"wrote {label} -> {output}")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def figures(
    output: str = typer.Option("runs/figures", help="Output directory for the PNG figures."),
    examples: int = typer.Option(0, help="Extra random-day forecast-example figures (Fig 3)."),
    seed: int = typer.Option(0, help="Random seed for the example days."),
    month: str = typer.Option(
        None, help="Also make rolling-24h figures from this start date, e.g. 2026-03-01."
    ),
    spans: str = typer.Option("28,15,7", help="Rolling spans in days (with --month)."),
    methods: str = typer.Option(
        "horizon,historical,hybrid,fixed", help="Window methods to render (subfolder each)."
    ),
) -> None:
    """Generate model-quality figures: decision, error-by-horizon, forecast example, calibration."""
    import os

    from green_observatory.exporters.plots import apply_style, generate, month_rolling_figure

    for path in generate(output, n_random_examples=examples, seed=seed):
        typer.echo(f"wrote {path}")

    if month:
        from green_observatory.carbon.model import ProjectCarbonModel
        from green_observatory.config import load_named
        from green_observatory.providers.carbon_base import CARBON
        from green_observatory.providers.carbon_odre import OdreCarbonProvider

        apply_style()
        df = OdreCarbonProvider.load_snapshot(DEFAULT_SNAPSHOT)
        dense = "models/project_carbon_hgb_forecast_dense24.joblib"
        model = ProjectCarbonModel.load(dense if os.path.exists(dense) else DEFAULT_MODEL)
        wcfg = load_named("window_scoring")
        ms = pd.Timestamp(month).tz_localize("UTC")
        hist = df[CARBON].loc[df.index < ms].to_numpy()
        if hist.size == 0:
            hist = df[CARBON].to_numpy()

        # One subfolder per window method; a 28/15/7-day rolling figure in each.
        single = {"enter_percentile": None, "exit_percentile": None, "percentile": 0.33}
        method_defs = {
            "horizon": dict(window_reference=None, window_overrides=single,
                            method_label="horizon-relative"),
            "historical": dict(window_reference=hist, window_overrides=single,
                               method_label="historical single-threshold"),
            "hybrid": dict(window_reference=hist, window_overrides=None,
                           method_label="historical + hysteresis"),
            "fixed": dict(
                window_reference=None,
                window_overrides={
                    "enter_gco2": wcfg.get("windows", {}).get("enter_gco2") or 20.0,
                    "exit_gco2": wcfg.get("windows", {}).get("exit_gco2") or 40.0,
                },
                method_label="fixed threshold + hysteresis",
            ),
        }
        span_list = [int(s) for s in spans.split(",")]
        for name in [x.strip() for x in methods.split(",")]:
            if name not in method_defs:
                typer.echo(f"skip unknown method: {name}")
                continue
            subdir = Path(output) / name
            subdir.mkdir(parents=True, exist_ok=True)
            for span in span_list:
                out = str(subdir / f"rolling_{month[:10]}_{span}d.png")
                wape = month_rolling_figure(df, model, month, out, days=span, wcfg=wcfg,
                                            **method_defs[name])
                typer.echo(f"wrote {out} (WAPE {wape:.0f}%)")


# --------------------------------------------------------------------------- #
# carbon import
# --------------------------------------------------------------------------- #
@carbon_app.command("import")
def carbon_import(
    output: str = typer.Option(DEFAULT_SNAPSHOT, help="Output parquet snapshot path."),
    start: str = typer.Option(None, help="Start (UTC, e.g. 2021-01-01). Default: ~5y ago."),
    end: str = typer.Option(None, help="End (UTC). Default: now."),
    realtime: bool = typer.Option(False, help="Use the near-real-time dataset instead of history."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
) -> None:
    """Fetch ODRE/eCO2mix carbon data and cache a canonical hourly snapshot."""
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    provider = OdreCarbonProvider.from_config(load_named(config))
    if realtime:
        df = provider.import_realtime()
    else:
        end_ts = _utc(end) or pd.Timestamp.now(tz="UTC")
        start_ts = _utc(start) or (end_ts - pd.Timedelta(days=365 * 5))
        typer.echo(f"fetching {start_ts.date()} .. {end_ts.date()} from ODRE ...")
        df = provider.import_history(start_ts, end_ts, progress=True)
    OdreCarbonProvider.save_snapshot(df, output)
    c = df["carbon_intensity_gco2_kwh"]
    typer.echo(
        f"saved {len(df)} hourly rows -> {output}  "
        f"[{df.index.min().date()}..{df.index.max().date()}] "
        f"carbon mean={c.mean():.1f} min={c.min():.0f} max={c.max():.0f} gCO2/kWh"
    )


# --------------------------------------------------------------------------- #
# carbon train
# --------------------------------------------------------------------------- #
@carbon_app.command("fetch-forecast")
def carbon_fetch_forecast(
    start: str = typer.Option(None, help="Start (UTC). Default: ~5y ago."),
    end: str = typer.Option(None, help="End (UTC). Default: now."),
    weather_out: str = typer.Option(DEFAULT_WEATHER, help="Weather snapshot output."),
    consumption_out: str = typer.Option(DEFAULT_CONSUMPTION, help="Consumption-forecast output."),
) -> None:
    """Fetch real-time-obtainable forecast features: Open-Meteo wind/solar + eco2mix prevision_j1."""
    from green_observatory.providers.carbon_odre import OdreCarbonProvider
    from green_observatory.providers.weather_openmeteo import WeatherProvider

    end_ts = _utc(end) or pd.Timestamp.now(tz="UTC")
    start_ts = _utc(start) or (end_ts - pd.Timedelta(days=365 * 5))
    typer.echo(f"fetching Open-Meteo wind/solar {start_ts.date()}..{end_ts.date()} ...")
    wx = WeatherProvider().fetch_archive(start_ts, end_ts, progress=True)
    WeatherProvider.save_snapshot(wx, weather_out)
    typer.echo(f"  weather -> {weather_out} ({len(wx)} rows)")
    typer.echo("fetching eco2mix prevision_j1 (day-ahead consumption) ...")
    cons = OdreCarbonProvider().import_consumption_forecast(start_ts, end_ts)
    Path(consumption_out).parent.mkdir(parents=True, exist_ok=True)
    cons.to_parquet(consumption_out)
    typer.echo(f"  consumption forecast -> {consumption_out} ({len(cons)} rows)")


@carbon_app.command("fetch-mix-forecast")
def carbon_fetch_mix_forecast(
    start: str = typer.Option("2021-07-11", help="Forecast target-period start (UTC)."),
    end: str = typer.Option(None, help="Forecast target-period end (UTC)."),
    output: str = typer.Option(DEFAULT_MIX_FORECAST, help="Output parquet snapshot."),
    production_types: str = typer.Option(
        "wind_onshore,wind_offshore,solar,load",
        help="Comma-separated Energy-Charts forecast types.",
    ),
) -> None:
    """Fetch public historical day-ahead wind/solar/load forecasts for France."""
    from green_observatory.providers.mix_forecast_energy_charts import (
        EnergyChartsMixForecastProvider,
    )

    start_ts = _utc(start)
    end_ts = _utc(end) or pd.Timestamp.now(tz="UTC")
    kinds = tuple(kind.strip() for kind in production_types.split(",") if kind.strip())
    typer.echo(
        f"fetching Energy-Charts day-ahead {list(kinds)} "
        f"for {start_ts.date()}..{end_ts.date()} ..."
    )
    provider = EnergyChartsMixForecastProvider()
    frame = provider.fetch(start_ts, end_ts, production_types=kinds, progress=True)
    provider.save_snapshot(frame, output)
    coverage = {column: round(float(frame[column].notna().mean()), 3) for column in frame}
    typer.echo(f"saved {len(frame)} hourly rows -> {output}; coverage={coverage}")


@carbon_app.command("fetch-rte-system")
def carbon_fetch_rte_system(
    start: str = typer.Option("2021-07-11", help="Historical publication/target start."),
    end: str = typer.Option(None, help="Historical publication/target end."),
    unavailability_output: str = typer.Option(
        DEFAULT_RTE_UNAVAILABILITY, help="Versioned RTE unavailability parquet."
    ),
    forecast_output: str = typer.Option(
        DEFAULT_RTE_GENERATION_FORECAST, help="RTE generation-forecast parquet."
    ),
    dotenv: str = typer.Option(
        ".env", help="Ignored local file containing RTE_CLIENT_ID/SECRET."
    ),
    unavailability: bool = typer.Option(
        True, help="Fetch versioned generation-unavailability messages."
    ),
    generation_forecast: bool = typer.Option(
        True, help="Fetch D-3/D-2/D-1/intraday generation forecasts."
    ),
) -> None:
    """Fetch causal, publication-versioned French system inputs from RTE."""
    from green_observatory.providers.rte_system_forecast import (
        RteSystemForecastProvider,
    )

    start_ts = _utc(start)
    end_ts = _utc(end) or pd.Timestamp.now(tz="UTC")
    provider = RteSystemForecastProvider.from_env(dotenv_path=dotenv)
    if unavailability:
        typer.echo(
            f"fetching RTE publication-versioned unavailability "
            f"{start_ts.date()}..{end_ts.date()} ..."
        )
        unavailability_frame = provider.fetch_unavailability(
            start_ts, end_ts, chunk_days=7, progress=True
        )
        provider.save_snapshot(unavailability_frame, unavailability_output)
        typer.echo(
            f"saved {len(unavailability_frame)} unavailability intervals -> "
            f"{unavailability_output}"
        )
    if generation_forecast:
        typer.echo("fetching RTE D-1 generation forecasts ...")
        forecasts = provider.fetch_generation_forecast(
            start_ts, end_ts, forecast_type="D-1", chunk_days=7, progress=True
        )
        provider.save_snapshot(forecasts, forecast_output)
        typer.echo(
            f"saved {len(forecasts)} forecast values -> {forecast_output}"
        )


@carbon_app.command("train")
def carbon_train(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    output: str = typer.Option(DEFAULT_MODEL, help="Output model path (joblib)."),
    test_start: str = typer.Option(None, help="Hold out data on/after this date from training."),
    forecast_features: bool = typer.Option(
        True, help="Use wind/solar/consumption forecast features if snapshots exist."
    ),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather snapshot (Open-Meteo)."),
    consumption_forecast: str = typer.Option(DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."),
) -> None:
    """Fit climatology + the project carbon model and save the model.

    ``--test-start D`` trains only on data before D (the rest is held out for
    honest evaluation); omit it to train on ALL data for the best deployable model.
    """
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.model import train_project_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts] if ts is not None else df
    ff = _forecast_frame(weather, consumption_forecast) if forecast_features else None
    span = f"{train.index.min().date()}..{train.index.max().date()}"
    typer.echo(f"training on {len(train)} rows ({span}); forecast features: "
               f"{list(ff.columns) if ff is not None else 'none'}")
    clim = climatology_from_config(train, cfg)
    model = train_project_model(train, cfg, climatology=clim, forecast_frame=ff)
    model.save(output)
    typer.echo(f"saved model ({model.algorithm}, horizons {list(model.horizons)}) -> {output}")


@carbon_app.command("train-physical")
def carbon_train_physical(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    output: str = typer.Option(DEFAULT_PHYSICAL_MODEL, help="Output Phase-C model path."),
    test_start: str = typer.Option(None, help="Hold out data on/after this date."),
    forecast_features: bool = typer.Option(True, help="Use available target forecasts."),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather snapshot."),
    consumption_forecast: str = typer.Option(
        DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."
    ),
) -> None:
    """Fit source-share forecasts + physical map + out-of-sample residual stage."""
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.physical import train_physical_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts] if ts is not None else df
    ff = _forecast_frame(weather, consumption_forecast) if forecast_features else None
    clim = climatology_from_config(train, cfg)
    typer.echo(
        f"training Phase-C model on {len(train)} rows; "
        f"forecast features: {list(ff.columns) if ff is not None else 'none'}"
    )
    model = train_physical_model(train, cfg, climatology=clim, forecast_frame=ff)
    model.save(output)
    typer.echo(
        "physical map: intercept="
        f"{model.mapper.intercept_:.3f}, coefficients={model.mapper.coefficients_}"
    )
    typer.echo(f"saved Phase-C model, horizons {list(model.horizons)} -> {output}")


@carbon_app.command("train-ranking")
def carbon_train_ranking(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    output: str = typer.Option(DEFAULT_RANKING_MODEL, help="Output Phase-D model path."),
    test_start: str = typer.Option(None, help="Hold out data on/after this date."),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather forecast snapshot."),
    consumption_forecast: str = typer.Option(
        DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."
    ),
    mix_forecast: str = typer.Option(
        DEFAULT_MIX_FORECAST, help="Day-ahead generation-forecast snapshot."
    ),
    system_forecast: str = typer.Option(
        DEFAULT_SYSTEM_FORECAST,
        help="Optional origin-safe nuclear/hydro/thermal forecast snapshot.",
    ),
) -> None:
    """Fit the opt-in exogenous point model and regret-weighted ranker."""
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.ranking import train_ranking_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts] if ts is not None else df
    forecast = _forecast_frame(
        weather, consumption_forecast, mix_forecast, system_forecast
    )
    clim = climatology_from_config(train, cfg)
    typer.echo(
        f"training Phase-D on {len(train)} rows; exogenous columns="
        f"{list(forecast.columns) if forecast is not None else []}"
    )
    model = train_ranking_model(
        train, cfg, climatology=clim, forecast_frame=forecast
    )
    model.save(output)
    typer.echo(
        f"saved Phase-D model -> {output}; calibration="
        f"{model.calibration_origins_} origins/{model.calibration_pairs_} pairs"
    )


@carbon_app.command("train-ensemble-ci")
def carbon_train_ensemble_ci(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    output: str = typer.Option(DEFAULT_ENSEMBLE_CI_MODEL, help="Output model path."),
    test_start: str = typer.Option(None, help="Hold out data on/after this date."),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather forecast snapshot."),
    consumption_forecast: str = typer.Option(
        DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."
    ),
    mix_forecast: str = typer.Option(
        DEFAULT_MIX_FORECAST, help="Day-ahead generation-forecast snapshot."
    ),
) -> None:
    """Fit the opt-in two-layer EnsembleCI adaptation."""
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.ensemble_ci import train_ensemble_ci_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts] if ts is not None else df
    forecast = _forecast_frame(weather, consumption_forecast, mix_forecast)
    clim = climatology_from_config(train, cfg)
    typer.echo(
        f"training EnsembleCI adaptation on {len(train)} rows; "
        f"forecast columns={list(forecast.columns) if forecast is not None else []}"
    )
    model = train_ensemble_ci_model(
        train, cfg, climatology=clim, forecast_frame=forecast
    )
    model.save(output)
    typer.echo(
        f"saved EnsembleCI adaptation -> {output}; backends={model.backends_}; "
        f"weights={model.weights_}"
    )


@carbon_app.command("train-fossil-regime")
def carbon_train_fossil_regime(
    carbon: str = typer.Option(
        DEFAULT_ENRICHED_SNAPSHOT,
        help="Enriched French carbon/mix snapshot.",
    ),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    output: str = typer.Option(
        DEFAULT_FOSSIL_REGIME_MODEL, help="Output France-24 regime model path."
    ),
    test_start: str = typer.Option(None, help="Hold out data on/after this date."),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather forecast snapshot."),
    consumption_forecast: str = typer.Option(
        DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."
    ),
    mix_forecast: str = typer.Option(
        DEFAULT_MIX_FORECAST, help="Day-ahead generation-forecast snapshot."
    ),
    system_forecast: str = typer.Option(
        DEFAULT_SYSTEM_FORECAST,
        help="Optional origin-safe nuclear/hydro/thermal forecast snapshot.",
    ),
    rte_unavailability: str = typer.Option(
        None, help="Optional versioned RTE unavailability snapshot (opt-in)."
    ),
    rte_generation_forecast: str = typer.Option(
        DEFAULT_RTE_GENERATION_FORECAST,
        help="Optional publication-timestamped RTE D-1 forecast snapshot.",
    ),
    rte_generation_sources: str = typer.Option(
        "WIND_ONSHORE,WIND_OFFSHORE,SOLAR",
        help="Comma-separated RTE D-1 renewable sources to use.",
    ),
) -> None:
    """Train the deployable dense French fossil-regime expert and ranker."""
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.fossil_regime import train_fossil_regime_model
    from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
    from green_observatory.carbon.rte_forecast_features import (
        RteGenerationForecastFeatureStore,
    )
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts] if ts is not None else df
    forecast = _forecast_frame(
        weather, consumption_forecast, mix_forecast, system_forecast
    )
    if forecast is None:
        raise typer.BadParameter(
            "fossil-regime training requires target-time forecast snapshots"
        )
    climatology = climatology_from_config(train, cfg)
    availability_store = (
        RteAvailabilityFeatureStore.from_parquet(rte_unavailability)
        if rte_unavailability and Path(rte_unavailability).exists()
        else None
    )
    rte_forecast_store = (
        RteGenerationForecastFeatureStore.from_parquet(
            rte_generation_forecast,
            production_types=tuple(
                source.strip()
                for source in rte_generation_sources.split(",")
                if source.strip()
            ),
        )
        if rte_generation_forecast and Path(rte_generation_forecast).exists()
        else None
    )
    typer.echo(
        f"training dense fossil-regime expert on {len(train)} rows; "
        f"forecast columns={list(forecast.columns)}"
    )
    model = train_fossil_regime_model(
        train,
        cfg,
        climatology=climatology,
        forecast_frame=forecast,
        availability_store=availability_store,
        rte_forecast_store=rte_forecast_store,
    )
    model.save(output)
    typer.echo(
        f"saved fossil-regime model -> {output}; "
        f"validation MAPE={model.validation_mape_:.2f}%; "
        f"direct/ranked regret={model.validation_regret_:.3f}/"
        f"{model.validation_ranked_regret_:.3f}; "
        f"risk/ranking weights={model.risk_weight_:.2f}/{model.ranking_weight_:.2f}"
    )


# --------------------------------------------------------------------------- #
# carbon forecast
# --------------------------------------------------------------------------- #
@carbon_app.command("forecast")
def carbon_forecast(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    horizon_hours: int = typer.Option(24, help="Forecast horizon in hours (24h primary; 48h secondary)."),
    strategy: str = typer.Option(
        "corrected", help="Dense forecaster: corrected | climatology (project is sparse; see compare)."
    ),
    at: str = typer.Option(None, help="Forecast origin (UTC). Default: last snapshot timestamp."),
    output: str = typer.Option(None, help="Optional JSON output for forecast + windows."),
) -> None:
    """Forecast the next hours and emit predicted low-carbon windows."""
    from green_observatory.carbon.climatology import ClimatologyForecaster, climatology_from_config
    from green_observatory.carbon.corrected_climatology import corrected_from_config
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_base import CARBON
    from green_observatory.providers.carbon_odre import OdreCarbonProvider
    from green_observatory.windows.scoring import low_carbon_windows_from_config

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    wcfg = load_named("window_scoring")
    origin = _utc(at) or df.index.max()
    history = df.loc[df.index < origin]

    clim = climatology_from_config(history, cfg)  # fit as-of origin (no leakage)
    forecaster = (
        ClimatologyForecaster(clim)
        if strategy == "climatology"
        else corrected_from_config(clim, cfg)
    )
    horizons = list(range(1, horizon_hours + 1))
    pred = forecaster.predict(history, origin, horizons)
    series = pd.Series(pred["prediction"].to_numpy(), index=pred.index, name=CARBON)

    from green_observatory.models import WindowType

    wins = low_carbon_windows_from_config(
        series, wcfg, window_type=WindowType.predicted_low_carbon_window,
        source_model=forecaster.name, issued_at=origin.to_pydatetime(),
    )
    typer.echo(
        f"origin {origin.isoformat()}  strategy={strategy}  horizon={horizon_hours}h\n"
        f"forecast gCO2/kWh: min={series.min():.0f} median={series.median():.0f} max={series.max():.0f}"
    )
    typer.echo(f"\n{len(wins)} predicted low-carbon window(s):")
    for w in wins:
        typer.echo(
            f"  #{w.rank} {w.start:%m-%d %H:%M}->{w.end:%m-%d %H:%M}Z  "
            f"score={w.carbon_score:.2f}  mean={w.mean_carbon_intensity_gco2_kwh:.0f} gCO2/kWh  "
            f"conf={w.confidence:.2f}"
        )
    _dump_json(
        {"origin": origin.isoformat(), "strategy": strategy,
         "windows": [w.model_dump(mode="json") for w in wins]},
        output, "forecast",
    )


# --------------------------------------------------------------------------- #
# carbon compare
# --------------------------------------------------------------------------- #
@carbon_app.command("compare")
def carbon_compare(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    test_start: str = typer.Option("2026-02-01", help="Backtest test-period start (UTC)."),
    stride_hours: int = typer.Option(6, help="Advance the forecast origin every N hours."),
    strategies: str = typer.Option(
        "persistence,climatology,corrected,project", help="Comma-separated strategies."
    ),
    output: str = typer.Option(None, help="Optional JSON output for the metric tables."),
    forecast_features: bool = typer.Option(
        True, help="Give the project model wind/solar/consumption forecast features."
    ),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather snapshot (Open-Meteo)."),
    consumption_forecast: str = typer.Option(DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."),
    mix_forecast: str = typer.Option(
        DEFAULT_MIX_FORECAST, help="Optional day-ahead generation-forecast snapshot."
    ),
) -> None:
    """Rolling-origin backtest: MAE + green-window selection vs perfect foresight."""
    from green_observatory.carbon import evaluation as ev
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.ensemble_ci import train_ensemble_ci_model
    from green_observatory.carbon.model import train_project_model
    from green_observatory.carbon.physical import train_physical_model
    from green_observatory.carbon.ranking import train_ranking_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_electricity_maps import ElectricityMapsProvider
    from green_observatory.providers.carbon_odre import OdreCarbonProvider
    from green_observatory.windows.oracle import window_selection_metrics

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    include = tuple(s.strip() for s in strategies.split(","))
    train = df.loc[df.index < ts]
    typer.echo(f"train<{ts.date()}={len(train)} rows; test>={ts.date()} (stride {stride_hours}h)")

    clim = climatology_from_config(train, cfg)
    ff = _forecast_frame(weather, consumption_forecast) if forecast_features else None
    project_requested = "project" in include or "physical_blend" in include
    project = (
        train_project_model(train, cfg, climatology=clim, forecast_frame=ff)
        if project_requested
        else None
    )
    physical_requested = any(
        name in include for name in ("physical", "physical_raw", "physical_blend")
    )
    physical = (
        train_physical_model(train, cfg, climatology=clim, forecast_frame=ff)
        if physical_requested
        else None
    )
    ranking_requested = any(
        name in include for name in ("exogenous_project", "ranking")
    )
    ranking_forecast = (
        _forecast_frame(weather, consumption_forecast, mix_forecast)
        if ranking_requested and forecast_features
        else None
    )
    ranking = (
        train_ranking_model(
            train, cfg, climatology=clim, forecast_frame=ranking_forecast
        )
        if ranking_requested
        else None
    )
    ensemble_requested = "ensemble_ci" in include
    ensemble_forecast = (
        _forecast_frame(weather, consumption_forecast, mix_forecast)
        if ensemble_requested and forecast_features
        else None
    )
    ensemble_ci = (
        train_ensemble_ci_model(
            train, cfg, climatology=clim, forecast_frame=ensemble_forecast
        )
        if ensemble_requested
        else None
    )
    if ff is not None:
        typer.echo(f"project model uses forecast features: {list(ff.columns)}")
    origins = ev.make_origins(df, ts, stride_hours=stride_hours)
    pred = ev.backtest_predictions(
        df, origins, climatology=clim, project_model=project,
        physical_model=physical, ranking_model=ranking,
        ensemble_ci_model=ensemble_ci,
        corrected_cfg=cfg.get("corrected_climatology"),
        include=include,
    )

    pm = ev.point_metrics(pred)
    mae = pm.pivot(index="model", columns="horizon", values="mae").round(2)
    wape = pm.pivot(index="model", columns="horizon", values="wape").round(1)
    sel = window_selection_metrics(pred, df)
    global_wape = (
        pred.groupby("model")
        .apply(lambda g: 100 * (g.prediction - g.actual).abs().sum() / g.actual.abs().sum(),
               include_groups=False)
        .round(1)
    )
    typer.echo(f"\n=== MAE (gCO2/kWh) by horizon  [{len(origins)} origins] ===")
    typer.echo(mae.to_string())
    typer.echo("\n=== WAPE (%) by horizon  [error as % of the real level] ===")
    typer.echo(wape.to_string())
    typer.echo("global WAPE (%):  "
               + "  ".join(f"{m}={v}" for m, v in global_wape.sort_values().items()))
    typer.echo("\n=== Green-window selection (pick greenest forecasted hour) ===")
    typer.echo(sel.to_string())

    em = ElectricityMapsProvider()
    typer.echo(
        "\nElectricity Maps: available (live consumption-based forecast comparison)."
        if em.available()
        else "\nElectricity Maps: no API key -> skipped (optional comparison)."
    )
    _dump_json(
        {"point_metrics": json.loads(pm.round(3).to_json(orient="records")),
         "global_wape_pct": {m: float(v) for m, v in global_wape.items()},
         "window_selection": json.loads(sel.reset_index().to_json(orient="records")),
         "physical_map": (
             {"intercept": physical.mapper.intercept_,
              "coefficients": physical.mapper.coefficients_,
              "residual_calibration_rows": physical.residual_calibration_rows_}
             if physical is not None else None
         ),
         "ranking_calibration": (
             {"origins": ranking.calibration_origins_,
              "pairs": ranking.calibration_pairs_,
              "ranking_weight": ranking.ranking_weight_,
              "validation_regret_by_weight": ranking.validation_regret_by_weight_,
              "candidate_features": ranking.candidate_feature_names_}
             if ranking is not None else None
         ),
         "ensemble_ci": (
             {"backends": ensemble_ci.backends_,
              "weights": ensemble_ci.weights_,
              "validation_mae": ensemble_ci.validation_mae_}
             if ensemble_ci is not None else None
         )},
        output, "metrics",
    )


# --------------------------------------------------------------------------- #
# carbon compare-france24 (isolated dense day-ahead experiment)
# --------------------------------------------------------------------------- #
@carbon_app.command("compare-france24")
def carbon_compare_france24(
    carbon: str = typer.Option(
        DEFAULT_ENRICHED_SNAPSHOT,
        help="Enriched French carbon/mix snapshot.",
    ),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    test_start: str = typer.Option("2026-02-01", help="Backtest test-period start (UTC)."),
    stride_hours: int = typer.Option(6, help="Advance the forecast origin every N hours."),
    output: str = typer.Option(
        "runs/compare_france24_metrics.json", help="JSON report output."
    ),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather snapshot."),
    consumption_forecast: str = typer.Option(
        DEFAULT_CONSUMPTION, help="Consumption-forecast snapshot."
    ),
    mix_forecast: str = typer.Option(
        DEFAULT_MIX_FORECAST, help="Day-ahead generation-forecast snapshot."
    ),
    system_forecast: str = typer.Option(
        DEFAULT_SYSTEM_FORECAST,
        help="Optional origin-safe nuclear/hydro/thermal forecast snapshot.",
    ),
    rte_unavailability: str = typer.Option(
        None, help="Optional versioned RTE unavailability snapshot (opt-in)."
    ),
    rte_generation_forecast: str = typer.Option(
        DEFAULT_RTE_GENERATION_FORECAST,
        help="Optional publication-timestamped RTE D-1 forecast snapshot.",
    ),
    rte_generation_sources: str = typer.Option(
        "WIND_ONSHORE,WIND_OFFSHORE,SOLAR",
        help="Comma-separated RTE D-1 renewable sources to use.",
    ),
    fossil_regime: bool = typer.Option(
        True, help="Train the probabilistic French CCG/TAC regime expert."
    ),
    controls: bool = typer.Option(
        True, help="Also train the dense project and France-24 control models."
    ),
) -> None:
    """Dense 1..24h French specialist: MAPE plus oracle decision quality."""
    import copy

    import numpy as np

    from green_observatory.carbon import evaluation as ev
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.france24 import (
        DENSE_DAY_AHEAD_HORIZONS,
        train_france24_model,
    )
    from green_observatory.carbon.fossil_regime import train_fossil_regime_model
    from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
    from green_observatory.carbon.rte_forecast_features import (
        RteGenerationForecastFeatureStore,
    )
    from green_observatory.carbon.model import train_project_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider
    from green_observatory.windows.oracle import window_selection_metrics

    if not controls and not fossil_regime:
        raise typer.BadParameter("enable at least one of controls or fossil-regime")

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts]
    forecasts = _forecast_frame(
        weather, consumption_forecast, mix_forecast, system_forecast
    )
    if forecasts is None:
        raise typer.BadParameter("France-24 requires target-time forecast snapshots")
    typer.echo(
        f"France-24 train<{ts.date()}={len(train)} rows; "
        f"dense horizons=1..24; stride={stride_hours}h"
    )
    typer.echo(f"forecast features: {list(forecasts.columns)}")

    climatology = climatology_from_config(train, cfg)
    availability_store = (
        RteAvailabilityFeatureStore.from_parquet(rte_unavailability)
        if rte_unavailability and Path(rte_unavailability).exists()
        else None
    )
    rte_forecast_store = (
        RteGenerationForecastFeatureStore.from_parquet(
            rte_generation_forecast,
            production_types=tuple(
                source.strip()
                for source in rte_generation_sources.split(",")
                if source.strip()
            ),
        )
        if rte_generation_forecast and Path(rte_generation_forecast).exists()
        else None
    )
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg.setdefault("model", {})["horizons_hours"] = list(
        DENSE_DAY_AHEAD_HORIZONS
    )
    baseline = None
    specialist = None
    if controls:
        typer.echo("training untouched dense project baseline ...")
        baseline = train_project_model(
            train, baseline_cfg, climatology=climatology, forecast_frame=forecasts
        )
        typer.echo("training modular France-24 specialist ...")
        specialist = train_france24_model(
            train, cfg, climatology=climatology, forecast_frame=forecasts
        )
    regime_model = None
    if fossil_regime:
        typer.echo("training shared probabilistic fossil-regime expert ...")
        regime_model = train_fossil_regime_model(
            train,
            cfg,
            climatology=climatology,
            forecast_frame=forecasts,
            availability_store=availability_store,
            rte_forecast_store=rte_forecast_store,
        )

    origins = ev.make_origins(
        df, ts, stride_hours=stride_hours, max_horizon=24
    )
    prediction_parts = []
    if baseline is not None and specialist is not None:
        baseline_predictions = ev._project_batch(
            baseline, df, origins, DENSE_DAY_AHEAD_HORIZONS
        ).assign(model="project_dense24")
        france = specialist.predict_batch(df, origins)
        point_predictions = france[
            ["origin", "horizon", "target_time", "point_prediction"]
        ].rename(columns={"point_prediction": "prediction"})
        point_predictions["model"] = "france24_point"
        decision_predictions = france[
            ["origin", "horizon", "target_time", "decision_prediction"]
        ].rename(columns={"decision_prediction": "prediction"})
        decision_predictions["model"] = "france24_decision"
        prediction_parts.extend(
            [baseline_predictions, point_predictions, decision_predictions]
        )
    if regime_model is not None:
        regime = regime_model.predict_batch(df, origins)
        regime_point = regime[
            ["origin", "horizon", "target_time", "point_prediction"]
        ].rename(columns={"point_prediction": "prediction"})
        regime_point["model"] = "fossil_regime_point"
        regime_decision = regime[
            ["origin", "horizon", "target_time", "decision_prediction"]
        ].rename(columns={"decision_prediction": "prediction"})
        regime_decision["model"] = "fossil_regime_decision"
        regime_ranked = regime[
            ["origin", "horizon", "target_time", "ranked_prediction"]
        ].rename(columns={"ranked_prediction": "prediction"})
        regime_ranked["model"] = "fossil_regime_ranked"
        prediction_parts.extend([regime_point, regime_decision, regime_ranked])
    pred = pd.concat(prediction_parts, ignore_index=True)
    pred["actual"] = df["carbon_intensity_gco2_kwh"].reindex(
        pd.DatetimeIndex(pred["target_time"])
    ).to_numpy()
    pred = pred.dropna(subset=["actual", "prediction"])

    point = ev.point_metrics(pred)
    landmarks = point[point["horizon"].isin((1, 3, 6, 12, 18, 24))]
    mape_landmarks = landmarks.pivot(
        index="model", columns="horizon", values="mape"
    ).round(1)
    aggregate_rows: list[dict] = []
    for model_name, group in pred.groupby("model"):
        error = group["prediction"] - group["actual"]
        aggregate_rows.append(
            {
                "model": model_name,
                "mape": float(
                    100.0
                    * np.mean(
                        error.abs()
                        / group["actual"].abs().clip(lower=1e-9)
                    )
                ),
                "wape": float(
                    100.0 * error.abs().sum() / group["actual"].abs().sum()
                ),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
                "n": int(len(group)),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).set_index("model").sort_values("mape")
    selection = window_selection_metrics(pred, df)

    typer.echo(f"\n=== Dense day-ahead MAPE (%) [{len(origins)} origins] ===")
    typer.echo(mape_landmarks.to_string())
    typer.echo("\n=== Aggregate over all 24 target hours ===")
    typer.echo(aggregate.round(3).to_string())
    typer.echo("\n=== 24-candidate decision vs perfect foresight ===")
    typer.echo(selection.to_string())
    if specialist is not None:
        typer.echo(
            "\nFrance-24 calibration: "
            f"validation MAPE={specialist.validation_mape_:.2f}%  "
            f"validation regret={specialist.validation_regret_:.3f}  "
            f"selector={specialist.selector_}"
        )
    if regime_model is not None:
        typer.echo(
            "Fossil-regime calibration: "
            f"validation MAPE={regime_model.validation_mape_:.2f}%  "
            f"validation regret={regime_model.validation_regret_:.3f}  "
            f"regime accuracy={regime_model.validation_regime_accuracy_:.3f}  "
            f"peak recall={regime_model.validation_peak_recall_:.3f}  "
            f"scale={regime_model.point_scale_:.2f}  "
            f"risk_weight={regime_model.risk_weight_:.2f}  "
            f"ranked regret={regime_model.validation_ranked_regret_:.3f}  "
            f"ranking_weight={regime_model.ranking_weight_:.2f}"
        )

    _dump_json(
        {
            "protocol": {
                "test_start": str(ts),
                "horizons": list(DENSE_DAY_AHEAD_HORIZONS),
                "origins": len(origins),
                "stride_hours": stride_hours,
            },
            "aggregate_metrics": json.loads(
                aggregate.reset_index().round(4).to_json(orient="records")
            ),
            "point_metrics": json.loads(
                point.round(4).to_json(orient="records")
            ),
            "window_selection": json.loads(
                selection.reset_index().to_json(orient="records")
            ),
            "france24_calibration": (
                {
                    "validation_mape_pct": specialist.validation_mape_,
                    "validation_regret": specialist.validation_regret_,
                    "point_scales": specialist.point_scales_,
                    "horizon_mae": specialist.horizon_mae_,
                    "selector": specialist.selector_,
                }
                if specialist is not None
                else None
            ),
            "fossil_regime_calibration": (
                {
                    "validation_mape_pct": regime_model.validation_mape_,
                    "validation_regret": regime_model.validation_regret_,
                    "validation_regime_accuracy": (
                        regime_model.validation_regime_accuracy_
                    ),
                    "validation_peak_recall": regime_model.validation_peak_recall_,
                    "point_scale": regime_model.point_scale_,
                    "risk_weight": regime_model.risk_weight_,
                    "ranking_weight": regime_model.ranking_weight_,
                    "validation_ranked_regret": (
                        regime_model.validation_ranked_regret_
                    ),
                    "regime_counts": regime_model.regime_counts_,
                    "physical_map": {
                        "intercept": regime_model.mapper.intercept_,
                        "coefficients": regime_model.mapper.coefficients_,
                    },
                }
                if regime_model is not None
                else None
            ),
        },
        output,
        "France-24 metrics",
    )


@carbon_app.command("compare-protocols")
def carbon_compare_protocols(
    carbon: str = typer.Option(
        DEFAULT_ENRICHED_SNAPSHOT, help="Consolidated training snapshot."
    ),
    holdout_carbon: str = typer.Option(
        None, help="Optional recent carbon snapshot appended only for evaluation."
    ),
    train_end: str = typer.Option(
        "2026-05-01", help="Exclusive end of model training data."
    ),
    test_start: str = typer.Option(
        "2026-06-24", help="First evaluated forecast origin."
    ),
    test_end: str = typer.Option(
        "2026-07-09", help="Last evaluated forecast origin."
    ),
    calibration_start: str = typer.Option(
        "2026-06-18", help="Start of recent pre-test scale calibration."
    ),
    calibration_end: str = typer.Option(
        "2026-06-23", help="Last calibration origin; targets must precede test."
    ),
    recent_calibration: bool = typer.Option(
        True, help="Fit recent pre-test multiplicative scales."
    ),
    weather: str = typer.Option(DEFAULT_WEATHER, help="Weather forecast snapshot."),
    consumption_forecast: str = typer.Option(
        DEFAULT_CONSUMPTION, help="Historical consumption-forecast snapshot."
    ),
    mix_forecast: str = typer.Option(
        DEFAULT_MIX_FORECAST, help="Historical day-ahead mix snapshot."
    ),
    holdout_mix_forecast: str = typer.Option(
        None, help="Optional recent day-ahead mix snapshot."
    ),
    rte_generation_forecast: str = typer.Option(
        DEFAULT_RTE_GENERATION_FORECAST,
        help="Historical publication-versioned RTE forecast snapshot.",
    ),
    holdout_rte_generation_forecast: str = typer.Option(
        None, help="Optional recent publication-versioned RTE snapshot."
    ),
    short_horizon_cutoff: int = typer.Option(
        2, help="Use the dense baseline through this horizon, then fossil-RTE."
    ),
    ensemble_ci: bool = typer.Option(
        True, help="Also train the dense EnsembleCI adaptation."
    ),
    output: str = typer.Option(
        "runs/compare_protocols_metrics.json", help="Protocol comparison JSON."
    ),
) -> None:
    """Compare rolling operational and fixed-daily dense 24h protocols."""
    import copy

    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.ensemble_ci import train_ensemble_ci_model
    from green_observatory.carbon.fossil_regime import train_fossil_regime_model
    from green_observatory.carbon.france24 import DENSE_DAY_AHEAD_HORIZONS
    from green_observatory.carbon.model import train_project_model
    from green_observatory.carbon.physical import (
        PhysicalCarbonMapper,
        generation_shares,
    )
    from green_observatory.carbon.protocols import (
        DAILY_UTC,
        ROLLING_6H,
        evaluate_protocol,
        fit_mape_scales,
        model_predictions,
        protocol_origins,
        regularize_hourly,
    )
    from green_observatory.carbon.rte_forecast_features import (
        RteGenerationForecastFeatureStore,
    )
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    consolidated = OdreCarbonProvider.load_snapshot(carbon)
    frames = [consolidated]
    if holdout_carbon and Path(holdout_carbon).exists():
        frames.append(OdreCarbonProvider.load_snapshot(holdout_carbon))
    full = regularize_hourly(pd.concat(frames).sort_index())
    train_cutoff = _utc(train_end)
    train = consolidated.loc[consolidated.index < train_cutoff]

    forecast_parts: list[pd.DataFrame] = []
    for path in (weather, consumption_forecast):
        if path and Path(path).exists():
            forecast_parts.append(pd.read_parquet(path))
    mix_parts = [
        pd.read_parquet(path)
        for path in (mix_forecast, holdout_mix_forecast)
        if path and Path(path).exists()
    ]
    if mix_parts:
        mix = pd.concat(mix_parts).sort_index()
        mix = mix[~mix.index.duplicated(keep="last")]
        forecast_parts.append(mix)
    forecasts = None
    for part in forecast_parts:
        part = part.copy()
        if part.index.tz is None:
            part.index = part.index.tz_localize("UTC")
        else:
            part.index = part.index.tz_convert("UTC")
        forecasts = part if forecasts is None else forecasts.join(part, how="outer")
    if forecasts is None:
        raise typer.BadParameter("at least one forecast snapshot is required")

    rte_parts = [
        pd.read_parquet(path)
        for path in (
            rte_generation_forecast,
            holdout_rte_generation_forecast,
        )
        if path and Path(path).exists()
    ]
    rte_store = (
        RteGenerationForecastFeatureStore(
            pd.concat(rte_parts, ignore_index=True),
            production_types=("WIND_ONSHORE", "WIND_OFFSHORE", "SOLAR"),
        )
        if rte_parts
        else None
    )

    cfg = load_named("carbon_model")
    climatology = climatology_from_config(train, cfg)
    dense_cfg = copy.deepcopy(cfg)
    dense_cfg.setdefault("model", {})["horizons_hours"] = list(
        DENSE_DAY_AHEAD_HORIZONS
    )
    typer.echo(
        f"protocol training rows={len(train)} through {train.index.max()}; "
        f"evaluation={test_start}..{test_end}"
    )
    typer.echo("training dense project baseline ...")
    baseline = train_project_model(
        train, dense_cfg, climatology=climatology, forecast_frame=forecasts
    )
    typer.echo("training fossil-regime + RTE signal model ...")
    fossil = train_fossil_regime_model(
        train,
        cfg,
        climatology=climatology,
        forecast_frame=forecasts,
        rte_forecast_store=rte_store,
    )
    ensemble = None
    if ensemble_ci:
        typer.echo("training dense EnsembleCI adaptation ...")
        ensemble = train_ensemble_ci_model(
            train,
            dense_cfg,
            climatology=climatology,
            forecast_frame=forecasts,
        )

    report: dict[str, object] = {
        "train_end": str(train_cutoff),
        "test_start": str(_utc(test_start)),
        "test_end": str(_utc(test_end)),
        "short_horizon_cutoff": short_horizon_cutoff,
        "protocols": {},
    }
    calibration_scales: dict[str, float] = {}
    recent_mapper = None
    if recent_calibration:
        calibration_rows = full.loc[
            (_utc(calibration_start) <= full.index)
            & (full.index < _utc(test_start))
        ]
        recent_shares = generation_shares(calibration_rows)
        recent_mapper = PhysicalCarbonMapper(recent_shares.columns).fit(
            recent_shares,
            calibration_rows["carbon_intensity_gco2_kwh"],
        )
        report["recent_physical_map"] = {
            "intercept": recent_mapper.intercept_,
            "coefficients": recent_mapper.coefficients_,
            "rows": len(calibration_rows),
        }
        calibration_origins = protocol_origins(
            full, calibration_start, calibration_end, ROLLING_6H
        )
        calibration_predictions = model_predictions(
            full,
            calibration_origins,
            baseline=baseline,
            fossil_regime=fossil,
            ensemble_ci=ensemble,
            short_horizon_cutoff=short_horizon_cutoff,
            recent_mapper=recent_mapper,
        )
        calibration_scales = fit_mape_scales(calibration_predictions)
        report["recent_scale_calibration"] = {
            "start": str(_utc(calibration_start)),
            "end": str(_utc(calibration_end)),
            "origins": len(calibration_origins),
            "scales": calibration_scales,
        }
        typer.echo(
            f"recent pre-test calibration: {len(calibration_origins)} origins; "
            f"scales={calibration_scales}"
        )
    for spec in (ROLLING_6H, DAILY_UTC):
        origins = protocol_origins(full, test_start, test_end, spec)
        result = evaluate_protocol(
            full,
            origins,
            baseline=baseline,
            fossil_regime=fossil,
            ensemble_ci=ensemble,
            short_horizon_cutoff=short_horizon_cutoff,
            calibration_scales=calibration_scales,
            recent_mapper=recent_mapper,
        )
        typer.echo(f"\n=== {spec.name}: {len(origins)} origins ===")
        typer.echo(result["aggregate"].round(3).to_string(index=False))
        typer.echo("\nOracle/window selection:")
        typer.echo(result["selection"].to_string())
        report["protocols"][spec.name] = {
            "origins": len(origins),
            "aggregate_metrics": json.loads(
                result["aggregate"].round(5).to_json(orient="records")
            ),
            "point_metrics": json.loads(
                result["point"].round(5).to_json(orient="records")
            ),
            "window_selection": json.loads(
                result["selection"].reset_index().to_json(orient="records")
            ),
        }
    _dump_json(report, output, "protocol comparison")


# --------------------------------------------------------------------------- #
# carbon compare-live (Electricity Maps)
# --------------------------------------------------------------------------- #
@carbon_app.command("compare-live")
def carbon_compare_live(
    model_path: str = typer.Option(DEFAULT_MODEL, help="Trained model (for its embedded climatology)."),
    horizon_hours: int = typer.Option(24, help="Comparison horizon (EM caps at ~24h)."),
    output: str = typer.Option(None, help="Optional JSON output."),
) -> None:
    """Live forward comparison of the project forecast vs Electricity Maps."""
    from green_observatory.carbon.live_compare import live_comparison
    from green_observatory.carbon.model import ProjectCarbonModel
    from green_observatory.providers.carbon_electricity_maps import ElectricityMapsProvider
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    em = ElectricityMapsProvider()
    if not em.available():
        typer.echo("Electricity Maps: no API key. Set ELECTRICITYMAPS_API_TOKEN and retry.")
        raise typer.Exit(code=1)

    model = ProjectCarbonModel.load(model_path)
    res = live_comparison(model, OdreCarbonProvider(), em, horizon_hours=horizon_hours)
    typer.echo(f"origin (last RTE hour): {res['origin'].isoformat()}")

    b = res.get("basis")
    if b:
        typer.echo(
            f"\n[BASIS] over {b['n']} recent shared hours:\n"
            f"  RTE production-based mean = {b['rte_production_mean']} gCO2/kWh\n"
            f"  EM  consumption-based mean = {b['em_consumption_mean']} gCO2/kWh "
            f"(diff {b['diff_em_minus_rte']:+}), shape correlation = {b['correlation']}"
        )
    a = res.get("agreement")
    if a:
        typer.echo(
            f"\n[FORECAST agreement] next {a['n']}h:\n"
            f"  Spearman(project ranking, EM ranking) = {a['spearman']}\n"
            f"  project greenest hour: {a['our_greenest_hour']:%m-%d %H:%M}Z ({a['our_greenest_gco2']} gCO2/kWh)\n"
            f"  EM  greenest hour: {a['em_greenest_hour']:%m-%d %H:%M}Z ({a['em_greenest_gco2']} gCO2/kWh)"
        )
    typer.echo(
        "\nThe two sources use different accounting bases (production vs consumption). "
        "For a live future window, ranking agreement is the comparable signal."
    )
    _dump_json(
        {"origin": res["origin"], "basis": res.get("basis"), "agreement": res.get("agreement")},
        output, "live-compare",
    )


# --------------------------------------------------------------------------- #
# windows analyze
# --------------------------------------------------------------------------- #
@windows_app.command("analyze")
def windows_analyze(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    horizon_hours: int = typer.Option(24, help="Look back this many hours of actual data."),
    at: str = typer.Option(None, help="Window horizon end (UTC). Default: last timestamp."),
    percentile: float = typer.Option(None, help="Override the low-carbon percentile (e.g. 0.25)."),
    output: str = typer.Option(None, help="Optional JSON output."),
) -> None:
    """Compute low-carbon windows over a horizon of actual carbon data."""
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_base import CARBON
    from green_observatory.providers.carbon_odre import OdreCarbonProvider
    from green_observatory.windows.scoring import low_carbon_windows_from_config

    df = OdreCarbonProvider.load_snapshot(carbon)
    wcfg = load_named("window_scoring")
    end = _utc(at) or df.index.max()
    start = end - pd.Timedelta(hours=horizon_hours - 1)
    series = df.loc[start:end, CARBON]

    overrides = {"percentile": percentile} if percentile is not None else {}
    wins = low_carbon_windows_from_config(series, wcfg, **overrides)
    typer.echo(
        f"horizon {series.index.min():%Y-%m-%d %H:%M}..{series.index.max():%H:%M}Z ({len(series)}h)  "
        f"carbon min={series.min():.0f} p25={series.quantile(.25):.0f} max={series.max():.0f} gCO2/kWh"
    )
    typer.echo(f"\n{len(wins)} low-carbon window(s):")
    for w in wins:
        typer.echo(
            f"  #{w.rank} {w.start:%m-%d %H:%M}->{w.end:%m-%d %H:%M}Z  "
            f"score={w.carbon_score:.2f}  mean={w.mean_carbon_intensity_gco2_kwh:.0f} gCO2/kWh"
        )
    _dump_json([w.model_dump(mode="json") for w in wins], output, "windows")


# --------------------------------------------------------------------------- #
# jobs: automatic Kepler + RTE accounting
# --------------------------------------------------------------------------- #
def _job_carbon_source(carbon_snapshot: str | None):
    from green_observatory.observability.rte import (
        OdreRealtimeCarbonSource,
        SnapshotCarbonSource,
    )

    return SnapshotCarbonSource(carbon_snapshot) if carbon_snapshot else OdreRealtimeCarbonSource()


def _build_job_reporter(
    url: str,
    kubectl,
    *,
    carbon_snapshot: str | None,
    zone: str,
    step: int,
    context: bool,
    capture_logs: bool,
    include_intervals: bool,
):
    """One place to wire the reporter, so `report` and `observe` stay identical."""
    from green_observatory.observability.prometheus import PrometheusClient
    from green_observatory.observability.reporter import JobReporter

    return JobReporter(
        PrometheusClient(url),
        _job_carbon_source(carbon_snapshot),
        zone=zone,
        step_seconds=step,
        kubectl=kubectl,
        collect_context=context,
        capture_logs=capture_logs,
        include_intervals=include_intervals,
    )


def _with_prometheus(
    kubectl,
    prometheus_url: str | None,
    prometheus_namespace: str,
    prometheus_service: str,
    prometheus_port: int,
    action,
):
    from green_observatory.observability.cluster import PrometheusPortForward

    if prometheus_url:
        return action(prometheus_url)
    with PrometheusPortForward(
        kubectl, prometheus_namespace, prometheus_service, prometheus_port
    ) as forward:
        return action(forward.url)


@jobs_app.command("report")
def jobs_report(
    job_name: str = typer.Argument(..., help="Kubernetes Job name."),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    output: str = typer.Option("runs/job-reports", help="Directory for report JSON files."),
    zone: str = typer.Option("package", help="Kepler RAPL zone used for attribution."),
    step: int = typer.Option(10, min=1, help="Prometheus range-query step in seconds."),
    carbon_snapshot: str = typer.Option(
        None, help="Optional parquet/CSV carbon snapshot; default fetches RTE near-real-time."
    ),
    kubeconfig: str = typer.Option(None, help="Optional kubeconfig; otherwise kubectl defaults apply."),
    context: bool = typer.Option(
        True, help="Collect provenance, node context and the node-isolation post-flight."
    ),
    capture_logs: bool = typer.Option(
        True, help="Capture container stdout + its sha256 (needs the pod to still exist)."
    ),
    include_intervals: bool = typer.Option(
        False, help="Persist the per-scrape-interval energy/carbon audit trail."
    ),
    prometheus_url: str = typer.Option(None, help="Existing Prometheus URL; otherwise port-forward."),
    prometheus_namespace: str = typer.Option("monitoring"),
    prometheus_service: str = typer.Option("monitoring-kube-prometheus-prometheus"),
    prometheus_port: int = typer.Option(9090),
) -> None:
    """Generate or refresh the energy/carbon JSON for one terminal Job."""
    from green_observatory.observability.cluster import KubectlClient
    from green_observatory.observability.observer import write_report

    kubectl = KubectlClient(kubeconfig)

    def execute(url: str):
        job = kubectl.get_job(namespace, job_name)
        pods = kubectl.pods_for_job(job)
        reporter = _build_job_reporter(
            url, kubectl, carbon_snapshot=carbon_snapshot, zone=zone, step=step,
            context=context, capture_logs=capture_logs, include_intervals=include_intervals,
        )
        report = reporter.build(job, pods)
        path = write_report(Path(output), report)
        typer.echo(report.model_dump_json(indent=2))
        typer.echo(f"wrote {'final' if report.quality.final else 'provisional'} report -> {path}")
        return path

    _with_prometheus(
        kubectl,
        prometheus_url,
        prometheus_namespace,
        prometheus_service,
        prometheus_port,
        execute,
    )


@jobs_app.command("observe")
def jobs_observe(
    selector: str = typer.Option(
        "sustainability.cern.ch/track=true",
        help="Label selector identifying Jobs to account.",
    ),
    namespace: str = typer.Option(None, "--namespace", "-n", help="Default: all namespaces."),
    output: str = typer.Option("runs/job-reports", help="Directory for report JSON files."),
    once: bool = typer.Option(False, help="Scan once and exit instead of observing continuously."),
    poll_seconds: int = typer.Option(30, min=5),
    max_job_age_hours: int = typer.Option(168, min=1, help="Ignore older terminal Jobs."),
    zone: str = typer.Option("package", help="Kepler RAPL zone used for attribution."),
    step: int = typer.Option(10, min=1, help="Prometheus range-query step in seconds."),
    carbon_snapshot: str = typer.Option(
        None, help="Optional parquet/CSV carbon snapshot; default fetches RTE near-real-time."
    ),
    kubeconfig: str = typer.Option(None, help="Optional kubeconfig; otherwise kubectl defaults apply."),
    context: bool = typer.Option(
        True, help="Collect provenance, node context and the node-isolation post-flight."
    ),
    capture_logs: bool = typer.Option(
        True, help="Capture container stdout + its sha256 (needs the pod to still exist)."
    ),
    include_intervals: bool = typer.Option(
        False, help="Persist the per-scrape-interval energy/carbon audit trail."
    ),
    prometheus_url: str = typer.Option(None, help="Existing Prometheus URL; otherwise port-forward."),
    prometheus_namespace: str = typer.Option("monitoring"),
    prometheus_service: str = typer.Option("monitoring-kube-prometheus-prometheus"),
    prometheus_port: int = typer.Option(9090),
) -> None:
    """Continuously write reports for labelled Jobs when they become terminal."""
    from green_observatory.observability.cluster import KubectlClient
    from green_observatory.observability.observer import JobObserver

    kubectl = KubectlClient(kubeconfig)

    def execute(url: str):
        reporter = _build_job_reporter(
            url, kubectl, carbon_snapshot=carbon_snapshot, zone=zone, step=step,
            context=context, capture_logs=capture_logs, include_intervals=include_intervals,
        )
        observer = JobObserver(
            kubectl,
            reporter,
            Path(output),
            selector=selector,
            namespace=namespace,
            max_job_age_seconds=max_job_age_hours * 3600,
        )
        return observer.run_once() if once else observer.run_forever(poll_seconds)

    _with_prometheus(
        kubectl,
        prometheus_url,
        prometheus_namespace,
        prometheus_service,
        prometheus_port,
        execute,
    )


@jobs_app.command("summarize")
def jobs_summarize(
    reports: str = typer.Option("runs/job-reports", help="Directory containing report JSON files."),
    output: str = typer.Option("runs/job-reports/summary.csv", help="Destination CSV."),
    include_provisional: bool = typer.Option(False, help="Include reports with quality.final=false."),
) -> None:
    """Flatten Job reports into a comparison-ready CSV."""
    from green_observatory.observability.summary import summarize_reports

    frame = summarize_reports(reports, include_provisional=include_provisional)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    typer.echo(f"wrote {len(frame)} Job report(s) -> {destination}")


if __name__ == "__main__":  # pragma: no cover
    app()
