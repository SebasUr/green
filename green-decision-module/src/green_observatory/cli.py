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
DEFAULT_MODEL = "models/project_carbon_hgb.joblib"
DEFAULT_WEATHER = "data/cache/weather_fr_hourly.parquet"
DEFAULT_CONSUMPTION = "data/cache/consumption_forecast_fr_hourly.parquet"


def _forecast_frame(weather: str | None, consumption: str | None):
    """Join available forecast-feature snapshots (wind/solar + consumption) or None."""
    parts = []
    for path in (weather, consumption):
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
    history = df.loc[df.index <= origin]

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
) -> None:
    """Rolling-origin backtest: MAE + green-window selection vs perfect foresight."""
    from green_observatory.carbon import evaluation as ev
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.model import train_project_model
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
    project = (
        train_project_model(train, cfg, climatology=clim, forecast_frame=ff)
        if "project" in include
        else None
    )
    if ff is not None:
        typer.echo(f"project model uses forecast features: {list(ff.columns)}")
    origins = ev.make_origins(df, ts, stride_hours=stride_hours)
    pred = ev.backtest_predictions(
        df, origins, climatology=clim, project_model=project,
        corrected_cfg=cfg.get("corrected_climatology"), include=include,
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
         "window_selection": json.loads(sel.reset_index().to_json(orient="records"))},
        output, "metrics",
    )


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
    prometheus_url: str = typer.Option(None, help="Existing Prometheus URL; otherwise port-forward."),
    prometheus_namespace: str = typer.Option("monitoring"),
    prometheus_service: str = typer.Option("monitoring-kube-prometheus-prometheus"),
    prometheus_port: int = typer.Option(9090),
) -> None:
    """Generate or refresh the energy/carbon JSON for one terminal Job."""
    from green_observatory.observability.cluster import KubectlClient
    from green_observatory.observability.observer import write_report
    from green_observatory.observability.prometheus import PrometheusClient
    from green_observatory.observability.reporter import JobReporter

    kubectl = KubectlClient(kubeconfig)

    def execute(url: str):
        job = kubectl.get_job(namespace, job_name)
        pods = kubectl.pods_for_job(job)
        reporter = JobReporter(
            PrometheusClient(url), _job_carbon_source(carbon_snapshot), zone=zone, step_seconds=step
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
    prometheus_url: str = typer.Option(None, help="Existing Prometheus URL; otherwise port-forward."),
    prometheus_namespace: str = typer.Option("monitoring"),
    prometheus_service: str = typer.Option("monitoring-kube-prometheus-prometheus"),
    prometheus_port: int = typer.Option(9090),
) -> None:
    """Continuously write reports for labelled Jobs when they become terminal."""
    from green_observatory.observability.cluster import KubectlClient
    from green_observatory.observability.observer import JobObserver
    from green_observatory.observability.prometheus import PrometheusClient
    from green_observatory.observability.reporter import JobReporter

    kubectl = KubectlClient(kubeconfig)

    def execute(url: str):
        reporter = JobReporter(
            PrometheusClient(url), _job_carbon_source(carbon_snapshot), zone=zone, step_seconds=step
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
