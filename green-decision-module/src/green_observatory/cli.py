"""``greenctl`` - command line interface for the Green Window Observatory.

Carbon track (Milestones 0-2):

    greenctl carbon import            --output data/cache/carbon_fr_hourly.parquet
    greenctl carbon fetch-mix-forecast
    greenctl carbon fetch-rte-system
    greenctl windows analyze --horizon-hours 24
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
    help="Green Window Observatory - carbon when-to-run intelligence.",
)
carbon_app = typer.Typer(no_args_is_help=True, help="Carbon data: import history and fetch day-ahead forecast inputs.")
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
DEFAULT_MIX_FORECAST = "data/cache/mix_day_ahead_fr_hourly.parquet"
DEFAULT_RTE_UNAVAILABILITY = "data/cache/rte_unavailability_messages.parquet"
DEFAULT_RTE_GENERATION_FORECAST = "data/cache/rte_generation_forecast.parquet"




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












# --------------------------------------------------------------------------- #
# carbon forecast
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# carbon compare
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# carbon compare-france24 (isolated dense day-ahead experiment)
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# carbon compare-live (Electricity Maps)
# --------------------------------------------------------------------------- #


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
