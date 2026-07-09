"""``greenctl`` - command line interface for the Green Window Observatory.

Carbon track (Milestones 0-2):

    greenctl carbon import   --output data/cache/carbon_fr_hourly.parquet
    greenctl carbon train    --config carbon_model --test-start 2026-02-01
    greenctl carbon forecast --horizon-hours 48
    greenctl carbon compare  --test-start 2026-02-01
    greenctl windows analyze --horizon-hours 48
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
app.add_typer(carbon_app, name="carbon")
app.add_typer(windows_app, name="windows")

DEFAULT_SNAPSHOT = "data/cache/carbon_fr_hourly.parquet"
DEFAULT_MODEL = "models/project_carbon_hgb.joblib"


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
@carbon_app.command("train")
def carbon_train(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    output: str = typer.Option(DEFAULT_MODEL, help="Output model path (joblib)."),
    test_start: str = typer.Option(None, help="Hold out data on/after this date from training."),
) -> None:
    """Fit climatology + the project carbon model and save the model."""
    from green_observatory.carbon.climatology import climatology_from_config
    from green_observatory.carbon.model import train_project_model
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider

    df = OdreCarbonProvider.load_snapshot(carbon)
    cfg = load_named(config)
    ts = _utc(test_start)
    train = df.loc[df.index < ts] if ts is not None else df
    typer.echo(f"training on {len(train)} rows (climatology embeds into the model) ...")
    clim = climatology_from_config(train, cfg)
    model = train_project_model(train, cfg, climatology=clim)
    model.save(output)
    typer.echo(f"saved model ({model.algorithm}, horizons {list(model.horizons)}) -> {output}")


# --------------------------------------------------------------------------- #
# carbon forecast
# --------------------------------------------------------------------------- #
@carbon_app.command("forecast")
def carbon_forecast(
    carbon: str = typer.Option(DEFAULT_SNAPSHOT, help="Carbon snapshot parquet."),
    config: str = typer.Option("carbon_model", help="Carbon config name or path."),
    horizon_hours: int = typer.Option(48, help="Forecast horizon in hours."),
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
) -> None:
    """Rolling-origin backtest: MAE + green-window selection vs the oracle."""
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
    project = train_project_model(train, cfg, climatology=clim) if "project" in include else None
    origins = ev.make_origins(df, ts, stride_hours=stride_hours)
    pred = ev.backtest_predictions(
        df, origins, climatology=clim, project_model=project,
        corrected_cfg=cfg.get("corrected_climatology"), include=include,
    )

    mae = ev.mae_table(pred)
    sel = window_selection_metrics(pred, df)
    typer.echo(f"\n=== MAE (gCO2/kWh) by horizon  [{len(origins)} origins] ===")
    typer.echo(mae.to_string())
    typer.echo("\n=== Green-window selection (pick greenest forecasted hour) ===")
    typer.echo(sel.to_string())

    em = ElectricityMapsProvider()
    typer.echo(
        "\nElectricity Maps: available (live consumption-based forecast comparison)."
        if em.available()
        else "\nElectricity Maps: no API key -> skipped (optional comparison)."
    )
    _dump_json(
        {"mae": json.loads(mae.reset_index().to_json(orient="records")),
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
    """Live forward comparison of our forecast vs Electricity Maps (needs API key)."""
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
            f"  Spearman(our ranking, EM ranking) = {a['spearman']}\n"
            f"  our greenest hour: {a['our_greenest_hour']:%m-%d %H:%M}Z ({a['our_greenest_gco2']} gCO2/kWh)\n"
            f"  EM  greenest hour: {a['em_greenest_hour']:%m-%d %H:%M}Z ({a['em_greenest_gco2']} gCO2/kWh)"
        )
    typer.echo(
        "\nNote: different basis (production vs consumption) and no scoring vs actuals "
        "yet (future window). Ranking agreement is the comparable signal."
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
    horizon_hours: int = typer.Option(48, help="Look back this many hours of actual data."),
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


if __name__ == "__main__":  # pragma: no cover
    app()
