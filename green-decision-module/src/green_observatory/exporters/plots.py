"""Scientific-but-intuitive figures for judging the carbon model.

Four figures, each answering one question at a glance:

1. ``decision_quality``   - how good is the *when-to-run decision* (vs run-now / oracle)?
2. ``error_by_horizon``   - how accurate is the *number*, and how does it decay with lead time?
3. ``forecast_example``   - what does one 48h forecast + its green windows look like?
4. ``calibration``        - is the model biased? (predicted vs actual)

Palette: Okabe-Ito (the standard colorblind-safe categorical set). Marks are
thin, grids recessive, series direct-labeled; text uses ink, not series colors.
Run ``python -m green_observatory.exporters.plots`` to regenerate everything.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from green_observatory.providers.carbon_base import CARBON  # noqa: E402

# Okabe-Ito, assigned to models in a FIXED order (never cycled).
MODEL_COLORS = {
    "run_now": "#9AA0A6",
    "persistence": "#E69F00",
    "climatology": "#56B4E9",
    "corrected": "#009E73",
    "sarimax": "#CC79A7",
    "project": "#0072B2",
    "oracle": "#111111",
}
MODEL_LABEL = {
    "run_now": "Run now",
    "persistence": "Persistence",
    "climatology": "Climatology",
    "corrected": "Corrected clim.",
    "sarimax": "SARIMAX",
    "project": "ML model",
    "oracle": "Oracle (ideal)",
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E9E9E9", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "font.family": "sans-serif",
    })


# --------------------------------------------------------------------------- #
def plot_decision_quality(sel: pd.DataFrame, ax) -> None:
    """Lollipop of realized carbon per strategy, framed by run-now and oracle."""
    order = [m for m in ["oracle", "project", "sarimax", "corrected", "climatology",
                         "persistence", "run_now"] if m in sel.index]
    y = np.arange(len(order))
    run_now = sel.loc["run_now", "mean_realized_gco2"]
    oracle = sel.loc["oracle", "mean_realized_gco2"]

    # reference band: everything between "do nothing" and "perfect foresight"
    ax.axvspan(oracle, run_now, color="#F2F7F4", zorder=0)
    ax.axvline(run_now, color=MODEL_COLORS["run_now"], ls="--", lw=1.5, zorder=1)
    ax.axvline(oracle, color=MODEL_COLORS["oracle"], ls="--", lw=1.5, zorder=1)

    for yi, m in zip(y, order):
        val = sel.loc[m, "mean_realized_gco2"]
        c = MODEL_COLORS[m]
        ax.plot([oracle, val], [yi, yi], color=c, lw=2.5, alpha=0.35, zorder=2)
        ax.scatter([val], [yi], s=130, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        pct = sel.loc[m, "pct_oracle_potential"]
        lab = f"{val:.1f}"
        if pd.notna(pct) and m not in ("run_now", "oracle"):
            lab += f"   ·   {pct:.0f}% of oracle"
        ax.annotate(lab, (val, yi), xytext=(9, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in order])
    ax.set_xlabel("REAL intensity you'd run at  (gCO₂/kWh)  ·  ← greener")
    ax.set_title("1 · How good is the when-to-run decision?")
    ax.annotate("run now\n(do nothing)", (run_now, -0.8),
                color=MUTED, fontsize=9, ha="center", va="top")
    ax.annotate("perfect\n(oracle)", (oracle, -0.8),
                color=MUTED, fontsize=9, ha="center", va="top")
    ax.set_xlim(oracle - 0.6, run_now + 3.2)
    ax.set_ylim(-1.5, len(order) - 0.4)


def plot_error_by_horizon(pm: pd.DataFrame, ax, value: str = "wape") -> None:
    """One line per model of error vs forecast lead time."""
    models = [m for m in ["persistence", "climatology", "corrected", "sarimax", "project"]
              if m in pm["model"].unique()]
    for m in models:
        sub = pm[pm.model == m].sort_values("horizon")
        ax.plot(sub.horizon, sub[value], "-o", color=MODEL_COLORS[m], lw=2, ms=6,
                markeredgecolor="white", markeredgewidth=1, label=MODEL_LABEL[m])
        ax.annotate(MODEL_LABEL[m], (sub.horizon.iloc[-1], sub[value].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=MODEL_COLORS[m], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(sorted(pm.horizon.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Forecast lead time (hours, log scale)")
    unit = "WAPE  (% of real level)" if value == "wape" else "MAE  (gCO₂/kWh)"
    ax.set_ylabel(unit)
    short = "Relative error — WAPE (%)" if value == "wape" else "Absolute error — MAE (gCO₂/kWh)"
    ax.set_title(short, fontsize=12)
    ax.set_xlim(right=sorted(pm.horizon.unique())[-1] * 1.9)
    ax.set_ylim(bottom=0)


def plot_forecast_example(actual: pd.Series, forecast: pd.Series, windows, oracle_wins, ax) -> None:
    """Actual vs forecast, with the model's predicted windows (green, full height)
    and the oracle's best-in-hindsight windows (amber ribbon on top)."""
    ax.plot(actual.index, actual.values, "-", color=INK, lw=2.2, label="Actual (what happened)")
    ax.plot(forecast.index, forecast.values, "--", color=MODEL_COLORS["corrected"], lw=2,
            label="Forecast")
    for i, w in enumerate(windows):
        ax.axvspan(w.start, w.end, color="#009E73", alpha=0.16, zorder=0,
                   label="Forecast green window" if i == 0 else None)
    for i, w in enumerate(oracle_wins):
        ax.axvspan(w.start, w.end, ymin=0.90, ymax=0.99, color="#6357B8", alpha=0.9, zorder=4,
                   label="Oracle window (best in hindsight)" if i == 0 else None)
    ax.set_ylabel("Carbon (gCO₂/kWh)")
    ax.set_title("3 · Real example: forecast, its green windows, and the oracle's")
    ax.set_ylim(0, float(actual.max()) * 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))


def plot_calibration(pred_df: pd.DataFrame, ax, model: str = "project") -> None:
    """Predicted vs actual scatter with the y=x diagonal, colored by horizon."""
    sub = pred_df[pred_df.model == model]
    sc = ax.scatter(sub.actual, sub.prediction, c=sub.horizon, cmap="viridis",
                    s=14, alpha=0.55, edgecolor="none")
    lim = [0, max(sub.actual.max(), sub.prediction.max()) * 1.05]
    ax.plot(lim, lim, "-", color=INK, lw=1.3)
    ax.annotate("perfect prediction", (lim[1] * 0.62, lim[1] * 0.62),
                rotation=45, color=MUTED, fontsize=9, ha="center", va="bottom")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("ACTUAL carbon (gCO₂/kWh)")
    ax.set_ylabel("PREDICTED carbon")
    wape = 100 * (sub.prediction - sub.actual).abs().sum() / sub.actual.sum()
    bias = (sub.prediction - sub.actual).mean()
    ax.set_title(f"4 · Does it predict the number well? ({MODEL_LABEL[model]})")
    ax.text(0.04, 0.96, f"WAPE {wape:.0f}%   ·   bias {bias:+.1f} gCO₂",
            transform=ax.transAxes, va="top", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d0d0"))
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("lead time (h)", fontsize=9)


# --------------------------------------------------------------------------- #
def _pick_illustrative_origin(df, test_start, horizon=48):
    best, best_range = None, -1.0
    grid = pd.date_range(test_start, df.index.max() - pd.Timedelta(hours=horizon), freq="12h")
    for origin in grid.intersection(df.index):
        seg = df[CARBON].loc[origin:origin + pd.Timedelta(hours=horizon)]
        r = float(seg.max() - seg.min())
        if r > best_range:
            best_range, best = r, origin
    return best


def generate(outdir: str = "runs/figures", *,
             snapshot: str = "data/cache/carbon_fr_hourly.parquet",
             model_path: str = "models/project_carbon_hgb.joblib",
             test_start: str = "2026-02-01", stride_hours: int = 6) -> list[str]:
    """Compute the backtest and write the four figures as PNGs."""
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore")
    from green_observatory.carbon import evaluation as ev
    from green_observatory.carbon.corrected_climatology import CorrectedClimatologyForecaster
    from green_observatory.carbon.model import ProjectCarbonModel
    from green_observatory.carbon.sarimax import SarimaxForecaster
    from green_observatory.config import load_named
    from green_observatory.providers.carbon_odre import OdreCarbonProvider
    from green_observatory.windows.oracle import window_selection_metrics
    from green_observatory.windows.scoring import low_carbon_windows_from_config

    apply_style()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = OdreCarbonProvider.load_snapshot(snapshot)
    cfg = load_named("carbon_model")
    ts = pd.Timestamp(test_start, tz="UTC")
    train = df.loc[df.index < ts]
    model = ProjectCarbonModel.load(model_path)
    clim = model.feature_builder.climatology
    sar = SarimaxForecaster().fit(train)

    origins = ev.make_origins(df, ts, stride_hours=stride_hours)
    base = ev.backtest_predictions(df, origins, climatology=clim, project_model=model,
                                   corrected_cfg=cfg.get("corrected_climatology"),
                                   include=("persistence", "climatology", "corrected", "project"))
    sarp = ev.forecaster_batch(sar, df, origins, [1, 3, 6, 12, 24, 48], "sarimax")
    pred = pd.concat([base, sarp], ignore_index=True)
    pm = ev.point_metrics(pred)
    sel = window_selection_metrics(pred, df)

    paths = []
    # Fig 1
    fig, ax = plt.subplots(figsize=(9, 4.2))
    plot_decision_quality(sel, ax)
    p = f"{outdir}/1_decision_quality.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    # Fig 2 (two panels: WAPE + MAE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_error_by_horizon(pm, axes[0], "wape")
    plot_error_by_horizon(pm, axes[1], "mae")
    fig.suptitle("2 · Number error vs forecast lead time",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = f"{outdir}/2_error_by_horizon.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    # Fig 3
    origin = _pick_illustrative_origin(df, ts)
    corr = CorrectedClimatologyForecaster(clim)
    fpred = corr.predict(df.loc[df.index <= origin], origin, list(range(1, 49)))
    fseries = pd.Series(fpred["prediction"].to_numpy(), index=fpred.index, name=CARBON)
    actual = df[CARBON].loc[origin:origin + pd.Timedelta(hours=48)]
    from green_observatory.models import WindowType
    wcfg = load_named("window_scoring")
    wins = low_carbon_windows_from_config(fseries, wcfg,
                                          window_type=WindowType.predicted_low_carbon_window)
    owins = low_carbon_windows_from_config(actual, wcfg, window_type=WindowType.oracle_window)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    plot_forecast_example(actual, fseries, wins, owins, ax)
    fig.tight_layout()
    p = f"{outdir}/3_forecast_example.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    # Fig 4
    fig, ax = plt.subplots(figsize=(6.4, 6))
    plot_calibration(pred, ax, "project")
    fig.tight_layout()
    p = f"{outdir}/4_calibration.png"; fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


if __name__ == "__main__":  # pragma: no cover
    for path in generate():
        print("wrote", path)
