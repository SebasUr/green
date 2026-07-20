"""Plot historical-threshold green windows from cached forecasts.

This module is deliberately independent from model training.  It consumes a
daily-refit predictions parquet, derives one threshold from pre-evaluation
historical carbon, applies it identically to forecast and realised curves, and
renders auditable 24-hour examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from green_observatory.models import WindowType
from green_observatory.windows.scoring import compute_low_carbon_windows


DEFAULT_INPUT = Path(
    "runs/daily_refit_2026/causal_operational_gate_tr_extended_ctx3/predictions.parquet"
)
DEFAULT_HISTORY = Path("data/cache/carbon_fr_hourly.parquet")
DEFAULT_OUTPUT = Path(
    "runs/figures/comparision/05_physical_alpha2_14d_historical_examples.png"
)
DEFAULT_MODEL = "physical_alpha2_calibrated_14d"


def _covered_hours(windows) -> set[pd.Timestamp]:
    covered: set[pd.Timestamp] = set()
    for window in windows:
        start = pd.Timestamp(window.start)
        last = pd.Timestamp(window.end) - pd.Timedelta(hours=1)
        covered.update(pd.date_range(start, last, freq="h"))
    return covered


def _hour_iou(predicted_windows, oracle_windows) -> float:
    predicted = _covered_hours(predicted_windows)
    oracle = _covered_hours(oracle_windows)
    union = predicted | oracle
    return float(len(predicted & oracle) / len(union)) if union else 1.0


def _windows(series: pd.Series, window_type: WindowType, reference: np.ndarray):
    # This exactly follows the CLI's `historical` method: one p33 threshold
    # anchored to actual history, without separate enter/exit thresholds.
    return compute_low_carbon_windows(
        series,
        percentile=0.33,
        enter_percentile=None,
        exit_percentile=None,
        reference=reference,
        min_duration_hours=1,
        max_duration_hours=24,
        merge_gap_hours=1,
        max_windows=24,
        window_type=window_type,
    )


def render(
    predictions: pd.DataFrame,
    reference: np.ndarray,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
    origins: tuple[str, ...] = ("2026-07-09", "2026-07-11", "2026-07-01"),
) -> Path:
    required = {"origin", "horizon", "target_time", "actual", model}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")

    frame = predictions.copy()
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(origins), 1, figsize=(13.5, 10.2), sharey=False)
    if len(origins) == 1:
        axes = [axes]

    labels = ("caso bueno", "caso típico", "caso difícil")
    for ax, origin_text, case_label in zip(axes, origins, labels):
        origin = pd.Timestamp(origin_text, tz="UTC")
        day = frame.loc[frame["origin"] == origin].sort_values("horizon")
        if len(day) != 24:
            raise ValueError(f"origin {origin_text} has {len(day)} rows, expected 24")

        actual = pd.Series(day["actual"].to_numpy(), index=day["target_time"], name="actual")
        forecast = pd.Series(day[model].to_numpy(), index=day["target_time"], name=model)
        predicted_windows = _windows(
            forecast, WindowType.predicted_low_carbon_window, reference
        )
        oracle_windows = _windows(actual, WindowType.oracle_window, reference)

        for index, window in enumerate(predicted_windows):
            ax.axvspan(
                window.start,
                window.end,
                color="#54B79A",
                alpha=0.20,
                zorder=0,
                label="Ventana predicha (histórico p33)" if index == 0 else None,
            )
        for index, window in enumerate(oracle_windows):
            ax.axvspan(
                window.start,
                window.end,
                ymin=0.91,
                ymax=0.99,
                color="#6657B8",
                alpha=0.90,
                zorder=5,
                label="Ventana oráculo (CO2 real)" if index == 0 else None,
            )

        ax.plot(actual.index, actual, color="#202124", linewidth=2.25, label="CO2 real")
        ax.plot(
            forecast.index,
            forecast,
            color="#007C78",
            linewidth=2.0,
            linestyle="--",
            label="Physical α2 + sys + calibración 14d",
        )
        historical_threshold = float(np.quantile(reference, 0.33))
        ax.axhline(
            historical_threshold,
            color="#C47A00",
            linewidth=1.2,
            linestyle=":",
            alpha=0.9,
            label=f"Umbral histórico p33 ({historical_threshold:.1f} g)"
            if ax is axes[0]
            else None,
        )

        selected_position = int(np.argmin(forecast.to_numpy()))
        oracle_position = int(np.argmin(actual.to_numpy()))
        ax.scatter(
            [actual.index[selected_position]],
            [actual.iloc[selected_position]],
            marker="*",
            s=150,
            color="#009E73",
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
            label="Hora elegida por el modelo" if ax is axes[0] else None,
        )
        ax.scatter(
            [actual.index[oracle_position]],
            [actual.iloc[oracle_position]],
            marker="*",
            s=150,
            color="#6657B8",
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
            label="Hora mínima real" if ax is axes[0] else None,
        )

        mape = float(100 * np.mean(np.abs((forecast - actual) / actual)))
        iou = _hour_iou(predicted_windows, oracle_windows)
        chosen_regret = float(actual.iloc[selected_position] - actual.iloc[oracle_position])
        ax.text(
            0.99,
            0.05,
            f"MAPE diario {mape:.1f}%  ·  solapamiento horario {100 * iou:.0f}%"
            f"  ·  regret 1h {chosen_regret:.2f} g",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            color="#202124",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D6D6D6"),
        )
        ax.set_title(f"{origin:%d-%b-%Y} — {case_label}", loc="left", fontsize=11.5)
        ax.set_ylabel("gCO2/kWh")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=mdates.UTC))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=mdates.UTC))

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Ventanas verdes en las 24 horas — método historical",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "Mismo umbral p33 del histórico real anterior al backtest aplicado a predicción y realidad",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    axes[-1].set_xlabel("Hora UTC dentro del horizonte de 24 h")
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--origins", nargs="+", default=["2026-07-09", "2026-07-11", "2026-07-01"])
    args = parser.parse_args()
    history = pd.read_parquet(args.history)
    reference = pd.to_numeric(
        history["carbon_intensity_gco2_kwh"], errors="coerce"
    ).dropna().to_numpy()
    result = render(
        pd.read_parquet(args.input),
        reference,
        args.output,
        model=args.model,
        origins=tuple(args.origins),
    )
    print(result)


if __name__ == "__main__":
    main()
