"""Visual explanation of MAPE versus green-window oracle performance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
OUT = RUNS / "figures" / "comparision"

COLORS = {
    "ink": "#172033",
    "muted": "#657083",
    "grid": "#E5EAF1",
    "project": "#8B95A7",
    "ensemble": "#3B82F6",
    "fossil": "#F59E0B",
    "hybrid": "#8B5CF6",
    "mape": "#E76F51",
    "oracle": "#159A72",
    "actual": "#172033",
    "low_mape": "#E76F51",
    "good_rank": "#159A72",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 135,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FBFCFE",
            "axes.facecolor": "#FBFCFE",
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.edgecolor": "#AEB7C5",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.9,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def load(name: str) -> dict:
    return json.loads((RUNS / name).read_text())


def metrics(report: dict, protocol: str) -> tuple[dict, dict]:
    block = report["protocols"][protocol]
    aggregate = {row["model"]: row for row in block["aggregate_metrics"]}
    selection = {row["strategy"]: row for row in block["window_selection"]}
    return aggregate, selection


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=fig.get_facecolor())
    plt.close(fig)


def tradeoff_scatter(holdout: dict, full: dict) -> None:
    """MAPE/oracle scatter showing that the metrics are not monotonic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    configs = [
        ("project_dense24", "Baseline", COLORS["project"], "o"),
        ("ensemble_ci_dense", "EnsembleCI", COLORS["ensemble"], "s"),
        ("fossil_regime_rte", "Fósil RTE crudo", COLORS["fossil"], "^"),
        ("hybrid_h2_mapper_delta", "Híbrido adaptativo", COLORS["hybrid"], "D"),
    ]
    for ax, protocol, title in zip(
        axes,
        ("rolling_6h", "daily_utc"),
        ("Operativo · actualización cada 6h", "Benchmark · una emisión diaria"),
    ):
        agg, sel = metrics(holdout, protocol)
        full_agg, full_sel = metrics(full, protocol)
        for model, label, color, marker in configs:
            source_agg = full_agg if model == "ensemble_ci_dense" else agg
            source_sel = full_sel if model == "ensemble_ci_dense" else sel
            x = source_agg[model]["mape"]
            y = source_sel[model]["pct_oracle_potential"]
            ax.scatter(
                x,
                y,
                s=155,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=1.7,
                zorder=3,
            )
            ax.annotate(
                f"{label}\n{x:.1f}% · {y:.1f}%",
                (x, y),
                xytext=(8, 7),
                textcoords="offset points",
                fontsize=9.5,
                color=COLORS["ink"],
            )

        ensemble_x = full_agg["ensemble_ci_dense"]["mape"]
        ensemble_y = full_sel["ensemble_ci_dense"]["pct_oracle_potential"]
        fossil_x = agg["fossil_regime_rte"]["mape"]
        fossil_y = sel["fossil_regime_rte"]["pct_oracle_potential"]
        ax.annotate(
            "",
            xy=(fossil_x, fossil_y),
            xytext=(ensemble_x, ensemble_y),
            arrowprops={
                "arrowstyle": "->",
                "lw": 2,
                "color": COLORS["fossil"],
                "connectionstyle": "arc3,rad=-0.16",
            },
        )
        ax.text(
            (ensemble_x + fossil_x) / 2,
            (ensemble_y + fossil_y) / 2 + 0.35,
            "MAPE peor,\npero mejor selección",
            ha="center",
            color=COLORS["fossil"],
            fontsize=9,
            fontweight="bold",
        )
        ax.set_title(title)
        ax.set_xlabel("MAPE (%)  ·  mejor hacia la izquierda")
        ax.set_xlim(15, 37)
        ax.set_ylim(80, 97)
        ax.axvspan(15, 23, color="#EAF7F2", alpha=0.65, zorder=0)
        ax.text(
            15.4,
            96.2,
            "zona deseada\n↓ MAPE  ·  ↑ oracle",
            color=COLORS["oracle"],
            fontsize=9,
            va="top",
            fontweight="bold",
        )
    axes[0].set_ylabel("Potencial del oráculo capturado (%)  ·  mejor hacia arriba")
    fig.suptitle(
        "MAPE y oracle miden capacidades distintas",
        fontsize=19,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Holdout nuevo · 24 jun–9 jul 2026. Un modelo puede equivocarse en el nivel "
        "y aun ordenar mejor las 24 horas.",
        ha="center",
        color=COLORS["muted"],
        fontsize=10,
    )
    save(fig, "01_mape_vs_oracle_scatter.png")


def period_shift(development: dict, holdout: dict) -> None:
    """Slope chart: the same model gets worse MAPE and better oracle."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    protocols = [
        ("rolling_6h", "Rolling 6h", 11.219, 22.428, 69.3, 88.6),
        ("daily_utc", "Diario", 11.081, 17.972, 51.9, 94.1),
    ]
    x = [0, 1]
    for protocol, label, dev_mape, hold_mape, dev_oracle, hold_oracle in protocols:
        color = COLORS["hybrid"] if protocol == "rolling_6h" else COLORS["ensemble"]
        axes[0].plot(x, [dev_mape, hold_mape], "-o", lw=3, ms=10, color=color)
        mape_offset = 16 if protocol == "rolling_6h" else -20
        axes[0].annotate(
            f"{label}  {dev_mape:.1f}%",
            (0, dev_mape),
            xytext=(8, mape_offset),
            textcoords="offset points",
            ha="left",
            va="center",
        )
        axes[0].text(1.04, hold_mape, f"{hold_mape:.1f}%", ha="left", va="center")
        axes[1].plot(x, [dev_oracle, hold_oracle], "-o", lw=3, ms=10, color=color)
        axes[1].text(
            -0.04, dev_oracle, f"{label}  {dev_oracle:.1f}%", ha="right", va="center"
        )
        axes[1].text(1.04, hold_oracle, f"{hold_oracle:.1f}%", ha="left", va="center")

    for ax in axes:
        ax.set_xlim(-0.18, 1.18)
        ax.set_xticks(x, ["Desarrollo\nfeb–abr", "Holdout\njun–jul"])
        ax.grid(axis="x", visible=False)
    axes[0].set_title("El valor exacto empeoró")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_ylim(5, 26)
    axes[0].annotate(
        "Cambio de régimen y factores RTE",
        xy=(1, 22.428),
        xytext=(0.45, 24.5),
        arrowprops={"arrowstyle": "->", "color": COLORS["mape"], "lw": 1.8},
        color=COLORS["mape"],
        ha="center",
        fontweight="bold",
    )
    axes[1].set_title("Pero la hora verde se identificó mejor")
    axes[1].set_ylabel("Potencial del oráculo (%)")
    axes[1].set_ylim(42, 100)
    axes[1].annotate(
        "Mayor contraste entre\nhoras limpias y sucias",
        xy=(1, 94.1),
        xytext=(0.47, 82),
        arrowprops={"arrowstyle": "->", "color": COLORS["oracle"], "lw": 1.8},
        color=COLORS["oracle"],
        ha="center",
        fontweight="bold",
    )
    fig.suptitle(
        "La misma familia de modelo puede subir en MAPE y mejorar frente al oráculo",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Híbrido: baseline en 1–2h + fósil RTE en 3–24h; en holdout incorpora "
        "la actualización modular del mapa físico.",
        ha="center",
        color=COLORS["muted"],
        fontsize=10,
    )
    save(fig, "02_same_model_period_shift.png")


def conceptual_example() -> None:
    """A didactic 24-hour example with low MAPE but wrong minimum."""
    actual = np.array(
        [30, 29, 27, 25, 21, 16, 10, 12, 14, 16, 19, 22,
         26, 30, 33, 35, 32, 28, 25, 22, 20, 23, 27, 30],
        dtype=float,
    )
    rng = np.random.default_rng(7)
    low_mape = actual * (1 + rng.normal(0, 0.045, len(actual)))
    low_mape[6] = 16.0
    low_mape[9] = 11.0
    good_rank = actual * 1.20 + 1.0 + np.linspace(0.4, -0.4, len(actual))

    def mape(pred):
        return 100 * np.mean(np.abs(pred - actual) / actual)

    run_now = actual[0]
    oracle = actual.min()

    def oracle_capture(pred):
        realized = actual[int(np.argmin(pred))]
        return 100 * (run_now - realized) / (run_now - oracle)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    h = np.arange(1, 25)
    ax.plot(h, actual, "-o", lw=3.2, ms=5, color=COLORS["actual"], label="Señal real")
    ax.plot(
        h,
        low_mape,
        "--o",
        lw=2.2,
        ms=4,
        color=COLORS["low_mape"],
        label=f"Modelo A · MAPE {mape(low_mape):.1f}% · oracle {oracle_capture(low_mape):.0f}%",
    )
    ax.plot(
        h,
        good_rank,
        "--o",
        lw=2.2,
        ms=4,
        color=COLORS["good_rank"],
        label=f"Modelo B · MAPE {mape(good_rank):.1f}% · oracle {oracle_capture(good_rank):.0f}%",
    )
    real_min = int(np.argmin(actual))
    a_min = int(np.argmin(low_mape))
    b_min = int(np.argmin(good_rank))
    ax.axvline(h[real_min], color=COLORS["actual"], alpha=0.18, lw=8)
    ax.scatter(
        h[a_min],
        low_mape[a_min],
        s=180,
        color=COLORS["low_mape"],
        edgecolor="white",
        linewidth=2,
        zorder=4,
    )
    ax.scatter(
        h[b_min],
        good_rank[b_min],
        s=180,
        color=COLORS["good_rank"],
        edgecolor="white",
        linewidth=2,
        zorder=4,
    )
    ax.annotate(
        f"A elige h{h[a_min]}\nreal = {actual[a_min]:.0f}",
        (h[a_min], low_mape[a_min]),
        xytext=(34, 48),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["low_mape"]},
        color=COLORS["low_mape"],
        fontweight="bold",
    )
    ax.annotate(
        f"B elige h{h[b_min]}\nla hora realmente más verde",
        (h[b_min], good_rank[b_min]),
        xytext=(-120, 55),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["good_rank"]},
        color=COLORS["good_rank"],
        fontweight="bold",
    )
    ax.set_xticks(h)
    ax.set_xlabel("Horizonte dentro de las próximas 24 horas")
    ax.set_ylabel("Intensidad de carbono")
    ax.set_title("Ejemplo conceptual: acertar valores no garantiza acertar el mínimo")
    ax.legend(loc="upper right")
    ax.text(
        0.015,
        0.03,
        "El modelo A está cerca casi siempre, pero falla justo en el valle.\n"
        "El modelo B tiene un error de nivel grande, pero conserva el orden.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.55", "fc": "white", "ec": COLORS["grid"]},
    )
    ax.set_ylim(8, 44.5)
    save(fig, "03_conceptual_24h_signal.png")


def opportunity_explainer(development: dict, holdout: dict) -> None:
    """Show why oracle capture rises when the available opportunity grows."""
    fig, ax = plt.subplots(figsize=(13, 6.3))
    rows = [
        ("Desarrollo · rolling", development, "rolling_6h", "hybrid_h2"),
        ("Holdout · rolling", holdout, "rolling_6h", "hybrid_h2_mapper_delta"),
    ]
    y_positions = [1, 0]
    for y, (label, report, protocol, model) in zip(y_positions, rows):
        _, sel = metrics(report, protocol)
        run_now = sel["run_now"]["mean_realized_gco2"]
        oracle = sel["oracle"]["mean_realized_gco2"]
        chosen = sel[model]["mean_realized_gco2"]
        capture = sel[model]["pct_oracle_potential"]
        ax.plot([oracle, run_now], [y, y], color="#CBD3DF", lw=14, solid_capstyle="round")
        ax.scatter(oracle, y, s=190, color=COLORS["oracle"], edgecolor="white", lw=2, zorder=3)
        ax.scatter(chosen, y, s=220, color=COLORS["hybrid"], edgecolor="white", lw=2, zorder=3)
        ax.scatter(run_now, y, s=190, color=COLORS["project"], edgecolor="white", lw=2, zorder=3)
        ax.text(oracle, y + 0.17, f"Oráculo\n{oracle:.2f}", ha="center", color=COLORS["oracle"])
        ax.text(
            chosen,
            y - 0.19,
            f"Modelo\n{chosen:.2f}\n{capture:.1f}%",
            ha="center",
            va="top",
            color=COLORS["hybrid"],
            fontweight="bold",
        )
        ax.text(run_now, y + 0.17, f"Ejecutar ya\n{run_now:.2f}", ha="center", color=COLORS["muted"])
        ax.text(
            (oracle + run_now) / 2,
            y + 0.36,
            f"oportunidad disponible = {run_now - oracle:.2f} gCO₂/kWh",
            ha="center",
            color=COLORS["ink"],
            fontweight="bold",
        )
        ax.text(7.8, y, label, ha="right", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(7.5, 29)
    ax.set_ylim(-0.65, 1.65)
    ax.set_yticks([])
    ax.set_xlabel("CO₂ real de la hora seleccionada  ·  menor es mejor")
    ax.set_title("El oracle mejora porque en el holdout había mucho más por ganar")
    ax.text(
        0.5,
        -0.18,
        "Oracle potential = reducción capturada por el modelo / reducción máxima posible.\n"
        "No depende de acertar exactamente el nivel de toda la curva.",
        transform=ax.transAxes,
        ha="center",
        color=COLORS["muted"],
        fontsize=10,
    )
    save(fig, "04_oracle_opportunity_explained.png")


def write_readme() -> None:
    text = """# MAPE vs oracle

Estas figuras separan dos preguntas:

1. **MAPE:** ¿qué tan cerca está cada valor pronosticado del CO₂ real?
2. **Oracle potential:** ¿qué fracción de la mejor reducción posible consigue
   la hora elegida por el modelo?

- `01_mape_vs_oracle_scatter.png`: resultados reales del holdout.
- `02_same_model_period_shift.png`: el cambio entre desarrollo y holdout.
- `03_conceptual_24h_signal.png`: ejemplo didáctico de por qué las métricas divergen.
- `04_oracle_opportunity_explained.png`: efecto del tamaño de la oportunidad verde.

El holdout diario contiene 16 orígenes y el rolling 61; por ahora debe
interpretarse como evidencia prospectiva preliminar.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    style()
    development = load("compare_protocols_development_metrics.json")
    holdout = load("compare_protocols_holdout_mapper_delta_metrics.json")
    full = load("compare_protocols_holdout_full_metrics.json")
    tradeoff_scatter(holdout, full)
    period_shift(development, holdout)
    conceptual_example()
    opportunity_explainer(development, holdout)
    write_readme()
    print(f"wrote comparison figures -> {OUT}")


if __name__ == "__main__":
    main()
