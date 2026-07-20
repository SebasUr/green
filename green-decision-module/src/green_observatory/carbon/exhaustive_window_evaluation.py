"""Evaluate every feasible green-workload window in a dense forecast curve.

This command is intentionally a post-processing layer: it consumes already
issued rolling forecasts and never refits a carbon model.  The primary track
uses a complete 24-hour candidate set.  An optional D-1 track uses the common
1..23 support because the hour labelled exactly at the next origin is not yet
closed and therefore cannot be a causal D-1 feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from green_observatory.windows.exhaustive import evaluate_exhaustive_windows


KEYS = ["origin", "horizon", "target_time"]
DEFAULT_MODELS = (
    "physical_alpha2_calibrated_14d",
    "physical_alpha2",
    "direct_ctx3_calibrated_14d",
    "physical_alpha2_calibrated_7d",
)
MODEL_LABELS = {
    "physical_alpha2_calibrated_14d": "Physical α2 + sys + calibración 14d",
    "physical_alpha2_calibrated_7d": "Physical α2 + sys + calibración 7d",
    "physical_alpha2": "Physical α2 + sys (sin calibrar)",
    "direct_raw": "Direct + sys + intercambios (ctx3)",
    "direct_ctx3_calibrated_14d": "Direct ctx3 + escala causal 14d",
    "d1_published": "Persistencia D-1 publicada",
    "prediction_convex_blend": "Blend convexo causal",
    "prediction_calibrated_gate_14d": "Gate calibrado 14d",
}
COLORS = {
    "physical_alpha2_calibrated_14d": "#00695C",
    "physical_alpha2": "#1565C0",
    "direct_raw": "#EF6C00",
    "direct_ctx3_calibrated_14d": "#EF6C00",
    "physical_alpha2_calibrated_7d": "#7B1FA2",
    "d1_published": "#616161",
    "prediction_convex_blend": "#C62828",
    "prediction_calibrated_gate_14d": "#5D4037",
}


def _tidy_predictions(
    frame: pd.DataFrame,
    model_columns: Iterable[str],
    *,
    horizon_hours: int,
) -> pd.DataFrame:
    """Convert one-column-per-strategy checkpoints into the tidy contract."""

    model_columns = tuple(dict.fromkeys(model_columns))
    required = set(KEYS + ["actual", *model_columns])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input predictions miss columns: {missing}")
    selected = frame.loc[
        pd.to_numeric(frame["horizon"], errors="coerce").between(1, horizon_hours),
        KEYS + ["actual", *model_columns],
    ].copy()
    selected["origin"] = pd.to_datetime(selected["origin"], utc=True)
    selected["target_time"] = pd.to_datetime(selected["target_time"], utc=True)
    tidy = selected.melt(
        id_vars=KEYS + ["actual"],
        value_vars=list(model_columns),
        var_name="model",
        value_name="prediction",
    )
    return tidy[KEYS + ["model", "prediction", "actual"]]


def _indexed_carbon(path: str) -> pd.Series:
    frame = pd.read_parquet(path)
    index = pd.DatetimeIndex(frame.index)
    frame.index = (
        index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    )
    column = next(
        (
            name
            for name in ("carbon_intensity_gco2_kwh", "actual", "taux_co2")
            if name in frame
        ),
        None,
    )
    if column is None:
        raise ValueError(f"{path} has no recognized carbon-intensity column")
    return pd.to_numeric(frame[column], errors="coerce").sort_index()


def _attach_operational_comparators(
    source: pd.DataFrame,
    *,
    direct_predictions: str | None,
    carbon_history: str | None,
    carbon_live: str | None,
) -> pd.DataFrame:
    """Attach the latest scaled Direct and true published D-1 fairly by key."""

    out = source.copy()
    for column in ("origin", "target_time"):
        out[column] = pd.to_datetime(out[column], utc=True)
    if direct_predictions:
        direct = pd.read_parquet(direct_predictions).copy()
        for column in ("origin", "target_time"):
            direct[column] = pd.to_datetime(direct[column], utc=True)
        if "prediction" not in direct:
            raise ValueError("Direct predictions need a 'prediction' column")
        direct = direct[KEYS + ["prediction"]].rename(
            columns={"prediction": "direct_ctx3_calibrated_14d"}
        )
        out = out.merge(direct, on=KEYS, how="left", validate="one_to_one")
        if out["direct_ctx3_calibrated_14d"].isna().any():
            raise ValueError("scaled Direct predictions do not cover every input key")

    if carbon_history and carbon_live:
        published = pd.concat(
            [_indexed_carbon(carbon_history), _indexed_carbon(carbon_live)]
        ).sort_index()
        published = published.loc[~published.index.duplicated(keep="last")]
        lag_times = pd.DatetimeIndex(out["target_time"]) - pd.Timedelta(hours=24)
        d1 = published.reindex(lag_times).to_numpy(dtype=float).copy()
        d1[lag_times >= pd.DatetimeIndex(out["origin"])] = np.nan
        out["d1_published"] = d1
    return out


def _point_metrics(frame: pd.DataFrame, models: Iterable[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    for model in models:
        prediction = pd.to_numeric(frame[model], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
        error = prediction[valid] - actual[valid]
        absolute = np.abs(error)
        rows[model] = {
            "mape": float(100.0 * np.mean(absolute / actual[valid])),
            "wape": float(100.0 * absolute.sum() / actual[valid].sum()),
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(error.mean()),
            "n": int(valid.sum()),
        }
    return rows


def _ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    total = float(denominator.sum())
    return 100.0 * float(numerator.sum()) / total if total > 1e-12 else float("nan")


def _tail_mean(values: pd.Series, quantile: float = 0.95) -> float:
    threshold = float(values.quantile(quantile))
    return float(values.loc[values >= threshold].mean())


def _summarize_decisions(
    decisions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Aggregate at query level; overlapping candidate windows are not samples."""

    rows: list[dict] = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in decisions.groupby(grouper, sort=True):
        keys = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = dict(zip(group_columns, keys))
        opportunity = group["oracle_opportunity_gco2_kwh"]
        random_opportunity = group["random_regret_gco2_kwh"]
        row.update(
            {
                "queries": int(len(group)),
                "origins": int(group["origin"].nunique()),
                "mean_regret_gco2_kwh": float(group["regret_gco2_kwh"].mean()),
                "p90_regret_gco2_kwh": float(group["regret_gco2_kwh"].quantile(0.90)),
                "cvar95_regret_gco2_kwh": _tail_mean(group["regret_gco2_kwh"]),
                "mean_selected_actual_cost": float(group["selected_actual_cost"].mean()),
                "mean_oracle_actual_cost": float(group["oracle_actual_cost"].mean()),
                "mean_asap_actual_cost": float(group["asap_actual_cost"].mean()),
                "mean_random_actual_cost": float(
                    group["random_expected_actual_cost"].mean()
                ),
                "oracle_potential_pct": _ratio(
                    group["savings_vs_asap_gco2_kwh"], opportunity
                ),
                "model_vs_random_oracle_potential_pct": _ratio(
                    group["improvement_vs_random_gco2_kwh"], random_opportunity
                ),
                "top1_accuracy_tie_aware": float(group["top1_tie_aware"].mean()),
                "epsilon_1g_optimal_rate": float(group["epsilon_optimal"].mean()),
                "top10pct_hit_rate": float(group["top10pct_hit"].mean()),
                "mean_selected_rank_percentile": float(
                    group["selected_actual_rank_percentile"].mean()
                ),
                "mean_window_cost_spearman": float(
                    group["window_cost_spearman"].mean()
                ),
                "mean_selected_delay_hours": float(
                    (group["selected_start_horizon"] - 1).mean()
                ),
                "positive_opportunity_queries": int((opportunity > 1e-12).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _circular_block_bootstrap(
    values: np.ndarray,
    *,
    block_days: int = 14,
    draws: int = 20_000,
    seed: int = 42,
) -> list[float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return [float("nan"), float("nan")]
    block = min(int(block_days), n)
    blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    offsets = np.arange(block)
    for draw in range(draws):
        starts = rng.integers(0, n, size=blocks)
        positions = np.concatenate([(start + offsets) % n for start in starts])[:n]
        samples[draw] = values[positions].mean()
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _paired_macro_comparisons(
    decisions: pd.DataFrame,
    primary_model: str,
    *,
    max_duration: int,
) -> dict[str, dict]:
    usable = decisions.loc[decisions["duration_hours"] < max_duration]
    daily = (
        usable.groupby(["origin", "model"], sort=True)["regret_gco2_kwh"]
        .mean()
        .unstack("model")
        .sort_index()
    )
    if primary_model not in daily:
        raise ValueError(f"primary model {primary_model!r} is absent from decisions")
    report: dict[str, dict] = {}
    for reference in daily.columns:
        if reference == primary_model:
            continue
        common = daily[[primary_model, reference]].dropna()
        # candidate-reference > 0 means the primary model has lower regret.
        delta = (common[reference] - common[primary_model]).to_numpy(dtype=float)
        report[str(reference)] = {
            "definition": "reference macro-regret minus primary macro-regret",
            "mean_delta_gco2_kwh": float(delta.mean()),
            "primary_better_days_pct": float(100.0 * np.mean(delta > 1e-12)),
            "tie_days_pct": float(100.0 * np.mean(np.abs(delta) <= 1e-12)),
            "circular_14d_block_bootstrap_95ci": _circular_block_bootstrap(delta),
            "origins": int(len(delta)),
        }
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def _plot_results(
    aggregate: pd.DataFrame,
    macro: pd.DataFrame,
    catalog: pd.DataFrame,
    primary_model: str,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    display_models = list(macro.sort_values("mean_regret_gco2_kwh")["model"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model in display_models:
        group = aggregate.loc[
            (aggregate["model"] == model) & (aggregate["duration_hours"] < 24)
        ].sort_values("duration_hours")
        ax.plot(
            group["duration_hours"],
            group["mean_regret_gco2_kwh"],
            marker="o",
            markersize=3,
            linewidth=2.1 if model == primary_model else 1.5,
            color=COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.set(title="Regret al elegir entre todas las ventanas", xlabel="Duración del workload (h)", ylabel="Regret medio (gCO2/kWh; menor es mejor)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "01_regret_por_duracion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model in display_models:
        group = aggregate.loc[
            (aggregate["model"] == model) & (aggregate["duration_hours"] < 24)
        ].sort_values("duration_hours")
        ax.plot(
            group["duration_hours"],
            group["oracle_potential_pct"],
            marker="o",
            markersize=3,
            linewidth=2.1 if model == primary_model else 1.5,
            color=COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.axhline(100.0, color="#212121", linewidth=1, linestyle="--", alpha=0.65)
    ax.set(title="Ahorro ASAP→oráculo capturado", xlabel="Duración del workload (h)", ylabel="Potencial de oráculo (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_potencial_oraculo_por_duracion.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for model in display_models:
        group = aggregate.loc[
            (aggregate["model"] == model) & (aggregate["duration_hours"] < 24)
        ].sort_values("duration_hours")
        label = MODEL_LABELS.get(model, model)
        color = COLORS.get(model)
        axes[0].plot(group["duration_hours"], group["mean_window_cost_spearman"], color=color, label=label)
        axes[1].plot(group["duration_hours"], 100.0 * group["top10pct_hit_rate"], color=color, label=label)
    axes[0].set(title="Ranking de todos los inicios", xlabel="Duración (h)", ylabel="Spearman predicho vs real")
    axes[1].set(title="¿La elegida quedó en el 10 % más verde?", xlabel="Duración (h)", ylabel="Hit rate (%)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "03_calidad_ranking_todas_las_ventanas.png", dpi=180)
    plt.close(fig)

    ordered = macro.sort_values("mean_regret_gco2_kwh")
    labels = [MODEL_LABELS.get(model, model) for model in ordered["model"]]
    colors = [COLORS.get(model, "#546E7A") for model in ordered["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(labels, ordered["mean_regret_gco2_kwh"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set(title="Macro 1–23 h", xlabel="Regret medio (gCO2/kWh)")
    axes[1].barh(labels, ordered["oracle_potential_pct"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set(title="Macro 1–23 h", xlabel="Potencial de oráculo (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_resumen_macro.png", dpi=180)
    plt.close(fig)

    # One auditable example: all candidate starts for several fixed workloads.
    primary = catalog.loc[catalog["model"] == primary_model]
    duration_one = primary.loc[primary["duration_hours"] == 1]
    opportunities = duration_one.groupby("origin")["actual_window_cost"].agg(
        lambda values: float(values.iloc[0] - values.min())
    )
    example_origin = opportunities.sort_values().index[len(opportunities) // 2]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for ax, duration in zip(axes.ravel(), (1, 3, 6, 12)):
        actual_group = primary.loc[
            (primary["origin"] == example_origin)
            & (primary["duration_hours"] == duration)
        ].sort_values("start_horizon")
        ax.plot(actual_group["start_horizon"], actual_group["actual_window_cost"], color="#212121", linewidth=2.3, label="Real")
        for model in display_models:
            group = catalog.loc[
                (catalog["model"] == model)
                & (catalog["origin"] == example_origin)
                & (catalog["duration_hours"] == duration)
            ].sort_values("start_horizon")
            ax.plot(group["start_horizon"], group["predicted_window_cost"], color=COLORS.get(model), alpha=0.82, linewidth=1.25, label=MODEL_LABELS.get(model, model))
        ax.set(title=f"Workload de {duration} h", xlabel="Hora inicial dentro del horizonte", ylabel="Media gCO2/kWh")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Todas las ventanas candidatas — origen {pd.Timestamp(example_origin):%Y-%m-%d}", y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(output_dir / "05_ejemplo_todas_las_ventanas.png", dpi=180)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    rows = frame.loc[:, columns].copy()
    headers = {
        "model": "Modelo",
        "month": "Mes",
        "duration_hours": "Duración",
        "mape": "MAPE",
        "mean_regret_gco2_kwh": "Regret",
        "oracle_potential_pct": "% oráculo",
        "mean_window_cost_spearman": "Spearman",
        "top10pct_hit_rate": "Top-10 %",
        "top1_accuracy_tie_aware": "Top-1",
        "epsilon_1g_optimal_rate": "≤1 g del oráculo",
        "epsilon_optimal_rate": "≤1 g del oráculo",
    }
    rows["model"] = rows["model"].map(lambda value: MODEL_LABELS.get(value, value))
    percentage_columns = {"mape", "oracle_potential_pct"}
    rate_columns = {
        "top10pct_hit_rate",
        "top1_accuracy_tie_aware",
        "epsilon_1g_optimal_rate",
        "epsilon_optimal_rate",
    }
    for column in rows:
        if column in {"model", "duration_hours"}:
            continue
        if column == "month":
            continue
        if column in percentage_columns:
            rows[column] = rows[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.3f}%"
            )
        elif column in rate_columns:
            rows[column] = rows[column].map(
                lambda value: "—" if pd.isna(value) else f"{100.0 * float(value):.1f}%"
            )
        else:
            rows[column] = rows[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.3f}"
            )
    rows = rows.rename(columns=headers)
    output = [
        "| " + " | ".join(rows.columns) + " |",
        "|" + "|".join("---" for _ in rows.columns) + "|",
    ]
    output.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in rows.itertuples(index=False, name=None)
    )
    return output


def _write_readme(
    output_dir: Path,
    *,
    input_path: str,
    point_metrics: dict[str, dict],
    macro: pd.DataFrame,
    by_duration: pd.DataFrame,
    paired: dict[str, dict],
    monthly: pd.DataFrame,
    origins: int,
    catalog_rows: int,
    primary_model: str,
    d1_macro: pd.DataFrame | None,
) -> None:
    point = pd.DataFrame(
        [{"model": model, **metrics} for model, metrics in point_metrics.items()]
    )
    macro_with_mape = macro.merge(point[["model", "mape"]], on="model", how="left")
    key_durations = by_duration.loc[
        by_duration["duration_hours"].isin([1, 2, 3, 4, 6, 8, 12, 18, 23])
        & by_duration["model"].eq(primary_model)
    ]
    raw_pair = paired.get("physical_alpha2", {})
    direct_pair = paired.get("direct_ctx3_calibrated_14d", {})
    raw_ci = raw_pair.get("circular_14d_block_bootstrap_95ci", [np.nan, np.nan])
    direct_ci = direct_pair.get(
        "circular_14d_block_bootstrap_95ci", [np.nan, np.nan]
    )
    lines = [
        "# Evaluación exhaustiva de ventanas verdes — 24 h",
        "",
        f"Entrada: `{input_path}`.",
        "",
        f"Se evaluaron **{origins} orígenes diarios** y **{catalog_rows:,} ventanas "
        "contiguas**. Para cada duración `d=1..24`, se enumeraron todos los "
        "inicios `h1..h(25-d)`. La duración se mantiene fija: una tarea de 1 h "
        "nunca compite contra una de 12 h.",
        "",
        "## Resultado macro",
        "",
        "El macro da el mismo peso a cada duración de 1 a 23 h y sirve como "
        "resumen sintético; la tabla por duración es la lectura operativa correcta. "
        "La duración 24 h se conserva en el catálogo, pero no es una decisión: sólo "
        "existe un inicio.",
        "",
        *_markdown_table(
            macro_with_mape,
            [
                "model",
                "mape",
                "mean_regret_gco2_kwh",
                "oracle_potential_pct",
                "mean_window_cost_spearman",
                "top10pct_hit_rate",
            ],
        ),
        "",
        "## Modelo principal por duración",
        "",
        *_markdown_table(
            key_durations,
            [
                "model",
                "duration_hours",
                "mean_regret_gco2_kwh",
                "oracle_potential_pct",
                "mean_window_cost_spearman",
                "top10pct_hit_rate",
                "top1_accuracy_tie_aware",
                "epsilon_optimal_rate",
            ],
        ),
        "",
        "## Cómo leerlo",
        "",
        "- **Regret:** carbono real de la ventana escogida menos el mínimo que "
        "habría conseguido el oráculo. Cero es perfecto.",
        "- **% oráculo:** fracción del ahorro entre ejecutar ASAP y el oráculo que "
        "captura el modelo. Se calcula como razón de sumas, sin recortar a `[0,100]`.",
        "- **Spearman:** si el modelo ordena bien *todos* los inicios posibles, no "
        "solamente el ganador.",
        "- **Top-10 %:** frecuencia con que su inicio elegido cae entre el 10 % de "
        "ventanas realmente más verdes.",
        "- **Top-1:** coincidencia exacta, sensible a diferencias diminutas entre "
        "ventanas solapadas; `≤1 g` es una lectura más estable.",
        "",
        "## Comparación pareada del modelo 14d",
        "",
        "El delta es `regret del baseline − regret del 14d`; positivo favorece al "
        "14d. Los intervalos remuestrean días en bloques circulares de 14 días, "
        "porque los orígenes consecutivos no son independientes.",
        "",
        f"- Contra Physical sin calibrar: {raw_pair.get('mean_delta_gco2_kwh', float('nan')):+.4f} "
        f"gCO₂/kWh, IC95 % [{raw_ci[0]:+.4f}, {raw_ci[1]:+.4f}].",
        f"- Contra Direct ctx3 escalado: {direct_pair.get('mean_delta_gco2_kwh', float('nan')):+.4f} "
        f"gCO₂/kWh, IC95 % [{direct_ci[0]:+.4f}, {direct_ci[1]:+.4f}].",
        "",
        "La calibración 14d minimiza MAPE de nivel. Sólo puede cambiar el ranking "
        "entre sus cuatro bloques de horizonte; dentro de cada bloque multiplica "
        "todas las horas por la misma escala. Por eso una mejora de MAPE puede ser "
        "neutra —o incluso cambiar de signo— para las ventanas.",
        "",
        "## Estabilidad por mes",
        "",
        *_markdown_table(
            monthly.loc[
                monthly["model"].isin(
                    [
                        primary_model,
                        "physical_alpha2",
                        "direct_ctx3_calibrated_14d",
                    ]
                )
            ],
            [
                "month",
                "model",
                "mean_regret_gco2_kwh",
                "oracle_potential_pct",
                "mean_window_cost_spearman",
            ],
        ),
        "",
        "Para ventanas largas el porcentaje de oráculo puede caer aunque el regret "
        "sea diminuto: ASAP y el oráculo casi coinciden y el denominador del "
        "porcentaje se vuelve muy pequeño. En ese tramo conviene leer primero regret.",
    ]
    if d1_macro is not None and not d1_macro.empty:
        lines += [
            "",
            "## Comparación causal con D-1 (soporte común h1–h23)",
            "",
            "D-1 no tiene h24: ese valor correspondería a una hora que aún no está "
            "cerrada en el origen. Se repitió la evaluación sobre h1–h23 para todos "
            "los modelos, sin imputarla.",
            "",
            *_markdown_table(
                d1_macro,
                [
                    "model",
                    "mean_regret_gco2_kwh",
                    "oracle_potential_pct",
                    "mean_window_cost_spearman",
                    "top10pct_hit_rate",
                ],
            ),
        ]
    lines += [
        "",
        "## Archivos",
        "",
        "- `all_windows_24h.parquet`: cada ventana candidata y sus costes predicho/real.",
        "- `decisions_24h.parquet`: una decisión por modelo, origen y duración.",
        "- `summary_by_duration_24h.csv`: métricas por duración.",
        "- `summary.json`: protocolo, macro, comparaciones pareadas y métricas puntuales.",
        "- `*.png`: curvas y un ejemplo con todos los inicios candidatos.",
        "",
        "Esta evaluación es causal por timestamp, pero sigue siendo post-live "
        "exploratoria: la arquitectura y la calibración se definieron después de "
        "inspeccionar parte de mayo–julio.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _attach_operational_comparators(
        pd.read_parquet(args.predictions),
        direct_predictions=args.direct_predictions,
        carbon_history=args.carbon_history,
        carbon_live=args.carbon_live,
    )
    models = tuple(dict.fromkeys(args.models))
    if args.primary_model not in models:
        raise ValueError("--primary-model must also appear in --models")

    tidy = _tidy_predictions(source, models, horizon_hours=24)
    result = evaluate_exhaustive_windows(
        tidy,
        durations=tuple(range(1, 25)),
        horizon_hours=24,
        epsilon_gco2=args.epsilon_gco2,
    )
    result.catalog.to_parquet(output_dir / "all_windows_24h.parquet", index=False)
    result.decisions.to_parquet(output_dir / "decisions_24h.parquet", index=False)
    result.aggregate.to_csv(output_dir / "summary_by_duration_24h.csv", index=False)

    decision_durations = result.decisions.loc[
        result.decisions["duration_hours"] < 24
    ]
    macro = _summarize_decisions(decision_durations, ["model"])
    monthly = _summarize_decisions(
        decision_durations.assign(
            month=decision_durations["origin"].dt.strftime("%Y-%m")
        ),
        ["month", "model"],
    )
    paired = _paired_macro_comparisons(
        result.decisions, args.primary_model, max_duration=24
    )

    d1_macro = None
    d1_payload = None
    if args.include_d1 and "d1_published" in source:
        common_models = (*models, "d1_published")
        tidy23 = _tidy_predictions(source, common_models, horizon_hours=23)
        result23 = evaluate_exhaustive_windows(
            tidy23,
            durations=tuple(range(1, 24)),
            horizon_hours=23,
            epsilon_gco2=args.epsilon_gco2,
        )
        result23.catalog.to_parquet(
            output_dir / "all_windows_common23_with_d1.parquet", index=False
        )
        result23.decisions.to_parquet(
            output_dir / "decisions_common23_with_d1.parquet", index=False
        )
        result23.aggregate.to_csv(
            output_dir / "summary_by_duration_common23_with_d1.csv", index=False
        )
        d1_decision = result23.decisions.loc[
            result23.decisions["duration_hours"] < 23
        ]
        d1_macro = _summarize_decisions(d1_decision, ["model"])
        d1_payload = {
            "horizon_hours": 23,
            "reason": "D-1 h24 is not closed at the issue origin",
            "macro_durations_1_to_22": _records(d1_macro),
        }

    point = _point_metrics(source, models)
    report = {
        "protocol": {
            "input": args.predictions,
            "target": "actual (RTE provisional real-time taux_co2)",
            "origin_rule": "daily rolling forecasts already issued at 00:00 UTC",
            "horizon_hours": 24,
            "durations_hours": list(range(1, 25)),
            "candidate_rule": "all contiguous starts within h1..h24",
            "window_cost": "arithmetic mean carbon intensity (constant-power workload)",
            "selection": "lowest predicted window mean; earliest deterministic tie-break",
            "oracle": "lowest actual window mean on the identical candidate set",
            "asap": "first feasible start h1",
            "random": "uniform expected cost over all feasible starts",
            "macro": "equal weight per origin-duration query, durations 1..23",
            "inference_unit": "origin day; candidate windows are not independent samples",
            "paired_uncertainty": "circular 14-day block bootstrap over daily macro regret",
            "duration_24_status": "catalogued but excluded from decision macro (one start)",
            "primary_model": args.primary_model,
            "models": list(models),
            "scaled_direct_predictions": args.direct_predictions,
            "published_d1_sources": [args.carbon_history, args.carbon_live],
        },
        "coverage": {
            "origins": int(result.decisions["origin"].nunique()),
            "origin_start": result.decisions["origin"].min().isoformat(),
            "origin_end": result.decisions["origin"].max().isoformat(),
            "candidate_windows_per_model": int(len(result.catalog) / len(models)),
            "candidate_windows_all_models": int(len(result.catalog)),
            "decision_queries_per_model": int(len(result.decisions) / len(models)),
        },
        "point_metrics": point,
        "macro_durations_1_to_23": _records(macro),
        "by_duration": _records(result.aggregate),
        "by_month_macro_durations_1_to_23": _records(monthly),
        "paired_macro_regret_vs_primary": paired,
        "common_23h_with_d1": d1_payload,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_results(
        result.aggregate,
        macro,
        result.catalog,
        args.primary_model,
        output_dir,
    )
    _write_readme(
        output_dir,
        input_path=args.predictions,
        point_metrics=point,
        macro=macro,
        by_duration=result.aggregate,
        paired=paired,
        monthly=monthly,
        origins=int(result.decisions["origin"].nunique()),
        catalog_rows=len(result.catalog),
        primary_model=args.primary_model,
        d1_macro=d1_macro,
    )
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default=(
            "runs/daily_refit_2026/"
            "causal_operational_gate_tr_extended_ctx3/predictions.parquet"
        ),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--direct-predictions",
        default=(
            "runs/daily_refit_2026/"
            "live_direct_daily_refit_tr_extended_ctx3/summary.predictions.parquet"
        ),
        help="latest rolling Direct output; set to an empty string to disable",
    )
    parser.add_argument(
        "--carbon-history", default="data/cache/carbon_fr_hourly_detailed.parquet"
    )
    parser.add_argument(
        "--carbon-live", default="data/cache/carbon_fr_realtime_2026_full.parquet"
    )
    parser.add_argument(
        "--primary-model", default="physical_alpha2_calibrated_14d"
    )
    parser.add_argument("--epsilon-gco2", type=float, default=1.0)
    parser.add_argument(
        "--include-d1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also compare every model to causal D-1 on the common h1..h23 support",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/exhaustive_windows_physical_alpha2_14d",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
