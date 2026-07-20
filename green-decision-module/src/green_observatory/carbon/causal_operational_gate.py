"""Causal expert selection for the 24-hour operational carbon forecast.

This module is deliberately downstream of the forecasting models.  It never
fits a new carbon model: at each daily origin it selects, independently for
four horizon blocks, the expert with the lowest MAPE among labels that would
already have been published.  The hard condition is ``target_time < origin``.

The experiment is post-live exploratory.  It is useful for finding a robust
production policy, but its June-July score must not be described as a pristine
holdout because the gate itself was proposed after inspecting that interval.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.regime_moe import select_mape_scale


KEYS = ["origin", "horizon", "target_time"]
HORIZON_BLOCKS = ((1, 7), (8, 15), (16, 21), (22, 24))
DEFAULT_EXPERTS = ("direct_raw", "d1", "physical_alpha2")
PHYSICAL_COMPONENT_COLUMNS = (
    "predicted_bioenergy_mw",
    "predicted_fuel_oil_mw",
    "predicted_total_generation_mw",
)


def _as_utc(values: Iterable) -> pd.DatetimeIndex:
    return pd.to_datetime(values, utc=True)


def _metrics(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
    actual = actual[valid]
    prediction = prediction[valid]
    if len(actual) == 0:
        return {
            "mape": float("nan"),
            "wape": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "actual_mean": float("nan"),
            "prediction_mean": float("nan"),
            "n": 0,
        }
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mape": float(100.0 * np.mean(absolute / actual)),
        "wape": float(100.0 * absolute.sum() / actual.sum()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(prediction.mean()),
        "n": int(len(actual)),
    }


def _load_direct(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    prediction_column = (
        "prediction_raw" if "prediction_raw" in frame else "direct"
    )
    missing = set(KEYS + ["actual", prediction_column]) - set(frame.columns)
    if missing:
        raise ValueError(f"direct predictions miss columns: {sorted(missing)}")
    frame["origin"] = _as_utc(frame["origin"])
    frame["target_time"] = _as_utc(frame["target_time"])
    columns = KEYS + ["actual", prediction_column]
    if "d1" in frame:
        columns.append("d1")
    return frame[columns].rename(columns={prediction_column: "direct_raw"})


def _load_physical(
    directory: str | Path,
    expert_columns: Sequence[str] = ("physical_alpha2",),
) -> pd.DataFrame:
    directory = Path(directory)
    summary_path = directory / "predictions.parquet"
    paths = [summary_path] if summary_path.exists() else sorted(
        directory.glob("2026-*.parquet")
    )
    if not paths:
        return pd.DataFrame(columns=KEYS + list(expert_columns))
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    requested = KEYS + list(expert_columns) + list(PHYSICAL_COMPONENT_COLUMNS)
    missing = set(requested) - set(frame.columns)
    if missing:
        raise ValueError(f"physical predictions miss columns: {sorted(missing)}")
    frame["origin"] = _as_utc(frame["origin"])
    frame["target_time"] = _as_utc(frame["target_time"])
    optional = ["d1"] if "d1" in frame else []
    return (
        frame[requested + optional]
        .sort_values(KEYS)
        .drop_duplicates(KEYS, keep="last")
    )


def _load_d1_column(
    carbon_live: str | Path,
    target_times: pd.Series,
    origins: pd.Series,
    column: str,
) -> np.ndarray:
    carbon = pd.read_parquet(carbon_live)
    if column not in carbon:
        raise ValueError(f"live carbon data miss {column}")
    index = pd.DatetimeIndex(carbon.index)
    carbon.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    source_times = _as_utc(target_times) - pd.Timedelta(hours=24)
    values = (
        carbon[column]
        .reindex(source_times)
        .to_numpy(dtype=float)
        .copy()
    )
    values[source_times >= _as_utc(origins)] = np.nan
    return values


def _load_d1(
    carbon_live: str | Path,
    target_times: pd.Series,
    origins: pd.Series,
) -> np.ndarray:
    return _load_d1_column(
        carbon_live,
        target_times,
        origins,
        "carbon_intensity_gco2_kwh",
    )


def load_expert_predictions(
    direct_path: str | Path,
    carbon_live: str | Path,
    physical_directory: str | Path | None = None,
    *,
    physical_expert: str = "physical_alpha2",
    additional_physical_experts: Sequence[str] = (),
    physical_d1_variants: bool = False,
    require_complete_physical: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Load aligned forecasts and only expose a complete physical expert.

    A long physical daily-refit run writes one checkpoint per day.  Silently
    using a partly completed run would make the set of experts change midway
    through the evaluation, so it is excluded until all direct keys exist.
    """

    frame = _load_direct(direct_path).sort_values(KEYS).reset_index(drop=True)
    embedded_d1 = "d1" in frame
    if not embedded_d1:
        frame["d1"] = _load_d1(carbon_live, frame["target_time"], frame["origin"])
    experts = ["direct_raw", "d1"]
    status = {
        "direct_rows": int(len(frame)),
        "d1_rows": int(np.isfinite(frame["d1"]).sum()),
        "d1_source": "embedded prediction input" if embedded_d1 else str(carbon_live),
        "physical_requested": physical_directory is not None,
        "physical_experts": list(dict.fromkeys((physical_expert, *additional_physical_experts))),
        "physical_complete": False,
        "physical_d1_variants": bool(physical_d1_variants),
        "physical_rows": 0,
    }
    if physical_directory is not None:
        physical_experts = list(
            dict.fromkeys((physical_expert, *additional_physical_experts))
        )
        physical = _load_physical(physical_directory, physical_experts)
        status["physical_rows"] = int(len(physical))
        direct_keys = pd.MultiIndex.from_frame(frame[KEYS])
        physical_keys = pd.MultiIndex.from_frame(physical[KEYS])
        complete = len(physical) == len(frame) and direct_keys.equals(physical_keys)
        status["physical_complete"] = bool(complete)
        if complete:
            if "d1" in physical:
                physical = physical.rename(columns={"d1": "d1_physical_input"})
            frame = frame.merge(physical, on=KEYS, how="left", validate="one_to_one")
            if "d1_physical_input" in frame:
                frame["d1"] = frame.pop("d1_physical_input")
                status["d1_rows"] = int(np.isfinite(frame["d1"]).sum())
                status["d1_source"] = "embedded physical prediction input"
            experts.extend(physical_experts)
        elif not require_complete_physical and not physical.empty:
            frame = frame.merge(physical, on=KEYS, how="left", validate="one_to_one")
            experts.extend(physical_experts)
        elif require_complete_physical:
            status["physical_exclusion_reason"] = (
                "checkpoint run is incomplete; physical expert excluded atomically"
            )
        if (complete or not require_complete_physical) and physical_d1_variants:
            bio_d1 = _load_d1_column(
                carbon_live, frame["target_time"], frame["origin"], "bioenergy_mw"
            )
            oil_d1 = _load_d1_column(
                carbon_live, frame["target_time"], frame["origin"], "fuel_oil_mw"
            )
            predicted_bio = frame["predicted_bioenergy_mw"].to_numpy(dtype=float)
            predicted_oil = frame["predicted_fuel_oil_mw"].to_numpy(dtype=float)
            denominator = frame["predicted_total_generation_mw"].to_numpy(dtype=float)
            usable_bio = np.where(np.isfinite(bio_d1), bio_d1, predicted_bio)
            usable_oil = np.where(np.isfinite(oil_d1), oil_d1, predicted_oil)
            for base_expert in physical_experts:
                bio_name = f"{base_expert}_bio_d1"
                oil_bio_name = f"{base_expert}_oil_bio_d1"
                base = frame[base_expert].to_numpy(dtype=float)
                bio_variant = base + 494.0 * (usable_bio - predicted_bio) / denominator
                oil_bio_variant = bio_variant + 777.0 * (
                    usable_oil - predicted_oil
                ) / denominator
                frame[bio_name] = np.clip(bio_variant, 0.0, None)
                frame[oil_bio_name] = np.clip(oil_bio_variant, 0.0, None)
                experts.extend((bio_name, oil_bio_name))
            status["generated_physical_d1_experts"] = [
                expert
                for expert in experts
                if expert.endswith("_bio_d1") or expert.endswith("_oil_bio_d1")
            ]
        for column in PHYSICAL_COMPONENT_COLUMNS:
            if column in frame:
                frame = frame.drop(columns=column)
    return frame, experts, status


def _history(
    frame: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    block: tuple[int, int],
    lookback_days: int,
) -> pd.DataFrame:
    start, end = block
    return frame[
        (frame["origin"] < origin)
        & (frame["target_time"] < origin)
        & (frame["target_time"] >= origin - pd.Timedelta(days=lookback_days))
        & frame["horizon"].between(start, end)
    ]


def _expert_mape(history: pd.DataFrame, expert: str, scale: float = 1.0) -> tuple[float, int]:
    actual = history["actual"].to_numpy(dtype=float)
    prediction = history[expert].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
    if not valid.any():
        return float("inf"), 0
    value = 100.0 * np.mean(
        np.abs(scale * prediction[valid] - actual[valid]) / actual[valid]
    )
    return float(value), int(valid.sum())


def _recent_scale(
    history: pd.DataFrame,
    expert: str,
    *,
    shrink: float,
    clip: tuple[float, float],
) -> tuple[float, float, int]:
    actual = history["actual"].to_numpy(dtype=float)
    prediction = history[expert].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
    count = int(valid.sum())
    if count == 0:
        return 1.0, 1.0, 0
    raw, _ = select_mape_scale(
        actual[valid],
        prediction[valid],
        grid=tuple(np.arange(0.75, 1.2501, 0.01)),
    )
    effective = float(np.clip(1.0 + shrink * (raw - 1.0), *clip))
    return float(raw), effective, count


@lru_cache(maxsize=None)
def _pairwise_simplex_weights(n_experts: int, resolution: int) -> np.ndarray:
    """Return all one-expert and two-expert convex combinations.

    A full simplex grows combinatorially once physical D-1 variants are added.
    Restricting support to at most two experts keeps the online policy cheap
    and is also a useful regularizer for a 14-day calibration window.
    """

    if n_experts < 1 or resolution < 1:
        raise ValueError("n_experts and resolution must be positive")

    rows = []
    for expert in range(n_experts):
        weights = np.zeros(n_experts, dtype=float)
        weights[expert] = 1.0
        rows.append(weights)
    for left in range(n_experts):
        for right in range(left + 1, n_experts):
            for units in range(1, resolution):
                weights = np.zeros(n_experts, dtype=float)
                weights[left] = units / resolution
                weights[right] = 1.0 - weights[left]
                rows.append(weights)
    return np.asarray(rows, dtype=float)


def causal_gate(
    frame: pd.DataFrame,
    experts: Sequence[str] = DEFAULT_EXPERTS,
    *,
    lookback_days: int = 14,
    blocks: Sequence[tuple[int, int]] = HORIZON_BLOCKS,
    fallback_expert: str = "direct_raw",
    calibrated_lookbacks: Sequence[int] = (7, 14),
    scale_shrink: float = 0.5,
    scale_clip: tuple[float, float] = (0.80, 1.20),
    blend_grid_step: float = 0.05,
) -> pd.DataFrame:
    """Apply raw and calibrated causal gates.

    Scale variants estimate one multiplicative factor per expert and horizon
    block, using exactly the same past-only visibility rule as the raw gate.
    The selected grid scale is shrunk halfway toward one and clipped.
    """

    frame = frame.copy()
    frame["origin"] = _as_utc(frame["origin"])
    frame["target_time"] = _as_utc(frame["target_time"])
    experts = [expert for expert in experts if expert in frame.columns]
    if fallback_expert not in experts:
        raise ValueError(f"fallback expert {fallback_expert!r} is unavailable")
    if not experts:
        raise ValueError("at least one expert is required")
    for column in ("prediction_gate", "selected_expert", "history_mape", "history_rows"):
        frame[column] = np.nan if column != "selected_expert" else ""
    frame["prediction_convex_blend"] = np.nan
    frame["blend_history_mape"] = np.nan
    frame["blend_history_rows"] = 0
    for expert in experts:
        frame[f"blend_weight_{expert}"] = np.nan
    for days in calibrated_lookbacks:
        frame[f"prediction_calibrated_gate_{days}d"] = np.nan
        frame[f"selected_expert_calibrated_{days}d"] = ""
        frame[f"selected_scale_calibrated_{days}d"] = np.nan
        for expert in experts:
            frame[f"{expert}_scale_{days}d"] = np.nan
            frame[f"{expert}_calibrated_{days}d"] = np.nan

    for origin in sorted(frame["origin"].unique()):
        origin = pd.Timestamp(origin)
        for start, end in blocks:
            current_mask = frame["origin"].eq(origin) & frame["horizon"].between(start, end)
            current = frame.loc[current_mask]
            if current.empty:
                continue

            history = _history(
                frame,
                origin=origin,
                block=(start, end),
                lookback_days=lookback_days,
            )
            raw_scores = {}
            raw_counts = {}
            for expert in experts:
                score, count = _expert_mape(history, expert)
                raw_scores[expert] = score
                raw_counts[expert] = count
            eligible = [
                expert
                for expert in experts
                if raw_counts[expert] > 0 and np.isfinite(current[expert]).all()
            ]
            selected = min(eligible, key=lambda expert: raw_scores[expert]) if eligible else fallback_expert
            selected_prediction = current[selected].to_numpy(dtype=float)
            missing = ~np.isfinite(selected_prediction)
            if missing.any():
                selected_prediction[missing] = current[fallback_expert].to_numpy(dtype=float)[missing]
            frame.loc[current_mask, "prediction_gate"] = selected_prediction
            frame.loc[current_mask, "selected_expert"] = selected
            frame.loc[current_mask, "history_mape"] = (
                raw_scores[selected] if np.isfinite(raw_scores[selected]) else np.nan
            )
            frame.loc[current_mask, "history_rows"] = raw_counts[selected]

            # Continuous counterpart of the discrete gate.  The weight grid is
            # fixed before seeing an origin and its loss is evaluated on the
            # exact same closed-label history.  Experts unavailable anywhere
            # in the current block are excluded from that block's simplex.
            blend_experts = [
                expert for expert in experts if np.isfinite(current[expert]).all()
            ]
            valid_history = np.isfinite(history["actual"].to_numpy(dtype=float)) & (
                history["actual"].to_numpy(dtype=float) > 0.0
            )
            for expert in blend_experts:
                valid_history &= np.isfinite(history[expert].to_numpy(dtype=float))
            blend_weights = {expert: 0.0 for expert in experts}
            if valid_history.any() and blend_experts:
                resolution = int(round(1.0 / blend_grid_step))
                if not np.isclose(resolution * blend_grid_step, 1.0):
                    raise ValueError("blend_grid_step must divide one exactly")
                grid = _pairwise_simplex_weights(len(blend_experts), resolution)
                hist_actual = history.loc[valid_history, "actual"].to_numpy(dtype=float)
                hist_matrix = history.loc[valid_history, blend_experts].to_numpy(dtype=float)
                candidate_predictions = grid @ hist_matrix.T
                losses = np.mean(
                    np.abs(candidate_predictions - hist_actual[None, :])
                    / hist_actual[None, :],
                    axis=1,
                )
                fallback_position = blend_experts.index(fallback_expert)
                chosen_position = int(
                    np.argmin(losses + 1e-12 * (1.0 - grid[:, fallback_position]))
                )
                chosen_weights = grid[chosen_position]
                blend_score = float(100.0 * losses[chosen_position])
                blend_rows = int(valid_history.sum())
                for expert, weight in zip(blend_experts, chosen_weights):
                    blend_weights[expert] = float(weight)
            else:
                blend_weights[fallback_expert] = 1.0
                blend_score = float("nan")
                blend_rows = 0
            current_prediction = sum(
                blend_weights[expert] * current[expert].to_numpy(dtype=float)
                for expert in experts
                if blend_weights[expert] > 0.0
            )
            frame.loc[current_mask, "prediction_convex_blend"] = current_prediction
            frame.loc[current_mask, "blend_history_mape"] = blend_score
            frame.loc[current_mask, "blend_history_rows"] = blend_rows
            for expert, weight in blend_weights.items():
                frame.loc[current_mask, f"blend_weight_{expert}"] = weight

            for days in calibrated_lookbacks:
                scale_history = _history(
                    frame,
                    origin=origin,
                    block=(start, end),
                    lookback_days=int(days),
                )
                calibrated_scores = {}
                calibrated_counts = {}
                effective_scales = {}
                for expert in experts:
                    _, effective, count = _recent_scale(
                        scale_history,
                        expert,
                        shrink=scale_shrink,
                        clip=scale_clip,
                    )
                    score, _ = _expert_mape(scale_history, expert, scale=effective)
                    calibrated_scores[expert] = score
                    calibrated_counts[expert] = count
                    effective_scales[expert] = effective
                    frame.loc[current_mask, f"{expert}_scale_{days}d"] = effective
                    frame.loc[current_mask, f"{expert}_calibrated_{days}d"] = (
                        current[expert].to_numpy(dtype=float) * effective
                    )
                eligible = [
                    expert
                    for expert in experts
                    if calibrated_counts[expert] > 0 and np.isfinite(current[expert]).all()
                ]
                calibrated_selected = (
                    min(eligible, key=lambda expert: calibrated_scores[expert])
                    if eligible
                    else fallback_expert
                )
                prediction = (
                    current[calibrated_selected].to_numpy(dtype=float)
                    * effective_scales[calibrated_selected]
                )
                missing = ~np.isfinite(prediction)
                if missing.any():
                    prediction[missing] = current[fallback_expert].to_numpy(dtype=float)[missing]
                frame.loc[current_mask, f"prediction_calibrated_gate_{days}d"] = prediction
                frame.loc[current_mask, f"selected_expert_calibrated_{days}d"] = calibrated_selected
                frame.loc[current_mask, f"selected_scale_calibrated_{days}d"] = effective_scales[
                    calibrated_selected
                ]

    return frame


def _selection_counts(
    frame: pd.DataFrame,
    selection_column: str,
    blocks: Sequence[tuple[int, int]] = HORIZON_BLOCKS,
) -> dict:
    result = {}
    for start, end in blocks:
        label = f"h{start}-{end}"
        decisions = frame.loc[
            frame["horizon"].between(start, end), ["origin", selection_column]
        ].drop_duplicates()
        result[label] = {
            str(expert): int(count)
            for expert, count in decisions[selection_column].value_counts().items()
        }
    return result


def _paired_comparison(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
) -> dict:
    actual = frame["actual"].to_numpy(dtype=float)
    delta = 100.0 * (
        np.abs(frame[candidate].to_numpy(dtype=float) - actual)
        - np.abs(frame[reference].to_numpy(dtype=float) - actual)
    ) / actual
    daily = pd.Series(delta, index=frame["origin"]).groupby(level=0).mean().to_numpy()
    rng = np.random.default_rng(42)
    n_days = len(daily)
    day_bootstrap = np.mean(
        daily[rng.integers(0, n_days, size=(20_000, n_days))], axis=1
    )
    n_blocks = int(np.ceil(n_days / 7))
    block_bootstrap = []
    for _ in range(20_000):
        starts = rng.integers(0, n_days, size=n_blocks)
        sample = np.concatenate(
            [daily[(start + np.arange(7)) % n_days] for start in starts]
        )[:n_days]
        block_bootstrap.append(sample.mean())
    return {
        "mape_delta_points": float(delta.mean()),
        "days_better_percent": float(100.0 * np.mean(daily < 0.0)),
        "day_bootstrap_95ci": [
            float(value) for value in np.quantile(day_bootstrap, [0.025, 0.975])
        ],
        "circular_7d_block_bootstrap_95ci": [
            float(value) for value in np.quantile(block_bootstrap, [0.025, 0.975])
        ],
    }


def summarize(
    frame: pd.DataFrame,
    experts: Sequence[str],
    *,
    load_status: dict,
    raw_lookback_days: int,
    calibrated_lookbacks: Sequence[int],
    scale_shrink: float,
    scale_clip: tuple[float, float],
    blend_grid_step: float,
) -> dict:
    model_columns = list(experts) + ["prediction_gate", "prediction_convex_blend"]
    for days in calibrated_lookbacks:
        model_columns += [f"prediction_calibrated_gate_{days}d"]
        model_columns += [f"{expert}_calibrated_{days}d" for expert in experts]
    metrics = {column: _metrics(frame["actual"], frame[column]) for column in model_columns}
    report = {
        "status": "post-live exploratory; not a pristine prospective holdout",
        "protocol": {
            "decision_frequency": "once per origin and horizon block",
            "horizon_blocks": [list(block) for block in HORIZON_BLOCKS],
            "raw_gate_lookback_days": int(raw_lookback_days),
            "visibility_rule": "origin_past < origin and target_time < origin",
            "fallback": "direct_raw only when no eligible past score exists",
            "expert_menu_status": (
                "post-live exploratory; causal online selection is distinct "
                "from the static Jan-Feb proxy preselection"
            ),
            "loss": "MAPE over already observable rows in the same horizon block",
            "convex_blend_grid_step": float(blend_grid_step),
            "convex_blend_support": "at most two experts",
            "calibrated_gate_lookback_days": [int(value) for value in calibrated_lookbacks],
            "calibration": {
                "grid": [0.75, 1.25, 0.01],
                "shrink_toward_one": float(scale_shrink),
                "effective_scale_clip": [float(value) for value in scale_clip],
            },
        },
        "input_status": load_status,
        "experts": list(experts),
        "overall": metrics,
        "by_block": {},
        "selection_counts_raw_gate": _selection_counts(frame, "selected_expert"),
        "selection_counts_calibrated_gate": {},
        "convex_blend_mean_weights_by_block": {},
        "paired_convex_blend_vs_direct_raw": _paired_comparison(
            frame, "prediction_convex_blend", "direct_raw"
        ),
        "by_month": {},
    }
    for start, end in HORIZON_BLOCKS:
        group = frame[frame["horizon"].between(start, end)]
        report["by_block"][f"h{start}-{end}"] = {
            column: _metrics(group["actual"], group[column]) for column in model_columns
        }
    for days in calibrated_lookbacks:
        report["selection_counts_calibrated_gate"][f"{days}d"] = _selection_counts(
            frame, f"selected_expert_calibrated_{days}d"
        )
    monthly_columns = list(experts) + [
        "prediction_gate",
        "prediction_convex_blend",
        *(f"prediction_calibrated_gate_{days}d" for days in calibrated_lookbacks),
    ]
    report["by_month"] = {
        str(month): {
            column: _metrics(group["actual"], group[column])
            for column in monthly_columns
        }
        for month, group in frame.groupby(frame["origin"].dt.strftime("%Y-%m"))
    }
    for start, end in HORIZON_BLOCKS:
        decisions = frame.loc[
            frame["horizon"].eq(start),
            [f"blend_weight_{expert}" for expert in experts],
        ]
        report["convex_blend_mean_weights_by_block"][f"h{start}-{end}"] = {
            expert: float(decisions[f"blend_weight_{expert}"].mean())
            for expert in experts
        }
    return report


def _readme(report: dict) -> str:
    rows = []
    for model, metrics in report["overall"].items():
        rows.append(
            f"| `{model}` | {metrics['mape']:.3f}% | {metrics['wape']:.3f}% | "
            f"{metrics['mae']:.3f} | {metrics['bias']:+.3f} | {metrics['n']} |"
        )
    experts = ", ".join(f"`{value}`" for value in report["experts"])
    physical_status = (
        "completo e incluido"
        if report["input_status"].get("physical_complete")
        else "incompleto y excluido"
    )
    return f"""# Selector causal operacional — exploratorio

Este experimento combina {experts}. Para cada origen diario y cada bloque
`h1-7`, `h8-15`, `h16-21` y `h22-24`, escoge el experto con menor MAPE en los
14 días previos. Una fila sólo entra a la calibración si
`target_time < origin`; el día que se está prediciendo y el futuro no se usan.

También se reportan dos ablaciones de escala, con memoria de 7 y 14 días. La
escala MAPE de cada experto se contrae 50% hacia 1 y queda acotada a [0.80,
1.20] antes de seleccionar el experto. Esto evita que una ventana corta aplique
íntegramente una corrección extrema.

`prediction_convex_blend` es la versión continua: en vez de elegir un solo
experto, busca en una rejilla de 0.05 mezclas convexas de hasta dos expertos,
siempre con el mismo historial causal de 14 días y por bloque. Limitarlo a dos
evita una explosión combinatoria y regulariza una ventana de calibración corta.

| Salida | MAPE | WAPE | MAE | sesgo | n |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Lectura correcta

La ejecución es causal fila por fila, pero el diseño del selector fue propuesto
después de inspeccionar este holdout de junio-julio. Por ello su resultado es
**exploratorio post-live**, no una estimación prospectiva limpia. Para promoverlo
habría que congelar bloques, memoria, shrink y clipping, y repetirlo en un
periodo futuro intacto.

Advertencia adicional del input: **{report['input_status'].get('input_caveat',
'ninguna')}**.

Esto tampoco es la preselección estática del proxy: aquella escogió una cabeza
con febrero antes de abrir el siguiente periodo. Aquí el menú de expertos y el
gate fueron definidos tras haber inspeccionado junio-julio; sólo la mecánica de
cada decisión individual es causal.

La elección del experto usa únicamente errores ya publicados. Si no existe
historia elegible, usa `direct_raw`. El proxy físico sólo se incorpora cuando
sus checkpoints cubren exactamente todas las claves del modelo directo; nunca
se mezcla una corrida física incompleta. En esta ejecución quedó:
**{physical_status}**.

## Reproducir

```bash
/Users/saur/miniconda3/envs/green-observatory/bin/python \\
  -m green_observatory.carbon.causal_operational_gate
```
"""


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, experts, load_status = load_expert_predictions(
        args.direct_predictions,
        args.carbon_live,
        args.physical_directory,
        physical_expert=args.physical_expert,
        additional_physical_experts=args.additional_physical_experts,
        physical_d1_variants=args.physical_d1_variants,
        require_complete_physical=not args.allow_partial_physical,
    )
    if args.input_caveat:
        load_status["input_caveat"] = str(args.input_caveat)
    calibrated_lookbacks = tuple(int(value) for value in args.calibrated_lookbacks)
    gated = causal_gate(
        frame,
        experts,
        lookback_days=args.lookback_days,
        calibrated_lookbacks=calibrated_lookbacks,
        scale_shrink=args.scale_shrink,
        scale_clip=(args.scale_clip_min, args.scale_clip_max),
        blend_grid_step=args.blend_grid_step,
    )
    report = summarize(
        gated,
        experts,
        load_status=load_status,
        raw_lookback_days=args.lookback_days,
        calibrated_lookbacks=calibrated_lookbacks,
        scale_shrink=args.scale_shrink,
        scale_clip=(args.scale_clip_min, args.scale_clip_max),
        blend_grid_step=args.blend_grid_step,
    )
    gated.to_parquet(output / "predictions.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.md").write_text(_readme(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-predictions",
        default="runs/daily_refit_2026/live_direct_daily_refit/summary.predictions.parquet",
    )
    parser.add_argument(
        "--carbon-live", default="data/cache/carbon_fr_realtime_holdout.parquet"
    )
    parser.add_argument(
        "--physical-directory",
        default="runs/daily_refit_2026/realtime_proxy_daily_refit_physical",
    )
    parser.add_argument("--physical-expert", default="physical_alpha2")
    parser.add_argument("--additional-physical-experts", nargs="*", default=())
    parser.add_argument("--physical-d1-variants", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--calibrated-lookbacks", nargs="+", type=int, default=(7, 14))
    parser.add_argument("--scale-shrink", type=float, default=0.5)
    parser.add_argument("--scale-clip-min", type=float, default=0.80)
    parser.add_argument("--scale-clip-max", type=float, default=1.20)
    parser.add_argument("--blend-grid-step", type=float, default=0.05)
    parser.add_argument("--allow-partial-physical", action="store_true")
    parser.add_argument(
        "--input-caveat",
        default=None,
        help="Free-text provenance warning copied into summary and README",
    )
    parser.add_argument(
        "--output-dir", default="runs/daily_refit_2026/causal_operational_gate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
