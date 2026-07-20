"""Causal residual calibration for the consolidated physical forecast.

The module is intentionally downstream of the expensive forecasting models.
It consumes their frozen predictions and estimates only a multiplicative
recent-level correction.  Every correction issued at ``origin`` is fitted on
rows satisfying both ``past_origin < origin`` and ``target_time < origin``.

This is an exploratory architecture: the usefulness of a source-disaggregated
physical oracle was first noticed on March-April.  The internal model choices
are nevertheless staged honestly: January selects the level gate, February
selects the online-calibration rule, and March-April is never used by either
selection step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


KEYS = ["origin", "horizon", "target_time"]
HORIZON_BLOCKS = ((1, 6), (7, 16), (17, 21), (22, 24))
DEFAULT_LOOKBACKS = (3, 7, 14, 21, 28)
DEFAULT_SHRINKS = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_SCALE_GRID = tuple(np.arange(0.70, 1.3001, 0.001))
DEFAULT_BLEND_GRID = tuple(np.arange(0.0, 1.001, 0.05))
DEFAULT_THRESHOLD_GRID = tuple(np.arange(5.0, 30.01, 0.5))
DEFAULT_SHARE_LEVEL_GRID = tuple(np.arange(8.0, 30.01, 1.0))


@dataclass(frozen=True)
class CalibrationSpec:
    """Frozen online-calibration hyperparameters."""

    lookback_days: int
    block_mode: str
    shrink: float

    @property
    def blocks(self) -> tuple[tuple[int, int], ...]:
        if self.block_mode == "global":
            return ((1, 24),)
        if self.block_mode == "four_blocks":
            return HORIZON_BLOCKS
        raise ValueError(f"unsupported block mode: {self.block_mode!r}")


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(
        KEYS
        + [
            "actual",
            "physical_lgbm",
            "direct_reference",
            "oracle_learned_factors",
        ]
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"consolidated predictions miss columns: {sorted(missing)}")
    out = frame.copy()
    out["origin"] = pd.to_datetime(out["origin"], utc=True)
    out["target_time"] = pd.to_datetime(out["target_time"], utc=True)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="raise").astype(int)
    out = out.sort_values(KEYS).reset_index(drop=True)
    if out.duplicated(KEYS).any():
        raise ValueError("consolidated predictions contain duplicate forecast keys")
    out["_row_id"] = np.arange(len(out), dtype=int)
    return out


def attach_share_predictions(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Align only the clean share head; ignore every precomputed ensemble column."""

    share = pd.read_parquet(path)
    required = set(KEYS + ["actual", "share_lgbm"])
    missing = required - set(share.columns)
    if missing:
        raise ValueError(f"share predictions miss columns: {sorted(missing)}")
    share = share[KEYS + ["actual", "share_lgbm"]].copy()
    share["origin"] = pd.to_datetime(share["origin"], utc=True)
    share["target_time"] = pd.to_datetime(share["target_time"], utc=True)
    share["horizon"] = pd.to_numeric(share["horizon"], errors="raise").astype(int)
    if share.duplicated(KEYS).any():
        raise ValueError("share predictions contain duplicate forecast keys")
    out = frame.merge(
        share.rename(columns={"actual": "share_actual"}),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if out["share_lgbm"].isna().any():
        raise ValueError("share predictions do not cover every physical/direct key")
    mismatch = np.abs(
        out["actual"].to_numpy(dtype=float)
        - out["share_actual"].to_numpy(dtype=float)
    )
    if np.nanmax(mismatch) > 1e-9:
        raise ValueError("share and physical prediction targets are not aligned")
    return out.drop(columns="share_actual")


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


def select_blend_weight(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    grid: Sequence[float] = DEFAULT_BLEND_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Select the physical weight of ``w * physical + (1-w) * Direct``."""

    candidates = []
    for physical_weight in grid:
        prediction = (
            float(physical_weight) * frame.loc[mask, "physical_lgbm"]
            + (1.0 - float(physical_weight)) * frame.loc[mask, "direct_reference"]
        )
        candidates.append(
            {
                "physical_weight": float(physical_weight),
                "mape": _metrics(frame.loc[mask, "actual"], prediction)["mape"],
            }
        )
    selected = min(candidates, key=lambda row: (row["mape"], row["physical_weight"]))
    return float(selected["physical_weight"]), candidates


def select_level_threshold(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Select a physical-prediction level gate on the supplied period only."""

    candidates = []
    physical = frame.loc[mask, "physical_lgbm"].to_numpy(dtype=float)
    direct = frame.loc[mask, "direct_reference"].to_numpy(dtype=float)
    actual = frame.loc[mask, "actual"].to_numpy(dtype=float)
    for threshold in grid:
        prediction = np.where(physical < float(threshold), physical, direct)
        candidates.append(
            {
                "threshold_gco2_kwh": float(threshold),
                "mape": _metrics(actual, prediction)["mape"],
            }
        )
    selected = min(
        candidates, key=lambda row: (row["mape"], row["threshold_gco2_kwh"])
    )
    return float(selected["threshold_gco2_kwh"]), candidates


def select_share_convex_weight(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    reference_column: str = "base_level_gate_january",
    grid: Sequence[float] = DEFAULT_BLEND_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Select a coarse one-dimensional share/reference blend on January."""

    candidates = []
    for share_weight in grid:
        prediction = (
            float(share_weight) * frame.loc[mask, "share_lgbm"]
            + (1.0 - float(share_weight)) * frame.loc[mask, reference_column]
        )
        candidates.append(
            {
                "share_weight": float(share_weight),
                "mape": _metrics(frame.loc[mask, "actual"], prediction)["mape"],
            }
        )
    selected = min(candidates, key=lambda row: (row["mape"], row["share_weight"]))
    return float(selected["share_weight"]), candidates


def select_share_level_thresholds(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    grid: Sequence[float] = DEFAULT_SHARE_LEVEL_GRID,
) -> tuple[tuple[float, float], list[dict[str, float]]]:
    """Select a simple three-regime gate using January and physical level.

    The expert order is fixed before selection: share at low physical levels,
    Direct in the middle and the MW-component physical model at high levels.
    Only the two thresholds are fitted.
    """

    physical = frame.loc[mask, "physical_lgbm"].to_numpy(dtype=float)
    share = frame.loc[mask, "share_lgbm"].to_numpy(dtype=float)
    direct = frame.loc[mask, "direct_reference"].to_numpy(dtype=float)
    actual = frame.loc[mask, "actual"].to_numpy(dtype=float)
    candidates = []
    for low_threshold in grid:
        for high_threshold in grid:
            if float(high_threshold) <= float(low_threshold):
                continue
            prediction = np.where(
                physical < float(low_threshold),
                share,
                np.where(physical < float(high_threshold), direct, physical),
            )
            candidates.append(
                {
                    "low_threshold_gco2_kwh": float(low_threshold),
                    "high_threshold_gco2_kwh": float(high_threshold),
                    "mape": _metrics(actual, prediction)["mape"],
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            row["mape"],
            row["high_threshold_gco2_kwh"] - row["low_threshold_gco2_kwh"],
            row["low_threshold_gco2_kwh"],
        ),
    )
    return (
        (
            float(selected["low_threshold_gco2_kwh"]),
            float(selected["high_threshold_gco2_kwh"]),
        ),
        candidates,
    )


def _scale_from_history(
    history: pd.DataFrame,
    base_column: str,
    *,
    grid: Sequence[float],
) -> tuple[float, int]:
    actual = history["actual"].to_numpy(dtype=float)
    prediction = history[base_column].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual > 0.0)
    count = int(valid.sum())
    if count == 0:
        return 1.0, 0
    actual = actual[valid]
    prediction = prediction[valid]
    scales = np.asarray(grid, dtype=float)
    losses = np.mean(
        np.abs(scales[:, None] * prediction[None, :] - actual[None, :])
        / actual[None, :],
        axis=1,
    )
    return float(scales[int(np.argmin(losses))]), count


def causal_scale(
    frame: pd.DataFrame,
    base_column: str,
    evaluation_origins: Iterable,
    spec: CalibrationSpec,
    *,
    history_origin_floor=None,
    scale_grid: Sequence[float] = DEFAULT_SCALE_GRID,
) -> pd.DataFrame:
    """Apply a frozen scale rule using labels strictly closed at each origin.

    ``history_origin_floor`` implements a fresh-start sensitivity.  For
    example, a floor of 2026-03-01 excludes every forecast issued before March
    even after its label becomes available.  Without a floor, January-February
    acts as a realistic warm seed for the March deployment.
    """

    if base_column not in frame:
        raise ValueError(f"unknown base prediction column: {base_column!r}")
    origins = {_utc(value) for value in evaluation_origins}
    floor = _utc(history_origin_floor) if history_origin_floor is not None else None
    pieces: list[pd.DataFrame] = []
    for origin in sorted(origins):
        current = frame.loc[frame["origin"].eq(origin)].copy()
        if current.empty:
            continue
        for start, end in spec.blocks:
            block_current = current.loc[current["horizon"].between(start, end)].copy()
            if block_current.empty:
                continue
            history_mask = (
                (frame["origin"] < origin)
                & (frame["target_time"] < origin)
                & (
                    frame["target_time"]
                    >= origin - pd.Timedelta(days=int(spec.lookback_days))
                )
                & frame["horizon"].between(start, end)
            )
            if floor is not None:
                history_mask &= frame["origin"] >= floor
            raw_scale, history_rows = _scale_from_history(
                frame.loc[history_mask], base_column, grid=scale_grid
            )
            effective_scale = float(1.0 + spec.shrink * (raw_scale - 1.0))
            block_current["prediction"] = np.clip(
                block_current[base_column].to_numpy(dtype=float) * effective_scale,
                0.0,
                None,
            )
            block_current["raw_scale"] = raw_scale
            block_current["effective_scale"] = effective_scale
            block_current["history_rows"] = history_rows
            block_current["calibration_block"] = f"h{start}-{end}"
            pieces.append(
                block_current[
                    [
                        "_row_id",
                        "prediction",
                        "raw_scale",
                        "effective_scale",
                        "history_rows",
                        "calibration_block",
                    ]
                ]
            )
    if not pieces:
        return pd.DataFrame(
            columns=[
                "_row_id",
                "prediction",
                "raw_scale",
                "effective_scale",
                "history_rows",
                "calibration_block",
            ]
        )
    return pd.concat(pieces, ignore_index=True).sort_values("_row_id")


def tune_calibration(
    frame: pd.DataFrame,
    base_column: str,
    validation_origins: Iterable,
    *,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    block_modes: Sequence[str] = ("global", "four_blocks"),
    shrinks: Sequence[float] = DEFAULT_SHRINKS,
    scale_grid: Sequence[float] = DEFAULT_SCALE_GRID,
) -> tuple[CalibrationSpec, list[dict]]:
    """Select calibration hyperparameters by causal February simulation."""

    origins = tuple(_utc(value) for value in validation_origins)
    candidates: list[dict] = []
    for lookback in lookbacks:
        for block_mode in block_modes:
            for shrink in shrinks:
                spec = CalibrationSpec(int(lookback), str(block_mode), float(shrink))
                calibrated = causal_scale(
                    frame,
                    base_column,
                    origins,
                    spec,
                    scale_grid=scale_grid,
                )
                rows = frame.set_index("_row_id").loc[calibrated["_row_id"]]
                metric = _metrics(rows["actual"], calibrated["prediction"])
                candidates.append({**asdict(spec), **metric})
    mode_order = {"global": 0, "four_blocks": 1}
    selected = min(
        candidates,
        key=lambda row: (
            row["mape"],
            row["shrink"],
            mode_order[row["block_mode"]],
            row["lookback_days"],
        ),
    )
    return (
        CalibrationSpec(
            int(selected["lookback_days"]),
            str(selected["block_mode"]),
            float(selected["shrink"]),
        ),
        candidates,
    )


def _assign_calibration(
    frame: pd.DataFrame,
    name: str,
    calibrated: pd.DataFrame,
) -> None:
    lookup = calibrated.set_index("_row_id")
    for source, suffix in (
        ("prediction", ""),
        ("raw_scale", "_raw_scale"),
        ("effective_scale", "_scale"),
        ("history_rows", "_history_rows"),
        ("calibration_block", "_block"),
    ):
        frame[name + suffix] = frame["_row_id"].map(lookup[source])


def _model_metrics(frame: pd.DataFrame, columns: Sequence[str]) -> dict:
    return {
        column: _metrics(frame["actual"], frame[column])
        for column in columns
        if column in frame and frame[column].notna().any()
    }


def _window_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    actual_by_time: pd.Series | None = None,
) -> dict:
    realized: list[float] = []
    oracle: list[float] = []
    run_now: list[float] = []
    regrets: list[float] = []
    top1: list[float] = []
    spearman: list[float] = []
    if actual_by_time is None:
        actual_by_time = (
            frame.dropna(subset=["actual"])
            .drop_duplicates("target_time", keep="last")
            .set_index("target_time")["actual"]
        )
    for origin, group in frame.groupby("origin"):
        group = group.sort_values("horizon")
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group[prediction_column].to_numpy(dtype=float)
        if (
            len(group) != 24
            or not np.isfinite(actual).all()
            or not np.isfinite(prediction).all()
        ):
            continue
        selected = int(np.argmin(prediction))
        best = float(np.min(actual))
        incurred = float(actual[selected])
        realized.append(incurred)
        oracle.append(best)
        regrets.append(incurred - best)
        top1.append(float(incurred <= best + 1e-9))
        now = actual_by_time.get(origin, np.nan)
        if np.isfinite(now):
            run_now.append(float(now))
        if np.std(prediction) > 0.0 and np.std(actual) > 0.0:
            spearman.append(float(spearmanr(prediction, actual).statistic))
    if not realized:
        return {}
    mean_realized = float(np.mean(realized))
    mean_oracle = float(np.mean(oracle))
    mean_now = float(np.mean(run_now)) if run_now else float("nan")
    denominator = mean_now - mean_oracle
    oracle_potential = (
        100.0 * (mean_now - mean_realized) / denominator
        if np.isfinite(mean_now) and denominator > 1e-9
        else float("nan")
    )
    return {
        "mean_realized_gco2": mean_realized,
        "mean_oracle_gco2": mean_oracle,
        "mean_run_now_gco2": mean_now,
        "mean_regret": float(np.mean(regrets)),
        "pct_oracle_potential": float(oracle_potential),
        "spearman": float(np.mean(spearman)) if spearman else float("nan"),
        "top1_accuracy": float(np.mean(top1)),
        "n": int(len(realized)),
    }


def _window_report(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    actual_by_time: pd.Series | None = None,
) -> dict:
    return {
        column: _window_metrics(
            frame, column, actual_by_time=actual_by_time
        )
        for column in columns
        if column in frame and frame[column].notna().any()
    }


def _bootstrap_ci(
    values: np.ndarray,
    *,
    block_days: int | None,
    samples: int = 20_000,
    seed: int = 20260719,
) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    if block_days is None:
        means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(
            axis=1
        )
    else:
        width = min(int(block_days), len(values))
        blocks_needed = int(np.ceil(len(values) / width))
        means = np.empty(samples, dtype=float)
        offsets = np.arange(width)
        for sample in range(samples):
            starts = rng.integers(0, len(values), size=blocks_needed)
            draw = np.concatenate([values[(start + offsets) % len(values)] for start in starts])
            means[sample] = draw[: len(values)].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
) -> dict:
    """Paired daily differences; negative values favour the candidate."""

    daily_rows = []
    for origin, group in frame.groupby("origin"):
        actual = group["actual"].to_numpy(dtype=float)
        cand = group[candidate].to_numpy(dtype=float)
        ref = group[reference].to_numpy(dtype=float)
        valid = (
            np.isfinite(actual)
            & np.isfinite(cand)
            & np.isfinite(ref)
            & (actual > 0.0)
        )
        if not valid.any():
            continue
        actual = actual[valid]
        cand = cand[valid]
        ref = ref[valid]
        daily_rows.append(
            {
                "origin": origin,
                "mape_delta_points": float(
                    100.0
                    * np.mean(np.abs(cand - actual) / actual - np.abs(ref - actual) / actual)
                ),
                "mae_delta": float(np.mean(np.abs(cand - actual) - np.abs(ref - actual))),
                "regret_delta": float(
                    actual[int(np.argmin(cand))] - actual[int(np.argmin(ref))]
                ),
            }
        )
    daily = pd.DataFrame(daily_rows)
    report = {
        "candidate": candidate,
        "reference": reference,
        "n_days": int(len(daily)),
        "interpretation": "negative deltas favour candidate",
    }
    for column in ("mape_delta_points", "mae_delta", "regret_delta"):
        values = daily[column].to_numpy(dtype=float)
        report[column] = {
            "mean": float(np.mean(values)),
            "days_candidate_better_percent": float(100.0 * np.mean(values < 0.0)),
            "days_equal_percent": float(100.0 * np.mean(np.isclose(values, 0.0))),
            "day_bootstrap_95ci": _bootstrap_ci(values, block_days=None),
            "circular_7d_block_bootstrap_95ci": _bootstrap_ci(
                values, block_days=7
            ),
        }
    return report


def _readme(report: dict) -> str:
    feb = report["metrics"]["calibration_february"]
    hold = report["metrics"]["evaluation_mar_apr"]
    rows = []
    labels = (
        ("direct_reference", "Direct"),
        ("physical_lgbm", "Físico por componentes"),
        ("base_blend_jan_feb", "Blend 0,60 físico / 0,40 Direct"),
        ("calibrated_blend_seeded", "Blend + calibración causal"),
        ("base_level_gate_january", "Gate de nivel fijado en enero"),
        ("calibrated_level_gate_seeded", "Gate + calibración, seed ene–feb"),
        ("calibrated_level_gate_fresh", "Gate + calibración, fresh marzo"),
        ("share_lgbm", "Share LGBM"),
        ("base_share_convex_january", "Share + gate, convex enero"),
        ("calibrated_share_convex_seeded", "Convex share + calibración"),
        ("base_share_level_gate_january", "Gate share/Direct/físico, enero"),
        ("calibrated_share_level_gate_seeded", "Gate share + calibración"),
    )
    for column, label in labels:
        feb_metric = feb.get(column)
        hold_metric = hold.get(column)
        rows.append(
            f"| {label} | "
            f"{feb_metric['mape']:.3f}%" if feb_metric else f"| {label} | —"
        )
        rows[-1] += (
            f" | {hold_metric['mape']:.3f}% | {hold_metric['wape']:.3f}% | "
            f"{hold_metric['mae']:.3f} |"
            if hold_metric
            else " | — | — | — |"
        )
    gate_spec = report["selection"]["level_gate_calibration_february"]["selected"]
    blend_spec = report["selection"]["blend_calibration_february"]["selected"]
    share_spec = report["selection"]["share_level_gate_calibration_february"][
        "selected"
    ]
    incumbent_oracle = report["window_selection"]["evaluation_mar_apr"][
        "calibrated_level_gate_seeded"
    ]
    share_oracle = report["window_selection"]["evaluation_mar_apr"][
        "calibrated_share_convex_seeded"
    ]
    share_pair = report["paired_bootstrap_evaluation"][
        "share_convex_vs_incumbent"
    ]["mape_delta_points"]
    return f"""# Calibración residual causal del modelo físico consolidado

Este módulo no vuelve a entrenar LightGBM. Combina sus pronósticos congelados y
corrige únicamente el nivel reciente con errores que ya eran observables al
emitir cada pronóstico: `past_origin < origin` y `target_time < origin`.

## Resultado

| Señal | Febrero MAPE | Marzo–abril MAPE | Marzo–abril WAPE | Marzo–abril MAE |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

La ruta mejor separada escoge en **enero** un gate: usa el modelo físico cuando
su propia predicción es menor de
`{report['selection']['level_gate_january']['selected_threshold_gco2_kwh']:.1f}`
gCO2/kWh y usa Direct en el resto. Después, **sólo febrero** elige su calibrador:
memoria de {gate_spec['lookback_days']} días, modo `{gate_spec['block_mode']}` y
shrink `{gate_spec['shrink']}`. Esos hiperparámetros quedan congelados antes de
calcular marzo–abril.

Como referencia también se conserva el blend 0,60 físico / 0,40 Direct. Su
calibrador de febrero eligió {blend_spec['lookback_days']} días, modo
`{blend_spec['block_mode']}` y shrink `{blend_spec['shrink']}`. La variante
`fresh` empieza marzo sin memoria; la principal usa enero–febrero como seed,
que es la regla operativa escogida de antemano porque en producción no se
descartan errores recientes disponibles.

## Ablación `share_lgbm`: rechazada

Se probaron dos familias pequeñas definidas sólo con enero: un blend convexo
unidimensional entre `share_lgbm` y el gate anterior, y un gate de tres
regímenes cuyo orden estaba fijado (share bajo, Direct medio, físico alto).
Este último seleccionó thresholds
`{report['selection']['share_level_gate_january']['selected_low_threshold_gco2_kwh']:.1f}`
y
`{report['selection']['share_level_gate_january']['selected_high_threshold_gco2_kwh']:.1f}`;
febrero eligió para él {share_spec['lookback_days']} días,
`{share_spec['block_mode']}` y shrink `{share_spec['shrink']}`.

El mejor resultado share fue el convexo calibrado: su MAPE de marzo–abril llega
a {hold['calibrated_share_convex_seeded']['mape']:.3f}%, sólo
{hold['calibrated_level_gate_seeded']['mape'] - hold['calibrated_share_convex_seeded']['mape']:.3f}
puntos por debajo del incumbent. No se promueve: el IC 95% bootstrap por
bloques de siete días para la diferencia MAPE es
[{share_pair['circular_7d_block_bootstrap_95ci'][0]:+.3f},
{share_pair['circular_7d_block_bootstrap_95ci'][1]:+.3f}] puntos, cruza cero,
y además empeoran WAPE
({hold['calibrated_share_convex_seeded']['wape']:.3f}% frente a
{hold['calibrated_level_gate_seeded']['wape']:.3f}%) y MAE
({hold['calibrated_share_convex_seeded']['mae']:.3f} frente a
{hold['calibrated_level_gate_seeded']['mae']:.3f}).

Sí existe un trade-off útil para ventanas: el incumbent captura
{incumbent_oracle['pct_oracle_potential']:.1f}% del potencial oracle, mientras
el convexo con share captura {share_oracle['pct_oracle_potential']:.1f}% y baja
el regret medio. Se conserva como candidato de ranking, pero se rechaza como
reemplazo del modelo cuyo objetivo principal es MAPE.

## Lectura metodológica

La cadena enero → febrero → marzo/abril evita que los targets de marzo–abril
seleccionen el threshold o el calibrador. Sin embargo, toda la arquitectura se
marca **exploratoria**, porque la idea de explotar componentes físicos nació al
ver el oracle físico de marzo–abril. Por eso 11,8% no debe venderse todavía como
un holdout prospectivo prístino; hay que congelar esta regla y comprobarla en
un periodo futuro intacto.

`oracle_learned_factors` usa los componentes reales futuros y sólo cuantifica
el techo físico: no es desplegable. Las métricas `pct_oracle_potential`, regret,
Spearman y top-1 miden por separado la selección de la hora más verde.

## Archivos y reproducción

- `predictions.parquet`: bases, calibraciones, escalas y tamaño del historial
  visible en cada fila.
- `summary.json`: MAPE/WAPE/MAE, meses, horizontes, métricas de ventanas y
  bootstrap pareado diario y por bloques de siete días.

```bash
/Users/saur/miniconda3/envs/green-observatory/bin/python \\
  -m green_observatory.carbon.consolidated_physical_calibration
```
"""


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare(pd.read_parquet(args.predictions))
    frame = attach_share_predictions(frame, args.share_predictions)

    jan_start, feb_start = _utc(args.january_start), _utc(args.february_start)
    mar_start, evaluation_end = _utc(args.march_start), _utc(args.evaluation_end)
    january = (frame["origin"] >= jan_start) & (frame["origin"] < feb_start)
    jan_feb = (frame["origin"] >= jan_start) & (frame["origin"] < mar_start)
    february = (frame["origin"] >= feb_start) & (frame["origin"] < mar_start)
    evaluation = (frame["origin"] >= mar_start) & (frame["origin"] < evaluation_end)
    if not january.any() or not february.any() or not evaluation.any():
        raise ValueError("January, February and March-April must all contain rows")

    blend_weight, blend_candidates = select_blend_weight(frame, jan_feb)
    january_weight, january_blend_candidates = select_blend_weight(frame, january)
    threshold, threshold_candidates = select_level_threshold(frame, january)
    frame["base_blend_jan_feb"] = (
        blend_weight * frame["physical_lgbm"]
        + (1.0 - blend_weight) * frame["direct_reference"]
    )
    frame["base_blend_january_only"] = (
        january_weight * frame["physical_lgbm"]
        + (1.0 - january_weight) * frame["direct_reference"]
    )
    physical_selected = frame["physical_lgbm"] < threshold
    frame["base_level_gate_january"] = np.where(
        physical_selected, frame["physical_lgbm"], frame["direct_reference"]
    )
    frame["level_gate_selected_expert"] = np.where(
        physical_selected, "physical_lgbm", "direct_reference"
    )
    share_weight, share_convex_candidates = select_share_convex_weight(
        frame, january
    )
    frame["base_share_convex_january"] = (
        share_weight * frame["share_lgbm"]
        + (1.0 - share_weight) * frame["base_level_gate_january"]
    )
    (
        share_low_threshold,
        share_high_threshold,
    ), share_level_candidates = select_share_level_thresholds(frame, january)
    frame["base_share_level_gate_january"] = np.where(
        frame["physical_lgbm"] < share_low_threshold,
        frame["share_lgbm"],
        np.where(
            frame["physical_lgbm"] < share_high_threshold,
            frame["direct_reference"],
            frame["physical_lgbm"],
        ),
    )
    frame["share_level_gate_selected_expert"] = np.select(
        [
            frame["physical_lgbm"] < share_low_threshold,
            frame["physical_lgbm"] < share_high_threshold,
        ],
        ["share_lgbm", "direct_reference"],
        default="physical_lgbm",
    )

    february_origins = frame.loc[february, "origin"].unique()
    holdout_origins = frame.loc[evaluation, "origin"].unique()
    paths = {
        "blend": "base_blend_jan_feb",
        "jan_weight": "base_blend_january_only",
        "level_gate": "base_level_gate_january",
        "share_convex": "base_share_convex_january",
        "share_level_gate": "base_share_level_gate_january",
    }
    selected_specs = {}
    calibration_candidates = {}
    for label, base_column in paths.items():
        spec, candidates = tune_calibration(
            frame, base_column, february_origins
        )
        selected_specs[label] = spec
        calibration_candidates[label] = candidates
        validation_result = causal_scale(
            frame, base_column, february_origins, spec
        )
        seeded_holdout = causal_scale(
            frame, base_column, holdout_origins, spec
        )
        fresh_holdout = causal_scale(
            frame,
            base_column,
            holdout_origins,
            spec,
            history_origin_floor=mar_start,
        )
        seeded = pd.concat([validation_result, seeded_holdout], ignore_index=True)
        _assign_calibration(frame, f"calibrated_{label}_seeded", seeded)
        _assign_calibration(frame, f"calibrated_{label}_fresh", fresh_holdout)

    frame["period"] = np.select(
        [january, february, evaluation],
        ["selection_january", "calibration_february", "evaluation_mar_apr"],
        default="outside_protocol",
    )
    model_columns = (
        "direct_reference",
        "physical_lgbm",
        "share_lgbm",
        "base_blend_jan_feb",
        "base_blend_january_only",
        "base_level_gate_january",
        "base_share_convex_january",
        "base_share_level_gate_january",
        "calibrated_blend_seeded",
        "calibrated_blend_fresh",
        "calibrated_jan_weight_seeded",
        "calibrated_jan_weight_fresh",
        "calibrated_level_gate_seeded",
        "calibrated_level_gate_fresh",
        "calibrated_share_convex_seeded",
        "calibrated_share_convex_fresh",
        "calibrated_share_level_gate_seeded",
        "calibrated_share_level_gate_fresh",
        "oracle_learned_factors",
    )
    jan_frame = frame.loc[january]
    feb_frame = frame.loc[february]
    holdout = frame.loc[evaluation]
    actual_by_time = (
        frame.dropna(subset=["actual"])
        .drop_duplicates("target_time", keep="last")
        .set_index("target_time")["actual"]
    )
    report = {
        "status": (
            "exploratory architecture; the physical-component oracle on Mar-Apr "
            "motivated the experiment, so this is not a pristine prospective holdout"
        ),
        "protocol": {
            "target": "RTE consolidated production carbon intensity",
            "base_inputs": (
                "causal-clean physical_lgbm, share_lgbm and causal-clean Direct only"
            ),
            "forbidden_inputs": "no full_models legacy signals",
            "visibility_rule": "past_origin < origin and target_time < origin",
            "architecture_selection": "January only",
            "calibration_selection": (
                "causal day-by-day February simulation only; MAPE objective"
            ),
            "evaluation": "origins March-April; no evaluation target selects a rule",
            "primary_initialization": (
                "seeded January-February, chosen before evaluation because an "
                "operational model retains its observable recent errors"
            ),
            "fresh_initialization": (
                "sensitivity only; forecasts issued before March are excluded"
            ),
            "scale_grid": [0.70, 1.30, 0.001],
            "lookbacks_days": list(DEFAULT_LOOKBACKS),
            "block_modes": {
                "global": [[1, 24]],
                "four_blocks": [list(block) for block in HORIZON_BLOCKS],
            },
            "shrinks": list(DEFAULT_SHRINKS),
        },
        "selection": {
            "blend_jan_feb": {
                "selected_physical_weight": blend_weight,
                "candidates": blend_candidates,
                "caveat": (
                    "uses all Jan-Feb and is therefore a less strictly nested "
                    "development path when its calibrator is also selected on February"
                ),
            },
            "blend_january_only": {
                "selected_physical_weight": january_weight,
                "candidates": january_blend_candidates,
            },
            "level_gate_january": {
                "rule": "physical_lgbm if physical_lgbm < threshold else Direct",
                "selected_threshold_gco2_kwh": threshold,
                "candidates": threshold_candidates,
            },
            "blend_calibration_february": {
                "selected": asdict(selected_specs["blend"]),
                "candidates": calibration_candidates["blend"],
            },
            "january_weight_calibration_february": {
                "selected": asdict(selected_specs["jan_weight"]),
                "candidates": calibration_candidates["jan_weight"],
            },
            "level_gate_calibration_february": {
                "selected": asdict(selected_specs["level_gate"]),
                "candidates": calibration_candidates["level_gate"],
            },
            "share_convex_january": {
                "rule": (
                    "share_weight * share_lgbm + (1-share_weight) * "
                    "base_level_gate_january"
                ),
                "selected_share_weight": share_weight,
                "candidates": share_convex_candidates,
            },
            "share_convex_calibration_february": {
                "selected": asdict(selected_specs["share_convex"]),
                "candidates": calibration_candidates["share_convex"],
            },
            "share_level_gate_january": {
                "rule": (
                    "share below low threshold; Direct in the middle; "
                    "physical above high threshold; driver=physical_lgbm"
                ),
                "selected_low_threshold_gco2_kwh": share_low_threshold,
                "selected_high_threshold_gco2_kwh": share_high_threshold,
                "candidates": share_level_candidates,
            },
            "share_level_gate_calibration_february": {
                "selected": asdict(selected_specs["share_level_gate"]),
                "candidates": calibration_candidates["share_level_gate"],
            },
        },
        "metrics": {
            "selection_january": _model_metrics(jan_frame, model_columns),
            "calibration_february": _model_metrics(feb_frame, model_columns),
            "evaluation_mar_apr": _model_metrics(holdout, model_columns),
        },
        "evaluation_by_month": {
            str(month): _model_metrics(group, model_columns)
            for month, group in holdout.groupby(holdout["origin"].dt.strftime("%Y-%m"))
        },
        "evaluation_by_horizon": {
            str(int(horizon)): _model_metrics(group, model_columns)
            for horizon, group in holdout.groupby("horizon")
        },
        "evaluation_by_block": {
            f"h{start}-{end}": _model_metrics(
                holdout.loc[holdout["horizon"].between(start, end)], model_columns
            )
            for start, end in HORIZON_BLOCKS
        },
        "window_selection": {
            "calibration_february": _window_report(
                feb_frame, model_columns, actual_by_time=actual_by_time
            ),
            "evaluation_mar_apr": _window_report(
                holdout, model_columns, actual_by_time=actual_by_time
            ),
            "evaluation_by_month": {
                str(month): _window_report(
                    group, model_columns, actual_by_time=actual_by_time
                )
                for month, group in holdout.groupby(
                    holdout["origin"].dt.strftime("%Y-%m")
                )
            },
        },
        "paired_bootstrap_evaluation": {
            "calibrated_gate_vs_raw_gate": paired_bootstrap(
                holdout,
                "calibrated_level_gate_seeded",
                "base_level_gate_january",
            ),
            "calibrated_gate_vs_direct": paired_bootstrap(
                holdout, "calibrated_level_gate_seeded", "direct_reference"
            ),
            "calibrated_gate_vs_blend": paired_bootstrap(
                holdout, "calibrated_level_gate_seeded", "base_blend_jan_feb"
            ),
            "calibrated_blend_vs_raw_blend": paired_bootstrap(
                holdout, "calibrated_blend_seeded", "base_blend_jan_feb"
            ),
            "share_convex_vs_incumbent": paired_bootstrap(
                holdout,
                "calibrated_share_convex_seeded",
                "calibrated_level_gate_seeded",
            ),
            "share_level_gate_vs_incumbent": paired_bootstrap(
                holdout,
                "calibrated_share_level_gate_seeded",
                "calibrated_level_gate_seeded",
            ),
        },
    }
    incumbent = report["metrics"]["evaluation_mar_apr"][
        "calibrated_level_gate_seeded"
    ]
    share_candidate = report["metrics"]["evaluation_mar_apr"][
        "calibrated_share_convex_seeded"
    ]
    share_pair = report["paired_bootstrap_evaluation"][
        "share_convex_vs_incumbent"
    ]["mape_delta_points"]
    report["share_integration_decision"] = {
        "promoted": False,
        "kept_model": "calibrated_level_gate_seeded",
        "best_share_candidate": "calibrated_share_convex_seeded",
        "robustness_rule": (
            "promote only if MAPE improves, the circular 7d bootstrap upper bound "
            "is below zero, and neither WAPE nor MAE worsens"
        ),
        "checks": {
            "mape_improves": bool(share_candidate["mape"] < incumbent["mape"]),
            "circular_7d_ci_entirely_below_zero": bool(
                share_pair["circular_7d_block_bootstrap_95ci"][1] < 0.0
            ),
            "wape_not_worse": bool(share_candidate["wape"] <= incumbent["wape"]),
            "mae_not_worse": bool(share_candidate["mae"] <= incumbent["mae"]),
        },
        "reason": (
            "rejected: the small MAPE gain is not bootstrap-robust and WAPE/MAE "
            "worsen; it also reverses sign between March and April"
        ),
    }
    frame.drop(columns=["_row_id"]).to_parquet(
        output_dir / "predictions.parquet", index=False
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_readme(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="runs/daily_refit_2026/consolidated_physical/predictions.parquet",
    )
    parser.add_argument(
        "--share-predictions",
        default="runs/daily_refit_2026/consolidated_share/predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/consolidated_physical_calibration",
    )
    parser.add_argument("--january-start", default="2026-01-01")
    parser.add_argument("--february-start", default="2026-02-01")
    parser.add_argument("--march-start", default="2026-03-01")
    parser.add_argument("--evaluation-end", default="2026-05-01")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
