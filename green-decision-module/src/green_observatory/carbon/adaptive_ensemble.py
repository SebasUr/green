"""Causal, lightweight ensembles over daily carbon-model predictions.

These helpers consume already generated tidy predictions.  They never refit or
modify a base model and only use outcomes strictly earlier than each origin.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_POINT_EXPERTS = (
    "fossil_regime_recent_mapper",
    "fossil_regime_decision",
    "hybrid_h2_mapper_delta",
    "hybrid_h2",
    "fossil_regime_point",
)

DEFAULT_RANK_EXPERTS = (
    "fossil_regime_recent_mapper",
    "fossil_regime_decision",
    "hybrid_h2_mapper_delta",
)


def mape_optimal_scale(
    prediction: np.ndarray,
    actual: np.ndarray,
    *,
    lower: float = 0.60,
    upper: float = 1.60,
) -> float:
    """Exact positive scalar minimizing MAPE for a fixed prediction curve.

    The objective is a weighted absolute deviation in ``actual/prediction``;
    therefore its minimizer is a weighted median, not the ordinary mean ratio.
    """
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    valid = (
        np.isfinite(prediction)
        & np.isfinite(actual)
        & (prediction > 1e-9)
        & (np.abs(actual) > 1e-9)
    )
    if not valid.any():
        return 1.0
    ratios = actual[valid] / prediction[valid]
    weights = prediction[valid] / np.abs(actual[valid])
    order = np.argsort(ratios)
    ratios = ratios[order]
    weights = weights[order]
    index = int(np.searchsorted(np.cumsum(weights), 0.5 * weights.sum()))
    return float(np.clip(ratios[min(index, len(ratios) - 1)], lower, upper))


def _history_before(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    start: pd.Timestamp,
) -> pd.DataFrame:
    history = frame.loc[(frame["origin"] < origin) & (frame["origin"] >= start)]
    if "target_time" in history:
        # With origins closer than 24 hours, the tail of the previous forecast
        # is not known yet.  The target cutoff is the actual causal invariant.
        history = history.loc[history["target_time"] < origin]
    return history


def causal_scaled_expert(
    predictions: pd.DataFrame,
    *,
    lookback_days: int = 7,
    candidates: Sequence[str] = DEFAULT_POINT_EXPERTS,
    default_expert: str = "fossil_regime_recent_mapper",
    scale_grid: Sequence[float] | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Select and scale one expert using only the preceding daily losses."""
    required = {"model", "origin", "prediction", "actual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    frame = predictions.loc[predictions["model"].isin(candidates)].copy()
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    grid = np.asarray(
        np.arange(0.60, 1.601, 0.01) if scale_grid is None else scale_grid,
        dtype=float,
    )
    origins = sorted(frame["origin"].unique())
    parts: list[pd.DataFrame] = []

    for origin in origins:
        start = origin - pd.Timedelta(days=lookback_days)
        past = _history_before(frame, origin, start=start)
        best: tuple[float, int, str, float] | None = None
        for order, expert in enumerate(candidates):
            history = past.loc[past["model"] == expert]
            if history.empty:
                continue
            prediction = history["prediction"].to_numpy(dtype=float)
            actual = history["actual"].to_numpy(dtype=float)
            losses = np.mean(
                np.abs(grid[:, None] * prediction[None, :] - actual[None, :])
                / np.clip(np.abs(actual)[None, :], 1e-9, None),
                axis=1,
            )
            index = int(np.argmin(losses))
            candidate = (float(losses[index]), order, expert, float(grid[index]))
            if best is None or candidate < best:
                best = candidate

        expert, scale = (
            (default_expert, 1.0) if best is None else (best[2], best[3])
        )
        current = frame.loc[
            (frame["origin"] == origin) & (frame["model"] == expert)
        ].copy()
        if current.empty:
            raise ValueError(f"expert {expert!r} is unavailable at {origin}")
        current["prediction"] = current["prediction"] * scale
        current["selected_expert"] = expert
        current["calibration_scale"] = scale
        current["model"] = name or f"causal_scaled_expert_{lookback_days}d"
        parts.append(current)

    return pd.concat(parts, ignore_index=True)


def causal_scaled_blend(
    predictions: pd.DataFrame,
    *,
    lookback_days: int = 7,
    candidates: Sequence[str] = DEFAULT_POINT_EXPERTS,
    temperature: float = 0.005,
    top_k: int | None = None,
    scale_grid: Sequence[float] | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Softly blend recent experts after causal MAPE scale calibration.

    Hard expert switching is unstable with only seven daily observations.  A
    softmax over recent losses retains diversity while assigning almost all
    weight to genuinely better experts when the evidence is strong.
    """
    required = {
        "model", "origin", "horizon", "target_time", "prediction", "actual"
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    frame = predictions.loc[predictions["model"].isin(candidates)].copy()
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    grid = np.asarray(
        np.arange(0.60, 1.601, 0.01) if scale_grid is None else scale_grid,
        dtype=float,
    )
    parts: list[pd.DataFrame] = []
    for origin in sorted(frame["origin"].unique()):
        start = origin - pd.Timedelta(days=lookback_days)
        history = _history_before(frame, origin, start=start)
        losses: dict[str, float] = {}
        scales: dict[str, float] = {}
        for expert in candidates:
            expert_history = history.loc[history["model"] == expert]
            if expert_history.empty:
                continue
            prediction = expert_history["prediction"].to_numpy(dtype=float)
            actual = expert_history["actual"].to_numpy(dtype=float)
            candidate_losses = np.mean(
                np.abs(grid[:, None] * prediction[None, :] - actual[None, :])
                / np.clip(np.abs(actual)[None, :], 1e-9, None),
                axis=1,
            )
            index = int(np.argmin(candidate_losses))
            losses[expert] = float(candidate_losses[index])
            scales[expert] = float(grid[index])

        available = [expert for expert in candidates if expert in losses]
        if not available:
            available = [
                expert
                for expert in candidates
                if not frame.loc[
                    (frame["origin"] == origin) & (frame["model"] == expert)
                ].empty
            ]
            if not available:
                raise ValueError(f"no blend expert is available at {origin}")
            weights = {available[0]: 1.0}
            scales = {available[0]: 1.0}
        else:
            if top_k is not None:
                available = sorted(available, key=lambda expert: losses[expert])[
                    :top_k
                ]
            centered = np.asarray(
                [losses[expert] for expert in available], dtype=float
            )
            centered -= centered.min()
            raw_weights = np.exp(-centered / temperature)
            raw_weights /= raw_weights.sum()
            weights = {
                expert: float(weight)
                for expert, weight in zip(available, raw_weights)
            }

        current_parts: list[pd.DataFrame] = []
        for expert, weight in weights.items():
            current = frame.loc[
                (frame["origin"] == origin) & (frame["model"] == expert),
                ["origin", "horizon", "target_time", "prediction", "actual"],
            ].copy()
            if current.empty:
                continue
            current["weighted_prediction"] = (
                current["prediction"] * scales.get(expert, 1.0) * weight
            )
            current_parts.append(current)
        if len(current_parts) != len(weights):
            raise ValueError(f"incomplete blend experts at {origin}")
        stacked = pd.concat(current_parts, ignore_index=True)
        out = (
            stacked.groupby(
                ["origin", "horizon", "target_time"], as_index=False
            )
            .agg(prediction=("weighted_prediction", "sum"), actual=("actual", "first"))
            .sort_values("horizon")
        )
        out["model"] = name or f"causal_scaled_blend_{lookback_days}d"
        out["ensemble_weights"] = json.dumps(weights, sort_keys=True)
        out["calibration_scales"] = json.dumps(
            {expert: scales.get(expert, 1.0) for expert in weights},
            sort_keys=True,
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(values[order[min(index, len(order) - 1)]])


def _decayed_mape_fit(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    half_life_days: float | None,
) -> tuple[float, float]:
    prediction = history["prediction"].to_numpy(dtype=float)
    actual = history["actual"].to_numpy(dtype=float)
    valid = (
        np.isfinite(prediction)
        & np.isfinite(actual)
        & (prediction > 1e-9)
        & (np.abs(actual) > 1e-9)
    )
    if not valid.any():
        return 1.0, np.inf
    prediction = prediction[valid]
    actual = actual[valid]
    target_time = pd.DatetimeIndex(history.loc[valid, "target_time"])
    age_days = (origin - target_time).total_seconds().to_numpy() / 86400.0
    time_weight = (
        np.ones(len(actual), dtype=float)
        if half_life_days is None
        else np.exp2(-age_days / half_life_days)
    )
    ratio = actual / prediction
    objective_weight = time_weight * prediction / np.abs(actual)
    scale = _weighted_median(ratio, objective_weight)
    scale = float(np.clip(np.round(scale, 2), 0.60, 1.60))
    loss = float(
        np.sum(time_weight * np.abs(scale * prediction - actual) / np.abs(actual))
        / time_weight.sum()
    )
    return scale, loss


def causal_block_scaled_expert(
    predictions: pd.DataFrame,
    *,
    lookback_days: int = 21,
    half_life_days: float | None = 3.0,
    candidates: Sequence[str] = DEFAULT_POINT_EXPERTS,
    blocks: Sequence[tuple[int, int]] = ((1, 6), (7, 16), (17, 21), (22, 24)),
    block_weight: float = 0.25,
    shape_expert: str | None = None,
    shape_weight: float = 0.0,
    name: str = "causal_block_scaled_expert",
) -> pd.DataFrame:
    """Select one daily expert and shrink horizon-block scales to its level.

    The default blocks isolate the evening solar-to-thermal ramp that dominates
    the current French error.  Exponential decay lets the level react to regime
    change without fitting a high-dimensional meta-regressor on only 119 days.
    """
    required = {
        "model", "origin", "horizon", "target_time", "prediction", "actual"
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if not 0.0 <= block_weight <= 1.0:
        raise ValueError("block_weight must be in [0, 1]")
    if not 0.0 <= shape_weight <= 1.0:
        raise ValueError("shape_weight must be in [0, 1]")
    frame = predictions.loc[
        predictions["model"].isin(
            set(candidates) | ({shape_expert} if shape_expert else set())
        )
    ].copy()
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    parts: list[pd.DataFrame] = []
    for origin in sorted(frame["origin"].unique()):
        start = origin - pd.Timedelta(days=lookback_days)
        history = frame.loc[
            (frame["target_time"] < origin)
            & (frame["target_time"] >= start)
            & frame["model"].isin(candidates)
        ]
        fitted: dict[str, tuple[float, float]] = {}
        for expert in candidates:
            expert_history = history.loc[history["model"] == expert]
            if not expert_history.empty:
                fitted[expert] = _decayed_mape_fit(
                    expert_history, origin, half_life_days=half_life_days
                )
        if fitted:
            selected = min(
                fitted,
                key=lambda expert: (fitted[expert][1], candidates.index(expert)),
            )
            global_scale = fitted[selected][0]
        else:
            selected = candidates[0]
            global_scale = 1.0
        current = frame.loc[
            (frame["origin"] == origin) & (frame["model"] == selected),
            ["origin", "horizon", "target_time", "prediction", "actual"],
        ].copy()
        if current.empty:
            raise ValueError(f"expert {selected!r} is unavailable at {origin}")
        current["block_scale"] = global_scale
        for low, high in blocks:
            block_history = history.loc[
                (history["model"] == selected)
                & history["horizon"].between(low, high)
            ]
            block_scale = (
                global_scale
                if block_history.empty
                else _decayed_mape_fit(
                    block_history, origin, half_life_days=half_life_days
                )[0]
            )
            scale = (1.0 - block_weight) * global_scale + block_weight * block_scale
            current.loc[current["horizon"].between(low, high), "block_scale"] = scale
        current["prediction"] *= current["block_scale"]

        if shape_expert is not None and shape_weight > 0.0:
            shape = frame.loc[
                (frame["origin"] == origin) & (frame["model"] == shape_expert),
                ["horizon", "prediction"],
            ].sort_values("horizon")
            ordered = current.sort_values("horizon")
            if len(shape) == len(ordered):
                base_values = ordered["prediction"].to_numpy(dtype=float)
                shape_values = shape["prediction"].to_numpy(dtype=float)
                base_level = float(np.median(base_values))
                base_shape = base_values / np.clip(base_level, 1e-9, None)
                other_shape = shape_values / np.clip(
                    np.median(shape_values), 1e-9, None
                )
                mixed = (1.0 - shape_weight) * base_shape + shape_weight * other_shape
                mixed /= np.clip(np.median(mixed), 1e-9, None)
                current.loc[ordered.index, "prediction"] = base_level * mixed
        current["model"] = name
        current["selected_expert"] = selected
        current["calibration_scale"] = global_scale
        parts.append(current)
    return pd.concat(parts, ignore_index=True)


def rank_consensus(
    predictions: pd.DataFrame,
    *,
    candidates: Sequence[str] = DEFAULT_RANK_EXPERTS,
    name: str = "rank_consensus",
) -> pd.DataFrame:
    """Average within-horizon ranks from fixed experts for each daily origin."""
    keys = ["origin", "horizon", "target_time"]
    required = {*keys, "model", "prediction", "actual"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")

    merged: pd.DataFrame | None = None
    for index, expert in enumerate(candidates):
        part = predictions.loc[
            predictions["model"] == expert, keys + ["prediction", "actual"]
        ].copy()
        if part.duplicated(keys).any():
            raise ValueError(f"duplicate predictions for expert {expert!r}")
        part[f"rank_{index}"] = part.groupby("origin")["prediction"].rank(
            method="average", pct=True
        )
        keep = keys + [f"rank_{index}"]
        if merged is None:
            merged = part[keys + ["actual", f"rank_{index}"]]
        else:
            merged = merged.merge(part[keep], on=keys, how="inner", validate="one_to_one")

    if merged is None:
        raise ValueError("at least one rank expert is required")
    rank_columns = [column for column in merged if column.startswith("rank_")]
    merged["prediction"] = merged[rank_columns].mean(axis=1)
    merged["model"] = name
    merged["selected_expert"] = "+".join(candidates)
    return merged.drop(columns=rank_columns)


__all__ = [
    "DEFAULT_POINT_EXPERTS",
    "DEFAULT_RANK_EXPERTS",
    "mape_optimal_scale",
    "causal_scaled_expert",
    "causal_scaled_blend",
    "causal_block_scaled_expert",
    "rank_consensus",
]
