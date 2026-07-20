"""Reusable evaluation protocols for dense French day-ahead models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon import evaluation as ev
from green_observatory.providers.carbon_base import CARBON
from green_observatory.windows.oracle import window_selection_metrics


@dataclass(frozen=True)
class ProtocolSpec:
    """Operational definition of a set of forecast origins."""

    name: str
    stride_hours: int
    origin_hour_utc: int | None = None


ROLLING_6H = ProtocolSpec("rolling_6h", stride_hours=6)
DAILY_UTC = ProtocolSpec("daily_utc", stride_hours=24, origin_hour_utc=0)


def regularize_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert missing UTC hours so positional lags remain genuine hour lags."""
    frame = frame.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    complete = pd.date_range(frame.index.min(), frame.index.max(), freq="1h")
    return frame.reindex(complete)


def protocol_origins(
    frame: pd.DataFrame,
    start,
    end,
    spec: ProtocolSpec,
    *,
    horizons: Sequence[int] = tuple(range(1, 25)),
) -> pd.DatetimeIndex:
    """Return origins whose current state and every requested target exist."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    grid = pd.date_range(start, end, freq=f"{spec.stride_hours}h")
    if spec.origin_hour_utc is not None:
        grid = grid[grid.hour == spec.origin_hour_utc]
    carbon = pd.to_numeric(frame[CARBON], errors="coerce")
    valid: list[pd.Timestamp] = []
    for origin in grid:
        times = pd.DatetimeIndex(
            [origin, *(origin + pd.Timedelta(hours=int(h)) for h in horizons)]
        )
        if carbon.reindex(times).notna().all():
            valid.append(origin)
    return pd.DatetimeIndex(valid)


def model_predictions(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    baseline,
    fossil_regime,
    ensemble_ci=None,
    horizons: Sequence[int] = tuple(range(1, 25)),
    short_horizon_cutoff: int = 2,
    calibration_scales: dict[str, float] | None = None,
    recent_mapper=None,
) -> pd.DataFrame:
    """Predict all modular candidates plus a fixed short/long horizon gate."""
    horizons = tuple(int(h) for h in horizons)
    baseline_pred = ev._project_batch(
        baseline, frame, origins, horizons
    ).assign(model="project_dense24")
    regime = fossil_regime.predict_batch(frame, origins)
    regime_point = regime[
        ["origin", "horizon", "target_time", "point_prediction"]
    ].rename(columns={"point_prediction": "prediction"})
    regime_point["model"] = "fossil_regime_rte"
    recent_physical = None
    mapper_delta = None
    if recent_mapper is not None:
        predicted_shares = pd.DataFrame(
            {
                name: regime[f"predicted_{name}"].to_numpy()
                for name in recent_mapper.share_names
            },
            index=regime.index,
        )
        recent_physical = regime[
            ["origin", "horizon", "target_time"]
        ].copy()
        recent_physical["prediction"] = recent_mapper.predict(
            predicted_shares
        )
        recent_physical = recent_physical.loc[
            :, ["origin", "horizon", "target_time", "prediction"]
        ]
        recent_physical["model"] = "fossil_regime_recent_mapper"
        mapper_delta = regime_point.copy()
        mapper_delta["prediction"] = np.clip(
            regime_point["prediction"].to_numpy(dtype=float)
            + float(fossil_regime.point_scale_)
            * (
                recent_physical["prediction"].to_numpy(dtype=float)
                - regime["physical_prediction"].to_numpy(dtype=float)
            ),
            0.0,
            None,
        )
        mapper_delta["model"] = "fossil_regime_mapper_delta"

    hybrid = pd.concat(
        [
            baseline_pred[baseline_pred["horizon"] <= short_horizon_cutoff],
            regime_point[regime_point["horizon"] > short_horizon_cutoff],
        ],
        ignore_index=True,
    )
    hybrid["model"] = f"hybrid_h{short_horizon_cutoff}"

    parts = [baseline_pred, regime_point, hybrid]
    if recent_physical is not None:
        parts.extend([recent_physical, mapper_delta])
        recent_mapper_hybrid = pd.concat(
            [
                baseline_pred[
                    baseline_pred["horizon"] <= short_horizon_cutoff
                ],
                recent_physical[
                    recent_physical["horizon"] > short_horizon_cutoff
                ],
            ],
            ignore_index=True,
        )
        recent_mapper_hybrid["model"] = (
            f"hybrid_h{short_horizon_cutoff}_recent_mapper"
        )
        parts.append(recent_mapper_hybrid)
        mapper_delta_hybrid = pd.concat(
            [
                baseline_pred[
                    baseline_pred["horizon"] <= short_horizon_cutoff
                ],
                mapper_delta[
                    mapper_delta["horizon"] > short_horizon_cutoff
                ],
            ],
            ignore_index=True,
        )
        mapper_delta_hybrid["model"] = (
            f"hybrid_h{short_horizon_cutoff}_mapper_delta"
        )
        parts.append(mapper_delta_hybrid)
    if ensemble_ci is not None:
        ensemble = ensemble_ci.predict_batch(
            frame, origins, horizons=horizons
        ).loc[:, ["origin", "horizon", "target_time", "prediction"]]
        ensemble["model"] = "ensemble_ci_dense"
        parts.append(ensemble)

    pred = pd.concat(parts, ignore_index=True)
    if calibration_scales:
        calibrated: list[pd.DataFrame] = []
        for model, scale in calibration_scales.items():
            part = pred[pred["model"].eq(model)].copy()
            if part.empty:
                continue
            part["prediction"] = np.clip(
                part["prediction"].to_numpy(dtype=float) * float(scale),
                0.0,
                None,
            )
            part["model"] = f"{model}_recent_scale"
            calibrated.append(part)
        if calibrated:
            pred = pd.concat([pred, *calibrated], ignore_index=True)
        baseline_scale = calibration_scales.get("project_dense24")
        fossil_scale = calibration_scales.get("fossil_regime_rte")
        if baseline_scale is not None and fossil_scale is not None:
            # Preserve the strong raw short-horizon nowcast and correct only
            # the long-horizon fossil model's recent level shift.
            component_hybrid = pd.concat(
                [
                    baseline_pred[
                        baseline_pred["horizon"] <= short_horizon_cutoff
                    ],
                    regime_point[
                        regime_point["horizon"] > short_horizon_cutoff
                    ].assign(
                        prediction=lambda value: np.clip(
                            value["prediction"].to_numpy(dtype=float)
                            * float(fossil_scale),
                            0.0,
                            None,
                        )
                    ),
                ],
                ignore_index=True,
            )
            component_hybrid["model"] = (
                f"hybrid_h{short_horizon_cutoff}_component_scale"
            )
            pred = pd.concat([pred, component_hybrid], ignore_index=True)
    pred["actual"] = pd.to_numeric(frame[CARBON], errors="coerce").reindex(
        pd.DatetimeIndex(pred["target_time"])
    ).to_numpy()
    return pred.dropna(subset=["prediction", "actual"]).reset_index(drop=True)


def fit_mape_scales(
    predictions: pd.DataFrame,
    *,
    scale_grid: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fit one recent multiplicative scale per model using only prior outcomes."""
    grid = np.asarray(
        scale_grid
        if scale_grid is not None
        else np.linspace(0.60, 1.80, 121),
        dtype=float,
    )
    scales: dict[str, float] = {}
    for model, group in predictions.groupby("model", sort=False):
        actual = group["actual"].to_numpy(dtype=float)
        raw = group["prediction"].to_numpy(dtype=float)
        denominator = np.clip(np.abs(actual), 1e-9, None)
        losses = np.mean(
            np.abs(grid[:, None] * raw[None, :] - actual[None, :])
            / denominator[None, :],
            axis=1,
        )
        scales[str(model)] = float(grid[int(np.argmin(losses))])
    return scales


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Signal metrics over all equally represented horizons."""
    rows: list[dict] = []
    for model, group in predictions.groupby("model", sort=False):
        error = group["prediction"] - group["actual"]
        actual_abs = group["actual"].abs()
        rows.append(
            {
                "model": model,
                "mape": 100.0
                * float((error.abs() / actual_abs.clip(lower=1e-9)).mean()),
                "wape": 100.0 * float(error.abs().sum() / actual_abs.sum()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(error.mean()),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def evaluate_protocol(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    baseline,
    fossil_regime,
    ensemble_ci=None,
    short_horizon_cutoff: int = 2,
    calibration_scales: dict[str, float] | None = None,
    recent_mapper=None,
) -> dict:
    """Evaluate signal accuracy and green-hour selection on one origin set."""
    pred = model_predictions(
        frame,
        origins,
        baseline=baseline,
        fossil_regime=fossil_regime,
        ensemble_ci=ensemble_ci,
        short_horizon_cutoff=short_horizon_cutoff,
        calibration_scales=calibration_scales,
        recent_mapper=recent_mapper,
    )
    aggregate = aggregate_metrics(pred)
    point = ev.point_metrics(pred)
    selection = window_selection_metrics(pred, frame)
    return {
        "predictions": pred,
        "aggregate": aggregate,
        "point": point,
        "selection": selection,
    }


__all__ = [
    "DAILY_UTC",
    "ROLLING_6H",
    "ProtocolSpec",
    "aggregate_metrics",
    "evaluate_protocol",
    "fit_mape_scales",
    "model_predictions",
    "protocol_origins",
    "regularize_hourly",
]
