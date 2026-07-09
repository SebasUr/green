"""Green-score normalization and low-carbon window detection.

Score convention (LOCKED): ``green_score`` in ``[0, 1]`` where **higher is
greener** (lower carbon). A low-carbon window is a contiguous block whose
intensity sits below the ``percentile``-th quantile of the reference horizon
(default p25), subject to duration and gap-merging rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from green_observatory.models import GreenWindow, ModelName, WindowType

_EPS = 1e-9


def green_score(
    values,
    reference: Sequence[float] | None = None,
    *,
    method: str = "quantile_rank",
    invert: bool = True,
    threshold: float | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Map carbon intensity (gCO2/kWh) to a green score in ``[0, 1]``.

    * ``quantile_rank`` - score = 1 - empirical CDF of the value within
      ``reference`` (rank-based, robust to the skew of carbon data). A value at
      the reference p10 scores ~0.90; at p90, ~0.10.
    * ``min_max`` - linear ``(hi - x) / (hi - lo)``.
    * ``threshold`` - 1 below ``threshold`` (or reference p25) else 0.

    With ``invert=True`` (default) lower carbon yields a higher score.
    ``reference`` defaults to ``values`` itself (self-normalization within the
    horizon).
    """
    v = np.asarray(values, dtype=float)
    ref = v if reference is None else np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(v.shape, np.nan)

    if method == "quantile_rank":
        order = np.sort(ref)
        cdf = np.searchsorted(order, v, side="right") / order.size
        score = 1.0 - cdf if invert else cdf
    elif method == "min_max":
        lo, hi = float(np.min(ref)), float(np.max(ref))
        if hi - lo < _EPS:
            score = np.full(v.shape, 0.5)
        else:
            norm = (v - lo) / (hi - lo)
            score = 1.0 - norm if invert else norm
    elif method == "threshold":
        thr = float(np.quantile(ref, 0.25)) if threshold is None else float(threshold)
        below = v <= thr
        score = below.astype(float) if invert else (~below).astype(float)
    else:
        raise ValueError(f"unknown green_score method: {method!r}")

    if clip:
        score = np.clip(score, 0.0, 1.0)
    return score


def green_score_series(
    carbon: pd.Series, reference: Sequence[float] | None = None, **kwargs
) -> pd.Series:
    """Green score for each point of a carbon Series (index preserved)."""
    return pd.Series(green_score(carbon.to_numpy(), reference, **kwargs), index=carbon.index)


def _merge_runs(
    selected: list[pd.Timestamp], step_hours: float, merge_gap_hours: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group sorted selected timestamps into runs, merging gaps <= merge_gap."""
    if not selected:
        return []
    tol = pd.Timedelta(hours=step_hours + merge_gap_hours)
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = selected[0]
    for t in selected[1:]:
        if (t - prev) <= tol:
            prev = t
        else:
            runs.append((start, prev))
            start = prev = t
    runs.append((start, prev))
    return runs


def _best_subwindow(carbon: pd.Series, max_len: int) -> pd.Series:
    """Return the length-``max_len`` contiguous slice with the lowest mean."""
    if len(carbon) <= max_len:
        return carbon
    best_start, best_mean = 0, np.inf
    vals = carbon.to_numpy()
    for i in range(0, len(carbon) - max_len + 1):
        m = float(np.nanmean(vals[i : i + max_len]))
        if m < best_mean:
            best_mean, best_start = m, i
    return carbon.iloc[best_start : best_start + max_len]


def compute_low_carbon_windows(
    carbon: pd.Series,
    *,
    percentile: float = 0.25,
    min_duration_hours: float = 1.0,
    max_duration_hours: float = 6.0,
    merge_gap_hours: float = 1.0,
    max_windows: int = 10,
    step_hours: float = 1.0,
    reference: Sequence[float] | None = None,
    window_type: WindowType = WindowType.low_carbon_window,
    source_model: ModelName | None = None,
    issued_at: datetime | None = None,
    zone: str = "FR",
) -> list[GreenWindow]:
    """Detect low-carbon windows in a (hourly) carbon Series over a horizon.

    ``carbon`` may be actual history (→ ``low_carbon_window``) or a forecast
    (→ ``predicted_low_carbon_window``). The percentile threshold and the
    green-score reference are computed on ``carbon`` itself unless a separate
    ``reference`` distribution is provided.
    """
    series = pd.to_numeric(carbon, errors="coerce").dropna().sort_index()
    if series.empty:
        return []

    ref = series.to_numpy() if reference is None else np.asarray(reference, dtype=float)
    threshold = float(np.quantile(ref, percentile))
    below = series[series <= threshold]
    runs = _merge_runs(list(below.index), step_hours, merge_gap_hours)

    windows: list[GreenWindow] = []
    for run_start, run_last in runs:
        block = series.loc[run_start:run_last]
        if block.empty:
            continue
        if len(block) * step_hours > max_duration_hours:
            block = _best_subwindow(block, int(round(max_duration_hours / step_hours)))
        duration = len(block) * step_hours
        if duration < min_duration_hours:
            continue

        start = block.index[0]
        end = block.index[-1] + pd.Timedelta(hours=step_hours)
        scores = green_score(block.to_numpy(), ref, method="quantile_rank")
        carbon_score = float(np.mean(scores))
        mean_intensity = float(block.mean())

        below_frac = float(np.mean(block.to_numpy() <= threshold))
        rel_spread = float(block.std(ddof=0)) / (float(np.std(ref)) + _EPS)
        confidence = float(np.clip(0.5 * below_frac + 0.5 * (1.0 - min(rel_spread, 1.0)), 0.0, 1.0))

        reasons = [
            f"Mean intensity {mean_intensity:.0f} gCO2/kWh, "
            f"below horizon p{int(round(percentile * 100))} ({threshold:.0f}).",
            f"Green score {carbon_score:.2f} (higher is greener).",
            f"{int(round(duration))}h window; internal variability "
            f"{'low' if rel_spread < 0.5 else 'moderate'}.",
        ]
        windows.append(
            GreenWindow(
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                zone=zone,
                window_type=window_type,
                carbon_score=carbon_score,
                mean_carbon_intensity_gco2_kwh=mean_intensity,
                confidence=confidence,
                source_model=source_model,
                issued_at=issued_at,
                reasons=reasons,
            )
        )

    windows.sort(key=lambda w: (w.carbon_score or 0.0), reverse=True)
    windows = windows[:max_windows]
    for i, w in enumerate(windows, start=1):
        w.rank = i
    return windows


def low_carbon_windows_from_config(
    carbon: pd.Series, window_cfg: dict, **overrides
) -> list[GreenWindow]:
    """Compute low-carbon windows using a ``window_scoring.yaml`` config dict."""
    w = window_cfg.get("windows", {})
    params = dict(
        percentile=w.get("percentile", 0.25),
        min_duration_hours=w.get("min_duration_hours", 1.0),
        max_duration_hours=w.get("max_duration_hours", 6.0),
        merge_gap_hours=w.get("merge_gap_hours", 1.0),
        max_windows=w.get("max_windows", 10),
    )
    params.update(overrides)
    return compute_low_carbon_windows(carbon, **params)
