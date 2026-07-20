"""Leakage guards for consolidated residual calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from green_observatory.carbon.consolidated_physical_calibration import (
    CalibrationSpec,
    _prepare,
    causal_scale,
    select_level_threshold,
    select_share_level_thresholds,
    tune_calibration,
)


def _frame(start: str = "2026-01-01", days: int = 7) -> pd.DataFrame:
    rows = []
    for day, origin in enumerate(pd.date_range(start, periods=days, freq="1D", tz="UTC")):
        for horizon in range(1, 25):
            actual = 10.0 + day + 0.05 * horizon
            rows.append(
                {
                    "origin": origin,
                    "horizon": horizon,
                    "target_time": origin + pd.Timedelta(hours=horizon),
                    "actual": actual,
                    "physical_lgbm": 0.90 * actual,
                    "share_lgbm": 0.95 * actual,
                    "direct_reference": 1.08 * actual,
                    "oracle_learned_factors": actual,
                }
            )
    out = _prepare(pd.DataFrame(rows))
    out["base"] = 0.6 * out["physical_lgbm"] + 0.4 * out["direct_reference"]
    return out


def test_causal_scale_ignores_target_at_origin_and_future() -> None:
    frame = _frame(days=5)
    evaluation_origin = pd.Timestamp("2026-01-05", tz="UTC")
    spec = CalibrationSpec(lookback_days=3, block_mode="four_blocks", shrink=1.0)
    before = causal_scale(
        frame, "base", [evaluation_origin], spec, scale_grid=(0.8, 1.0, 1.2)
    )

    changed = frame.copy()
    unavailable = changed["target_time"] >= evaluation_origin
    changed.loc[unavailable, "actual"] *= 100.0
    # Forecasts issued after the evaluated origin are not part of the current
    # prediction either, so changing them must not alter its fitted scale.
    changed.loc[changed["origin"] > evaluation_origin, "base"] *= 0.01
    after = causal_scale(
        changed, "base", [evaluation_origin], spec, scale_grid=(0.8, 1.0, 1.2)
    )
    pdt.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_fresh_start_excludes_seed_history() -> None:
    frame = _frame(days=6)
    origin = pd.Timestamp("2026-01-06", tz="UTC")
    floor = pd.Timestamp("2026-01-05", tz="UTC")
    spec = CalibrationSpec(lookback_days=5, block_mode="global", shrink=1.0)
    seeded_before = causal_scale(
        frame, "base", [origin], spec, scale_grid=(0.8, 1.0, 1.2)
    )
    fresh_before = causal_scale(
        frame,
        "base",
        [origin],
        spec,
        history_origin_floor=floor,
        scale_grid=(0.8, 1.0, 1.2),
    )

    changed = frame.copy()
    seed = changed["origin"] < floor
    changed.loc[seed, "actual"] *= 2.0
    seeded_after = causal_scale(
        changed, "base", [origin], spec, scale_grid=(0.8, 1.0, 1.2)
    )
    fresh_after = causal_scale(
        changed,
        "base",
        [origin],
        spec,
        history_origin_floor=floor,
        scale_grid=(0.8, 1.0, 1.2),
    )
    assert not np.allclose(
        seeded_before["prediction"], seeded_after["prediction"]
    )
    pdt.assert_frame_equal(
        fresh_before.reset_index(drop=True), fresh_after.reset_index(drop=True)
    )


def test_validation_selection_cannot_see_later_actuals() -> None:
    frame = _frame(days=8)
    validation_origins = pd.date_range(
        "2026-01-05", periods=2, freq="1D", tz="UTC"
    )
    selected_before, candidates_before = tune_calibration(
        frame,
        "base",
        validation_origins,
        lookbacks=(2, 3),
        block_modes=("global", "four_blocks"),
        shrinks=(0.0, 0.5),
        scale_grid=(0.8, 1.0, 1.2),
    )
    changed = frame.copy()
    changed.loc[changed["origin"] > validation_origins.max(), "actual"] *= 50.0
    selected_after, candidates_after = tune_calibration(
        changed,
        "base",
        validation_origins,
        lookbacks=(2, 3),
        block_modes=("global", "four_blocks"),
        shrinks=(0.0, 0.5),
        scale_grid=(0.8, 1.0, 1.2),
    )
    assert selected_before == selected_after
    assert candidates_before == candidates_after


def test_january_gate_selection_cannot_see_february() -> None:
    frame = _frame(days=6)
    january = frame["origin"] < pd.Timestamp("2026-01-04", tz="UTC")
    selected_before, scores_before = select_level_threshold(
        frame, january, grid=(8.0, 12.0, 16.0)
    )
    changed = frame.copy()
    changed.loc[~january, "actual"] *= 100.0
    selected_after, scores_after = select_level_threshold(
        changed, january, grid=(8.0, 12.0, 16.0)
    )
    assert selected_before == selected_after
    assert scores_before == scores_after


def test_share_gate_selection_cannot_see_post_selection_targets() -> None:
    frame = _frame(days=6)
    january = frame["origin"] < pd.Timestamp("2026-01-04", tz="UTC")
    selected_before, scores_before = select_share_level_thresholds(
        frame, january, grid=(8.0, 12.0, 16.0, 20.0)
    )
    changed = frame.copy()
    changed.loc[~january, "actual"] *= 100.0
    changed.loc[~january, "share_lgbm"] *= 0.01
    selected_after, scores_after = select_share_level_thresholds(
        changed, january, grid=(8.0, 12.0, 16.0, 20.0)
    )
    assert selected_before == selected_after
    assert scores_before == scores_after
