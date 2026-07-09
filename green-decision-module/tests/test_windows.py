"""Green-score and low-carbon window detection tests."""

import numpy as np
import pandas as pd

from green_observatory.models import WindowType
from green_observatory.windows.scoring import compute_low_carbon_windows, green_score


def test_green_score_monotonic_and_bounded():
    scores = green_score([10, 20, 30, 40])
    assert scores.tolist() == [0.75, 0.5, 0.25, 0.0]  # lower carbon -> higher score
    assert (scores >= 0).all() and (scores <= 1).all()


def _series(values, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="1h", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _two_dip_horizon():
    v = np.full(48, 50.0)
    v[8:14] = 10.0    # dip 1 (6h, greener)
    v[30:36] = 12.0   # dip 2 (6h)
    return _series(v)


def test_detects_and_ranks_two_windows():
    wins = compute_low_carbon_windows(
        _two_dip_horizon(), percentile=0.25, min_duration_hours=2, max_duration_hours=6
    )
    assert len(wins) == 2
    # greener dip (mean 10) ranks first
    assert wins[0].rank == 1
    assert wins[0].mean_carbon_intensity_gco2_kwh < wins[1].mean_carbon_intensity_gco2_kwh
    assert wins[0].carbon_score > wins[1].carbon_score
    assert wins[0].window_type is WindowType.low_carbon_window
    assert wins[0].carbon_score > 0.6


def test_min_duration_filters_short_windows():
    v = np.full(48, 50.0)
    v[10] = 5.0  # single-hour dip
    wins = compute_low_carbon_windows(_series(v), percentile=0.25, min_duration_hours=2)
    assert all(w.duration_hours >= 2 for w in wins)


def test_merge_gap_joins_adjacent_blocks():
    v = np.full(48, 50.0)
    v[8:14] = 10.0
    v[14] = 50.0      # one high hour...
    v[15:18] = 10.0   # ...between two low blocks
    merged = compute_low_carbon_windows(
        _series(v), percentile=0.25, min_duration_hours=2, max_duration_hours=24, merge_gap_hours=1
    )
    # the single-hour gap is bridged into one longer window
    assert any(w.duration_hours >= 9 for w in merged)


def test_max_windows_limit():
    wins = compute_low_carbon_windows(
        _two_dip_horizon(), percentile=0.25, min_duration_hours=2, max_windows=1
    )
    assert len(wins) == 1
