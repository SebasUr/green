import numpy as np
import pandas as pd

from green_observatory.carbon.green_window_ranker import (
    blend_query_percentiles,
    complete_query_positions,
    ordinal_relevance,
    within_query_percentile,
)
from green_observatory.carbon.green_window_ranker_evaluation import (
    paired_daily_bootstrap,
    select_blend_weight,
    window_metrics,
)


def _meta(origins: int = 2) -> pd.DataFrame:
    origin = pd.date_range("2026-01-01", periods=origins, freq="1D", tz="UTC")
    repeated = pd.DatetimeIndex(np.repeat(origin.asi8, 24), tz="UTC")
    horizon = np.tile(np.arange(1, 25), origins)
    return pd.DataFrame(
        {
            "origin": repeated,
            "horizon": horizon,
            "target_time": repeated + pd.to_timedelta(horizon, unit="h"),
            "actual": np.tile(np.arange(24.0, 0.0, -1.0), origins),
        }
    )


def test_complete_query_positions_rejects_cutoff_split_query():
    meta = _meta(2)
    # Full first query plus only h1..h12 of the second query.
    mask = np.r_[np.ones(24, dtype=bool), np.arange(1, 25) <= 12]
    positions = complete_query_positions(meta, mask=mask)
    assert positions.tolist() == list(range(24))


def test_ordinal_relevance_is_highest_for_lowest_carbon_and_preserves_ties():
    meta = _meta(1)
    meta.index = np.arange(100, 124)  # public helper must not require RangeIndex
    meta["actual"] = np.arange(1.0, 25.0)
    meta.loc[meta.index[:2], "actual"] = 1.0
    relevance = ordinal_relevance(meta)
    assert relevance[0] == relevance[1] == 23
    assert relevance[-1] < relevance[0]


def test_within_query_percentile_has_zero_at_greenest_value():
    origins = np.repeat(["a", "b"], 3)
    intensity = np.array([30.0, 10.0, 20.0, 4.0, 6.0, 5.0])
    rank = within_query_percentile(
        intensity, origins, higher_is_greener=False
    )
    np.testing.assert_allclose(rank, [1.0, 0.0, 0.5, 0.0, 1.0, 0.5])
    score_rank = within_query_percentile(
        -intensity, origins, higher_is_greener=True
    )
    np.testing.assert_allclose(rank, score_rank)


def test_percentile_blend_does_not_mix_rank_score_and_carbon_units():
    origins = np.repeat(["a"], 3)
    score = np.array([1000.0, 3000.0, 2000.0])
    carbon = np.array([1.0, 3.0, 2.0])
    ranker_only = blend_query_percentiles(
        score, carbon, origins, ranker_weight=1.0
    )
    share_only = blend_query_percentiles(
        score, carbon, origins, ranker_weight=0.0
    )
    np.testing.assert_allclose(ranker_only, [1.0, 0.0, 0.5])
    np.testing.assert_allclose(share_only, [0.0, 1.0, 0.5])


def test_window_metrics_and_blend_selection_operate_per_complete_day():
    frame = _meta(3)
    # Share selects the dirtiest hour; ranker selects the actual greenest.
    frame["share_lgbm"] = -frame["actual"]
    frame["ranker_score"] = -frame["actual"]
    frame["green_window_ranker"] = within_query_percentile(
        frame["ranker_score"], frame["origin"], higher_is_greener=True
    )
    actual_by_time = pd.Series(
        [30.0, 30.0, 30.0], index=frame["origin"].drop_duplicates()
    )
    metrics = window_metrics(
        frame, "green_window_ranker", actual_by_time=actual_by_time
    )
    assert metrics["mean_regret"] == 0.0
    assert metrics["top1_accuracy"] == 1.0
    weight, candidates = select_blend_weight(
        frame, actual_by_time=actual_by_time, grid=(0.0, 1.0)
    )
    assert weight == 1.0
    assert len(candidates) == 2


def test_paired_daily_bootstrap_reports_exact_deterministic_delta():
    frame = _meta(3)
    frame["candidate"] = frame["actual"]
    frame["reference"] = -frame["actual"]
    report = paired_daily_bootstrap(
        frame, "candidate", "reference", samples=100, seed=7
    )
    assert report["n_days"] == 3
    assert report["mean_realized_delta_gco2"] == -23.0
    assert report["day_bootstrap_95ci"] == [-23.0, -23.0]
