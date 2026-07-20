import numpy as np
import pandas as pd
import pytest

from green_observatory.windows.exhaustive import (
    aggregate_window_metrics,
    enumerate_contiguous_windows,
    evaluate_exhaustive_windows,
    validate_complete_queries,
)


def _predictions(
    actual: np.ndarray,
    prediction: np.ndarray | None = None,
    *,
    models: tuple[str, ...] = ("model",),
    origin: pd.Timestamp = pd.Timestamp("2026-03-01", tz="UTC"),
) -> pd.DataFrame:
    prediction = actual if prediction is None else prediction
    parts = []
    for model in models:
        parts.append(
            pd.DataFrame(
                {
                    "model": model,
                    "origin": origin,
                    "horizon": np.arange(1, 25),
                    "target_time": origin
                    + pd.to_timedelta(np.arange(1, 25), unit="h"),
                    "prediction": prediction,
                    "actual": actual,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_prefix_catalog_enumerates_every_start_for_each_duration() -> None:
    actual = np.arange(1.0, 25.0)
    catalog = enumerate_contiguous_windows(
        _predictions(actual), durations=(1, 3, 24)
    )

    assert len(catalog) == 24 + 22 + 1
    three = catalog[catalog["duration_hours"].eq(3)]
    assert three["start_horizon"].tolist() == list(range(1, 23))
    assert three.iloc[0]["actual_window_cost"] == 2.0
    assert three.iloc[-1]["actual_window_cost"] == 23.0
    assert three.iloc[-1]["end_horizon"] == 24
    assert (
        three.iloc[-1]["window_end_time"]
        - three.iloc[-1]["window_start_time"]
    ) == pd.Timedelta(hours=3)


def test_validation_rejects_incomplete_and_misaligned_queries() -> None:
    actual = np.arange(1.0, 25.0)
    incomplete = _predictions(actual).iloc[:-1]
    with pytest.raises(ValueError, match="incomplete 24h query"):
        validate_complete_queries(incomplete)

    misaligned = _predictions(actual)
    misaligned.loc[5, "target_time"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match=r"origin\+horizon"):
        validate_complete_queries(misaligned)


def test_validation_requires_common_support_and_identical_actuals() -> None:
    actual = np.arange(1.0, 25.0)
    common = _predictions(actual, models=("a", "b"))
    common.loc[(common["model"].eq("b")) & common["horizon"].eq(4), "actual"] += 1
    with pytest.raises(ValueError, match="actual curve differs"):
        validate_complete_queries(common)

    second_origin = _predictions(
        actual,
        models=("a",),
        origin=pd.Timestamp("2026-03-02", tz="UTC"),
    )
    unequal_support = pd.concat([_predictions(actual, models=("a", "b")), second_origin])
    with pytest.raises(ValueError, match="identical origin support"):
        validate_complete_queries(unequal_support)


def test_decision_is_tie_aware_and_reports_random_and_ranking_metrics() -> None:
    # Two equally green actual starts at h2/h3 for a one-hour job.  The model
    # selects the later one, which remains a correct top-1 decision.
    actual = np.asarray([10.0, 1.0, 1.0, *([8.0] * 21)])
    prediction = np.asarray([9.0, 2.0, 0.0, *([7.0] * 21)])
    result = evaluate_exhaustive_windows(
        _predictions(actual, prediction), durations=(1,), epsilon_gco2=0.5
    )
    row = result.decisions.iloc[0]

    assert row["selected_start_horizon"] == 3
    assert row["oracle_start_horizon"] == 2
    assert row["oracle_tie_count"] == 2
    assert bool(row["top1_tie_aware"])
    assert bool(row["epsilon_optimal"])
    assert row["selected_actual_rank_min"] == 1
    assert row["selected_actual_rank_percentile"] == 0.0
    assert bool(row["top10pct_hit"])
    assert row["asap_actual_cost"] == 10.0
    assert row["random_expected_actual_cost"] == pytest.approx(actual.mean())
    assert np.isfinite(row["window_cost_spearman"])


def test_oracle_potential_is_ratio_of_sums_and_is_not_clipped() -> None:
    # The second origin is made deliberately worse than ASAP.  Ratio of sums:
    # (5 + -1) / (10 + 1) = 36.36%, not mean(50%, -100%) = -25%.
    decisions = pd.DataFrame(
        {
            "model": ["m", "m"],
            "origin": pd.to_datetime(["2026-03-01", "2026-03-02"], utc=True),
            "duration_hours": [2, 2],
            "selected_actual_cost": [5.0, 3.0],
            "oracle_actual_cost": [0.0, 1.0],
            "asap_actual_cost": [10.0, 2.0],
            "random_expected_actual_cost": [6.0, 2.5],
            "regret_gco2_kwh": [5.0, 2.0],
            "savings_vs_asap_gco2_kwh": [5.0, -1.0],
            "oracle_opportunity_gco2_kwh": [10.0, 1.0],
            "improvement_vs_random_gco2_kwh": [1.0, -0.5],
            "random_regret_gco2_kwh": [6.0, 1.5],
            "top1_tie_aware": [False, False],
            "epsilon_optimal": [False, False],
            "selected_actual_rank_percentile": [0.5, 1.0],
            "top10pct_hit": [False, False],
            "window_cost_spearman": [0.5, -0.5],
        }
    )
    aggregate = aggregate_window_metrics(decisions).iloc[0]

    assert aggregate["oracle_potential_pct"] == pytest.approx(100.0 * 4.0 / 11.0)

    decisions["savings_vs_asap_gco2_kwh"] = [-20.0, -2.0]
    negative = aggregate_window_metrics(decisions).iloc[0]
    assert negative["oracle_potential_pct"] == -200.0


def test_duration_24_has_one_candidate_and_undefined_oracle_potential() -> None:
    actual = np.linspace(1.0, 24.0, 24)
    result = evaluate_exhaustive_windows(
        _predictions(actual, actual[::-1]), durations=(24,)
    )
    decision = result.decisions.iloc[0]
    aggregate = result.aggregate.iloc[0]

    assert decision["candidate_count"] == 1
    assert decision["regret_gco2_kwh"] == 0.0
    assert bool(decision["top1_tie_aware"])
    assert np.isnan(decision["window_cost_spearman"])
    assert np.isnan(aggregate["oracle_potential_pct"])


def test_custom_horizon_supports_the_common_23h_d1_track() -> None:
    origin = pd.Timestamp("2026-03-01", tz="UTC")
    actual = np.arange(1.0, 24.0)
    predictions = pd.DataFrame(
        {
            "model": "d1",
            "origin": origin,
            "horizon": np.arange(1, 24),
            "target_time": origin + pd.to_timedelta(np.arange(1, 24), unit="h"),
            "prediction": actual[::-1],
            "actual": actual,
        }
    )

    result = evaluate_exhaustive_windows(
        predictions,
        durations=(1, 2, 23),
        horizon_hours=23,
    )

    assert len(result.catalog) == 23 + 22 + 1
    assert result.catalog["deadline_horizon"].eq(23).all()
    assert set(result.decisions["duration_hours"]) == {1, 2, 23}
    with pytest.raises(ValueError, match=r"\[1, 23\]"):
        evaluate_exhaustive_windows(
            predictions,
            durations=(24,),
            horizon_hours=23,
        )
