import numpy as np
import pandas as pd

from green_observatory.carbon.extra_trees_carbon import ExtraTreesCarbonRegressor
from green_observatory.carbon.extra_trees_evaluation import (
    ERTSpec,
    select_spec,
    window_metrics,
)


def test_imputer_is_fit_only_on_training_rows() -> None:
    x = pd.DataFrame({"a": [1.0, np.nan, 3.0], "empty": [np.nan] * 3})
    model = ExtraTreesCarbonRegressor(n_estimators=5, min_samples_leaf=1)
    model.fit(x, np.asarray([10.0, 11.0, 12.0]))

    assert model.feature_names_ == ["a"]
    assert float(model.imputer_.statistics_[0]) == 2.0
    assert np.isfinite(model.predict(pd.DataFrame({"a": [1_000_000.0]}))).all()
    assert float(model.imputer_.statistics_[0]) == 2.0


def test_relative_error_weights_use_the_declared_floor() -> None:
    model = ExtraTreesCarbonRegressor(inverse_level_floor=8.0)
    np.testing.assert_allclose(
        model.sample_weight(np.asarray([2.0, 8.0, 16.0])),
        [1.0 / 8.0, 1.0 / 8.0, 1.0 / 16.0],
    )


def test_spec_selection_uses_mape_then_declared_order() -> None:
    specs = (
        ERTSpec("first", 5, 1.0, 1),
        ERTSpec("second", 5, 1.0, 1),
    )
    selected = select_spec(
        {"first": {"mape": 9.0}, "second": {"mape": 9.0}}, specs
    )
    assert selected.name == "first"


def test_window_metrics_reports_perfect_oracle_selection() -> None:
    origin = pd.Timestamp("2026-03-01", tz="UTC")
    actual = np.arange(24.0, 0.0, -1.0)
    frame = pd.DataFrame(
        {
            "origin": origin,
            "horizon": np.arange(1, 25),
            "actual": actual,
            "prediction": actual,
        }
    )
    actual_by_time = pd.Series([30.0], index=pd.DatetimeIndex([origin]))

    metrics = window_metrics(
        frame, "prediction", actual_by_time=actual_by_time
    )

    assert metrics["mean_regret"] == 0.0
    assert metrics["pct_oracle_potential"] == 100.0
    assert metrics["top1_accuracy"] == 1.0
