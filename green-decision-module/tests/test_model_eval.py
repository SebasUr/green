"""Model + evaluation tests: batch == per-origin forecaster, metric sanity."""

import numpy as np
import pandas as pd

from green_observatory.carbon import evaluation as ev
from green_observatory.carbon.climatology import ClimatologyForecaster, ClimatologyModel
from green_observatory.carbon.features import FeatureBuilder
from green_observatory.carbon.model import ProjectCarbonModel
from green_observatory.providers.carbon_base import CARBON
from green_observatory.windows.oracle import window_selection_metrics


def _frame(periods=600, start="2025-01-01"):
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(idx.tz_convert("Europe/Paris").hour)
    carbon = 25 + 12 * np.sin(2 * np.pi * hour / 24.0) + 0.5 * np.random.default_rng(0).normal(size=periods)
    return pd.DataFrame({CARBON: np.clip(carbon, 0, None), "consumption_mw": 60000.0}, index=idx)


def _tiny_model(frame):
    fb = FeatureBuilder(climatology=None)
    return ProjectCarbonModel(
        fb, horizons=(1, 3), algorithm="hist_gradient_boosting",
        params={"max_iter": 40, "early_stopping": False},
    ).fit(frame)


def test_project_batch_equals_forecaster():
    frame = _frame()
    model = _tiny_model(frame)
    forecaster = model.make_forecaster(frame)
    origins = frame.index[[300, 400, 500]]

    batch = ev._project_batch(model, frame, origins, [1, 3]).set_index(["origin", "horizon"])[
        "prediction"
    ]
    for origin in origins:
        fc = forecaster.predict(frame.loc[:origin], origin, [1, 3])
        for h in (1, 3):
            got = fc.loc[origin + pd.Timedelta(hours=h), "prediction"]
            assert abs(got - batch.loc[(origin, h)]) < 1e-9


def test_climatology_batch_equals_forecaster():
    frame = _frame()
    clim = ClimatologyModel(min_samples=3).fit(frame)
    fc = ClimatologyForecaster(clim)
    origins = frame.index[[300, 450]]
    batch = ev._climatology_batch(clim, origins, [1, 6, 24]).set_index(["origin", "horizon"])[
        "prediction"
    ]
    for origin in origins:
        out = fc.predict(frame.loc[:origin], origin, [1, 6, 24])
        for h in (1, 6, 24):
            assert abs(out.loc[origin + pd.Timedelta(hours=h), "prediction"] - batch.loc[(origin, h)]) < 1e-9


def test_learns_diurnal_pattern_better_than_naive_mean():
    frame = _frame()
    model = _tiny_model(frame)
    origins = frame.index[400:560:8]
    pred = ev.backtest_predictions(
        frame, origins, horizons=(1, 3), project_model=model, include=("project",)
    )
    mae = ev.point_metrics(pred)["mae"].mean()
    # a model that learned the +/-12 diurnal swing should beat the ~8 gCO2 MAE
    # of predicting the global mean
    assert mae < 6.0


def test_window_selection_perfect_model_captures_full_potential():
    origins = pd.DatetimeIndex(["2026-02-01T00:00:00Z", "2026-02-01T06:00:00Z"])
    df = pd.DataFrame({CARBON: [25.0, 25.0]}, index=origins)
    rows = []
    for origin in origins:
        # candidate actuals across 3 horizons; a perfectly-ranking prediction
        acts = [30.0, 10.0, 20.0]
        preds = [3.0, 1.0, 2.0]  # same ranking as actuals -> picks the 10
        for h, a, p in zip((1, 3, 6), acts, preds):
            rows.append({"model": "project", "origin": origin, "horizon": h,
                         "target_time": origin + pd.Timedelta(hours=h), "prediction": p, "actual": a})
    pred_df = pd.DataFrame(rows)
    m = window_selection_metrics(pred_df, df)
    assert m.loc["project", "mean_regret"] == 0.0
    assert m.loc["project", "top1_accuracy"] == 1.0
    assert m.loc["project", "pct_oracle_potential"] == 100.0
    assert m.loc["run_now", "mean_realized_gco2"] == 25.0
    assert m.loc["oracle", "mean_realized_gco2"] == 10.0
