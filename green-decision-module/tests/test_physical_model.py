"""Physics-guided Phase-C model tests."""

import numpy as np
import pandas as pd

from green_observatory.carbon.features import FeatureBuilder
from green_observatory.carbon.physical import (
    PhysicalCarbonMapper,
    PhysicalCarbonModel,
    generation_shares,
)
from green_observatory.providers.carbon_base import CARBON


GENERATION = [
    "nuclear_mw", "gas_mw", "coal_mw", "fuel_oil_mw",
    "wind_mw", "solar_mw", "hydro_mw", "bioenergy_mw",
]


def _physical_frame(periods=900):
    idx = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    hour = np.asarray(idx.hour)
    day = np.arange(periods) / 24.0
    rng = np.random.default_rng(4)
    data = pd.DataFrame(index=idx)
    data["nuclear_mw"] = 42000 + 1000 * np.sin(2 * np.pi * day / 30)
    data["wind_mw"] = 6000 + 2500 * np.sin(2 * np.pi * day / 5)
    data["solar_mw"] = np.maximum(0, 9000 * np.sin(np.pi * (hour - 6) / 13))
    data["hydro_mw"] = 6000 + 1000 * np.cos(2 * np.pi * hour / 24)
    data["bioenergy_mw"] = 1000.0
    data["gas_mw"] = np.maximum(
        100.0, 3500 - 0.18 * data["wind_mw"] - 0.10 * data["solar_mw"]
        + 150 * rng.normal(size=periods)
    )
    data["coal_mw"] = 120.0
    data["fuel_oil_mw"] = 60.0
    shares = generation_shares(
        data, generation_columns=GENERATION,
        share_columns=["gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw"],
    )
    data[CARBON] = (
        4.0 + 390 * shares["gas_mw_share"] + 980 * shares["coal_mw_share"]
        + 780 * shares["fuel_oil_mw_share"] + 180 * shares["bioenergy_mw_share"]
    )
    data["consumption_mw"] = data[GENERATION].sum(axis=1)
    return data


def test_physical_mapper_recovers_non_negative_emission_factors():
    frame = _physical_frame()
    shares = generation_shares(
        frame, generation_columns=GENERATION,
        share_columns=["gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw"],
    )
    mapper = PhysicalCarbonMapper(shares.columns).fit(shares, frame[CARBON])
    prediction = mapper.predict(shares)
    assert np.mean(np.abs(prediction - frame[CARBON])) < 1e-6
    assert all(value >= 0 for value in mapper.coefficients_.values())


def test_physical_model_trains_residual_out_of_sample_and_predicts_batch():
    frame = _physical_frame()
    train, full = frame.iloc[:720], frame
    fb = FeatureBuilder(
        climatology=None,
        lags_hours=(1, 24),
        rolling_means_hours=(3, 24),
        use_system=GENERATION,
    )
    params = {"max_iter": 25, "max_leaf_nodes": 7, "early_stopping": False}
    model = PhysicalCarbonModel(
        fb,
        horizons=(1, 6),
        generation_columns=GENERATION,
        share_columns=("gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw"),
        source_params=params,
        residual_params=params,
        residual_holdout_fraction=0.2,
    ).fit(train)
    origins = full.index[740:800:10]
    prediction = model.predict_batch(full, origins)
    assert len(prediction) == len(origins) * 2
    assert prediction["prediction"].notna().all()
    assert (prediction["prediction"] >= 0).all()
    assert model.residual_calibration_rows_[1] > 100
