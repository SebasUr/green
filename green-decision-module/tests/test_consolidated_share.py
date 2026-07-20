import numpy as np
import pandas as pd

from green_observatory.carbon.consolidated_share import (
    add_causal_detailed_share_features,
    detailed_generation_shares,
    detailed_share_targets,
)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2025-12-20", "2026-01-03", freq="1h", tz="UTC")
    base = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "nuclear_mw": 40_000.0 + base,
            "gas_mw": 2_000.0 + base,
            "coal_mw": 100.0 + base,
            "fuel_oil_mw": 200.0 + base,
            "wind_mw": 5_000.0 + base,
            "solar_mw": 1_000.0 + base,
            "hydro_mw": 8_000.0 + base,
            "bioenergy_mw": 1_200.0 + base,
            "gas_turbine_mw": 100.0 + base / 10,
            "gas_cogeneration_mw": 400.0 + base / 10,
            "gas_ccg_mw": 1_300.0 + base / 2,
            "gas_other_mw": 200.0 + base / 10,
            "bioenergy_waste_mw": 600.0 + base / 10,
        },
        index=index,
    )
    return frame


def test_detailed_generation_shares_use_aggregate_domestic_denominator():
    frame = _frame().iloc[:1]
    shares = detailed_generation_shares(frame)
    denominator = frame[
        [
            "nuclear_mw",
            "gas_mw",
            "coal_mw",
            "fuel_oil_mw",
            "wind_mw",
            "solar_mw",
            "hydro_mw",
            "bioenergy_mw",
        ]
    ].sum(axis=1).iloc[0]
    assert np.isclose(shares["gas_ccg_mw_share"].iloc[0], 1300.0 / denominator)
    targets = detailed_share_targets(frame, frame.index)
    assert np.isclose(targets["total_generation_mw"].iloc[0], denominator)


def test_detailed_share_features_mask_open_hour_and_ignore_its_mutation():
    frame = _frame()
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    meta = pd.DataFrame(
        {
            "origin": [origin, origin],
            "target_time": [origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=24)],
        }
    )
    x = pd.DataFrame({"base": [1.0, 2.0]})
    before = add_causal_detailed_share_features(x, meta, frame)
    mutated = frame.copy()
    mutated.loc[origin, "gas_ccg_mw"] *= 100
    after = add_causal_detailed_share_features(x, meta, mutated)
    assert np.allclose(
        before["detail_share_origin_gas_ccg_mw_share"],
        after["detail_share_origin_gas_ccg_mw_share"],
    )
    assert np.isnan(before.loc[1, "detail_share_tgtlag24_gas_ccg_mw_share"])
