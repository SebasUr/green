import numpy as np
import pandas as pd

from green_observatory.carbon.source_share_proxy import (
    add_causal_share_features,
    source_share_targets,
    source_share_variants,
)


def _generation_frame() -> pd.DataFrame:
    index = pd.date_range("2025-12-20", "2026-01-03", freq="1h", tz="UTC")
    base = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "nuclear_mw": 40_000.0 + base,
            "gas_mw": 2_000.0 + base,
            "coal_mw": 100.0 + base,
            "fuel_oil_mw": 200.0 + base,
            "wind_mw": 5_000.0 + base,
            "solar_mw": 1_000.0 + base,
            "hydro_mw": 8_000.0 + base,
            "bioenergy_mw": 1_200.0 + base,
        },
        index=index,
    )


def test_share_features_use_last_closed_hour_and_mask_open_d1():
    frame = _generation_frame()
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    meta = pd.DataFrame(
        {
            "origin": [origin, origin],
            "target_time": [origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=24)],
        }
    )
    x = pd.DataFrame({"base": [1.0, 2.0]})
    before = add_causal_share_features(x, meta, frame)

    mutated = frame.copy()
    mutated.loc[origin, ["gas_mw", "coal_mw", "fuel_oil_mw", "bioenergy_mw"]] *= 100.0
    after = add_causal_share_features(x, meta, mutated)

    assert np.allclose(
        before["share_origin_gas_mw_share"],
        after["share_origin_gas_mw_share"],
    )
    assert np.isnan(before.loc[1, "share_tgtlag24_gas_mw_share"])
    assert np.isfinite(before.loc[0, "share_tgtlag24_gas_mw_share"])


def test_source_share_targets_sum_to_emitting_fraction():
    frame = _generation_frame()
    target = frame.index[-2:]
    shares = source_share_targets(frame, target)
    assert list(shares.columns) == [
        "gas_mw_share",
        "coal_mw_share",
        "fuel_oil_mw_share",
        "bioenergy_mw_share",
    ]
    assert (shares.sum(axis=1) < 1.0).all()


def test_source_share_variants_preserve_fixed_non_gas_contribution():
    matrix = pd.DataFrame(
        {
            "prediction": [20.0],
            "prediction_pooled": [21.0],
            "predicted_gas_share": [10.0 / 429.0],
            "prob_baseload": [0.5],
            "prob_ccg": [0.3],
            "prob_peak": [0.2],
            "gas_expert_baseload_contribution": [8.0],
            "gas_expert_ccg_contribution": [12.0],
            "gas_expert_peak_contribution": [20.0],
        }
    )
    variants = source_share_variants(matrix)
    assert variants["share_physical"][0] == 20.0
    assert variants["share_physical_pooled"][0] == 21.0
    assert variants["share_physical_hard"][0] == 18.0
