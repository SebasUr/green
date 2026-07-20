import numpy as np
import pandas as pd
import pytest

from green_observatory.carbon.realtime_proxy import (
    PhysicalProxyMoE,
    rte_realtime_carbon_proxy,
)


def test_rte_realtime_proxy_uses_fixed_factors_and_total_generation():
    frame = pd.DataFrame(
        {
            "nuclear_mw": [50.0],
            "gas_mw": [10.0],
            "coal_mw": [5.0],
            "fuel_oil_mw": [2.0],
            "wind_mw": [10.0],
            "solar_mw": [5.0],
            "hydro_mw": [13.0],
            "bioenergy_mw": [5.0],
        },
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"]),
    )
    expected = (986 * 5 + 777 * 2 + 429 * 10 + 494 * 5) / 100
    assert np.isclose(rte_realtime_carbon_proxy(frame).iloc[0], expected)


def test_rte_realtime_proxy_does_not_treat_missing_fuel_as_zero():
    frame = pd.DataFrame(
        {
            "nuclear_mw": [50.0],
            "gas_mw": [10.0],
            "coal_mw": [np.nan],
            "fuel_oil_mw": [2.0],
            "wind_mw": [10.0],
            "solar_mw": [5.0],
            "hydro_mw": [13.0],
            "bioenergy_mw": [5.0],
        }
    )
    assert np.isnan(rte_realtime_carbon_proxy(frame).iloc[0])


def test_physical_proxy_rejects_non_positive_recency_half_life():
    with pytest.raises(ValueError, match="recency_half_life_days"):
        PhysicalProxyMoE(recency_half_life_days=0.0)
