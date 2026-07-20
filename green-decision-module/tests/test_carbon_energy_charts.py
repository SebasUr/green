import numpy as np
import pandas as pd
import pytest

from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_energy_charts import (
    EnergyChartsCarbonGapProvider,
)


def _payload(start="2026-05-01T00:00:00Z", periods=8):
    index = pd.date_range(start, periods=periods, freq="15min")
    names = {
        "Load": 40_000.0,
        "Nuclear": 30_000.0,
        "Fossil gas": 1_000.0,
        "Fossil oil": 50.0,
        "Wind onshore": 2_000.0,
        "Wind offshore": 500.0,
        "Solar": 3_000.0,
        "Hydro Run-of-River": 4_000.0,
        "Hydro water reservoir": 1_000.0,
        "Hydro pumped storage": 500.0,
        "Hydro pumped storage consumption": -200.0,
        "Biomass": 300.0,
        "Waste": 400.0,
        "Cross border electricity trading": -5_000.0,
    }
    return {
        "unix_seconds": [int(value.timestamp()) for value in index],
        "production_types": [
            {"name": name, "data": [value] * periods}
            for name, value in names.items()
        ],
    }


def test_parse_energy_charts_gap_to_rte_like_hourly_aggregates():
    history = pd.Series(
        [290.0, 310.0],
        index=pd.DatetimeIndex(
            ["2026-04-29T00:00:00Z", "2026-04-30T00:00:00Z"]
        ),
    )
    out = EnergyChartsCarbonGapProvider.parse_payloads(
        [_payload()], biogas_history=history
    )

    assert len(out) == 2
    assert out.index.tz is not None
    assert out["coal_mw"].eq(0.0).all()
    assert out["wind_mw"].eq(2_500.0).all()
    assert out["hydro_mw"].eq(5_500.0).all()
    assert out["bioenergy_biogas_mw"].eq(300.0).all()
    assert out["bioenergy_mw"].eq(1_000.0).all()
    # Energy-Charts' cross-border trading is a commercial schedule, whereas
    # this canonical column is RTE's physical flow.  Never alias the two.
    assert out["physical_exchange_mw"].isna().all()
    assert out[CARBON].isna().all()


def test_biogas_imputation_ignores_values_at_or_after_gap_start():
    history = pd.Series(
        [300.0, 900.0],
        index=pd.DatetimeIndex(
            ["2026-04-30T23:00:00Z", "2026-05-01T00:00:00Z"]
        ),
    )
    out = EnergyChartsCarbonGapProvider.parse_payloads(
        [_payload()], biogas_history=history
    )
    assert out["bioenergy_biogas_mw"].eq(300.0).all()


def test_partial_energy_charts_hour_is_not_manufactured():
    history = pd.Series(
        [300.0], index=pd.DatetimeIndex(["2026-04-30T00:00:00Z"])
    )
    out = EnergyChartsCarbonGapProvider.parse_payloads(
        [_payload(periods=7)], biogas_history=history
    )
    assert np.isfinite(out.iloc[0]["nuclear_mw"])
    assert np.isnan(out.iloc[1]["nuclear_mw"])


def test_parse_rejects_missing_required_series():
    payload = _payload()
    payload["production_types"] = [
        item for item in payload["production_types"] if item["name"] != "Nuclear"
    ]
    with pytest.raises(ValueError, match="Nuclear"):
        EnergyChartsCarbonGapProvider.parse_payloads(
            [payload],
            biogas_history=pd.Series(
                [300.0], index=pd.DatetimeIndex(["2026-04-30T00:00:00Z"])
            ),
        )
