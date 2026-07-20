import numpy as np
import pandas as pd

from green_observatory.carbon.realtime_proxy_live_refit import (
    _build_state_features_with_rte_supervision,
)
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder
from green_observatory.providers.carbon_base import CARBON


def test_gap_state_never_becomes_supervision():
    origin = pd.Timestamp("2026-06-17", tz="UTC")
    index = pd.date_range(origin - pd.Timedelta(days=8), periods=240, freq="1h")
    state = pd.DataFrame(
        {
            CARBON: 999.0,
            "gas_turbine_mw": 0.0,
            "gas_cogeneration_mw": 0.0,
            "gas_ccg_mw": 600.0,
            "gas_other_mw": 0.0,
        },
        index=index,
    )
    labels = state.copy()
    labels[CARBON] = np.nan
    labels[[
        "gas_turbine_mw", "gas_cogeneration_mw", "gas_ccg_mw", "gas_other_mw"
    ]] = np.nan
    only_rte_target = origin + pd.Timedelta(hours=1)
    labels.loc[only_rte_target, CARBON] = 12.0
    labels.loc[only_rte_target, [
        "gas_turbine_mw", "gas_cogeneration_mw", "gas_ccg_mw", "gas_other_mw"
    ]] = [0.0, 0.0, 600.0, 0.0]
    forecasts = pd.DataFrame({"known": 1.0}, index=index)

    x, meta = _build_state_features_with_rte_supervision(
        RegimeMoEFeatureBuilder(forecasts),
        state,
        labels,
        pd.DatetimeIndex([origin]),
    )

    assert len(x) == len(meta) == 1
    assert meta.loc[0, "target_time"] == only_rte_target
    assert meta.loc[0, "actual"] == 12.0
    assert meta.loc[0, "regime"] == 1
    # The synthetic state remains allowed as a feature, never as the target.
    assert x.loc[0, f"origin_{CARBON}"] == 999.0
