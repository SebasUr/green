"""Live forward comparison: our forecast vs Electricity Maps (next ~24h).

Electricity Maps exposes only its *current* forecast (no historical forecasts)
and uses a *consumption-based* basis, so it cannot join the historical backtest.
What it can do, live, is:

1. **Basis gap** - EM's consumption-based intensity vs RTE ``taux_co2``
   (production-based) on the recent shared hours: how different are the two
   "ground truths" in level and in shape (correlation)?
2. **Forecast agreement** - over the next ~24h, which hours does each method
   rank greenest (Spearman of the two forecasts + each side's greenest pick)?

Neither side is scored against actuals here: the forecast window is in the
future. To score, persist both forecasts and re-evaluate once the hours pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from green_observatory.carbon.corrected_climatology import CorrectedClimatologyForecaster
from green_observatory.providers.carbon_base import CARBON


def live_comparison(model, odre_provider, em_provider, *, horizon_hours: int = 24) -> dict:
    """Compare our dense forecast against Electricity Maps for the next hours.

    ``model`` supplies the embedded climatology for a dense (corrected) forecast;
    ``odre_provider`` fetches fresh production-based data; ``em_provider`` must be
    ``available()``.
    """
    rt = odre_provider.import_realtime(hourly=True)
    origin = rt.index.max()
    corrected = CorrectedClimatologyForecaster(model.feature_builder.climatology)
    ours = corrected.predict(rt, origin, list(range(1, horizon_hours + 8)))
    our_fc = pd.Series(ours["prediction"].to_numpy(), index=ours.index, name="ours")

    em_fc = em_provider.forecast_series()
    em_hist = em_provider.history_series()

    result: dict = {"origin": origin, "our_forecast": our_fc, "em_forecast": em_fc}

    shared = em_hist.index.intersection(rt.index)
    if len(shared) >= 3:
        rte = rt[CARBON].reindex(shared)
        emh = em_hist.reindex(shared)
        result["basis"] = {
            "n": int(len(shared)),
            "rte_production_mean": round(float(rte.mean()), 1),
            "em_consumption_mean": round(float(emh.mean()), 1),
            "diff_em_minus_rte": round(float((emh - rte).mean()), 1),
            "correlation": round(float(rte.corr(emh)), 3),
        }

    common = our_fc.index.intersection(em_fc.index)
    if len(common) >= 3:
        o = our_fc.reindex(common).astype(float)
        e = em_fc.reindex(common).astype(float)
        result["agreement"] = {
            "n": int(len(common)),
            "spearman": round(float(spearmanr(o.to_numpy(), e.to_numpy()).statistic), 3),
            "our_greenest_hour": common[int(o.to_numpy().argmin())],
            "our_greenest_gco2": round(float(o.min()), 1),
            "em_greenest_hour": common[int(e.to_numpy().argmin())],
            "em_greenest_gco2": round(float(e.min()), 1),
        }
    return result
