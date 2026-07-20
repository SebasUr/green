"""Corrected climatology (baseline ladder rung 3).

Climatology captures the recurring seasonal/diurnal pattern but ignores whether
*today's* grid runs above or below its usual level (e.g. a windy week, an
outage). The corrected forecaster nudges the climatology by the recent residual:

    residual(t)  = actual(t) - climatology_center(t)      for t <= origin
    correction   = time-weighted EWMA of recent residuals (as-of origin)
    forecast(t0+h) = climatology_center(t0+h) + correction * decay(h)

The correction fades with horizon via ``decay(h) = 0.5 ** (h / decay_halflife)``
so it dominates at short horizons and hands back to pure climatology far out.
Everything is computed strictly as-of ``origin`` (no look-ahead).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import ClimatologyModel, _target_index
from green_observatory.models import ModelName
from green_observatory.providers.carbon_base import CARBON


class CorrectedClimatologyForecaster:
    """Climatology plus a decayed EWMA of recent residuals."""

    name = ModelName.corrected_climatology

    def __init__(
        self,
        model: ClimatologyModel,
        residual_halflife_hours: float = 12.0,
        correction_decay_halflife_hours: float = 24.0,
        max_lookback_hours: float = 168.0,
    ) -> None:
        self.model = model
        self.residual_halflife_hours = float(residual_halflife_hours)
        self.correction_decay_halflife_hours = float(correction_decay_halflife_hours)
        self.max_lookback_hours = float(max_lookback_hours)

    def _recent_correction(self, history: pd.DataFrame, origin: pd.Timestamp) -> float:
        lo = origin - pd.Timedelta(hours=self.max_lookback_hours)
        mask = (history.index < origin) & (history.index > lo)
        window = pd.to_numeric(history.loc[mask, CARBON], errors="coerce").dropna()
        if window.empty:
            return 0.0
        clim = self.model.predict_carbon(window.index).to_numpy()
        resid = pd.Series(window.to_numpy() - clim, index=window.index)
        # Time-aware EWMA: weight by real elapsed time, robust to small gaps.
        ewm = resid.ewm(
            halflife=pd.Timedelta(hours=self.residual_halflife_hours), times=window.index
        ).mean()
        return float(ewm.iloc[-1])

    def predict(
        self, history: pd.DataFrame, origin: pd.Timestamp, horizons_hours: Sequence[float]
    ) -> pd.DataFrame:
        correction = self._recent_correction(history, origin)
        targets = _target_index(origin, horizons_hours)
        base = self.model.predict(targets)

        decay = np.array(
            [0.5 ** (float(h) / self.correction_decay_halflife_hours) for h in horizons_hours]
        )
        adj = correction * decay
        center = np.asarray(base["center"], dtype=float) + adj
        center = np.clip(center, 0.0, None)  # carbon intensity is non-negative

        lower = np.clip(np.asarray(base.get("p10", np.nan), dtype=float) + adj, 0.0, None)
        upper = np.clip(np.asarray(base.get("p90", np.nan), dtype=float) + adj, 0.0, None)
        return pd.DataFrame(
            {
                "prediction": center,
                "lower": lower,
                "upper": upper,
                "horizon_hours": list(horizons_hours),
            },
            index=targets,
        )


def corrected_from_config(model: ClimatologyModel, carbon_cfg: dict) -> CorrectedClimatologyForecaster:
    """Build a corrected-climatology forecaster from a carbon-model config dict."""
    cc = carbon_cfg.get("corrected_climatology", {})
    return CorrectedClimatologyForecaster(
        model=model,
        residual_halflife_hours=cc.get("residual_halflife_hours", 12.0),
        correction_decay_halflife_hours=cc.get("correction_decay_halflife_hours", 24.0),
        max_lookback_hours=cc.get("max_lookback_hours", 168.0),
    )
