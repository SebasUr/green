"""Rolling-origin backtest and point-forecast metrics.

The backtest advances a forecast origin across a held-out test period and, for
each origin, forecasts every horizon. Predictions are produced with **vectorized
batch predictors** (one call per horizon over all origins) that are exactly
equivalent to the per-origin ``Forecaster`` classes but far faster - and, like
them, strictly causal (no feature or correction uses data after the origin).

Model objects (climatology, project model) must have been fit on data preceding
``test_start`` so the whole backtest is leakage-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from green_observatory.carbon.climatology import ClimatologyModel
from green_observatory.carbon.model import ProjectCarbonModel
from green_observatory.providers.carbon_base import CARBON

DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48)


def make_origins(
    df: pd.DataFrame,
    test_start,
    test_end=None,
    *,
    stride_hours: int = 6,
    max_horizon: int = 48,
) -> pd.DatetimeIndex:
    """Forecast origins in ``[test_start, test_end - max_horizon]`` on a grid."""
    test_start = pd.Timestamp(test_start).tz_convert("UTC") if pd.Timestamp(test_start).tzinfo else pd.Timestamp(test_start, tz="UTC")
    end = pd.Timestamp(test_end).tz_convert("UTC") if test_end is not None else df.index.max()
    last_origin = end - pd.Timedelta(hours=max_horizon)
    grid = pd.date_range(test_start, last_origin, freq=f"{stride_hours}h")
    return grid.intersection(df.index)


# --------------------------------------------------------------------------- #
# Vectorized batch predictors (one call per horizon)
# --------------------------------------------------------------------------- #
def _persistence_batch(df, origins, horizons):
    y_origin = df[CARBON].reindex(origins).to_numpy()
    frames = []
    for h in horizons:
        frames.append(
            pd.DataFrame(
                {
                    "origin": origins,
                    "horizon": h,
                    "target_time": origins + pd.Timedelta(hours=h),
                    "prediction": y_origin,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _climatology_batch(clim: ClimatologyModel, origins, horizons):
    frames = []
    for h in horizons:
        tt = origins + pd.Timedelta(hours=h)
        frames.append(
            pd.DataFrame(
                {"origin": origins, "horizon": h, "target_time": tt,
                 "prediction": clim.predict_carbon(tt).to_numpy()}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _corrected_batch(clim, df, origins, horizons, *, residual_halflife_hours=12.0,
                     correction_decay_halflife_hours=24.0):
    y = pd.to_numeric(df[CARBON], errors="coerce")
    resid = y - clim.predict_carbon(y.index).to_numpy()
    # Causal, time-aware EWMA: correction(t) uses only residuals up to t.
    corr = resid.ewm(halflife=pd.Timedelta(hours=residual_halflife_hours), times=y.index).mean()
    corr_o = corr.reindex(origins).to_numpy()
    frames = []
    for h in horizons:
        decay = 0.5 ** (h / correction_decay_halflife_hours)
        tt = origins + pd.Timedelta(hours=h)
        pred = np.clip(clim.predict_carbon(tt).to_numpy() + corr_o * decay, 0.0, None)
        frames.append(
            pd.DataFrame({"origin": origins, "horizon": h, "target_time": tt, "prediction": pred})
        )
    return pd.concat(frames, ignore_index=True)


def _project_batch(model: ProjectCarbonModel, df, origins, horizons):
    origin_feats = model.feature_builder.origin_features(df).reindex(origins)
    frames = []
    for h in horizons:
        if h not in model.estimators_:
            continue
        tgt = model.feature_builder.target_block(
            origins + pd.Timedelta(hours=h), h, index=origins
        )
        x = pd.concat([tgt, origin_feats], axis=1)
        x = x.reindex(columns=model.feature_names_[h])
        pred = np.clip(model.estimators_[h].predict(x), 0.0, None)
        frames.append(
            pd.DataFrame(
                {"origin": origins, "horizon": h,
                 "target_time": origins + pd.Timedelta(hours=h), "prediction": pred}
            )
        )
    return pd.concat(frames, ignore_index=True)


def backtest_predictions(
    df: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    climatology: ClimatologyModel | None = None,
    project_model: ProjectCarbonModel | None = None,
    corrected_cfg: dict | None = None,
    include: Sequence[str] = ("persistence", "climatology", "corrected", "project"),
) -> pd.DataFrame:
    """Return a tidy frame: ``[model, origin, horizon, target_time, prediction, actual]``."""
    horizons = [int(h) for h in horizons]
    parts: list[pd.DataFrame] = []
    cc = corrected_cfg or {}

    if "persistence" in include:
        parts.append(_persistence_batch(df, origins, horizons).assign(model="persistence"))
    if "climatology" in include and climatology is not None:
        parts.append(_climatology_batch(climatology, origins, horizons).assign(model="climatology"))
    if "corrected" in include and climatology is not None:
        parts.append(
            _corrected_batch(
                climatology, df, origins, horizons,
                residual_halflife_hours=cc.get("residual_halflife_hours", 12.0),
                correction_decay_halflife_hours=cc.get("correction_decay_halflife_hours", 24.0),
            ).assign(model="corrected")
        )
    if "project" in include and project_model is not None:
        parts.append(_project_batch(project_model, df, origins, horizons).assign(model="project"))

    pred = pd.concat(parts, ignore_index=True)
    actual = df[CARBON].reindex(pd.DatetimeIndex(pred["target_time"])).to_numpy()
    pred["actual"] = actual
    return pred.dropna(subset=["actual", "prediction"])


def forecaster_batch(
    forecaster, df: pd.DataFrame, origins: pd.DatetimeIndex, horizons, model_name: str
) -> pd.DataFrame:
    """Run any (non-vectorized) ``Forecaster`` over origins into the tidy frame.

    Used for forecasters without a vectorized batch predictor (e.g. SARIMAX).
    Each call receives as-of history (``timestamp <= origin``).
    """
    frames: list[pd.DataFrame] = []
    for origin in origins:
        hist = df.loc[df.index <= origin]
        p = forecaster.predict(hist, origin, horizons)
        frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "origin": origin,
                    "horizon": [int(h) for h in p["horizon_hours"]],
                    "target_time": p.index,
                    "prediction": p["prediction"].to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["actual"] = df[CARBON].reindex(pd.DatetimeIndex(out["target_time"])).to_numpy()
    return out.dropna(subset=["actual", "prediction"])


def point_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Point-forecast metrics per (model, horizon).

    Includes two percentage views (computed, not assumed):

    * ``wape`` = ``MAE / mean(actual) * 100`` - the robust one. "Typical error
      as a share of the typical level"; the denominator is the real mean of what
      is being predicted, so no tiny-denominator blow-ups.
    * ``mape`` = ``mean(|err| / |actual|) * 100`` - the classic per-point one,
      reported for reference but **unreliable on low-carbon hours** (dividing by
      values near a few gCO2/kWh inflates it).
    """
    err = pred_df["prediction"] - pred_df["actual"]
    ape = err.abs() / pred_df["actual"].abs().clip(lower=1e-9)
    tmp = pred_df.assign(err=err, abserr=err.abs(), sqerr=err**2, ape=ape)
    g = tmp.groupby(["model", "horizon"])
    out = g.agg(
        mae=("abserr", "mean"),
        rmse=("sqerr", lambda s: float(np.sqrt(s.mean()))),
        bias=("err", "mean"),
        mean_actual=("actual", "mean"),
        mape=("ape", "mean"),
        n=("err", "size"),
    ).reset_index()
    out["mape"] = 100.0 * out["mape"]
    out["wape"] = 100.0 * out["mae"] / out["mean_actual"]  # MAE as % of the true mean
    return out


def mae_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Wide MAE table: rows=model, cols=horizon (for quick display)."""
    m = point_metrics(pred_df)
    return m.pivot(index="model", columns="horizon", values="mae").round(2)
