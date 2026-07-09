"""Oracle windows and green-window *selection* metrics.

Point error (MAE/RMSE) is necessary but not sufficient: a forecast is only
useful here if it **ranks** future hours by greenness correctly. These metrics
frame a concrete decision - *"run a job at one of the forecasted candidate hours
in the horizon; pick the greenest"* - and score each strategy by the carbon it
would actually incur, against two references:

* **run-now**: run immediately at the origin (no shifting);
* **oracle**: pick the truly greenest candidate using known actuals (an upper
  bound, not deployable).

Reported per strategy:

* ``mean_realized_gco2`` - average actual intensity you'd run at;
* ``mean_regret`` - realized minus oracle (0 = perfect selection);
* ``pct_oracle_potential`` - share of the run-now → oracle savings captured;
* ``spearman`` - rank correlation of predicted vs actual candidate intensity;
* ``top1_accuracy`` - how often the picked hour is the truly greenest one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from green_observatory.models import GreenWindow, ModelName, WindowType
from green_observatory.providers.carbon_base import CARBON
from green_observatory.windows.scoring import compute_low_carbon_windows


def oracle_windows(actual_carbon: pd.Series, **kwargs) -> list[GreenWindow]:
    """Best-possible low-carbon windows using known actuals (upper bound)."""
    kwargs.setdefault("window_type", WindowType.oracle_window)
    kwargs.setdefault("source_model", ModelName.oracle)
    return compute_low_carbon_windows(actual_carbon, **kwargs)


def _select_metrics_for_model(g: pd.DataFrame, now_by_origin: pd.Series) -> dict:
    regrets: list[float] = []
    realized: list[float] = []
    oracle_real: list[float] = []
    now_costs: list[float] = []
    spears: list[float] = []
    top1: list[float] = []

    for origin, gg in g.groupby("origin"):
        gg = gg.sort_values("horizon")
        pred = gg["prediction"].to_numpy()
        act = gg["actual"].to_numpy()
        if len(gg) < 2 or not np.isfinite(act).all():
            continue
        i_model = int(np.argmin(pred))
        r_model = float(act[i_model])
        r_oracle = float(np.min(act))
        realized.append(r_model)
        oracle_real.append(r_oracle)
        regrets.append(r_model - r_oracle)
        top1.append(1.0 if r_model <= r_oracle + 1e-9 else 0.0)
        if np.std(pred) > 0 and np.std(act) > 0:
            spears.append(float(spearmanr(pred, act).statistic))
        now = now_by_origin.get(origin, np.nan)
        if np.isfinite(now):
            now_costs.append(float(now))

    if not realized:
        return {}
    mean_now = float(np.mean(now_costs)) if now_costs else np.nan
    mean_real = float(np.mean(realized))
    mean_oracle = float(np.mean(oracle_real))
    denom = mean_now - mean_oracle
    pct = 100.0 * (mean_now - mean_real) / denom if np.isfinite(mean_now) and denom > 1e-9 else np.nan
    return {
        "mean_realized_gco2": round(mean_real, 2),
        "mean_regret": round(float(np.mean(regrets)), 2),
        "pct_oracle_potential": round(pct, 1) if np.isfinite(pct) else np.nan,
        "spearman": round(float(np.mean(spears)), 3) if spears else np.nan,
        "top1_accuracy": round(float(np.mean(top1)), 3),
        "n": len(realized),
    }


def window_selection_metrics(pred_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Green-window selection metrics per strategy, plus run-now / oracle rows.

    ``pred_df`` is the tidy backtest frame from
    :func:`green_observatory.carbon.evaluation.backtest_predictions`; ``df`` is
    the canonical carbon frame (used for the run-now cost at each origin).
    """
    now_by_origin = df[CARBON]
    rows: list[dict] = []
    for model, g in pred_df.groupby("model"):
        metrics = _select_metrics_for_model(g, now_by_origin)
        if metrics:
            rows.append({"strategy": model, **metrics})

    # Reference rows derived from actuals (model-independent).
    any_model = pred_df["model"].iloc[0]
    g0 = pred_df[pred_df["model"] == any_model]
    ref_real, ref_now = [], []
    for origin, gg in g0.groupby("origin"):
        act = gg["actual"].to_numpy()
        if len(gg) < 2 or not np.isfinite(act).all():
            continue
        ref_real.append(float(np.min(act)))
        now = now_by_origin.get(origin, np.nan)
        if np.isfinite(now):
            ref_now.append(float(now))
    if ref_real:
        rows.append({
            "strategy": "oracle", "mean_realized_gco2": round(float(np.mean(ref_real)), 2),
            "mean_regret": 0.0, "pct_oracle_potential": 100.0, "spearman": 1.0,
            "top1_accuracy": 1.0, "n": len(ref_real),
        })
    if ref_now:
        mean_now = float(np.mean(ref_now))
        mean_oracle = float(np.mean(ref_real))
        rows.append({
            "strategy": "run_now", "mean_realized_gco2": round(mean_now, 2),
            "mean_regret": round(mean_now - mean_oracle, 2), "pct_oracle_potential": 0.0,
            "spearman": np.nan, "top1_accuracy": np.nan, "n": len(ref_now),
        })

    out = pd.DataFrame(rows).set_index("strategy")
    preferred = ["run_now", "persistence", "climatology", "corrected", "sarimax", "project", "oracle"]
    order = [s for s in preferred if s in out.index]
    order += [s for s in out.index if s not in order]  # keep any extra strategies
    return out.reindex(order)
