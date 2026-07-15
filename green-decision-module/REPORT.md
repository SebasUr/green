# Green Window Observatory: Methodology

Carbon-intensity forecasting and low-carbon window selection for carbon-aware
scheduling of deferrable workloads on the French electricity grid. This document
is the detailed methodology reference; `report/report.tex` is a condensed
three-page technical note with the same results.

All numbers come from a leakage-free rolling-origin backtest: training on data
before `2026-02-01` and evaluating 348 forecast origins over February to April
2026.

## 1. Scope

The system answers one question: over the next 1 to 48 hours, when is it greener
to run a deferrable workload? It produces an hourly carbon-intensity forecast for
the French bidding zone and a ranked set of low-carbon windows. Per-workload
actions (suspend, power cap, co-locate) are out of scope for this version.

## 2. Data

### 2.1 Ground truth: RTE / ODRE eCO2mix

The target is RTE `taux_co2`, a production-based carbon intensity (gCO2/kWh)
published on the open ODRE portal, aggregated to hourly UTC over roughly five
years. The generation mix (nuclear, gas, coal, wind, solar, hydro, exchanges) is
available as auxiliary features. `taux_co2` is published at 30-minute cadence;
the `:15` and `:45` slots and the not-yet-consolidated tail arrive null and are
dropped.

### 2.2 Forecast features (the main driver)

French carbon intensity is driven mainly by wind (nuclear is stable, solar is
daytime-deterministic). The model therefore uses forecast features evaluated at
the target time `T`, all obtainable in real time and leakage-free:

| Feature | Source | Real-time availability |
|---|---|---|
| Wind speed @100 m | Open-Meteo (free, no key) | forecast to 16 days, all horizons |
| Solar irradiance | Open-Meteo | same |
| Consumption forecast | RTE `prevision_j1` | day-ahead, gated to horizons ≤ 24 h |

A forecast for `T` is published before `T`, so using it at decision time
`t0 < T` is not leakage. The national weather signal averages several France
points; the default is a 6-point equal-weighted mean.

### 2.3 Comparative provider: Electricity Maps (optional)

Electricity Maps reports a consumption-based intensity (a different accounting
basis) and exposes only its current forecast. It is used for a live ranking
comparison when an API token is configured; it cannot join the historical
backtest.

## 3. Data pipeline

```
ODRE download -> field mapping -> UTC (tz-aware, enforced)
  -> drop rows without taux_co2 -> resample 30 min to hourly (mean)
  -> canonical frame (sorted UTC index, no duplicates, fixed columns)
  -> parquet snapshot (reproducible replay)
Open-Meteo (France points -> national mean, hourly, UTC)
prevision_j1 (indexed by target time, hourly)
```

The single invariant is no look-ahead leakage: at decision time `t0`, features
use only information available at or before `t0`, and the realized value
`y(t0+h)` is used only as the supervised label. A unit test verifies that
origin features are identical whether computed on the full series or only on the
past.

## 4. Forecasting models

A ladder of forecasters shares one interface, so the backtest treats them
uniformly.

| Model | Description |
|---|---|
| Persistence | future = last value (minimum baseline) |
| Climatology | historical median by (month, day-of-week, hour) in local time |
| Corrected climatology | climatology plus a decayed EWMA of recent residuals |
| SARIMAX (optional) | ARIMA with Fourier seasonality (statsmodels) |
| **Project model (primary)** | direct multi-horizon gradient boosting |
| LSTM (optional) | recent-sequence encoder plus exogenous-forecast head (torch) |
| Perfect foresight | selects on realized values; upper bound, not deployable |

### 4.1 Project model

A direct multi-horizon regressor: six independent `HistGradientBoosting`
estimators, one per horizon `h in {1,3,6,12,24,48}`. Each uses about 37 features
that are known at `t0` or deterministic: target-time calendar (hour, day-of-week,
month, weekend, holiday, cyclical encodings); recent signal (value, lags at
1/2/3/24/168 h, rolling means, slope, residual from climatology); system state at
`t0`; and the target-time forecast features above. For a given horizon, examples
`(X = features at t0, y = intensity at t0+h)` are fit on data before the test
period. At prediction time each estimator emits its horizon in one pass from a
single origin, so overlapping horizons stay consistent.

Gradient boosting is preferred over ARIMA or deep learning: the problem is
tabular with nonlinear calendar-by-weather interactions, handles missing values
natively, and stays fast and explainable. Recursive forecasting is avoided
because it would also require forecasting the mix.

## 5. Low-carbon windows

Each forecast is mapped to a green score in `[0,1]` (higher is greener) by
inverting the empirical CDF within a reference: `green_score(x) = 1 - F(x)`. A
window is a contiguous block below a threshold, gap-merged and duration-filtered,
ranked by mean green score. The detector supports several methods (defaults
reduce to a single horizon percentile):

- Reference: the horizon itself (relative), or a historical distribution
  (anchored to the grid's typical level, so a uniformly green or dirty horizon is
  handled correctly).
- Hysteresis: a strict enter percentile and a looser exit percentile, for stable
  windows across brief upticks.
- Fixed absolute thresholds in gCO2/kWh, and optional absolute guard-rails.

## 6. Evaluation and metrics

Protocol: a leakage-free rolling-origin backtest. Fit on data before
`2026-02-01` (44,563 hourly rows); advance an origin every 6 h over Feb to Apr
2026 (348 origins), predicting all horizons from information up to the origin
only.

Point accuracy: MAE, RMSE, bias, and a relative error using WAPE rather than
MAPE, `WAPE = sum|pred - actual| / sum|actual|`. MAPE weights each hour by the
inverse of its own magnitude, so on a grid that is usually at low intensity
(5 to 15 gCO2/kWh) it is dominated by already-clean hours and diverges as the
value goes to zero; WAPE normalizes total error by total intensity and is stable.

Decision quality (the operational metric): each strategy selects the hour it
predicts greenest within the horizon; the realized intensity there is compared to
run-now (`C_now`) and perfect foresight (`C_PF`). Reported metrics are mean
realized intensity, regret `C_model - C_PF`, Spearman rank correlation, top-1
accuracy, and the share of achievable savings captured,
`(C_now - C_model) / (C_now - C_PF) * 100`, which is 0% for run-now and 100% for
perfect foresight.

## 7. Results

### 7.1 Decision quality

| Strategy | Realized gCO2/kWh | Regret | % of perfect foresight | Spearman | Top-1 |
|---|---:|---:|---:|---:|---:|
| Run-now | 14.91 | 3.16 | 0.0 | -- | -- |
| Persistence | 15.04 | 3.29 | -4.1 | -- | 0.24 |
| Climatology | 13.85 | 2.09 | 33.7 | 0.05 | 0.20 |
| Corrected climatology | 13.36 | 1.61 | 49.1 | 0.05 | 0.26 |
| SARIMAX | 13.36 | 1.60 | 49.2 | 0.16 | 0.25 |
| LSTM | 12.90 | 1.14 | 63.9 | 0.28 | 0.37 |
| **Project model** | **12.80** | **1.04** | **67.0** | **0.34** | **0.41** |
| Perfect foresight | 11.76 | 0.00 | 100.0 | 1.00 | 1.00 |

The project model captures 67% of perfect-foresight potential at 48 h, and 73%
when restricted to the primary 24 h horizon. Following its recommendations runs
at about 13% lower carbon intensity than executing immediately.

### 7.2 Point error (WAPE, %)

| Model | 1 h | 3 h | 6 h | 12 h | 24 h | 48 h |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 5.3 | 12.1 | 26.9 | 22.3 | 21.9 | 26.1 |
| Corrected climatology | 34.1 | 35.2 | 40.6 | 49.6 | 68.6 | 94.8 |
| **Project model** | **5.5** | **10.7** | **18.5** | **20.3** | **24.0** | **30.3** |

Global WAPE for the project model is 18%.

### 7.3 Contribution of the forecast features

| Project model | % of perfect foresight | Spearman | WAPE @48 h |
|---|---:|---:|---:|
| Without forecast features | 61.8 | 0.23 | 37.7 |
| With wind / solar / consumption | 67.0 | 0.34 | 30.3 |

The wind and solar forecast features are the main driver of ranking quality,
especially at 12 to 48 h where the recent signal is weak.

## 8. Comparison experiments (negative results)

Two additional models and one data change were tested; none improved on the
gradient-boosting model.

- **SARIMAX** performs on par with corrected climatology (49% of perfect
  foresight) and loses to gradient boosting.
- **LSTM** did not beat gradient boosting (63.9% vs 67.0% of perfect foresight,
  and roughly twice the point error), consistent with the tabular nature of the
  problem and the modest data volume.
- **Denser, capacity-weighted wind sampling** (6 to 14 Open-Meteo points) was a
  wash (Spearman +0.01, share of perfect foresight -1.5), because the national
  wind average is already well captured by 6 points. The ceiling is set by the
  temporal error of the wind forecast at 24 to 48 h, not by spatial sampling.

These results indicate the current model is close to the ceiling that the
available information permits; the main lever for improvement is weather-forecast
quality (ensemble NWP), not the model architecture.

## 9. Comparison with external providers

A direct backtest against a commercial provider such as Electricity Maps is not
feasible: their historical forecasts are not exposed for replay, and they report
a consumption-based intensity. A live comparison shows about 0.7 Spearman
agreement between the two 24 h forecasts. The value of this model is that it is
built entirely from free, open data with no vendor dependency, and every
prediction is reproducible and auditable.

## 10. Limitations

- The Feb to Apr 2026 test period is a persistently low-carbon regime, so the
  absolute headroom is small (perfect foresight saves about 3.15 gCO2/kWh versus
  run-now). The percentage of perfect foresight measures skill; the absolute
  benefit is regime-dependent.
- The backtest uses ERA5 reanalysis as a near-actual proxy for the weather
  forecast, which is mildly optimistic relative to a real 24 to 48 h forecast.
- The day-ahead consumption forecast is not available two days out, so it is
  dropped beyond the 24 h horizon.

## 11. Reproduction

```bash
conda activate green-observatory

greenctl carbon import                              # carbon + generation mix (~5y)
greenctl carbon fetch-forecast                      # Open-Meteo wind/solar + prevision_j1
greenctl carbon train                               # deployable model (all data)
greenctl carbon compare --test-start 2026-02-01     # MAE/WAPE + decision metrics
greenctl carbon forecast --horizon-hours 24         # predicted low-carbon windows
greenctl figures --examples 4 --month 2026-03-01    # report and rolling figures
```

SARIMAX and LSTM are optional experiments (`pip install -e ".[stats,dl]"`) and
are compared by script rather than in the default CLI.
