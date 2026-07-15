# CLAUDE.md: Green Window Observatory (carbon forecasting track)

Project context for contributors and AI assistants. This document covers the
**carbon forecasting** track only: forecasting the carbon intensity of the
French grid and ranking low-carbon time windows for carbon-aware scheduling of
deferrable workloads.

> The repository also contains a separate, opt-in Kubernetes Job observability
> track (`observability/`, `greenctl jobs …`, `docs/JOB_OBSERVABILITY.md`). It is
> out of scope for this document.

## What this is

Package `green_observatory` (src layout). It answers one bounded question:

> Over the next 1-48 hours, when is the lowest-carbon time to run a deferrable
> workload (batch jobs, Monte-Carlo, offline training/inference)?

Outputs: an hourly carbon-intensity forecast for the French bidding zone and a
ranked set of low-carbon **windows**. It does **not** make workload-level
decisions (suspend, power-cap, co-locate, mutate Kubernetes): those are a later
layer.

## Locked conventions (do not silently change)

- **Ground truth** = RTE `taux_co2` (production-based gCO2/kWh) from the open
  ODRE/eCO2mix portal, aggregated to hourly UTC.
- **`green_score` ∈ [0,1], higher = greener** (lower carbon). Raw gCO2/kWh is
  always kept alongside it.
- **No look-ahead leakage.** At decision time `t0`, features use only data ≤ `t0`.
  Target-time forecast features (weather/consumption) are allowed because they
  are published before the target time. `y(t0+h)` is used only as a label.
- **Calendar/climatology grouping in `Europe/Paris`** local time; instants stay UTC.
- **Evaluation = rolling-origin backtest, no leakage** (train `< test_start`).

## Architecture

```
providers/            data adapters
  carbon_base.py        canonical carbon-frame schema + Provider protocol
  carbon_odre.py        RTE/eCO2mix (taux_co2 + mix + prevision_j1)   [primary]
  weather_openmeteo.py  Open-Meteo wind@100m + solar irradiance (free, no key)
  carbon_electricity_maps.py  Electricity Maps (optional, needs API key)
carbon/               forecasting
  features.py           leakage-safe feature construction (FeatureBuilder)
  climatology.py        persistence + historical climatology + Forecaster protocol
  corrected_climatology.py  climatology + recent-residual EWMA correction
  model.py              project model: direct multi-horizon gradient boosting  [primary]
  sarimax.py            SARIMAX comparison (optional, needs statsmodels)
  lstm.py               LSTM comparison (optional, needs torch)
  evaluation.py         rolling-origin backtest + point metrics (MAE/RMSE/WAPE)
windows/
  scoring.py            green_score + low-carbon window detection
  oracle.py             perfect-foresight windows + decision metrics
exporters/plots.py    report figures (matplotlib)
models.py             pydantic data contracts
config.py             YAML config loading
cli.py                greenctl
configs/              carbon_model.yaml, window_scoring.yaml, simulation.yaml
```

`facility/` and `simulation/` are scaffolded placeholders (later milestones).

## The forecasting ladder

Every forecaster implements the same `Forecaster` interface, so the backtest
treats them uniformly. Ordered from weakest to strongest:

1. **Persistence**: future = last value. Minimum baseline.
2. **Climatology**: historical median by (month, day-of-week, hour) in local time.
3. **Corrected climatology**: climatology + decayed EWMA of recent residuals.
4. **SARIMAX** *(optional)*: ARIMA + Fourier seasonality. On par with corrected.
5. **Project model (primary)**: per-horizon `HistGradientBoosting`, direct
   multi-horizon over `{1,3,6,12,24,48}` h, ~37 leakage-safe features
   (target calendar, recent signal, system state, and target-time weather /
   consumption forecasts).
6. **LSTM** *(optional)*: recent-sequence encoder + exogenous-forecast head.
7. **Perfect foresight**: selects on realized values; upper bound, not deployable
   (internal id `oracle`).

The project model (gradient boosting) is primary. SARIMAX and LSTM are
comparison experiments only; **neither beats gradient boosting** on this data.

## Metrics

Two families (see `carbon/evaluation.py`, `windows/oracle.py`):

- **Point accuracy**: `MAE`, `RMSE`, `bias`, and `WAPE = MAE/mean(actual)`.
  Prefer **WAPE over MAPE**: MAPE divides by each hour's own value, so on a grid
  that is usually at low intensity (5-15 gCO2/kWh) it is dominated by already-clean
  hours and diverges as intensity → 0.
- **Decision quality** (the operational metric): each strategy picks the hour it
  predicts greenest; realized intensity is compared to run-now and perfect
  foresight. Reported: `mean_realized_gco2`, `regret`, `% of perfect foresight`
  = `(C_now − C_model)/(C_now − C_PF)·100`, `spearman`, `top1_accuracy`.

Ranking quality matters more than point error: persistence has the best point MAE
at long horizons yet cannot rank windows (it forecasts a flat line).

## Low-carbon windows (`windows/scoring.py`)

`green_score(x) = 1 − F(x)` (invert the empirical CDF within a reference). A
window is a contiguous block below a threshold, gap-merged and duration-filtered,
ranked by mean green score. `compute_low_carbon_windows` supports, all in one
function (defaults reduce to the original single horizon-percentile behaviour):

- **reference**: the horizon itself (relative) or a historical distribution
  (anchored to the grid's typical level: a uniformly green/dirty horizon is then
  handled correctly).
- **hysteresis**: `enter_percentile` / `exit_percentile` (open strict, stay loose).
- **fixed absolute thresholds**: `enter_gco2` / `exit_gco2`.
- **guard-rails**: `absolute_green_gco2` (always green below), `absolute_dirty_gco2`
  (never green above).

## Headline result

Rolling-origin backtest, train `< 2026-02-01`, 348 origins over Feb-Apr 2026:

- Project model captures **67.0%** of perfect-foresight potential at the 48 h
  horizon and **73%** at the primary 24 h horizon.
- Wind/solar forecast features are the main driver (Spearman 0.23→0.34, 48 h WAPE
  37.7→30.3%). See `README.md` and `REPORT.md` for full tables.
- Honest caveats: the backtest uses ERA5 near-actual weather as a proxy for a real
  forecast (mildly optimistic); the ceiling is set by wind-forecast quality, not
  the model type (LSTM and denser wind sampling did not help).

## Environment and commands

Conda env `green-observatory` (Python 3.12). Core deps: numpy, pandas,
scikit-learn, scipy, pyarrow, pydantic, pyyaml, typer, httpx, holidays, joblib.
Optional: `statsmodels` (SARIMAX), `torch` (LSTM), matplotlib (figures).

```bash
conda activate green-observatory
pip install -e ".[viz,stats,dl]"

greenctl carbon import            # ODRE carbon + generation mix (~5y)
greenctl carbon fetch-forecast    # Open-Meteo wind/solar + eco2mix prevision_j1
greenctl carbon train             # deployable model (all data + forecast features)
greenctl carbon compare --test-start 2026-02-01   # backtest: MAE/WAPE + decision table
greenctl carbon forecast --horizon-hours 24       # predicted low-carbon windows
greenctl carbon compare-live      # vs Electricity Maps (needs ELECTRICITYMAPS_API_TOKEN)
greenctl figures --examples 4 --month 2026-03-01  # report + rolling figures
pytest -q                         # test suite
```

`carbon train` / `carbon compare` use forecast-feature snapshots by default when
present; pass `--no-forecast-features` to disable.

## Gotchas

- **`taux_co2`** is published at 30-min cadence (`:00`/`:30`); `:15`/`:45` and the
  unconsolidated tail arrive null and are dropped. `date_heure` is genuine UTC.
  `ech_physiques` negative = export from France.
- **`torch`/conda OpenMP clash**: set `KMP_DUPLICATE_LIB_OK=TRUE` before running
  anything that imports torch (the LSTM module sets it on import).
- Data snapshots and models live in `data/cache/` and `models/` (git-ignored).
  The default weather is the 6-point equal-weighted mean; a 14-point
  capacity-weighted variant exists as an experiment (`*_v2`, on par in results).

## Contributor notes

- Keep new forecasters leakage-safe and behind the `Forecaster` interface.
- Core numeric functions take explicit parameters; YAML config only maps into them.
- Heavy model dependencies (`statsmodels`, `torch`) stay optional; their tests use
  `pytest.importorskip`.
- Prefer editing config over hard-coding; keep runs reproducible (fixed seeds,
  cached snapshots).
