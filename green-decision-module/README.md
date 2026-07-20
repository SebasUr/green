# Green Window Observatory

Carbon-intensity forecasting and low-carbon window selection for carbon-aware
workload scheduling on the French electricity grid.

This repository contains the carbon forecasting track of the CERN Green Decision
Module. It answers a bounded operational question:

> Given the next 1-48 hours, when is the lowest-carbon time to run a deferrable
> workload?

The current version produces hourly carbon-intensity forecasts for the French
bidding zone and ranks future low-carbon windows. It does not yet perform
workload-level actions such as suspending jobs, applying power caps, or mutating
Kubernetes resources.

## Summary

- **Ground truth:** RTE/ODRE eCO2mix `taux_co2`, production-based
  gCO2/kWh, aggregated to hourly UTC.
- **Model:** direct multi-horizon gradient-boosted tree regressors for
  1, 3, 6, 12, 24, and 48 hour horizons.
- **Forecast features:** Open-Meteo wind speed and solar irradiance, plus RTE
  day-ahead consumption forecast where available.
- **Evaluation:** leakage-free rolling-origin backtest over Feb-Apr 2026.
- **Primary decision metric:** share of savings captured relative to
  run-now and perfect foresight.

In the reference backtest, the ML model captures **67.0%** of the
perfect-foresight saving potential over a 48 hour decision horizon and **73%**
over the primary 24 hour horizon.

## Method

The model is trained on approximately five years of historical French grid data.
For each forecast origin `t0`, all features are restricted to information known
at or before `t0`, except target-time weather and consumption forecasts that are
published before the target time and are therefore available to an operational
system. The realized carbon value at `t0 + h` is used only as the supervised
label.

Low-carbon windows are derived from the forecast by ranking candidate hours
within the forecast horizon. A green score in `[0, 1]` is computed by inverting
the empirical CDF of forecasted carbon intensity within the horizon; higher
scores correspond to lower carbon intensity.

## Results

Rolling-origin backtest, training on data before `2026-02-01` and evaluating
348 forecast origins from Feb-Apr 2026:

| Strategy | Realized gCO2/kWh | Regret | % of perfect foresight | Spearman | Top-1 |
|---|---:|---:|---:|---:|---:|
| Run-now | 14.91 | 3.16 | 0.0 | -- | -- |
| Persistence | 15.04 | 3.29 | -4.1 | -- | 0.24 |
| Climatology | 13.85 | 2.09 | 33.7 | 0.05 | 0.20 |
| Corrected climatology | 13.36 | 1.61 | 49.1 | 0.05 | 0.26 |
| SARIMAX | 13.36 | 1.60 | 49.2 | 0.16 | 0.25 |
| LSTM | 12.90 | 1.14 | 63.9 | 0.28 | 0.37 |
| **ML model** | **12.80** | **1.04** | **67.0** | **0.34** | **0.41** |
| Perfect foresight | 11.76 | 0.00 | 100.0 | 1.00 | 1.00 |

Relative point error for the ML model is 5.5% WAPE at 1 hour, 18.5% at 6 hours,
24.0% at 24 hours, and 30.3% at 48 hours. The operational metric is the ranking
quality of candidate windows, because scheduling decisions depend on selecting
the lowest-carbon future hour rather than only minimizing point forecast error.

The wind and solar forecast features are the main driver of the ML model, raising
Spearman from 0.23 to 0.34 and reducing 48-hour WAPE from 37.7% to 30.3%. SARIMAX
and LSTM are optional comparison experiments; neither outperforms the
gradient-boosting model, which indicates the model is close to the ceiling set by
the available forecast information rather than by the model architecture. See
[REPORT.md](REPORT.md) for the full methodology and [CLAUDE.md](CLAUDE.md) for a
developer-oriented overview.

An experimental **Phase-C physics-guided model** is also available. It forecasts
the future gas, coal, fuel-oil, and bioenergy shares, maps them to carbon with
learned non-negative emission factors, and fits its residual stage on a separate
temporal calibration block. On the same Feb-Apr 2026 holdout it reaches 62.6% of
oracle potential (18.4% global WAPE), versus 66.7% (18.3% WAPE) for the direct
model in that run. The result is intentionally retained as a negative ablation:
the physical decomposition is accurate once the future mix is known, but the
remaining bottleneck is forecasting and ranking that future mix.

The opt-in **Phase-D exogenous model** adds historical day-ahead French wind,
solar, and load forecasts from Energy-Charts. These forecasts are gated to
horizons up to 24 hours; the 48-hour model cannot see a day-ahead vintage that
would not yet have been published. On the same holdout, exogenous data raises
captured oracle potential from 66.7% to **69.7%**, Top-1 accuracy from 40.2% to
45.4%, and reduces global WAPE from 18.3% to 17.6%. A separate pairwise selector
is trained on out-of-sample predictions with realized carbon-gap sample weights;
its influence is chosen on an internal temporal validation block. It reaches
68.6%, so the direct exogenous model remains the current winner and the ranker
is retained as a modular ablation rather than replacing it.

An **EnsembleCI-inspired** branch reproduces the paper's central structure:
24-hour CI/source history, LightGBM + CatBoost + neural-network sublearners,
raw-feature/base-prediction stacking, and greedy static ensemble weights. It is
adapted to this project's six causal horizons rather than the paper's dense
96-hour recursion. With the real LightGBM/CatBoost backends it reaches 65.1% of
oracle potential and 17.9% global WAPE. It improves 48-hour WAPE to 29.4%, but
does not improve candidate-hour selection, so it remains an opt-in negative
ablation and does not replace the 69.7% exogenous model.

The isolated **France-24 branch** predicts every hour from `t0+1` through
`t0+24` without changing any sparse-horizon experiment. Its probabilistic
fossil-regime expert classifies future dispatchable gas as baseload, CCG or peak,
fits conditional emitting shares and a physical carbon map, then calibrates an
optional uncertainty score and regret-weighted pairwise ranker on temporal OOS
blocks. Against an untouched dense project baseline on 352 Feb-Apr 2026 origins,
the RTE-enhanced fossil point model reduces aggregate MAPE from 18.26% to
**11.40%** and WAPE from 17.64% to **13.20%**. Its RTE inputs are D-1 onshore
wind, offshore wind and solar revisions selected as-of each origin. The
separately reported no-RTE ranked experiment captures
**72.9%** of perfect-foresight potential versus 67.6% for the dense baseline,
with regret falling from 1.10 to **0.92 gCO2/kWh**. Use
the RTE `fossil_regime_point` for value accuracy. RTE unit availability remains
an opt-in ablation because it worsened MAPE despite reaching 74.0% oracle capture
in one test configuration. See [REPORT_FRANCE24.md](REPORT_FRANCE24.md).

## Installation

```bash
conda create -y --override-channels -c conda-forge -n green-observatory \
  python=3.12 numpy pandas scikit-learn scipy pyarrow \
  pydantic pyyaml typer httpx holidays joblib pytest tzdata python-dateutil
conda activate green-observatory
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[viz,stats]"

# Exact tree sublearners for the optional EnsembleCI adaptation.
pip install -e ".[ensemble]"
```

## Reproduction

Run commands from the `green-decision-module` directory so that the relative
data, model, and figure paths resolve consistently.

```bash
# Import historical carbon and generation-mix data.
greenctl carbon import

# Import forecast features used by the ML model.
greenctl carbon fetch-forecast

# Import historical day-ahead wind/solar/load forecasts used only by Phase D.
greenctl carbon fetch-mix-forecast \
  --start 2021-07-11 --end 2026-05-01

# With RTE_CLIENT_ID and RTE_CLIENT_SECRET in the ignored local .env, download
# publication-versioned unit availability and D-1 generation forecasts.
greenctl carbon fetch-rte-system \
  --start 2021-07-11 --end 2026-05-01

# Train a deployable model on all available data.
greenctl carbon train

# Train the optional physics-guided Phase-C model.
greenctl carbon train-physical \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet

# Train the optional exogenous point model + regret-weighted selector.
greenctl carbon train-ranking \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet

# Train the optional two-layer EnsembleCI adaptation.
greenctl carbon train-ensemble-ci \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet

# Train the deployable dense French fossil-regime expert and pairwise ranker.
greenctl carbon train-fossil-regime \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet \
  --test-start 2026-02-01

# Reproduce the leakage-free rolling-origin backtest.
greenctl carbon compare --test-start 2026-02-01 --output runs/compare_metrics.json

# Compare Phase C, its raw physical stage, and a fixed direct/physical ensemble.
greenctl carbon compare \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet \
  --test-start 2026-02-01 \
  --strategies persistence,climatology,corrected,project,physical_raw,physical,physical_blend \
  --output runs/compare_physical_metrics.json

# Isolate the value of exogenous data from the value of the ranking objective.
greenctl carbon compare \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet \
  --test-start 2026-02-01 \
  --strategies project,exogenous_project,ranking \
  --output runs/compare_ranking_metrics.json

# Evaluate the EnsembleCI adaptation without changing the winning model.
greenctl carbon compare \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet \
  --test-start 2026-02-01 \
  --strategies ensemble_ci \
  --output runs/compare_ensemble_ci_metrics.json

# Evaluate the isolated dense French day-ahead specialist and its oracle choice.
greenctl carbon compare-france24 \
  --carbon data/cache/carbon_fr_hourly_enriched.parquet \
  --test-start 2026-02-01 \
  --output runs/compare_france24_rte_forecast_only_metrics.json

# Generate figures for the report.
greenctl figures --examples 4
```

The `carbon train` and `carbon compare` commands use forecast-feature snapshots
by default when they are present. Pass `--no-forecast-features` to disable them.

## Command Reference

```bash
greenctl version
greenctl carbon import
greenctl carbon fetch-forecast
greenctl carbon fetch-mix-forecast
greenctl carbon fetch-rte-system
greenctl carbon train
greenctl carbon train-physical
greenctl carbon train-ranking
greenctl carbon train-ensemble-ci
greenctl carbon train-fossil-regime
greenctl carbon forecast --horizon-hours 48
greenctl carbon compare --test-start 2026-02-01
greenctl carbon compare-france24 --test-start 2026-02-01
greenctl carbon compare-live
greenctl windows analyze --horizon-hours 48
greenctl figures --examples 4
greenctl jobs report JOB_NAME --namespace NAMESPACE
greenctl jobs observe --namespace NAMESPACE
greenctl jobs summarize
```

`compare-live` optionally compares the current forecast ranking with Electricity
Maps when `ELECTRICITYMAPS_API_TOKEN` is configured. This comparison is live-only:
historical commercial forecasts are not available for replay, and Electricity
Maps reports consumption-based intensity while this model is trained on
production-based RTE intensity.

## Repository Layout

```text
src/green_observatory/
  cli.py                         Typer command-line interface
  config.py                      YAML configuration loading
  models.py                      Pydantic data contracts
  providers/                     ODRE, Open-Meteo, Electricity Maps adapters
  carbon/                        feature engineering, models, evaluation
  windows/                       window scoring and perfect-foresight metrics
  observability/                 Kepler/RTE accounting for Kubernetes Jobs
  exporters/                     report figures
configs/                         model and window-scoring configuration
tests/                           unit tests
runs/                            generated metrics and figures
report/                          LaTeX technical note
```

## Scope

The current release combines the carbon-signal module with opt-in, read-only
Kubernetes Job observability. Facility telemetry, workload simulation, API
serving, and Kubernetes scheduling or mutation remain downstream components and
are not required to reproduce the carbon forecasting results.

## Per-Job energy and carbon observability

The repository now includes an opt-in Kubernetes Job observer. A Job labelled
`sustainability.cern.ch/track=true` receives a JSON report after termination
with reset-aware Kepler CPU energy, energy-weighted realized RTE intensity,
operational emissions without PUE, and explicit measurement-quality fields.

```bash
export KUBECONFIG=~/config
greenctl jobs observe --namespace green-experiment --output runs/job-reports
```

See [docs/JOB_OBSERVABILITY.md](docs/JOB_OBSERVABILITY.md) for labels, one-shot
reporting, Prometheus configuration, the JSON schema, and accounting limits.
