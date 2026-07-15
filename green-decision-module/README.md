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
```

## Reproduction

Run commands from the `green-decision-module` directory so that the relative
data, model, and figure paths resolve consistently.

```bash
# Import historical carbon and generation-mix data.
greenctl carbon import

# Import forecast features used by the ML model.
greenctl carbon fetch-forecast

# Train a deployable model on all available data.
greenctl carbon train

# Reproduce the leakage-free rolling-origin backtest.
greenctl carbon compare --test-start 2026-02-01 --output runs/compare_metrics.json

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
greenctl carbon train
greenctl carbon forecast --horizon-hours 48
greenctl carbon compare --test-start 2026-02-01
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
