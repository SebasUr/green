# Green Observatory and Decision Module - Implementation Plan

## Architecture Decision

Split the project into two layers:

```text
V1.0 Green Window Observatory
  carbon temporal intelligence + carbon forecasting + facility context
  + green-window simulation + dashboards

V1.1 Green Decision Module
  workload-aware decisions that consume V1.0 green windows
```

The core V1.0 question is:

```text
When is it greener to run?
```

Not:

```text
Which datacenter should I choose?
```

The first version should build a temporal intelligence layer. It should identify, forecast, compare and simulate green windows using the French electricity carbon signal as the main source, then enrich those windows with CERN facility context.

The decision module comes later:

```text
Green Window Observatory:
  Which time windows are lower carbon?
  Which windows are likely to be lower carbon in the next 24-48h?
  How good is our forecast versus Electricity Maps and oracle?
  Which windows also look favorable from facility load/cooling/PUE context?
  How would simple "run now vs wait" policies behave in simulation?

Green Decision Module:
  Given a specific workload and the V1.0 windows, should it run now, wait,
  sleep, backfill, run with limits, or only be observed?
```

## V1.0 Goal - Green Window Observatory

Build an application that measures, models, forecasts and simulates green execution windows.

V1.0 answers:

```text
What is the current French grid CO2 signal?
Which historical hours/days tend to have lower CO2?
Which future windows are predicted to be lower CO2?
How well does the project model perform against baselines?
How does it compare with Electricity Maps when available?
How much of the oracle best-window potential is captured?
How do CDC facility signals change or qualify those windows?
How would synthetic workload arrivals behave under different window policies?
```

V1.0 does not answer:

```text
Should this specific Monte Carlo job run now?
Should this notebook be suspended?
Should this AI workload be power capped?
Which Kubernetes object should be mutated?
```

Those are V1.1/V1.2 questions.

## V1.0 Product Shape

Build V1.0 as a Python application with five faces:

1. CLI for importing data, training models, analyzing windows and generating reports.
2. FastAPI service exposing carbon signals, forecasts, windows, facility context and simulations.
3. Forecasting/modeling layer for the French carbon signal.
4. Facility context layer for CERN CDC exports.
5. Offline simulation layer for comparing when-to-run policies.

This gives a useful artifact before workload-level telemetry from Kepler or real Kubernetes job traces exist.

## V1.0 Conceptual Flow

```text
RTE/ODRE/eCO2mix data
  -> carbon history
  -> carbon forecast model
  -> low-carbon windows

CERN CDC exports
  -> facility context analysis
  -> low-facility-pressure windows

carbon windows + facility context
  -> combined green windows
  -> simulations
  -> dashboard/report/API
```

The carbon model is the primary model. The facility model is contextual: it makes green-window analysis more realistic, but it should not become the heart of V1.0.

## Two Model Tracks

### Carbon Model

Source:

```text
RTE/ODRE/eCO2mix historical and near-real-time French electricity/carbon data
```

Optional comparison:

```text
Electricity Maps forecast API
```

Outputs:

```text
carbon_score(t)
carbon_forecast(t)
low_carbon_windows
forecast_confidence
oracle_comparison
```

### Facility Context Model

Source:

```text
CERN CDC exports
optional live Grafana/Prometheus later
```

Outputs:

```text
facility_score(t)
POD/IT load patterns
temperature/humidity/cooling-pressure patterns
PUE/facility proxy patterns
low_facility_pressure_windows
facility_data_quality_report
```

The observatory should show these tracks separately, then combine them only for window ranking:

```text
combined_green_score(t) =
    carbon_weight * carbon_score(t)
  + facility_weight * facility_score(t)
```

For V1.0, the recommended starting weights are:

```text
carbon_weight = 0.75
facility_weight = 0.25
```

These weights are configuration, not truth. They should be varied in simulation.

## Available CDC Facility Exports

The project has one year of hourly CERN CDC exports in `green-decision-module/`.

```text
cdc_humidity_1y_hourly.csv          148,920 rows, 17 series
cdc_power_1y_hourly.csv           1,251,955 rows, 151 series
cdc_pue_components_1y_hourly.csv     26,280 rows, 3 series
cdc_temperature_1y_hourly.csv     1,459,102 rows, 220 series
cdc_wue_water_1y_hourly.csv          15,924 rows, 6 series
```

Observed integrity:

```text
main time range:
  2025-07-08T00:00:00Z to 2026-07-08T00:00:00Z

basic quality:
  no missing columns
  no non-numeric values
  no duplicated (time, plugin_instance) pairs
  hourly resolution

known caveats:
  temperature has dead/outlier sensors with zeros and negative values
  power has zero-only or near-zero series that must be filtered
  some power series have large gaps
  WUE/water starts only on 2026-03-19 and is cumulative counter data
  raw power series may mix feeders, rooms, PDUs and outlets, so naive summation can double-count
```

The strongest immediately useful file is:

```text
cdc_pue_components_1y_hourly.csv
```

It contains three `POD.CONS` series:

```text
PDCITR5.POD.CONS
PDCITR6.POD.CONS
PDCMMR.POD.CONS
```

The summed POD load is physically plausible:

```text
mean total PDC POD load: about 1.96 MW
max total PDC POD load: about 2.16 MW
```

Use these exports for:

```text
facility context analysis
load and thermal pattern analysis
facility-window discovery
facility-aware green-window ranking
synthetic workload replay over real facility traces
```

Do not use them for:

```text
per-pod energy attribution
per-namespace energy attribution
exact per-workload CO2 savings
direct Kubernetes admission decisions
```

Those require Kepler, Prometheus, Kubernetes metadata, benchmark traces or V1.1 workload intent data.

## Carbon Data And Forecasting

### Primary Provider

Use open French electricity/carbon data:

```text
ODRE/eCO2mix
RTE/ODRE-derived data where useful
```

The project should rely on these as the reproducible core. Electricity Maps should not be required for V1.0.

### Comparative Provider

Use Electricity Maps only as a benchmark when an API key is available:

```text
project model vs Electricity Maps
project model vs climatology
project model vs oracle
```

This makes the scientific contribution stronger:

```text
We build a reproducible carbon-window forecast from open French data and compare
it with a professional external forecast provider.
```

## Carbon Forecasting Models

Implement the forecasting ladder in this order.

### 1. Persistence Baseline

```text
future value = current/recent value
```

Purpose:

```text
minimum baseline
```

### 2. Historical Climatology

Use historical carbon intensity grouped by:

```text
month
day_of_week
hour_of_day
```

Compute:

```text
median
p25 / p75
p10 / p90
standard deviation
sample count
```

Prediction:

```text
for each future timestamp:
  base = median(month, day_of_week, hour_of_day)
  uncertainty = p75 - p25
  confidence = f(sample_count, uncertainty)
```

Purpose:

```text
Which windows are usually lower CO2?
How stable are those patterns?
Which windows are historically risky?
```

### 3. Corrected Climatology

Use recent observations to adjust the historical baseline:

```text
recent_residual = recent_actual - recent_climatology
correction = exponentially_weighted_mean(recent_residual)
forecast = future_climatology + correction_decay(correction)
```

Purpose:

```text
capture days where the grid is currently above or below its usual pattern
```

### 4. Project Carbon Model

Train a small reproducible model from open French electricity/carbon data.

Recommended first model:

```text
gradient boosted regression or random forest regression
```

Reason:

```text
works well on tabular time-series features
handles nonlinear hour/day/season effects
does not require huge data volume
is easier to explain than a deep learning model
```

Features:

```text
calendar:
  hour_of_day
  day_of_week
  month
  weekend
  holiday flag if available

recent signal:
  latest carbon value
  rolling mean 1h / 3h / 6h / 24h
  rolling slope
  recent residual from climatology

electricity system:
  consumption
  consumption forecast D-1 / intraday if available
  nuclear
  gas
  coal
  wind
  solar
  hydro
  imports/exports
```

Targets:

```text
carbon_intensity_gco2_kwh at horizon h
carbon_score at horizon h
probability that a future window is in the lowest X percentile
```

Horizons:

```text
1h, 3h, 6h, 12h, 24h, 48h
```

The most important metric is not only point forecast error. The key question is whether the model ranks green windows correctly.

### 5. Oracle

Oracle chooses the best possible future window using known historical future data.

Purpose:

```text
upper bound for window selection
not deployable
```

## Carbon Model Evaluation

Compare:

```text
Persistence
Historical climatology
Corrected climatology
Project carbon model
Electricity Maps forecast if available
Oracle
```

Metrics:

```text
MAE / RMSE for carbon intensity
top-k low-carbon window accuracy
green-window ranking regret
percentage of oracle potential captured
uncertainty calibration
```

Expected result format:

```text
The project model reduced green-window selection regret by X% versus climatology
and captured Y% of the oracle potential. Electricity Maps captured Z% when
available as an external comparison.
```

## Facility Context Model

The facility model is a V1.0 context layer for CERN CDC behavior. It should not be framed as the core of the project and should not be framed as "which datacenter to choose".

It answers:

```text
Which hours have lower load pressure?
Which hours have higher cooling/thermal pressure?
Are there recurring facility-friendly windows?
Does facility context change the ranking of low-carbon windows?
```

### Facility Cleaning Rules

```text
drop zero-only series
drop impossible temperature values, e.g. below -20 C or above 70 C unless explicitly validated
flag rather than silently delete outliers
interpolate short gaps only when needed for hourly models
do not sum raw power series unless they are known to be non-overlapping
prefer curated aggregate series such as POD.CONS for load modeling
convert water counters to hourly/daily deltas before analysis
```

### Facility Features

```text
calendar:
  hour_of_day
  day_of_week
  month
  weekend

load:
  curated POD load
  room-level POD load if available
  rolling mean load 1h / 6h / 24h
  load percentile within trailing 30d

thermal:
  representative internal temperature
  representative humidity
  rolling temperature/humidity trends

facility:
  PUE if available
  PUE proxy if only components are available
  cooling pressure proxy
  WUE delta if usable
```

### Facility Outputs

```text
facility_score(t)
low_facility_pressure_windows
high_facility_pressure_windows
load_forecast(t)
cooling_pressure_forecast(t)
data_quality_report
```

### Facility Prediction

Start simple:

```text
facility_climatology:
  median POD load / temperature / humidity by hour-of-day and day-of-week

corrected_facility_climatology:
  climatology + recent residual correction

project_facility_model:
  random forest or gradient boosting for load/facility_score forecasting
```

This facility model is an approximation for data analysis and simulation. It is not workload-level energy accounting.

## Green Windows

V1.0 should produce several window types:

```text
low_carbon_windows:
  based only on the French carbon signal

predicted_low_carbon_windows:
  based on the project carbon forecast

low_facility_pressure_windows:
  based only on CDC facility context

combined_green_windows:
  based on carbon score + facility score

oracle_windows:
  best possible windows using known historical future
```

A window result should look like:

```yaml
start: "2026-07-07T22:00:00Z"
end: "2026-07-08T02:00:00Z"
window_type: combined_green_window
carbon_score: 0.18
facility_score: 0.42
combined_score: 0.24
confidence: 0.71
reason:
  - Predicted carbon score is below daily median.
  - Facility pressure is moderate.
  - Historical variance for this hour is low.
```

## Simulation Layer

The CDC data are sufficient to simulate facility-aware when-to-run behavior at the observatory level.

### What Can Be Simulated Now

```text
synthetic workload arrivals over real historical carbon/facility traces
run-now vs wait-for-green-window policies
carbon-only window selection
facility-only window selection
combined carbon/facility window selection
oracle upper bound using known historical future
delay vs greener-window tradeoff
future V1.1 decision policies in dry-run mode
```

### What Cannot Be Simulated Honestly Yet

```text
real per-workload energy savings
real pod-level CO2 attribution
actual Kubernetes queue behavior
actual user disruption
actual power change caused by shifting one job
```

Those require Kepler/Prometheus/Kubernetes job traces or controlled benchmark runs.

### Simulation Inputs

```yaml
historical_carbon:
  provider: ODRE/eCO2mix
  resolution: hourly_or_15min

facility_trace:
  provider: CERN_CDC_exports
  files:
    - cdc_pue_components_1y_hourly.csv
    - cdc_temperature_1y_hourly.csv
    - cdc_humidity_1y_hourly.csv

synthetic_workload_trace:
  arrivals: poisson_or_replayed_schedule
  classes:
    - generic_deferrable
    - generic_non_deferrable
  duration_distribution: configurable
  energy_proxy: fixed_kw_by_class
  max_delay: configurable
```

### Simulation Policies

```text
baseline:
  run at arrival time

carbon_only:
  choose best carbon window before max delay/deadline

facility_only:
  choose best facility-score window before max delay/deadline

combined_green_window:
  choose best weighted carbon + facility window before max delay/deadline

oracle:
  choose best possible combined window using known future
```

### Simulation Outputs

```text
selected_start_time
selected_window_type
carbon_score_at_start
facility_score_at_start
combined_green_score_at_start
delay_minutes
deadline_violation
oracle_regret
percentage_of_oracle_potential_captured
estimated_green_score_improvement
estimated_CO2_proxy
```

This lets V1.0 test the signal layer before V1.1 makes workload-specific decisions.

## V1.0 Data Sources

### Required

1. ODRE/eCO2mix carbon provider
   - Historical and near-real-time French electricity/carbon data.
   - Main source for carbon modeling.

2. CDC facility CSV provider
   - Reads one-year hourly CDC exports.
   - Produces facility signals, data quality reports and facility windows.

3. Carbon model trainer
   - Trains project-owned carbon forecasting models.

4. Window analyzer
   - Computes carbon, facility, combined and oracle windows.

5. Simulation engine
   - Replays synthetic workload arrivals over historical carbon/facility traces.

### Optional

6. Electricity Maps provider
   - Comparative external provider only.
   - Requires API key.
   - Not a core dependency.

7. Live Grafana/Prometheus provider
   - Optional later live facility context.
   - Not needed because CDC exports are already available.

## V1.0 Project Structure

```text
green-decision-module/
  pyproject.toml
  README.md
  configs/
    carbon_model.yaml
    facility_model.yaml
    window_scoring.yaml
    simulation.yaml
  data/
    sample_carbon_timeseries.csv
    cdc_pue_components_1y_hourly.csv
    cdc_temperature_1y_hourly.csv
    cdc_humidity_1y_hourly.csv
    cdc_power_1y_hourly.csv
    cdc_wue_water_1y_hourly.csv
  src/green_observatory/
    __init__.py
    cli.py
    api.py
    models.py
    providers/
      carbon_base.py
      carbon_odre.py
      carbon_electricity_maps.py
      cdc_csv.py
    carbon/
      features.py
      climatology.py
      corrected_climatology.py
      model.py
      evaluation.py
    facility/
      cleaning.py
      features.py
      model.py
      windows.py
      quality.py
    windows/
      scoring.py
      oracle.py
      combine.py
    simulation/
      workload_trace.py
      replay.py
      policies.py
      metrics.py
    exporters/
      csv_export.py
      json_export.py
      grafana_dashboard.py
      markdown_report.py
  dashboards/
    green_observatory_v1.json
  examples/
    synthetic_workloads.yaml
    analyze_windows.yaml
  tests/
```

The repository can still be named `green-decision-module` because that is the long-term project. The V1.0 Python package should be named `green_observatory`.

## V1.0 CLI

```bash
greenctl carbon import --source odre --output data/carbon_fr.csv
greenctl carbon train --carbon data/carbon_fr.csv --config configs/carbon_model.yaml
greenctl carbon forecast --horizon-hours 48
greenctl carbon compare --against electricity-maps --if-available

greenctl facility inspect --data-dir data
greenctl facility analyze --data-dir data
greenctl facility windows --data-dir data

greenctl windows analyze --carbon data/carbon_fr.csv --facility data
greenctl windows combine --carbon data/carbon_fr.csv --facility data
greenctl windows compare --strategies baseline,climatology,corrected,project-model,electricity-maps,oracle

greenctl simulate windows --carbon data/carbon_fr.csv --facility data --trace examples/synthetic_workloads.yaml
greenctl dashboard export --output dashboards/green_observatory_v1.json
greenctl report --output runs/latest/report.md
```

The CLI should produce:

```text
best recurring low-carbon windows
predicted low-carbon windows
facility context report
facility-friendly windows
combined green windows
model comparison against oracle
model comparison against Electricity Maps when available
simulation report for run-now vs wait policies
```

## V1.0 API

```text
GET  /health

GET  /carbon/current
GET  /carbon/forecast?horizon_hours=48
GET  /carbon/windows
GET  /carbon/model/evaluation

GET  /facility/context
GET  /facility/windows
GET  /facility/quality

GET  /windows/low-carbon
GET  /windows/facility-context
GET  /windows/combined
GET  /windows/oracle

POST /simulate/windows
GET  /reports/latest
```

Example response:

```json
{
  "horizon_hours": 48,
  "windows": [
    {
      "start": "2026-07-07T22:00:00Z",
      "end": "2026-07-08T02:00:00Z",
      "window_type": "combined_green_window",
      "carbon_score": 0.18,
      "facility_score": 0.42,
      "combined_score": 0.24,
      "confidence": 0.71,
      "reason": "low predicted carbon with moderate facility pressure"
    }
  ]
}
```

## V1.0 Dashboard

Dashboard panels:

```text
Current French carbon signal
Carbon intensity timeline
Historical low-carbon window heatmap
Carbon forecast next 24-48h
Project model vs climatology vs oracle
Project model vs Electricity Maps if available

CDC facility data quality
POD load timeline
Temperature/humidity patterns
Facility score timeline
Low-facility-pressure windows

Combined green windows
Simulation selected windows by policy
Delay vs green-score improvement
Oracle regret by policy
Data freshness and provider status
```

Avoid framing the dashboard as "choose the best datacenter". The dashboard should frame facility data as temporal context for better when-to-run analysis.

## V1.0 Milestones

### Milestone 0 - Reframe And Scaffold

- Create package skeleton under `src/green_observatory`.
- Add config files for carbon model, facility model, window scoring and simulation.
- Define typed models for CarbonSignal, CarbonForecast, FacilitySignal, GreenWindow, SimulationResult and DataQualityReport.

### Milestone 1 - Carbon Data And Baselines

- Implement ODRE/eCO2mix import/replay provider.
- Implement persistence baseline.
- Implement historical climatology.
- Implement corrected climatology.
- Compute low-carbon windows.

### Milestone 2 - Project Carbon Model

- Build carbon forecasting features.
- Train first model: random forest or gradient boosting.
- Evaluate MAE/RMSE and green-window ranking.
- Compare against oracle.
- Add Electricity Maps comparison only if an API key is available.

### Milestone 3 - CDC Facility Data Quality

- Implement CDC CSV provider.
- Generate per-series quality report.
- Filter zero-only, impossible, duplicate and high-gap series.
- Build curated POD load aggregate from `POD.CONS`.
- Convert WUE counters to hourly/daily deltas.

### Milestone 4 - Facility Context Model

- Compute facility climatology by hour/day.
- Compute low/high facility-pressure windows.
- Implement facility score from load, temperature, humidity and PUE proxy.
- Produce facility context report.

### Milestone 5 - Combined Windows

- Implement carbon-only windows.
- Implement facility-context windows.
- Implement combined carbon/facility window scoring.
- Implement oracle best-window evaluation.
- Evaluate sensitivity to carbon/facility weights.

### Milestone 6 - Simulation

- Generate synthetic workload arrivals.
- Replay arrivals over historical carbon and CDC facility traces.
- Compare baseline, carbon-only, facility-only, combined and oracle policies.
- Report delay, selected-window score, oracle regret and CO2 proxy.

### Milestone 7 - Visualization And Reports

- Export CSV/JSON results.
- Generate Grafana dashboard JSON.
- Generate Markdown report summarizing carbon model, facility context, combined windows and simulation results.

### Milestone 8 - FastAPI Observatory

- Add API endpoints for carbon forecast, facility context, combined windows and simulation results.
- Add Dockerfile only after the API shape is stable.

## V1.0 Success Criteria

V1.0 is successful when it can produce:

```text
1. Current and historical French carbon signal.
2. Project carbon forecast model.
3. Model comparison against persistence, climatology, oracle and Electricity Maps when available.
4. Low-carbon windows for France.
5. CDC facility data quality report.
6. Facility context patterns and low-facility-pressure windows.
7. Combined carbon/facility green windows.
8. Simulation comparing run-now, carbon-only, facility-only, combined and oracle policies.
9. Dashboard and report explaining the tradeoff between delay and greener windows.
```

Example result:

```text
Using open French carbon data, the project model identifies recurring and
forecasted low-carbon windows, captures X% of oracle best-window potential,
and improves green-window ranking by Y% versus climatology. CERN CDC exports
provide facility context showing which low-carbon windows also have lower
facility pressure. Simulation shows the delay vs green-score improvement
tradeoff for synthetic workloads.
```

## V1.1 Goal - Green Decision Module

V1.1 consumes V1.0 green windows and adds workload-specific decisions.

```text
V1.0 green windows
+ workload intent
+ estimated duration
+ deadline/max delay
+ criticality
+ interruptibility
+ resource request
+ energy/runtime estimate
= workload decision
```

V1.1 answers:

```text
Should this batch job run now or wait?
Should this notebook namespace be a sleep candidate?
Should this service be observe-only?
Should this workload be recommended for power limits?
Should this job be used as backfill?
```

Decision outputs:

```text
RUN_NOW
DEFER
BACKFILL
SLEEP_CANDIDATE
RUN_WITH_LIMITS
OBSERVE_ONLY
```

Workload classes:

```text
observe_only
idle_prone_interactive
deferrable_batch
energy_shapeable
colocation_tolerant
checkpointable_long_running
```

Example:

```text
if workload is deferrable
and deadline allows waiting
and V1.0 predicts a better low-carbon or combined green window before deadline:
  DEFER
else:
  RUN_NOW
```

## V1.2 Kubernetes Advisory

V1.2 turns decisions into reviewable Kubernetes artifacts:

```text
Kueue advisory output
kube-green SleepInfo candidates
dry-run annotations
human-readable reports
```

No automatic mutation by default.

## V2 Controller

V2 can become a Kubernetes controller only after V1.0 and V1.1 are useful:

```text
watch labeled Jobs, Workloads and Namespaces
query Green Window Observatory and Green Decision Module
write status annotations
optionally create/update Kueue or kube-green resources
start in dry-run mode
```

## What Not To Do In V1.0

- Do not frame the project as choosing between datacenters.
- Do not build a mutating Kubernetes controller.
- Do not implement per-workload decisions yet.
- Do not claim precise workload-level CO2 without Kepler or calibrated benchmarks.
- Do not depend on Electricity Maps as a core dependency.
- Do not sum raw CDC power series without checking for double counting.

## Immediate Next Step

Implement the Green Window Observatory scaffold first:

```text
1. Typed models for CarbonSignal, CarbonForecast, FacilitySignal, GreenWindow and SimulationResult.
2. ODRE/eCO2mix carbon provider.
3. Persistence and climatology carbon baselines.
4. CDC CSV quality inspection.
5. Facility score from curated POD load + cleaned temperature/humidity.
6. Carbon-only and facility-context windows.
7. Combined green-window scoring.
8. First simulation command: greenctl simulate windows.
```

That creates the real V1 artifact: a measurement, forecasting and simulation application for when-to-run intelligence. The workload decision layer then becomes a clean V1.1 module instead of being mixed into the observatory.
