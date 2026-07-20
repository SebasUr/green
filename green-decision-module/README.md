# Green Window Observatory

Carbon-intensity forecasting and low-carbon window selection for carbon-aware
scheduling of deferrable workloads on the French grid, plus an opt-in
Kubernetes Job energy/CO₂ accounting track.

**Full technical report (results, methodology, ledger of everything tested):
[REPORTE_FINAL.md](REPORTE_FINAL.md).** Historical experiments and superseded
code live in the `snapshot2007` branch.

## What it does

- Forecasts hourly carbon intensity (gCO₂/kWh, RTE `taux_co2`) for the next
  24 h with a **physical mixture-of-experts** (per-fuel MW forecasts through
  the official RTE formula) retrained daily. Live MAPE ≈ 13 % (May–Jul 2026);
  a Direct intensity model is kept as the secondary signal and for
  consolidated-era backtests.
- Selects **low-carbon windows** for deferrable jobs (exhaustive
  any-duration scheduling evaluation + hysteresis-based green-hour detection).
- Accounts **per-Job energy and CO₂** on Kubernetes via Kepler + live RTE
  intensity (`greenctl jobs …`, see [docs/JOB_OBSERVABILITY.md](docs/JOB_OBSERVABILITY.md)).

## Layout

```
src/green_observatory/
  carbon/       forecasting stack
    realtime_proxy(.py,_live_refit.py)   physical MoE + daily-refit evaluator  [primary]
    regime_moe(.py,_live_refit.py)       feature builder + Direct model        [secondary]
    causal_operational_gate.py           causal calibration / gating layer
    thermal_margin.py                    D-1 implied-tightness features
    exhaustive_window_evaluation.py      all-durations scheduling oracle
    hysteresis_window_figure.py          detection-vs-oracle figures
  providers/    ODRE/eCO2mix, RTE (OAuth), ENTSO-E FMS, Energy Charts
  windows/      green-window scoring, oracle, exhaustive enumeration
  observability/  Kubernetes Job Kepler/RTE accounting (greenctl jobs)
```

## Environment

Conda env `green-observatory` (Python 3.12): numpy, pandas, scikit-learn,
lightgbm, pyarrow, pydantic, pyyaml, typer, httpx, holidays. Optional:
matplotlib (figures). Credentials in `.env` (gitignored):
`RTE_CLIENT_ID/SECRET`, `ENTSOE_EMAIL/PASSWORD`.

```bash
pip install -e ".[model,viz]"
```

## Data refresh

```bash
greenctl carbon import              # ODRE consolidated history
greenctl carbon fetch-mix-forecast  # Energy Charts D-1 wind/solar/load
greenctl carbon fetch-rte-system    # RTE versioned unavailability (needs .env)
python -m green_observatory.providers.rte_exchange_schedule --start ... --end ... --output data/cache/rte_exchange_schedule_da.parquet
python -m green_observatory.providers.entsoe_fms --start ... --end ... --output data/cache/entsoe_a71_generation_forecast_fr.parquet
```

The live snapshot `data/cache/carbon_fr_realtime_2026_full.parquet` comes from
the ODRE real-time dataset, which **retains data only until consolidation** —
re-archive it periodically (see REPORTE_FINAL §4).

## Run the production evaluation

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m green_observatory.carbon.realtime_proxy_live_refit \
  --model physical --pooled-gas \
  --carbon-live data/cache/carbon_fr_realtime_2026_full.parquet \
  --mix-forecast data/cache/mix_day_ahead_fr_hourly_bridged.parquet \
  --price-forecast data/cache/day_ahead_price_fr_hourly_bridged.parquet \
  --thermal-margin --entsoe-a71 data/cache/entsoe_a71_generation_forecast_fr.parquet \
  --eval-start 2026-05-01 --eval-end 2026-07-18 \
  --parallel-origins 5 --model-threads 1 \
  --output-dir runs/daily_refit_2026/realtime_proxy_daily_refit_tr_extended_sys
```

Then the calibrated headline via
`python -m green_observatory.carbon.causal_operational_gate` and the
scheduling-oracle report via
`python -m green_observatory.carbon.exhaustive_window_evaluation`.

`pytest -q` runs the suite (73 tests).
