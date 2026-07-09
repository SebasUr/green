# Green Window Observatory (V1.0)

Carbon temporal intelligence for **when-to-run** decisions on the French grid.
Part of the CERN Green Decision Module. V1.0 answers *"when is it greener to
run?"* using open French electricity/carbon data; workload-specific decisions
(*"should this job run now?"*) are the later V1.1 Green Decision Module.

> **Scope note.** Milestones 0-2 (the carbon track) are the current focus. The
> CERN CDC facility track (M3-4), simulation (M6), exporters (M7) and API (M8)
> are scaffolded but not yet built. See [`IMPLEMENTATION_PLAN_V1.md`](IMPLEMENTATION_PLAN_V1.md).

## Locked V1.0 conventions

| Decision | Choice |
|---|---|
| Carbon ground truth | RTE `taux_co2` (production-based gCO2/kWh) from ODRE/eCO2mix |
| Score | `green_score` in `[0, 1]`, **higher = greener**; raw gCO2/kWh kept separately |
| Green window | contiguous block below the **p25 of the forecast horizon** (configurable) |
| Climatology grouping | `Europe/Paris` local time (instants stored in UTC) |
| Evaluation | rolling-origin backtest, no look-ahead leakage |
| Electricity Maps | optional comparison only, needs `ELECTRICITYMAPS_API_TOKEN` |

## Setup (conda)

```bash
conda create -y --override-channels -c conda-forge -n green-observatory \
  python=3.12 numpy pandas scikit-learn scipy pyarrow \
  pydantic pyyaml typer httpx holidays joblib pytest tzdata python-dateutil
conda activate green-observatory
pip install -e .
```

## CLI (`greenctl`)

```bash
greenctl version
greenctl carbon import   --output data/carbon_fr.parquet      # fetch ODRE snapshot
greenctl carbon train    --config configs/carbon_model.yaml   # train project model
greenctl carbon forecast --horizon-hours 48
greenctl carbon compare  --strategies persistence,climatology,corrected,project-model,oracle
greenctl windows analyze --carbon data/carbon_fr.parquet
```

## Layout

```
src/green_observatory/
  models.py            typed contracts (pydantic v2)
  config.py            YAML config loading
  cli.py               greenctl
  providers/           odre (primary), electricity_maps (optional), cdc (M3+)
  carbon/              features, climatology, corrected_climatology, model, evaluation
  windows/             scoring, oracle, combine
  facility/            deferred (M3-4)
  simulation/          deferred (M6)
  exporters/           deferred (M7)
configs/               carbon_model.yaml, window_scoring.yaml, simulation.yaml
data/                  ODRE snapshots + CDC exports (CDC not used in M0-2)
tests/
```

## Results (illustrative backtest)

Rolling-origin backtest on the open French signal, training `< 2026-02-01`,
testing on 348 origins (Feb–Apr 2026, a persistently low-carbon regime):

**Green-window selection** — *"pick the greenest of the forecasted horizon hours"*:

| strategy | realized gCO₂/kWh | regret | % oracle potential | Spearman | top-1 |
|---|---|---|---|---|---|
| run-now | 14.91 | 3.16 | 0% | — | — |
| persistence | 15.04 | 3.29 | −4.1% | — | 0.24 |
| climatology | 13.85 | 2.09 | 33.7% | 0.05 | 0.20 |
| corrected | 13.36 | 1.61 | 49.1% | 0.05 | 0.26 |
| **project** | **12.96** | **1.21** | **61.8%** | **0.23** | **0.35** |
| oracle | 11.76 | 0.00 | 100% | 1.00 | 1.00 |

The project model captures **61.8% of the oracle's best-window potential** and
gives the lowest ranking regret. Note the headline lesson: **persistence has the
best point MAE at 24/48 h yet is useless for window selection** (it forecasts a
flat line and cannot rank future hours), which is exactly why ranking metrics —
not MAE alone — decide the "when to run" question. Reproduce with
`greenctl carbon compare --test-start 2026-02-01`.

## Scientific notes

- **No leakage.** Every forecast records its `issued_at`; features are built
  strictly as-of that instant, and the backtest advances a rolling origin.
- **Honest baselines.** The project model is only meaningful relative to
  persistence, climatology and corrected climatology, with an *oracle* upper
  bound. Green-window *ranking* quality matters as much as point MAE/RMSE.
- **Reproducible.** Fixed seeds, cached data snapshots and YAML-driven configs.
