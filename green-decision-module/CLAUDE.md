# CLAUDE.md: Green Window Observatory

Project context for contributors and AI assistants. Two tracks: carbon
forecasting (primary, below) and an opt-in Kubernetes Job observability track
(`observability/`, `greenctl jobs …`, `docs/JOB_OBSERVABILITY.md`).

> All superseded experiments (V1.0 HistGBM ladder, SARIMAX/LSTM/ERT/ranker
> comparisons, consolidated-track evaluators) live in the **`snapshot2007`**
> branch. Do not re-create them here; cite `REPORTE_FINAL.md` instead.

## What this is

Package `green_observatory` (src layout). Forecast hourly carbon intensity of
the French grid for the next 24 h and rank low-carbon windows for deferrable
workloads.

## Locked conventions (do not silently change)

- **Ground truth** = RTE `taux_co2` (production-based gCO₂/kWh) from ODRE.
  Two targets exist and must never be mixed in one metric: the
  **consolidated** history and the **provisional live feed** (different
  numeric definitions; see REPORTE_FINAL §2).
- **No look-ahead leakage.** State features use the last fully closed hour
  (row `t` summarizes `[t, t+1h)` and is masked at origin `t`); day-ahead
  products are masked for target hours on the next local delivery day;
  publication-versioned inputs filter `updated/publication_date <= origin`.
- **Origins at 00:00 UTC**, horizons h1–h24, daily expanding refit.
- **Evaluation = paired comparisons** with 7-day circular block bootstrap;
  MAPE/WAPE/MAE for signal, exhaustive scheduling oracle + regret for windows.

## Architecture (current production)

- `carbon/realtime_proxy.py` + `realtime_proxy_live_refit.py` — **primary**:
  physical MoE (gas via 3 regime experts + classifier, alpha2 sharpening;
  coal/fuel/bioenergy/total pooled) through the fixed RTE provisional formula.
  Production signal = `physical_alpha2` + 14d causal calibration (≈13.06 %
  MAPE live May–Jul 2026).
- `carbon/regime_moe.py` + `regime_moe_live_refit.py` — feature builder
  (`RegimeMoEFeatureBuilder`) + Direct intensity model; secondary signal and
  the evaluator for consolidated-era backtests (e.g. winter).
- `carbon/causal_operational_gate.py` — causal calibration/gating layer.
- Context features (opt-in flags on both refits): `--thermal-margin`
  (residual demand + versioned outages), `--entsoe-a71` (D-1 total scheduled
  generation via ENTSO-E File Library), `--exchange-schedule` (RTE DA
  exchange programs). Column names carry `day_ahead` so the builder's mask
  applies.
- `windows/` — scoring (hysteresis detection), oracle, exhaustive
  all-durations scheduling evaluation.
- `--parallel-origins N` parallelizes independent origin refits
  (byte-identical to sequential; use with `--model-threads 1`).

## Environment & gotchas

- Conda env **`green-observatory`** (py3.12), python at
  `/Users/saur/miniconda3/envs/green-observatory/bin/python`.
- Credentials in `.env`: `RTE_CLIENT_ID/SECRET` (OAuth),
  `ENTSOE_EMAIL/PASSWORD` (FMS Keycloak). Never log them.
- **Parquet datetimes round-trip as ms**; always `.as_unit("ns")` before
  `.asi8` comparisons or joins with ns-resolution frames.
- ODRE real-time dataset (`eco2mix-national-tr`) retains data only until
  consolidation → re-archive `carbon_fr_realtime_2026_full.parquet`
  periodically; it cannot be re-downloaded once purged.
- ENTSO-E FMS: dates are Paris-midnight in UTC Z-notation; max 7 days/call;
  historical `UpdateTime` stamps were destroyed by the 2025 platform
  migration (trust policy in `entsoe_fms.A71_TRUSTED_STAMPS_FROM`).
- RTE unavailability: dedupe outages by `identifier` (`message_id` embeds the
  version suffix).
- Data snapshots live in `data/cache/` (gitignored, partially
  irreplaceable — see README).

## Contributor rules

- Keep new features opt-in flags with vintage-safe construction; measure by
  paired ablation before promoting (CI must not cross zero, or document why).
- Prefer editing config over hard-coding; keep runs reproducible.
- `pytest -q` must stay green (73 tests).
