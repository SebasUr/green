"""Causal static evaluation of the direct source-share operational proxy.

The experiment uses the exact temporal splits of ``realtime_proxy_evaluation``
but fits only the isolated source-share model.  Baseline predictions are read
from a completed causal-clean physical-proxy run, which avoids retraining the
existing models and guarantees an identical row-level comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.realtime_proxy import proxy_training_frame
from green_observatory.carbon.realtime_proxy_evaluation import (
    _combine_indexed,
    _indexed_parquet,
    _metrics,
)
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.source_share_proxy import (
    SHARE_TARGET_COLUMNS,
    SourceShareProxyMoE,
    add_causal_share_features,
    source_share_targets,
    source_share_variants,
)
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


KEYS = ["origin", "horizon", "target_time"]
SHARE_VARIANTS = (
    "share_physical",
    "share_physical_pooled",
    "share_physical_alpha2",
    "share_physical_alpha3",
    "share_physical_alpha5",
    "share_physical_hard",
)
BASELINE_COLUMNS = (
    "direct",
    "physical",
    "physical_pooled",
    "physical_alpha2",
    "physical_alpha3",
    "physical_alpha5",
    "physical_hard",
    "d1",
)
# D-1 has one missing row per origin under the strict open-hour mask.  It is
# useful diagnostically but is not a paired full-coverage comparator.
FULL_COVERAGE_BASELINE_COLUMNS = tuple(
    column for column in BASELINE_COLUMNS if column != "d1"
)


def _attach_share_targets(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    targets = source_share_targets(frame, out["target_time"])
    for column in SHARE_TARGET_COLUMNS:
        out[column] = targets[column].to_numpy()
    return out


def _fit(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    train: pd.Series,
    *,
    threads: int,
) -> SourceShareProxyMoE:
    x_train = x.loc[train].reset_index(drop=True)
    meta_train = _attach_share_targets(frame, meta.loc[train])
    model = SourceShareProxyMoE(
        warm_season_coal_persistence=True,
        classifier_params={"n_jobs": threads},
        source_params={"n_jobs": threads},
    )
    return model.fit(x_train, meta_train)


def _predict(
    model: SourceShareProxyMoE,
    x: pd.DataFrame,
    meta: pd.DataFrame,
    mask: pd.Series,
) -> pd.DataFrame:
    x_eval = x.loc[mask].reset_index(drop=True)
    meta_eval = meta.loc[mask].reset_index(drop=True)
    out = meta_eval.loc[:, KEYS + ["actual", "regime"]].copy()
    matrix = model.predict_matrix(x_eval)
    for name, values in source_share_variants(matrix).items():
        out[name] = values
    for column in matrix:
        if column in {"prediction", "prediction_pooled"}:
            continue
        out[column] = matrix[column].to_numpy()
    return out


def _load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    for column in ("origin", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    keep = KEYS + [
        column
        for column in ("actual", "proxy_actual", *BASELINE_COLUMNS)
        if column in frame
    ]
    return frame.loc[:, keep].drop_duplicates(KEYS, keep="last")


def _merge_baselines(
    share: pd.DataFrame, baseline: pd.DataFrame, *, live: bool = False
) -> pd.DataFrame:
    baseline_values = baseline.drop(
        columns=[column for column in ("actual", "proxy_actual") if column in baseline]
    )
    out = share.merge(baseline_values, on=KEYS, how="left", validate="one_to_one")
    if live:
        actual = baseline.loc[:, KEYS + ["actual"]]
        out = out.drop(columns="actual").merge(
            actual, on=KEYS, how="left", validate="one_to_one"
        )
    return out


def _period_report(frame: pd.DataFrame) -> dict:
    model_columns = [
        column
        for column in (*SHARE_VARIANTS, *BASELINE_COLUMNS)
        if column in frame
    ]
    report = {
        "overall": {
            column: _metrics(frame["actual"], frame[column])
            for column in model_columns
        },
        "by_month": {},
        "by_horizon_block": {},
    }
    for month, group in frame.groupby(frame["origin"].dt.strftime("%Y-%m")):
        report["by_month"][month] = {
            column: _metrics(group["actual"], group[column])
            for column in model_columns
        }
    blocks = ((1, 6), (7, 16), (17, 21), (22, 24))
    for start, end in blocks:
        group = frame[frame["horizon"].between(start, end)]
        report["by_horizon_block"][f"h{start}-{end}"] = {
            column: _metrics(group["actual"], group[column])
            for column in model_columns
        }
    return report


def _best(report: dict, columns: tuple[str, ...]) -> tuple[str, float]:
    rows = report["overall"]
    available = [(column, rows[column]["mape"]) for column in columns if column in rows]
    return min(available, key=lambda item: item[1])


def _write_readme(output_dir: Path, report: dict) -> None:
    selected = report["selection"]["share_variant"]
    rows = []
    for label, key in (
        ("Desarrollo enero-febrero", "dev_jan_feb"),
        ("Confirmacion marzo-abril", "retrospective_mar_apr"),
        ("Diagnostico live junio-julio", "live_jun_jul"),
    ):
        section = report.get(key)
        if not section:
            continue
        own = section["overall"][selected]["mape"]
        baseline_name, baseline_mape = _best(
            section, FULL_COVERAGE_BASELINE_COLUMNS
        )
        rows.append(f"| {label} | {own:.3f}% | {baseline_name} ({baseline_mape:.3f}%) |")
    verdict = (
        "PROMOVIDO a daily-refit"
        if report["selection"]["robust_improvement"]
        else "RECHAZADO: no mejora de forma robusta los baselines"
    )
    text = f"""# Proxy operacional por shares fisicos

Este experimento predice directamente `gas/coal/fuel_oil/bioenergy` como
fraccion de la generacion domestica total. La intensidad final usa solo la
formula fija de RTE; no aprende un mapper de carbono.

| Periodo | Share seleccionado | Mejor baseline existente |
|---|---:|---:|
{chr(10).join(rows)}

**Decision:** {verdict}.

El modelo se selecciono exclusivamente con enero-febrero. Marzo-abril y el
periodo live no cambian la seleccion. Los features de estado usan la ultima
hora cerrada (`origin-1h`); D-1/D-7 estan alineados con la hora objetivo y D-1
en horizonte 24 se enmascara porque coincide con la hora aun abierta.

Reproduccion:

```bash
/Users/saur/miniconda3/envs/green-observatory/bin/python \\
  -m green_observatory.carbon.source_share_proxy_evaluation \\
  --model-threads 4 \\
  --output-dir {output_dir}
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir)

    historical_published = regularize_hourly(
        OdreCarbonProvider.load_snapshot(args.carbon)
    )
    historical = proxy_training_frame(historical_published, carbon_column=CARBON)
    forecast = _indexed_parquet(args.mix_forecast).join(
        _indexed_parquet(args.price_forecast), how="outer"
    )
    builder = RegimeMoEFeatureBuilder(
        forecast,
        availability_store=RteAvailabilityFeatureStore.from_parquet(
            args.rte_unavailability
        ),
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=True,
    )
    historical_origins = pd.date_range(
        _utc(args.feature_start), _utc(args.retrospective_end), freq="1D"
    )
    x_base, meta = builder.build(historical, historical_origins, supervised=True)
    x = add_causal_share_features(x_base, meta, historical)

    train_dev = meta["target_time"] < _utc("2026-01-01")
    dev_mask = (meta["origin"] >= _utc("2026-01-01")) & (
        meta["origin"] < _utc("2026-03-01")
    )
    dev_model = _fit(
        x, meta, historical, train_dev, threads=args.model_threads
    )
    dev = _predict(dev_model, x, meta, dev_mask)
    dev = _merge_baselines(
        dev, _load_baseline(baseline_dir / "dev_predictions.parquet")
    )
    dev_report = _period_report(dev)
    selected_share, _ = _best(dev_report, SHARE_VARIANTS)

    train_retrospective = meta["target_time"] < _utc("2026-03-01")
    retrospective_mask = (meta["origin"] >= _utc("2026-03-01")) & (
        meta["origin"] <= _utc(args.retrospective_end)
    )
    retrospective_model = _fit(
        x, meta, historical, train_retrospective, threads=args.model_threads
    )
    retrospective = _predict(
        retrospective_model, x, meta, retrospective_mask
    )
    retrospective = _merge_baselines(
        retrospective,
        _load_baseline(baseline_dir / "retrospective_predictions.parquet"),
    )
    retrospective_report = _period_report(retrospective)

    live_report = None
    live = pd.DataFrame()
    if args.carbon_live:
        live_published = _indexed_parquet(args.carbon_live)
        combined_published = regularize_hourly(
            _combine_indexed(historical_published, live_published)
        )
        combined = proxy_training_frame(combined_published, carbon_column=CARBON)
        live_forecast = _combine_indexed(
            _indexed_parquet(args.mix_forecast),
            _indexed_parquet(args.mix_forecast_live),
        ).join(
            _combine_indexed(
                _indexed_parquet(args.price_forecast),
                _indexed_parquet(args.price_forecast_live),
            ),
            how="outer",
        )
        intervals = pd.concat(
            [
                pd.read_parquet(args.rte_unavailability),
                pd.read_parquet(args.rte_unavailability_live),
            ],
            ignore_index=True,
        ).drop_duplicates()
        live_builder = RegimeMoEFeatureBuilder(
            live_forecast,
            availability_store=RteAvailabilityFeatureStore(intervals),
            availability_feature_mode="all",
            include_curve_summaries=True,
            include_detailed_state=True,
        )
        live_origins = pd.date_range(
            _utc(args.live_start), _utc(args.live_end), freq="1D"
        )
        x_live_base, meta_live = live_builder.build(
            combined, live_origins, supervised=True
        )
        x_live = add_causal_share_features(x_live_base, meta_live, combined)
        final_train = meta["target_time"] < _utc("2026-05-01")
        final_model = _fit(
            x, meta, historical, final_train, threads=args.model_threads
        )
        live = _predict(
            final_model,
            x_live,
            meta_live,
            pd.Series(True, index=meta_live.index),
        ).rename(columns={"actual": "proxy_actual"})
        live["actual"] = combined_published[CARBON].reindex(
            pd.DatetimeIndex(live["target_time"])
        ).to_numpy(dtype=float)
        live = _merge_baselines(
            live,
            _load_baseline(baseline_dir / "live_diagnostic_predictions.parquet"),
            live=True,
        )
        live = live.loc[live["actual"].notna()].reset_index(drop=True)
        live_report = _period_report(live)
        live_report["against_reconstructed_proxy"] = {
            column: _metrics(live["proxy_actual"], live[column])
            for column in SHARE_VARIANTS
        }

    # A route is called robust only if its preselected variant beats the best
    # existing baseline independently in all three forward slices.  This is a
    # deliberately demanding gate before spending hours on daily refits.
    comparisons = {}
    robust = True
    for key, section in (
        ("dev_jan_feb", dev_report),
        ("retrospective_mar_apr", retrospective_report),
        ("live_jun_jul", live_report),
    ):
        if section is None:
            continue
        selected_mape = section["overall"][selected_share]["mape"]
        baseline_name, baseline_mape = _best(
            section, FULL_COVERAGE_BASELINE_COLUMNS
        )
        win = selected_mape < baseline_mape
        robust &= win
        comparisons[key] = {
            "selected_share_mape": selected_mape,
            "best_baseline": baseline_name,
            "best_baseline_mape": baseline_mape,
            "absolute_mape_delta": selected_mape - baseline_mape,
            "wins": bool(win),
        }

    report = {
        "protocol": {
            "target": "operational RTE physical proxy; live scored against published provisional taux_co2",
            "formula_gco2_kwh": "986*coal_share + 777*fuel_oil_share + 429*gas_share + 494*bioenergy_share",
            "feature_visibility": "origin-1h closed state plus target-aligned D-1/D-7; target D-1 h24 masked",
            "coal_head": "D-1 share persistence in Feb-Oct; learned share regressor in Nov-Jan",
            "fit_before_dev": "2026-01-01T00:00:00Z",
            "dev_selection": "minimum raw MAPE on January-February among fixed share variants",
            "retrospective_refit_before": "2026-03-01T00:00:00Z",
            "live_static_fit_before": "2026-05-01T00:00:00Z",
            "scientific_status": "exploratory; all 2026 periods and the warm-season coal heuristic had been inspected by earlier model families",
        },
        "selection": {
            "share_variant": selected_share,
            "robust_rule": "preselected share variant beats the best full-coverage baseline MAPE in every reported forward slice; masked D-1 remains diagnostic",
            "comparisons": comparisons,
            "robust_improvement": bool(robust),
            "daily_refit_recommended": bool(robust),
        },
        "dev_jan_feb": dev_report,
        "retrospective_mar_apr": retrospective_report,
        "live_jun_jul": live_report,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    dev.to_parquet(output_dir / "dev_predictions.parquet", index=False)
    retrospective.to_parquet(
        output_dir / "retrospective_predictions.parquet", index=False
    )
    if not live.empty:
        live.to_parquet(output_dir / "live_diagnostic_predictions.parquet", index=False)
    _write_readme(output_dir, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_detailed.parquet")
    parser.add_argument("--carbon-live", default="data/cache/carbon_fr_realtime_holdout.parquet")
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument("--mix-forecast-live", default="data/cache/mix_day_ahead_fr_holdout.parquet")
    parser.add_argument("--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet")
    parser.add_argument("--price-forecast-live", default="data/cache/day_ahead_price_fr_holdout.parquet")
    parser.add_argument("--rte-unavailability", default="data/cache/rte_unavailability_messages.parquet")
    parser.add_argument("--rte-unavailability-live", default="data/cache/rte_unavailability_messages_holdout.parquet")
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument("--retrospective-end", default="2026-04-29")
    parser.add_argument("--live-start", default="2026-06-17")
    parser.add_argument("--live-end", default="2026-07-15")
    parser.add_argument(
        "--baseline-dir",
        default="runs/daily_refit_2026/realtime_proxy_v3_causal_clean",
    )
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/source_share_proxy_causal_clean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
