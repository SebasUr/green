"""Strict causal evaluation of a 24-hour green-window learning-to-rank head.

Protocol
--------
1. Fit each small ranker candidate on complete queries with
   ``target_time < 2026-01-01``.
2. Select the ranker configuration using January only.
3. Freeze that identity and select a percentile-rank blend with the clean
   consolidated ``share_lgbm`` head using February only.
4. Refit the selected ranker on complete queries with
   ``target_time < 2026-03-01`` and evaluate once on March-April.

The ranker is evaluated only with window/ranking metrics.  Its score has no
gCO2/kWh interpretation and therefore this module deliberately reports no
MAPE for it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from green_observatory.carbon.annual_evaluation import _utc
from green_observatory.carbon.consolidated_physical_evaluation import (
    KEYS,
    _indexed_parquet,
)
from green_observatory.carbon.green_window_ranker import (
    DEFAULT_RANKER_SPECS,
    GreenWindowRanker,
    RankerSpec,
    blend_query_percentiles,
    complete_query_positions,
    within_query_percentile,
)
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.carbon.rte_forecast_features import (
    RteGenerationForecastFeatureStore,
)
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


BASELINE_COLUMNS = (
    "direct_reference",
    "physical_lgbm",
    "selected_direct_physical_blend",
    "share_lgbm",
)
RANKING_COLUMNS = (*BASELINE_COLUMNS, "green_window_ranker", "ranker_share_blend")
DEFAULT_BLEND_GRID = tuple(np.arange(0.0, 1.0001, 0.05))


def _complete_period_positions(
    meta: pd.DataFrame,
    *,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
) -> np.ndarray:
    mask = meta["origin"].between(origin_start, origin_end)
    return complete_query_positions(meta, mask=mask)


def _prediction_frame(
    meta: pd.DataFrame,
    positions: np.ndarray,
    score: np.ndarray,
) -> pd.DataFrame:
    out = meta.iloc[positions][KEYS + ["actual"]].reset_index(drop=True)
    if len(out) != len(score):
        raise ValueError("ranker score does not align with evaluation metadata")
    out["ranker_score"] = np.asarray(score, dtype=float)
    out["green_window_ranker"] = within_query_percentile(
        out["ranker_score"], out["origin"], higher_is_greener=True
    )
    return out


def _load_clean_references(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = set(KEYS + ["actual", "split", *BASELINE_COLUMNS])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"clean share predictions miss columns: {sorted(missing)}")
    keep = KEYS + ["actual", "split", *BASELINE_COLUMNS]
    out = frame[keep].copy()
    out["origin"] = pd.to_datetime(out["origin"], utc=True)
    out["target_time"] = pd.to_datetime(out["target_time"], utc=True)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="raise").astype(int)
    if out.duplicated(KEYS).any():
        raise ValueError("clean share predictions contain duplicate keys")
    return out


def _attach_reference(
    ranker: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    subset = reference.loc[reference["split"].eq(split)].drop(columns="split")
    out = ranker.merge(
        subset.rename(columns={"actual": "reference_actual"}),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    missing = out[list(BASELINE_COLUMNS)].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"reference predictions miss {int(missing.sum())} ranker rows")
    mismatch = np.abs(
        out["actual"].to_numpy(dtype=float)
        - out["reference_actual"].to_numpy(dtype=float)
    )
    if np.nanmax(mismatch) > 1e-9:
        raise ValueError("ranker and reference targets are not aligned")
    return out.drop(columns="reference_actual")


def window_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    actual_by_time: pd.Series,
) -> dict[str, float | int]:
    """Measure one-hour green selection over complete 24-hour queries."""

    realized: list[float] = []
    oracle: list[float] = []
    run_now: list[float] = []
    correlations: list[float] = []
    top1: list[float] = []
    for origin, group in frame.groupby("origin", sort=True):
        group = group.sort_values("horizon")
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group[prediction_column].to_numpy(dtype=float)
        if (
            len(group) != 24
            or not np.isfinite(actual).all()
            or not np.isfinite(prediction).all()
        ):
            continue
        selected = int(np.argmin(prediction))
        best = float(np.min(actual))
        incurred = float(actual[selected])
        realized.append(incurred)
        oracle.append(best)
        top1.append(float(incurred <= best + 1e-9))
        now = actual_by_time.get(origin, np.nan)
        if np.isfinite(now):
            run_now.append(float(now))
        if np.std(prediction) > 0.0 and np.std(actual) > 0.0:
            correlations.append(float(spearmanr(prediction, actual).statistic))
    if not realized:
        return {}
    mean_realized = float(np.mean(realized))
    mean_oracle = float(np.mean(oracle))
    mean_now = float(np.mean(run_now)) if run_now else float("nan")
    denominator = mean_now - mean_oracle
    potential = (
        100.0 * (mean_now - mean_realized) / denominator
        if np.isfinite(mean_now) and denominator > 1e-9
        else float("nan")
    )
    return {
        "mean_realized_gco2": mean_realized,
        "mean_oracle_gco2": mean_oracle,
        "mean_run_now_gco2": mean_now,
        "mean_regret": mean_realized - mean_oracle,
        "pct_oracle_potential": potential,
        "spearman": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "top1_accuracy": float(np.mean(top1)),
        "n": int(len(realized)),
    }


def _window_report(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    actual_by_time: pd.Series,
) -> dict[str, dict]:
    return {
        column: window_metrics(
            frame, column, actual_by_time=actual_by_time
        )
        for column in columns
        if column in frame.columns
    }


def select_ranker_spec(
    metrics_by_spec: dict[str, dict],
    specs: Sequence[RankerSpec],
) -> RankerSpec:
    """Select on regret, then whole-query Spearman, then stable name."""

    by_name = {spec.name: spec for spec in specs}
    selected_name = min(
        metrics_by_spec,
        key=lambda name: (
            metrics_by_spec[name]["mean_regret"],
            -metrics_by_spec[name]["spearman"],
            name,
        ),
    )
    return by_name[selected_name]


def select_blend_weight(
    frame: pd.DataFrame,
    *,
    actual_by_time: pd.Series,
    grid: Sequence[float] = DEFAULT_BLEND_GRID,
) -> tuple[float, list[dict]]:
    """Choose the ranker percentile weight using a frozen validation period."""

    candidates: list[dict] = []
    for weight in grid:
        prediction = blend_query_percentiles(
            frame["ranker_score"],
            frame["share_lgbm"],
            frame["origin"],
            ranker_weight=float(weight),
        )
        candidate = frame.assign(_candidate=prediction)
        metrics = window_metrics(
            candidate, "_candidate", actual_by_time=actual_by_time
        )
        candidates.append({"ranker_weight": float(weight), **metrics})
    selected = min(
        candidates,
        key=lambda row: (
            row["mean_regret"],
            -row["spearman"],
            row["ranker_weight"],
        ),
    )
    return float(selected["ranker_weight"]), candidates


def _selection_costs(frame: pd.DataFrame, column: str) -> pd.Series:
    costs = {}
    for origin, group in frame.groupby("origin", sort=True):
        group = group.sort_values("horizon")
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group[column].to_numpy(dtype=float)
        if (
            len(group) == 24
            and np.isfinite(actual).all()
            and np.isfinite(prediction).all()
        ):
            costs[origin] = float(actual[int(np.argmin(prediction))])
    return pd.Series(costs, dtype=float)


def paired_daily_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    samples: int = 20_000,
    seed: int = 20260719,
) -> dict[str, float | int | list[float]]:
    """Bootstrap the paired daily realized-carbon difference."""

    candidate_cost = _selection_costs(frame, candidate)
    reference_cost = _selection_costs(frame, reference)
    common = candidate_cost.index.intersection(reference_cost.index)
    delta = (
        candidate_cost.reindex(common) - reference_cost.reindex(common)
    ).to_numpy(dtype=float)
    if len(delta) == 0:
        return {
            "mean_realized_delta_gco2": float("nan"),
            "day_bootstrap_95ci": [float("nan"), float("nan")],
            "days_candidate_better_percent": float("nan"),
            "days_tied_percent": float("nan"),
            "n_days": 0,
        }
    rng = np.random.default_rng(seed)
    draws = delta[
        rng.integers(0, len(delta), size=(int(samples), len(delta)))
    ].mean(axis=1)
    return {
        "mean_realized_delta_gco2": float(delta.mean()),
        "day_bootstrap_95ci": [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ],
        "days_candidate_better_percent": float(100.0 * np.mean(delta < 0.0)),
        "days_tied_percent": float(100.0 * np.mean(delta == 0.0)),
        "n_days": int(len(delta)),
    }


def _fit_candidate(
    x: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    train_before: pd.Timestamp,
    spec: RankerSpec,
    threads: int,
) -> tuple[GreenWindowRanker, np.ndarray]:
    positions = complete_query_positions(
        meta, mask=meta["target_time"] < train_before
    )
    model = GreenWindowRanker(spec, n_jobs=threads)
    model.fit(
        x.iloc[positions].reset_index(drop=True),
        meta.iloc[positions].reset_index(drop=True),
    )
    return model, positions


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    forecasts = _indexed_parquet(args.mix_forecast).join(
        _indexed_parquet(args.price_forecast), how="outer"
    )
    rte_forecast_store = (
        RteGenerationForecastFeatureStore.from_parquet(
            args.rte_generation_forecast,
            production_types=args.rte_production_type or None,
        )
        if args.rte_generation_forecast
        else None
    )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        availability_store=RteAvailabilityFeatureStore.from_parquet(
            args.rte_unavailability
        ),
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=not args.base_state_only,
        rte_forecast_store=rte_forecast_store,
    )
    origins = pd.date_range(
        _utc(args.feature_start), _utc("2026-04-29"), freq="1D"
    )
    x, meta = builder.build(frame, origins, supervised=True)
    reference = _load_clean_references(args.share_predictions)
    actual_by_time = frame[CARBON]

    january_positions = _complete_period_positions(
        meta,
        origin_start=_utc("2026-01-01"),
        origin_end=_utc("2026-01-31"),
    )
    february_positions = _complete_period_positions(
        meta,
        origin_start=_utc("2026-02-01"),
        origin_end=_utc("2026-02-28"),
    )
    dev_positions = np.concatenate([january_positions, february_positions])

    candidate_predictions: dict[str, pd.DataFrame] = {}
    candidate_reports: dict[str, dict] = {}
    pre_january_train_positions: np.ndarray | None = None
    for spec in DEFAULT_RANKER_SPECS:
        model, train_positions = _fit_candidate(
            x,
            meta,
            train_before=_utc("2026-01-01"),
            spec=spec,
            threads=args.model_threads,
        )
        pre_january_train_positions = train_positions
        score = model.predict_score(x.iloc[dev_positions].reset_index(drop=True))
        candidate = _prediction_frame(meta, dev_positions, score)
        candidate = _attach_reference(
            candidate, reference, split="dev_jan_feb"
        )
        january = candidate["origin"] < _utc("2026-02-01")
        candidate_predictions[spec.name] = candidate
        candidate_reports[spec.name] = {
            "spec": spec.to_dict(),
            "selection_january": window_metrics(
                candidate.loc[january],
                "green_window_ranker",
                actual_by_time=actual_by_time,
            ),
            "validation_february": window_metrics(
                candidate.loc[~january],
                "green_window_ranker",
                actual_by_time=actual_by_time,
            ),
        }

    selected_spec = select_ranker_spec(
        {
            name: report["selection_january"]
            for name, report in candidate_reports.items()
        },
        DEFAULT_RANKER_SPECS,
    )
    dev = candidate_predictions[selected_spec.name].copy()
    february = dev["origin"] >= _utc("2026-02-01")
    blend_weight, blend_candidates = select_blend_weight(
        dev.loc[february], actual_by_time=actual_by_time
    )
    dev["ranker_share_blend"] = blend_query_percentiles(
        dev["ranker_score"],
        dev["share_lgbm"],
        dev["origin"],
        ranker_weight=blend_weight,
    )

    final_model, final_train_positions = _fit_candidate(
        x,
        meta,
        train_before=_utc("2026-03-01"),
        spec=selected_spec,
        threads=args.model_threads,
    )
    holdout_positions = _complete_period_positions(
        meta,
        origin_start=_utc("2026-03-01"),
        origin_end=_utc("2026-04-29"),
    )
    holdout_score = final_model.predict_score(
        x.iloc[holdout_positions].reset_index(drop=True)
    )
    holdout = _prediction_frame(meta, holdout_positions, holdout_score)
    holdout = _attach_reference(
        holdout, reference, split="holdout_mar_apr"
    )
    holdout["ranker_share_blend"] = blend_query_percentiles(
        holdout["ranker_score"],
        holdout["share_lgbm"],
        holdout["origin"],
        ranker_weight=blend_weight,
    )
    importance = final_model.feature_importance().head(40)
    final_model.save_model(output_dir / "ranker_model.txt")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(final_model.feature_columns_, indent=2), encoding="utf-8"
    )

    january = dev["origin"] < _utc("2026-02-01")
    february = ~january
    report = {
        "status": (
            "causal-clean ranking experiment; rank scores are not carbon "
            "intensity and have no MAPE"
        ),
        "protocol": {
            "target": "RTE consolidated production carbon intensity",
            "query": "one complete UTC-origin group with horizons 1..24",
            "relevance": (
                "integer ordinal relevance 0..23; lower actual carbon receives "
                "higher relevance; tied actuals receive equal relevance"
            ),
            "feature_visibility": (
                "RegimeMoEFeatureBuilder origin-1h closed state; aligned D-1/D-7 "
                "with D-1 h24 masked; versioned RTE availability"
            ),
            "detailed_state": not args.base_state_only,
            "rte_generation_forecast": (
                {
                    "path": args.rte_generation_forecast,
                    "production_types": args.rte_production_type or "all supported",
                    "visibility": "latest updated_date <= origin",
                }
                if args.rte_generation_forecast
                else None
            ),
            "configuration_selection": (
                "fit target_time < 2026-01-01; select minimum January regret"
            ),
            "blend_selection": (
                "selected ranker frozen; choose ranker/share percentile weight "
                "on February only"
            ),
            "holdout": (
                "refit selected ranker on complete groups target_time < "
                "2026-03-01; evaluate origins 2026-03-01..2026-04-29"
            ),
            "legacy_full_models_used": False,
            "feature_count_final": len(final_model.feature_columns_),
            "pre_january_train_groups": int(
                len(pre_january_train_positions) // 24
                if pre_january_train_positions is not None
                else 0
            ),
            "pre_march_refit_groups": int(len(final_train_positions) // 24),
        },
        "candidate_selection": candidate_reports,
        "selected_spec": selected_spec.to_dict(),
        "selected_ranker_weight_in_percentile_blend": blend_weight,
        "top_feature_importance_gain": json.loads(
            importance.to_json(orient="records")
        ),
        "february_blend_candidates": blend_candidates,
        "window_selection": {
            "selection_january": _window_report(
                dev.loc[january], RANKING_COLUMNS, actual_by_time=actual_by_time
            ),
            "validation_february": _window_report(
                dev.loc[february], RANKING_COLUMNS, actual_by_time=actual_by_time
            ),
            "dev_jan_feb": _window_report(
                dev, RANKING_COLUMNS, actual_by_time=actual_by_time
            ),
            "evaluation_mar_apr": _window_report(
                holdout, RANKING_COLUMNS, actual_by_time=actual_by_time
            ),
            "evaluation_by_month": {
                str(month): _window_report(
                    group, RANKING_COLUMNS, actual_by_time=actual_by_time
                )
                for month, group in holdout.groupby(
                    holdout["origin"].dt.strftime("%Y-%m")
                )
            },
        },
        "paired_daily_bootstrap_evaluation": {
            "ranker_vs_share_lgbm": paired_daily_bootstrap(
                holdout, "green_window_ranker", "share_lgbm"
            ),
            "ranker_share_blend_vs_share_lgbm": paired_daily_bootstrap(
                holdout, "ranker_share_blend", "share_lgbm"
            ),
            "ranker_share_blend_vs_existing_point_blend": paired_daily_bootstrap(
                holdout,
                "ranker_share_blend",
                "selected_direct_physical_blend",
            ),
        },
    }

    predictions = pd.concat(
        [
            dev.assign(split="dev_jan_feb"),
            holdout.assign(split="holdout_mar_apr"),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    holdout_metrics = report["window_selection"]["evaluation_mar_apr"]
    february_metrics = report["window_selection"]["validation_february"]
    lines = [
        "# Ranker causal para ventanas verdes de 24 horas",
        "",
        "Este head aprende sólo el orden de las 24 horas. Su score no está en "
        "gCO2/kWh y, por diseño, no tiene MAPE.",
        "",
        "## Protocolo",
        "",
        "- Configuraciones entrenadas antes de enero y elegidas únicamente con enero.",
        "- Peso ranker/`share_lgbm` elegido únicamente con febrero.",
        "- Ranker elegido refiteado antes de marzo; marzo-abril se abre una sola vez.",
        "- Estado físico de origen tomado de la última hora cerrada; disponibilidad RTE versionada.",
        "- No se usa `full_models` ni `rank_consensus` legacy.",
        "",
        f"Configuración elegida: `{selected_spec.name}`.",
        f"Peso del ranker en el blend de percentiles: {blend_weight:.2f}.",
        "",
        "## Resultado",
        "",
        "| Señal | Feb potencial oracle | Mar-Abr potencial oracle | Mar-Abr regret | Mar-Abr top-1 | Mar-Abr Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "direct_reference": "Direct clean",
        "physical_lgbm": "MW physical",
        "selected_direct_physical_blend": "Blend físico+Direct",
        "share_lgbm": "Share LGBM",
        "green_window_ranker": "Ranker",
        "ranker_share_blend": "Ranker+share",
    }
    for column in RANKING_COLUMNS:
        feb = february_metrics[column]
        hold = holdout_metrics[column]
        lines.append(
            f"| {labels[column]} | {feb['pct_oracle_potential']:.2f}% | "
            f"{hold['pct_oracle_potential']:.2f}% | {hold['mean_regret']:.3f} | "
            f"{100.0 * hold['top1_accuracy']:.1f}% | {hold['spearman']:.3f} |"
        )
    ranker_vs_share = report["paired_daily_bootstrap_evaluation"][
        "ranker_vs_share_lgbm"
    ]
    blend_vs_share = report["paired_daily_bootstrap_evaluation"][
        "ranker_share_blend_vs_share_lgbm"
    ]
    important = ", ".join(
        f"`{row['feature']}` ({100.0 * row['share']:.1f}%)"
        for row in report["top_feature_importance_gain"][:8]
    )
    lines += [
        "",
        "## Decisión",
        "",
        "No promover este ranker: en el holdout, `share_lgbm` conserva "
        f"{holdout_metrics['share_lgbm']['pct_oracle_potential']:.2f}% del "
        "potencial oracle, frente a "
        f"{holdout_metrics['green_window_ranker']['pct_oracle_potential']:.2f}% "
        "del ranker y "
        f"{holdout_metrics['ranker_share_blend']['pct_oracle_potential']:.2f}% "
        "del blend. "
        "El ranker añade "
        f"{ranker_vs_share['mean_realized_delta_gco2']:.3f} gCO2/kWh frente "
        "a share (IC95% bootstrap diario "
        f"{ranker_vs_share['day_bootstrap_95ci']}); el blend añade "
        f"{blend_vs_share['mean_realized_delta_gco2']:.3f} gCO2/kWh "
        f"(IC95% {blend_vs_share['day_bootstrap_95ci']}).",
        "",
        "La ablación de estado base está guardada aparte. Mejoró febrero, pero "
        "también falló al abrir marzo-abril; no se usa para escoger el resultado "
        "después de ver el holdout.",
        "",
        "## Qué aprendió",
        "",
        f"Las mayores importancias por ganancia fueron: {important}. La fuerte "
        "dependencia de la forma horaria y del precio/VRE, con poca señal fósil "
        "prospectiva, ayuda a explicar por qué el orden no se trasladó bien del "
        "invierno al régimen bajo en carbono de primavera.",
        "",
        "La comparación bootstrap completa está en `summary.json`; el booster "
        "final en `ranker_model.txt` y el contrato de columnas en "
        "`feature_columns.json`.",
        "",
        "## Reproducir",
        "",
        "```bash",
        "/Users/saur/miniconda3/envs/green-observatory/bin/python -m \\",
        "  green_observatory.carbon.green_window_ranker_evaluation \\",
        "  --model-threads 4",
        "```",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carbon", default="data/cache/carbon_fr_hourly_detailed.parquet"
    )
    parser.add_argument(
        "--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet"
    )
    parser.add_argument(
        "--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet"
    )
    parser.add_argument(
        "--rte-unavailability",
        default="data/cache/rte_unavailability_messages.parquet",
    )
    parser.add_argument("--rte-generation-forecast", default="")
    parser.add_argument(
        "--rte-production-type",
        action="append",
        choices=("WIND_ONSHORE", "WIND_OFFSHORE", "SOLAR"),
        default=[],
    )
    parser.add_argument(
        "--base-state-only",
        action="store_true",
        help="Disable the opt-in detailed closed-hour physical state columns",
    )
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument(
        "--share-predictions",
        default="runs/daily_refit_2026/consolidated_share/predictions.parquet",
    )
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/green_window_ranker_causal_clean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
