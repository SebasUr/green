"""Causal-clean ERT ablation for consolidated French 24-hour forecasts.

Protocol (fixed before opening the holdout):

* build exactly the consolidated benchmark's closed-hour feature matrix;
* fit three predeclared ERT configurations on targets before 2026-01-01;
* select the configuration by January MAPE only;
* estimate one multiplicative MAPE scale on February only;
* refit the frozen configuration on targets before 2026-03-01;
* open March-April once and compare with Direct and the physical-share head.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    _metrics,
    _paired_comparison,
)
from green_observatory.carbon.extra_trees_carbon import ExtraTreesCarbonRegressor
from green_observatory.carbon.protocols import regularize_hourly
from green_observatory.carbon.regime_moe import RegimeMoEFeatureBuilder, select_mape_scale
from green_observatory.carbon.rte_availability import RteAvailabilityFeatureStore
from green_observatory.providers.carbon_base import CARBON
from green_observatory.providers.carbon_odre import OdreCarbonProvider


@dataclass(frozen=True)
class ERTSpec:
    name: str
    n_estimators: int
    max_features: float
    min_samples_leaf: int
    max_depth: int | None = None


# Small and fixed: this is an ablation, not a holdout hyperparameter search.
ERT_SPECS = (
    ERTSpec("ert_balanced", 300, 0.70, 6, None),
    ERTSpec("ert_local", 300, 1.00, 3, None),
    ERTSpec("ert_smooth", 350, 0.55, 14, None),
)


def select_spec(
    metrics_by_name: dict[str, dict[str, float]],
    specs: Sequence[ERTSpec] = ERT_SPECS,
) -> ERTSpec:
    """Select by January MAPE, resolving ties by declared order."""

    order = {spec.name: position for position, spec in enumerate(specs)}
    return min(
        specs,
        key=lambda spec: (metrics_by_name[spec.name]["mape"], order[spec.name]),
    )


def window_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    actual_by_time: pd.Series,
) -> dict[str, float | int]:
    """One-hour green-window metrics over complete dense 24-hour queries."""

    realized: list[float] = []
    oracle: list[float] = []
    run_now: list[float] = []
    top1: list[float] = []
    ranks: list[float] = []
    for origin, group in frame.groupby("origin", sort=True):
        group = group.sort_values("horizon")
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group[prediction_column].to_numpy(dtype=float)
        if (
            len(group) != 24
            or set(group["horizon"].astype(int)) != set(range(1, 25))
            or not np.isfinite(actual).all()
            or not np.isfinite(prediction).all()
        ):
            continue
        selected = int(np.argmin(prediction))
        best = float(actual.min())
        incurred = float(actual[selected])
        realized.append(incurred)
        oracle.append(best)
        top1.append(float(incurred <= best + 1e-9))
        now = actual_by_time.get(origin, np.nan)
        if np.isfinite(now):
            run_now.append(float(now))
        if np.std(actual) > 0.0 and np.std(prediction) > 0.0:
            ranks.append(float(spearmanr(prediction, actual).statistic))
    if not realized:
        return {}
    mean_realized = float(np.mean(realized))
    mean_oracle = float(np.mean(oracle))
    mean_now = float(np.mean(run_now)) if run_now else float("nan")
    available = mean_now - mean_oracle
    potential = (
        100.0 * (mean_now - mean_realized) / available
        if np.isfinite(mean_now) and available > 1e-9
        else float("nan")
    )
    return {
        "mean_realized_gco2": mean_realized,
        "mean_oracle_gco2": mean_oracle,
        "mean_run_now_gco2": mean_now,
        "mean_regret": mean_realized - mean_oracle,
        "pct_oracle_potential": float(potential),
        "spearman": float(np.mean(ranks)) if ranks else float("nan"),
        "top1_accuracy": float(np.mean(top1)),
        "n": int(len(realized)),
    }


def _model(spec: ERTSpec, *, floor: float, threads: int) -> ExtraTreesCarbonRegressor:
    return ExtraTreesCarbonRegressor(
        n_estimators=spec.n_estimators,
        max_features=spec.max_features,
        min_samples_leaf=spec.min_samples_leaf,
        max_depth=spec.max_depth,
        inverse_level_floor=floor,
        n_jobs=threads,
        random_state=42,
    )


def _reference_predictions(path: str, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    for column in ("origin", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    keep = [*KEYS, "split", *columns]
    missing = sorted(set(keep).difference(frame.columns))
    if missing:
        raise ValueError(f"reference predictions miss columns: {missing}")
    return frame.loc[:, keep]


def _attach_references(
    predictions: pd.DataFrame,
    *,
    physical_reference: pd.DataFrame,
    share_reference: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    physical = physical_reference[physical_reference["split"].eq(split)].drop(
        columns="split"
    )
    share = share_reference[share_reference["split"].eq(split)].drop(columns="split")
    out = predictions.merge(physical, on=KEYS, how="left", validate="one_to_one")
    return out.merge(share, on=KEYS, how="left", validate="one_to_one")


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = regularize_hourly(OdreCarbonProvider.load_snapshot(args.carbon))
    forecasts = _indexed_parquet(args.mix_forecast).join(
        _indexed_parquet(args.price_forecast), how="outer"
    )
    builder = RegimeMoEFeatureBuilder(
        forecasts,
        availability_store=RteAvailabilityFeatureStore(
            pd.read_parquet(args.rte_unavailability)
        ),
        availability_feature_mode="all",
        include_curve_summaries=True,
        include_detailed_state=True,
    )
    origins = pd.date_range(_utc(args.feature_start), _utc("2026-04-29"), freq="1D")
    x, meta = builder.build(frame, origins, supervised=True)

    pre_january = meta["target_time"] < _utc("2026-01-01")
    january = meta["origin"].between(_utc("2026-01-01"), _utc("2026-01-31"))
    february = meta["origin"].between(_utc("2026-02-01"), _utc("2026-02-28"))
    pre_march = meta["target_time"] < _utc("2026-03-01")
    holdout_mask = meta["origin"].between(_utc("2026-03-01"), _utc("2026-04-29"))

    selection_metrics: dict[str, dict[str, float]] = {}
    dev_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in ERT_SPECS:
        estimator = _model(
            spec, floor=args.inverse_level_floor, threads=args.model_threads
        )
        estimator.fit(
            x.loc[pre_january].reset_index(drop=True),
            meta.loc[pre_january, "actual"].to_numpy(dtype=float),
        )
        january_prediction = estimator.predict(
            x.loc[january].reset_index(drop=True)
        )
        february_prediction = estimator.predict(
            x.loc[february].reset_index(drop=True)
        )
        selection_metrics[spec.name] = _metrics(
            meta.loc[january, "actual"], january_prediction
        )
        dev_predictions[spec.name] = (january_prediction, february_prediction)

    selected = select_spec(selection_metrics)
    january_prediction, february_prediction = dev_predictions[selected.name]
    scale, scale_fit_mape = select_mape_scale(
        meta.loc[february, "actual"].to_numpy(dtype=float), february_prediction
    )

    final_model = _model(
        selected, floor=args.inverse_level_floor, threads=args.model_threads
    )
    final_model.fit(
        x.loc[pre_march].reset_index(drop=True),
        meta.loc[pre_march, "actual"].to_numpy(dtype=float),
    )
    holdout_prediction = final_model.predict(
        x.loc[holdout_mask].reset_index(drop=True)
    )

    def rows(mask: pd.Series, raw: np.ndarray, split: str) -> pd.DataFrame:
        out = meta.loc[mask, KEYS + ["actual"]].reset_index(drop=True)
        out["ert_raw"] = raw
        out["ert_feb_scaled"] = scale * raw
        out["split"] = split
        return out

    january_frame = rows(january, january_prediction, "selection_january")
    february_frame = rows(february, february_prediction, "calibration_february")
    holdout = rows(holdout_mask, holdout_prediction, "holdout_mar_apr")

    physical_reference = _reference_predictions(
        args.physical_reference, ("direct_reference",)
    )
    share_reference = _reference_predictions(args.share_reference, ("share_lgbm",))
    holdout = _attach_references(
        holdout,
        physical_reference=physical_reference,
        share_reference=share_reference,
        split="holdout_mar_apr",
    )
    if holdout[["direct_reference", "share_lgbm"]].isna().any().any():
        raise ValueError("reference predictions do not cover the entire ERT holdout")

    point_columns = ("ert_raw", "ert_feb_scaled", "direct_reference", "share_lgbm")
    holdout_point = {
        column: _metrics(holdout["actual"], holdout[column])
        for column in point_columns
    }
    actual_by_time = pd.to_numeric(frame[CARBON], errors="coerce")
    holdout_window = {
        column: window_metrics(
            holdout, column, actual_by_time=actual_by_time
        )
        for column in point_columns
    }
    best_ert_column = min(
        ("ert_raw", "ert_feb_scaled"),
        key=lambda column: holdout_point[column]["mape"],
    )
    # Holdout chooses nothing operationally: this alias is only a concise
    # post-hoc diagnostic.  Deployment remains the February-scaled variant.
    decision = {
        "deployed_variant": "ert_feb_scaled",
        "posthoc_best_ert_variant": best_ert_column,
        "beats_direct_mape": bool(
            holdout_point["ert_feb_scaled"]["mape"]
            < holdout_point["direct_reference"]["mape"]
        ),
        "beats_best_known_point_mape_11_834314": bool(
            holdout_point["ert_feb_scaled"]["mape"] < 11.8343145
        ),
        "beats_share_oracle_potential_57_7947": bool(
            holdout_window["ert_feb_scaled"]["pct_oracle_potential"] > 57.7947
        ),
    }
    decision["accepted"] = bool(
        decision["beats_best_known_point_mape_11_834314"]
        or decision["beats_share_oracle_potential_57_7947"]
    )

    report = {
        "status": (
            "accepted as a complementary checkpoint"
            if decision["accepted"]
            else "rejected; does not improve either current checkpoint"
        ),
        "protocol": {
            "target": "RTE consolidated production intensity",
            "feature_builder": "RegimeMoEFeatureBuilder",
            "visibility": "origin-1h closed state; target D-1/D-7; D-1 h24 masked",
            "features": "same mix+price+RTE availability+curve summaries+detailed state as consolidated benchmark",
            "imputation": "median fit only on each training slice; all-empty train columns dropped",
            "objective": "ERT squared error with sample_weight=1/max(actual, floor)",
            "inverse_level_floor": float(args.inverse_level_floor),
            "configuration_selection": "fit target_time < 2026-01-01; January MAPE",
            "scale_calibration": "selected configuration's February predictions only",
            "final_fit": "target_time < 2026-03-01",
            "holdout": "origins 2026-03-01 through 2026-04-29, opened once",
        },
        "specs": [asdict(spec) for spec in ERT_SPECS],
        "january_selection": selection_metrics,
        "selected_spec": asdict(selected),
        "february_calibration": {
            "raw": _metrics(meta.loc[february, "actual"], february_prediction),
            "scale": float(scale),
            "scaled_mape": float(scale_fit_mape),
            "scaled": _metrics(
                meta.loc[february, "actual"], scale * february_prediction
            ),
        },
        "holdout_mar_apr": holdout_point,
        "holdout_window_selection": holdout_window,
        "paired_holdout_ert_vs_direct": _paired_comparison(
            holdout, "ert_feb_scaled", "direct_reference"
        ),
        "paired_holdout_ert_vs_share": _paired_comparison(
            holdout, "ert_feb_scaled", "share_lgbm"
        ),
        "decision": decision,
    }

    predictions = pd.concat(
        [january_frame, february_frame, holdout], ignore_index=True
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    hold = report["holdout_mar_apr"]
    windows = report["holdout_window_selection"]
    lines = [
        "# Ablación causal-clean Extremely Randomized Trees",
        "",
        f"Decisión: **{report['status']}**.",
        "",
        "| Señal | MAPE Mar-Abr | WAPE | MAE | Potencial oracle |",
        "|---|---:|---:|---:|---:|",
    ]
    for column in point_columns:
        lines.append(
            f"| `{column}` | {hold[column]['mape']:.3f}% | "
            f"{hold[column]['wape']:.3f}% | {hold[column]['mae']:.3f} | "
            f"{windows[column]['pct_oracle_potential']:.1f}% |"
        )
    lines += [
        "",
        f"Configuración seleccionada sólo en enero: `{selected.name}`.",
        f"Escala aprendida sólo en febrero: {scale:.3f}.",
        "El escalado no cambia el ranking horario; por eso su potencial oracle es idéntico al ERT crudo.",
        "Los detalles, bootstrap pareado y protocolo completo están en `summary.json`.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", default="data/cache/carbon_fr_hourly_detailed.parquet")
    parser.add_argument("--mix-forecast", default="data/cache/mix_day_ahead_fr_hourly.parquet")
    parser.add_argument("--price-forecast", default="data/cache/day_ahead_price_fr_hourly.parquet")
    parser.add_argument(
        "--rte-unavailability",
        default="data/cache/rte_unavailability_messages.parquet",
    )
    parser.add_argument("--feature-start", default="2021-08-15")
    parser.add_argument(
        "--physical-reference",
        default="runs/daily_refit_2026/consolidated_physical/predictions.parquet",
    )
    parser.add_argument(
        "--share-reference",
        default="runs/daily_refit_2026/consolidated_share/predictions.parquet",
    )
    parser.add_argument("--inverse-level-floor", type=float, default=8.0)
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/extra_trees_causal_clean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
