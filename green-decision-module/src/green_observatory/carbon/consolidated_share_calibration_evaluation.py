"""Apply the existing nested causal calibrator to consolidated share forecasts.

This is a thin, reproducible adapter: it aliases a chosen share prediction to
the calibrator's generic ``physical_lgbm`` input and leaves the calibration
implementation untouched.  January still selects the architecture, February
selects the online scaling rule, and March-April remains evaluation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from green_observatory.carbon.consolidated_physical_calibration import run as run_calibration


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions).copy()
    if args.share_column not in frame:
        raise ValueError(f"share predictions miss {args.share_column!r}")
    frame["physical_mw_reference"] = frame["physical_lgbm"]
    frame["physical_lgbm"] = frame[args.share_column]
    adapter_input = output_dir / "share_calibration_input.parquet"
    frame.to_parquet(adapter_input, index=False)
    report = run_calibration(
        SimpleNamespace(
            predictions=str(adapter_input),
            output_dir=str(output_dir),
            january_start="2026-01-01",
            february_start="2026-02-01",
            march_start="2026-03-01",
            evaluation_end="2026-05-01",
        )
    )
    report["share_adapter"] = {
        "source_predictions": str(args.predictions),
        "share_column_aliased_as_physical_lgbm": args.share_column,
        "physical_mw_reference_preserved_in_input": "physical_mw_reference",
        "scientific_status": (
            "exploratory extension; uses exactly the already frozen nested "
            "calibration protocol, but was tried after inspecting prior results"
        ),
    }
    baseline_path = Path(args.baseline_calibration_summary)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current = report["metrics"]["evaluation_mar_apr"]
        previous = baseline["metrics"]["evaluation_mar_apr"]
        report["share_adapter"]["comparison_to_mw_calibrated_gate"] = {
            "share_calibrated_gate_mape": current[
                "calibrated_level_gate_seeded"
            ]["mape"],
            "mw_calibrated_gate_mape": previous[
                "calibrated_level_gate_seeded"
            ]["mape"],
            "delta_mape_points": (
                current["calibrated_level_gate_seeded"]["mape"]
                - previous["calibrated_level_gate_seeded"]["mape"]
            ),
        }
        current_window = report["window_selection"]["evaluation_mar_apr"][
            "physical_lgbm"
        ]
        previous_window = baseline["window_selection"]["evaluation_mar_apr"][
            "physical_lgbm"
        ]
        report["share_adapter"]["comparison_to_mw_window_selection"] = {
            "share_oracle_potential_percent": current_window[
                "pct_oracle_potential"
            ],
            "mw_oracle_potential_percent": previous_window[
                "pct_oracle_potential"
            ],
            "share_mean_regret_gco2": current_window["mean_regret"],
            "mw_mean_regret_gco2": previous_window["mean_regret"],
        }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    comparison = report["share_adapter"].get("comparison_to_mw_calibrated_gate")
    comparison_text = ""
    if comparison is not None:
        verdict = (
            "mejora"
            if comparison["delta_mape_points"] < 0.0
            else "no mejora"
        )
        comparison_text = (
            f"El gate calibrado da {comparison['share_calibrated_gate_mape']:.3f}% "
            f"MAPE frente a {comparison['mw_calibrated_gate_mape']:.3f}% del "
            f"gate MW: **{verdict}** ({comparison['delta_mape_points']:+.3f} puntos).\n\n"
        )
        window = report["share_adapter"].get("comparison_to_mw_window_selection")
        if window is not None:
            comparison_text += (
                f"Para ventanas verdes, en cambio, captura "
                f"{window['share_oracle_potential_percent']:.1f}% del potencial "
                f"frente a {window['mw_oracle_potential_percent']:.1f}% del MW "
                f"(regret {window['share_mean_regret_gco2']:.3f} vs "
                f"{window['mw_mean_regret_gco2']:.3f} gCO2/kWh).\n\n"
            )
    preface = f"""# Calibracion causal del modelo consolidado por shares

Adaptador sobre `{args.share_column}`. Toda referencia a `physical_lgbm` en el
reporte generado debajo significa ese forecast por shares, no el modelo MW.

{comparison_text}
"""
    readme = output_dir / "README.md"
    readme.write_text(preface + readme.read_text(encoding="utf-8"), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="runs/daily_refit_2026/consolidated_share/predictions.parquet",
    )
    parser.add_argument("--share-column", default="share_lgbm")
    parser.add_argument(
        "--baseline-calibration-summary",
        default="runs/daily_refit_2026/consolidated_physical_calibration/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/daily_refit_2026/consolidated_share_calibration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
