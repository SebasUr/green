import numpy as np
import pandas as pd

from green_observatory.carbon.causal_operational_gate import (
    _load_d1,
    causal_gate,
    load_expert_predictions,
)


def _two_day_frame() -> pd.DataFrame:
    rows = []
    for day in range(2):
        origin = pd.Timestamp("2026-06-17", tz="UTC") + pd.Timedelta(days=day)
        for horizon in range(1, 25):
            rows.append(
                {
                    "origin": origin,
                    "horizon": horizon,
                    "target_time": origin + pd.Timedelta(hours=horizon),
                    "actual": 10.0,
                    "direct_raw": 20.0,
                    "d1": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_gate_falls_back_on_first_origin_and_uses_only_closed_targets():
    result = causal_gate(
        _two_day_frame(),
        experts=("direct_raw", "d1"),
        calibrated_lookbacks=(),
    )

    first = result[result["origin"].eq(pd.Timestamp("2026-06-17", tz="UTC"))]
    second = result[result["origin"].eq(pd.Timestamp("2026-06-18", tz="UTC"))]
    assert set(first["selected_expert"]) == {"direct_raw"}
    assert set(second["selected_expert"]) == {"d1"}
    # h24 from the first origin lands exactly at the second origin, so strict
    # target_time < origin leaves only h22 and h23 in that block.
    assert second.loc[second["horizon"].between(22, 24), "history_rows"].eq(2).all()


def test_future_labels_cannot_change_an_earlier_decision():
    frame = _two_day_frame()
    baseline = causal_gate(
        frame, experts=("direct_raw", "d1"), calibrated_lookbacks=()
    )
    changed = frame.copy()
    changed.loc[changed["origin"].eq(pd.Timestamp("2026-06-18", tz="UTC")), "actual"] = 1_000.0
    rerun = causal_gate(
        changed, experts=("direct_raw", "d1"), calibrated_lookbacks=()
    )
    second = baseline["origin"].eq(pd.Timestamp("2026-06-18", tz="UTC"))
    assert np.array_equal(
        baseline.loc[second, "selected_expert"], rerun.loc[second, "selected_expert"]
    )
    assert np.allclose(
        baseline.loc[second, "prediction_convex_blend"],
        rerun.loc[second, "prediction_convex_blend"],
    )


def test_recent_scaling_is_shrunk_toward_one():
    result = causal_gate(
        _two_day_frame(),
        experts=("direct_raw", "d1"),
        calibrated_lookbacks=(14,),
        scale_shrink=0.5,
    )
    second = result[result["origin"].eq(pd.Timestamp("2026-06-18", tz="UTC"))]
    # The unconstrained optimum for direct_raw would be 0.5, outside the
    # search grid.  The grid reaches 0.75 and shrink gives 0.875.
    assert np.allclose(second["direct_raw_scale_14d"], 0.875)
    assert np.allclose(second["d1_scale_14d"], 1.0)


def test_d1_excludes_hour_labelled_exactly_at_origin(tmp_path):
    index = pd.date_range("2026-06-16", periods=49, freq="1h", tz="UTC")
    carbon = pd.DataFrame(
        {"carbon_intensity_gco2_kwh": np.arange(len(index), dtype=float)},
        index=index,
    )
    path = tmp_path / "carbon.parquet"
    carbon.to_parquet(path)
    origin = pd.Timestamp("2026-06-17", tz="UTC")
    targets = pd.Series([origin + pd.Timedelta(hours=23), origin + pd.Timedelta(hours=24)])
    origins = pd.Series([origin, origin])

    values = _load_d1(path, targets, origins)

    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_physical_d1_variants_fallback_at_origin(tmp_path):
    origin = pd.Timestamp("2026-06-17", tz="UTC")
    target_times = [origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=24)]
    direct = pd.DataFrame(
        {
            "origin": [origin, origin],
            "horizon": [1, 24],
            "target_time": target_times,
            "actual": [10.0, 10.0],
            "prediction_raw": [10.0, 10.0],
        }
    )
    direct_path = tmp_path / "direct.parquet"
    direct.to_parquet(direct_path, index=False)
    source_times = pd.DatetimeIndex(target_times) - pd.Timedelta(hours=24)
    carbon = pd.DataFrame(
        {
            "carbon_intensity_gco2_kwh": [9.0, 9.0],
            "bioenergy_mw": [200.0, 900.0],
            "fuel_oil_mw": [20.0, 90.0],
        },
        index=source_times,
    )
    carbon_path = tmp_path / "carbon.parquet"
    carbon.to_parquet(carbon_path)
    physical_dir = tmp_path / "physical"
    physical_dir.mkdir()
    physical = direct[["origin", "horizon", "target_time"]].copy()
    physical["physical_alpha2"] = 10.0
    physical["predicted_bioenergy_mw"] = 100.0
    physical["predicted_fuel_oil_mw"] = 10.0
    physical["predicted_total_generation_mw"] = 1_000.0
    physical.to_parquet(physical_dir / "predictions.parquet", index=False)

    frame, experts, status = load_expert_predictions(
        direct_path,
        carbon_path,
        physical_dir,
        physical_d1_variants=True,
    )

    assert status["physical_complete"]
    assert "physical_alpha2_bio_d1" in experts
    assert np.isclose(frame.loc[0, "physical_alpha2_bio_d1"], 59.4)
    assert np.isclose(frame.loc[0, "physical_alpha2_oil_bio_d1"], 67.17)
    # h24's nominal D-1 source is exactly the origin, hence both components
    # fall back to their physical-model values and leave the base unchanged.
    assert np.isclose(frame.loc[1, "physical_alpha2_bio_d1"], 10.0)
    assert np.isclose(frame.loc[1, "physical_alpha2_oil_bio_d1"], 10.0)


def test_loader_accepts_realtime_runner_direct_and_embedded_gap_d1(tmp_path):
    origin = pd.Timestamp("2026-06-17", tz="UTC")
    direct = pd.DataFrame(
        {
            "origin": [origin],
            "horizon": [1],
            "target_time": [origin + pd.Timedelta(hours=1)],
            "actual": [12.0],
            "direct": [11.0],
            "d1": [10.0],
        }
    )
    direct_path = tmp_path / "direct.parquet"
    direct.to_parquet(direct_path, index=False)
    # This file deliberately contains another value.  The embedded D-1 is the
    # one constructed with the opt-in Energy-Charts state bridge and must win.
    carbon_path = tmp_path / "carbon.parquet"
    pd.DataFrame(
        {"carbon_intensity_gco2_kwh": [99.0]},
        index=pd.DatetimeIndex([origin - pd.Timedelta(hours=23)]),
    ).to_parquet(carbon_path)

    frame, experts, status = load_expert_predictions(
        direct_path, carbon_path, physical_directory=None
    )

    assert experts == ["direct_raw", "d1"]
    assert frame.loc[0, "direct_raw"] == 11.0
    assert frame.loc[0, "d1"] == 10.0
    assert status["d1_source"] == "embedded prediction input"


def test_complete_physical_input_can_supply_gap_d1(tmp_path):
    origin = pd.Timestamp("2026-06-17", tz="UTC")
    target = origin + pd.Timedelta(hours=1)
    direct = pd.DataFrame(
        {
            "origin": [origin],
            "horizon": [1],
            "target_time": [target],
            "actual": [12.0],
            "prediction_raw": [11.0],
        }
    )
    direct_path = tmp_path / "direct.parquet"
    direct.to_parquet(direct_path, index=False)
    carbon_path = tmp_path / "carbon.parquet"
    pd.DataFrame(
        {"carbon_intensity_gco2_kwh": [99.0]},
        index=pd.DatetimeIndex([target - pd.Timedelta(hours=24)]),
    ).to_parquet(carbon_path)
    physical_dir = tmp_path / "physical"
    physical_dir.mkdir()
    physical = direct[["origin", "horizon", "target_time"]].copy()
    physical["physical_alpha2"] = 10.5
    physical["predicted_bioenergy_mw"] = 100.0
    physical["predicted_fuel_oil_mw"] = 10.0
    physical["predicted_total_generation_mw"] = 1_000.0
    physical["d1"] = 8.0
    physical.to_parquet(physical_dir / "predictions.parquet", index=False)

    frame, _, status = load_expert_predictions(
        direct_path, carbon_path, physical_dir
    )

    assert frame.loc[0, "d1"] == 8.0
    assert status["d1_source"] == "embedded physical prediction input"
