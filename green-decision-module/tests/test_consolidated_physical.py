import numpy as np
import pandas as pd

from green_observatory.carbon.consolidated_physical import (
    EMITTING_COMPONENTS,
    GAS_COMPONENTS,
    GENERATION_COLUMNS,
    PHYSICAL_TARGETS,
    ConsolidatedPhysicalRegressor,
    detailed_physical_targets,
)


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value)


def test_positive_ols_recovers_effective_factors():
    rng = np.random.default_rng(42)
    design = rng.uniform(0.0, 2_000.0, size=(500, len(EMITTING_COMPONENTS)))
    expected = np.asarray([994.0, 760.0, 419.0, 371.0, 350.0, 601.0, 517.0])
    fitted = ConsolidatedPhysicalRegressor._positive_ols(design, design @ expected)
    assert np.allclose(fitted, expected, rtol=1e-8, atol=1e-8)


def test_detailed_targets_sum_domestic_generation():
    index = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    values = {column: [100.0, 200.0] for column in GENERATION_COLUMNS}
    for column in EMITTING_COMPONENTS:
        values.setdefault(column, [10.0, 20.0])
    frame = pd.DataFrame(values, index=index)

    targets = detailed_physical_targets(frame, index)

    assert np.allclose(targets["total_generation_mw"], [800.0, 1_600.0])
    assert np.allclose(targets["gas_mw"], [100.0, 200.0])
    assert list(targets.loc[:, EMITTING_COMPONENTS].columns) == list(
        EMITTING_COMPONENTS
    )


def test_gas_total_reconciliation_preserves_component_proportions():
    raw_values = {
        "coal_mw": 5.0,
        "fuel_oil_mw": 5.0,
        "gas_turbine_mw": 1.0,
        "gas_cogeneration_mw": 2.0,
        "gas_ccg_mw": 3.0,
        "gas_other_mw": 4.0,
        "bioenergy_waste_mw": 5.0,
        "total_generation_mw": 100.0,
        "gas_mw": 20.0,
    }
    model = ConsolidatedPhysicalRegressor(gas_total_reconciliation=True)
    model.feature_names_ = ["feature"]
    model.emission_factors_ = np.ones(len(EMITTING_COMPONENTS))
    model.gas_component_shares_ = np.full(len(GAS_COMPONENTS), 0.25)
    model.regressors_ = {
        column: _ConstantRegressor(raw_values[column])
        for column in (*PHYSICAL_TARGETS, "gas_mw")
    }

    prediction = model.predict_matrix(pd.DataFrame({"feature": [0.0, 1.0]}))

    reconciled = prediction[
        [f"predicted_{column}" for column in GAS_COMPONENTS]
    ].to_numpy()
    assert np.allclose(reconciled.sum(axis=1), 20.0)
    assert np.allclose(reconciled / reconciled[:, [0]], [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(prediction["predicted_gas_components_raw_sum_mw"], 10.0)
    assert np.allclose(prediction["prediction_unreconciled"], 0.25)
    assert np.allclose(prediction["prediction"], 0.35)


def test_gas_total_reconciliation_is_opt_in():
    model = ConsolidatedPhysicalRegressor()
    assert model.gas_total_reconciliation is False
