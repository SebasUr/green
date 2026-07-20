"""Pure parsing/standardization tests for the ODRE provider (no network)."""

import pandas as pd

from green_observatory.providers.carbon_base import CANONICAL_COLUMNS, CARBON, TIMESTAMP
from green_observatory.providers.carbon_odre import OdreCarbonProvider

RAW = [
    {"date_heure": "2026-01-15T10:00:00+00:00", "taux_co2": 35, "consommation": 60000,
     "nucleaire": 45000, "solaire": 100, "ech_physiques": -5000,
     "gaz_ccg": 1200, "hydraulique_lacs": 900, "ech_comm_espagne": -300},
    {"date_heure": "2026-01-15T10:15:00+00:00", "taux_co2": None, "consommation": None,
     "nucleaire": None, "solaire": None, "ech_physiques": None},
    {"date_heure": "2026-01-15T10:30:00+00:00", "taux_co2": 33, "consommation": 60500,
     "nucleaire": 45100, "solaire": 120, "ech_physiques": -5200},
    {"date_heure": "2026-01-15T11:00:00+00:00", "taux_co2": 31, "consommation": 61000,
     "nucleaire": 45200, "solaire": 300, "ech_physiques": -5300},
    {"date_heure": "2026-01-15T11:30:00+00:00", "taux_co2": 29, "consommation": 61200,
     "nucleaire": 45300, "solaire": 500, "ech_physiques": -5400},
]


def test_standardize_canonical_shape_and_utc_index():
    df = OdreCarbonProvider.parse_records(RAW)
    assert df.index.name == TIMESTAMP
    assert str(df.index.tz) == "UTC"
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df.index.is_monotonic_increasing


def test_field_mapping_and_export_sign_convention():
    df = OdreCarbonProvider.parse_records(RAW)
    row = df.loc["2026-01-15T10:00:00+00:00"]
    assert row[CARBON] == 35
    assert row["consumption_mw"] == 60000
    assert row["nuclear_mw"] == 45000
    assert row["physical_exchange_mw"] == -5000  # negative = export from France
    assert row["gas_ccg_mw"] == 1200
    assert row["hydro_reservoir_mw"] == 900
    assert row["commercial_exchange_es_mw"] == -300


def test_hourly_aggregation_averages_halfhour_points():
    df = OdreCarbonProvider.parse_records(RAW)
    df = df[df[CARBON].notna()]
    hourly = OdreCarbonProvider.to_hourly(df)
    # 10:00 hour = mean(35, 33) = 34 ; 11:00 hour = mean(31, 29) = 30
    assert hourly.loc["2026-01-15T10:00:00+00:00", CARBON] == 34.0
    assert hourly.loc["2026-01-15T11:00:00+00:00", CARBON] == 30.0
    assert hourly.loc["2026-01-15T10:00:00+00:00", "solar_mw"] == 110.0


def test_missing_columns_filled_with_nan():
    df = OdreCarbonProvider.parse_records(
        [{"date_heure": "2026-01-15T10:00:00+00:00", "taux_co2": 20}]
    )
    assert list(df.columns) == CANONICAL_COLUMNS
    assert pd.isna(df.iloc[0]["gas_mw"])
    assert df.iloc[0][CARBON] == 20


def test_snapshot_roundtrip(tmp_path):
    df = OdreCarbonProvider.parse_records(RAW)
    path = tmp_path / "snap.parquet"
    OdreCarbonProvider.save_snapshot(df, path)
    loaded = OdreCarbonProvider.load_snapshot(path)
    assert str(loaded.index.tz) == "UTC"
    assert list(loaded.columns) == CANONICAL_COLUMNS
    assert loaded[CARBON].dropna().tolist() == df[CARBON].dropna().tolist()


def test_empty_records_yield_empty_canonical_frame():
    df = OdreCarbonProvider.parse_records([])
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 0
