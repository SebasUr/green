"""Electricity Maps provider: availability guard (no network)."""

import pytest

from green_observatory.providers.carbon_electricity_maps import ElectricityMapsProvider


def test_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("ELECTRICITYMAPS_API_TOKEN", raising=False)
    monkeypatch.delenv("ELECTRICITY_MAPS_TOKEN", raising=False)
    provider = ElectricityMapsProvider()
    assert provider.available() is False
    with pytest.raises(RuntimeError):  # guarded before any network call
        provider._get("carbon-intensity/latest", {"zone": "FR"})


def test_available_with_env_token(monkeypatch):
    monkeypatch.delenv("ELECTRICITY_MAPS_TOKEN", raising=False)
    monkeypatch.setenv("ELECTRICITYMAPS_API_TOKEN", "dummy-token")
    assert ElectricityMapsProvider().available() is True


def test_explicit_token_overrides_env(monkeypatch):
    monkeypatch.delenv("ELECTRICITYMAPS_API_TOKEN", raising=False)
    monkeypatch.delenv("ELECTRICITY_MAPS_TOKEN", raising=False)
    assert ElectricityMapsProvider(token="abc").available() is True
