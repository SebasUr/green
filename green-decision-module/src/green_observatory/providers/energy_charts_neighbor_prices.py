"""French-neighbour day-ahead prices and causal market-spread features.

Neighbouring market prices are a compact proxy for the cross-border merit
order and expected exchange pressure.  The output remains indexed by delivery
time; consumers must apply the same local delivery-day visibility mask as for
the French day-ahead price.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from green_observatory.providers.energy_charts_price import (
    EnergyChartsPriceProvider,
    PRICE_COLUMN,
)


DEFAULT_ZONES = ("BE", "DE-LU", "ES", "CH")


def _slug(zone: str) -> str:
    return zone.lower().replace("-", "_")


class EnergyChartsNeighborPriceProvider:
    """Download neighbour prices and derive France-minus-neighbour spreads."""

    def __init__(self, *, timeout: float = 300.0) -> None:
        self.price_provider = EnergyChartsPriceProvider(timeout=timeout)

    def fetch(
        self,
        start: str,
        end: str,
        *,
        france_price: pd.DataFrame,
        zones: tuple[str, ...] = DEFAULT_ZONES,
        chunk_days: int = 5000,
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for zone in zones:
            slug = _slug(zone)
            frame = self.price_provider.fetch(
                start,
                end,
                bidding_zone=zone,
                chunk_days=chunk_days,
            ).rename(columns={PRICE_COLUMN: f"day_ahead_price_{slug}_eur_mwh"})
            parts.append(frame)
        if not parts:
            return pd.DataFrame()
        neighbours = pd.concat(parts, axis=1).sort_index()
        french = france_price[PRICE_COLUMN].reindex(neighbours.index)
        for zone in zones:
            slug = _slug(zone)
            neighbours[f"day_ahead_price_spread_fr_minus_{slug}_eur_mwh"] = (
                french - neighbours[f"day_ahead_price_{slug}_eur_mwh"]
            )
        neighbours.index.name = "target_time"
        return neighbours

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--fr-price", required=True)
    parser.add_argument("--zones", default=",".join(DEFAULT_ZONES))
    parser.add_argument("--chunk-days", type=int, default=5000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    france = pd.read_parquet(args.fr_price)
    zones = tuple(zone.strip() for zone in args.zones.split(",") if zone.strip())
    provider = EnergyChartsNeighborPriceProvider()
    frame = provider.fetch(
        args.start,
        args.end,
        france_price=france,
        zones=zones,
        chunk_days=args.chunk_days,
    )
    provider.save_snapshot(frame, args.output)
    coverage = {
        column: round(float(frame[column].notna().mean()), 3) for column in frame
    }
    print(f"saved {len(frame)} hourly neighbour prices -> {args.output}")
    print(f"coverage={coverage}")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_ZONES", "EnergyChartsNeighborPriceProvider"]
