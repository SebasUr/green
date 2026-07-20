"""Public Energy-Charts day-ahead price snapshots.

Prices are indexed by their delivery instant, not by publication vintage.  At
the project's midnight UTC origin the current *local* delivery-day auction is
already known, but the next local day's auction is not.  Consumers must mask
targets that cross that local-date boundary; ``RegimeMoEFeatureBuilder`` does
so before using this snapshot as an exogenous feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pandas as pd


DEFAULT_URL = "https://api.energy-charts.info/price"
PRICE_COLUMN = "day_ahead_price_eur_mwh"


class EnergyChartsPriceProvider:
    def __init__(self, *, base_url: str = DEFAULT_URL, timeout: float = 180.0):
        self.base_url = base_url
        self.timeout = float(timeout)

    def fetch(
        self,
        start: str,
        end: str,
        *,
        bidding_zone: str = "FR",
        chunk_days: int = 180,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("price end must not precede start")
        parts: list[pd.DataFrame] = []
        with httpx.Client(timeout=self.timeout) as client:
            cursor = start_ts
            while cursor <= end_ts:
                chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
                response = client.get(
                    self.base_url,
                    params={
                        "bzn": bidding_zone,
                        "start": cursor.strftime("%Y-%m-%d"),
                        "end": chunk_end.strftime("%Y-%m-%d"),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                timestamps = payload.get("unix_seconds", [])
                prices = payload.get("price", [])
                if len(timestamps) != len(prices):
                    raise ValueError("Energy-Charts price arrays have different lengths")
                parts.append(
                    pd.DataFrame(
                        {
                            PRICE_COLUMN: pd.to_numeric(prices, errors="coerce")
                        },
                        index=pd.to_datetime(timestamps, unit="s", utc=True),
                    )
                )
                print(
                    f"Energy-Charts {bidding_zone}: {cursor.date()}.."
                    f"{chunk_end.date()} rows={len(timestamps)}"
                )
                cursor = chunk_end + pd.Timedelta(days=1)
        if not parts:
            return pd.DataFrame(columns=[PRICE_COLUMN])
        raw = pd.concat(parts).sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        # The market changed from hourly to quarter-hourly products; a common
        # hourly mean matches the carbon target resolution in both eras.
        hourly = raw.resample("1h").mean().dropna(how="all")
        hourly.index.name = "target_time"
        return hourly

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bidding-zone", default="FR")
    parser.add_argument("--chunk-days", type=int, default=180)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = EnergyChartsPriceProvider()
    frame = provider.fetch(
        args.start,
        args.end,
        bidding_zone=args.bidding_zone,
        chunk_days=args.chunk_days,
    )
    provider.save_snapshot(frame, args.output)
    print(f"saved {len(frame)} hourly prices -> {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["EnergyChartsPriceProvider", "PRICE_COLUMN"]
