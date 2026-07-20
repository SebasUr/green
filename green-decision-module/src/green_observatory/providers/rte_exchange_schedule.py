"""RTE Exchange Schedule v2 provider (scheduled commercial exchanges).

Publication-versioned cross-border exchange *programs* for the French hub.
Unlike realized physical flows, the day-ahead (``DA``) program for delivery
day ``D`` is published on ``D-1`` around 13:15-13:30 UTC with an explicit
``updated_date``, so at a daily origin ``t = 00:00 UTC`` it is a genuinely
prospective, vintage-causal signal of next-day dispatch. ``LT`` (long-term)
arrives even earlier (~07:30 UTC on ``D-1``); ``ID`` (intraday) is finalized
*after* delivery and must never feed same-day features.

API notes verified against the live endpoint (2026-07):

* ``start_date``/``end_date`` must be **midnight Europe/Paris expressed in
  UTC** (``2026-07-16T22:00:00Z`` style); anything else returns
  ``EXCHSCHED_SCHED_F08``.
* Maximum 7 days per call; history depth back to 2020-10-01.
* The live response root is ``time_series`` (the sandbox's ``schedules``
  wrapper does not appear on real data).
* Values are 15-min MW points per (sender, receiver) direction; both
  directions of each border are separate series.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pandas as pd

from green_observatory.providers.rte_system_forecast import RteSystemForecastProvider

SCHEDULES_PATH = "/open_api/exchange_schedule/v2/schedules"
PARIS_TZ = "Europe/Paris"
MAX_CHUNK_DAYS = 7

#: Country/interconnector EIC -> short border label (FR perspective).
EIC_LABELS = {
    "10YFR-RTE------C": "FR",
    "10YIT-GRTN-----B": "IT",
    "10YBE----------2": "BE",
    "10YES-REE------0": "ES",
    "10YCH-SWISSGRIDZ": "CH",
    "10YCB-GERMANY--8": "DE",
    "10Y1001C--000255": "GB_IFA",
    "10Y1001C--000344": "GB_ELECLINK",
    "10Y1001C--000263": "GB_IFA2",
}


def _paris_midnight(value: pd.Timestamp | str) -> pd.Timestamp:
    """Snap an instant/date to midnight Europe/Paris, returned as UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(PARIS_TZ)
    else:
        ts = ts.tz_convert(PARIS_TZ)
    return ts.normalize().tz_convert("UTC")


def _rte_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_paris_chunks(start, end, chunk_days: int = MAX_CHUNK_DAYS):
    """Yield [a, b) chunks stepped on the Paris calendar (DST-safe).

    The API rejects any boundary that is not midnight Europe/Paris, so chunk
    edges must be advanced in local calendar days and re-normalized: adding
    7 absolute days across a DST switch would land on 23:00/01:00 local.
    """
    cursor = _paris_midnight(start).tz_convert(PARIS_TZ)
    stop = _paris_midnight(end).tz_convert(PARIS_TZ)
    while cursor < stop:
        chunk_end = min((cursor + pd.Timedelta(days=chunk_days)).normalize(), stop)
        yield cursor, chunk_end
        cursor = chunk_end


class RteExchangeScheduleProvider(RteSystemForecastProvider):
    """Fetch scheduled commercial exchange programs with their vintages."""

    def fetch_schedules(
        self,
        start,
        end,
        *,
        process_type: str = "DA",
        progress: bool = False,
    ) -> pd.DataFrame:
        if _paris_midnight(end) <= _paris_midnight(start):
            raise ValueError("exchange-schedule end must be after start")
        rows: list[dict] = []
        with httpx.Client(timeout=self.timeout) as client:
            for cursor, chunk_end in iter_paris_chunks(start, end):
                payload, _ = self._get(
                    client,
                    SCHEDULES_PATH,
                    {
                        "start_date": _rte_utc(cursor),
                        "end_date": _rte_utc(chunk_end),
                        "process_type": process_type,
                    },
                )
                series = payload.get("time_series", [])
                rows.extend(self.normalize_schedules(series, process_type))
                if progress:
                    print(
                        f"  RTE exchange {process_type}: "
                        f"{cursor.date()}..{chunk_end.date()} ({len(series)} series)"
                    )
        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["value_start", "border", "direction"]).reset_index(
            drop=True
        )

    @staticmethod
    def normalize_schedules(series: list[dict], process_type: str) -> list[dict]:
        """Flatten time_series payloads to long (one row per 15-min value)."""
        rows: list[dict] = []
        for ts in series:
            sender = ts.get("sender_country_eic_code", "")
            receiver = ts.get("receiver_country_eic_code", "")
            if EIC_LABELS.get(sender) == "FR":
                border, direction = EIC_LABELS.get(receiver, receiver), "export"
            else:
                border, direction = EIC_LABELS.get(sender, sender), "import"
            updated = pd.Timestamp(ts["updated_date"])
            for value in ts.get("values", []):
                rows.append(
                    {
                        "value_start": pd.Timestamp(value["start_date"]),
                        "value_end": pd.Timestamp(value["end_date"]),
                        "border": border,
                        "direction": direction,
                        "process_type": process_type,
                        "updated_date": updated,
                        "value_mw": float(value.get("value") or 0.0),
                    }
                )
        return rows

    @staticmethod
    def save_snapshot(frame: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


#: GB interconnectors are aggregated into one border for features.
_FEATURE_BORDERS = {"DE": "de", "BE": "be", "CH": "ch", "ES": "es", "IT": "it",
                    "GB_IFA": "gb", "GB_ELECLINK": "gb", "GB_IFA2": "gb"}


def day_ahead_hourly_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Long DA snapshot -> hourly target-time feature frame.

    Keeps only the **first published version** per (delivery quarter-hour,
    border, direction). The first DA publication lands ~13:15-13:30 UTC on
    D-1, i.e. before the Paris midnight that starts delivery day D, so the
    result is available at any 00:00 UTC origin for target hours on the
    origin's local delivery day; later target hours are the feature
    builder's job to mask (column names carry ``day_ahead`` on purpose).
    Any value whose first publication is *not* before its Paris delivery-day
    start is dropped as a safety net.
    """
    df = frame.copy()
    df = df.sort_values("updated_date").drop_duplicates(
        ["value_start", "border", "direction"], keep="first"
    )
    paris_day_start = (
        df["value_start"].dt.tz_convert(PARIS_TZ).dt.normalize().dt.tz_convert("UTC")
    )
    df = df[df["updated_date"] < paris_day_start]
    df["group"] = df["border"].map(_FEATURE_BORDERS)
    df = df.dropna(subset=["group"])
    df["hour"] = df["value_start"].dt.floor("h")
    per_direction = (
        df.groupby(["hour", "border", "direction"])["value_mw"]
        .mean()
        .unstack("direction")
        .fillna(0.0)
    )
    net = per_direction.get("import", 0.0) - per_direction.get("export", 0.0)
    net = net.reset_index(name="net_mw")
    net["group"] = net["border"].map(_FEATURE_BORDERS)
    hourly = net.groupby(["hour", "group"])["net_mw"].sum().unstack("group")
    hourly.columns = [
        f"exchange_day_ahead_net_import_{name}_mw" for name in hourly.columns
    ]
    hourly["exchange_day_ahead_net_import_total_mw"] = hourly.sum(axis=1)
    hourly.index.name = None
    return hourly.sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--process-type", default="DA", choices=("DA", "LT", "ID"))
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = RteExchangeScheduleProvider.from_env(dotenv_path=args.dotenv)
    frame = provider.fetch_schedules(
        args.start, args.end, process_type=args.process_type, progress=True
    )
    provider.save_snapshot(frame, args.output)
    if frame.empty:
        print("saved 0 rows")
        return
    print(
        f"saved {len(frame)} rows -> {args.output}  "
        f"[{frame['value_start'].min()}..{frame['value_end'].max()}] "
        f"borders={sorted(frame['border'].unique())}"
    )


if __name__ == "__main__":
    main()
