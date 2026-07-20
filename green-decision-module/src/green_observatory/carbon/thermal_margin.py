"""Day-ahead implied thermal-margin features (cheap merit-order proxy).

The gas-ramp bottleneck needs a prospective scarcity signal. Until ENTSO-E
A71 (total scheduled generation) is available, this module derives the same
physics from inputs already cached and causal:

    residual demand forecast   (D-1 load - wind - solar, published D-1)
  + unavailable capacity       (RTE outage messages, publication-versioned)
  = implied tightness          (higher -> dispatchable margin is thinner ->
                                gas/hydro must fire)

Vintage rule: for every delivery hour, outage state uses the latest version
of each message published strictly before the **Paris midnight that starts
the delivery day**. That instant precedes every 00:00 UTC origin of the
daily protocol, so the columns are causal wherever the feature builder's
``day_ahead`` mask exposes them (target hours on the origin's local day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARIS_TZ = "Europe/Paris"

NUCLEAR_FUELS = frozenset({"NUCLEAR"})
#: Everything the operator can dispatch against residual demand.
DISPATCHABLE_FUELS = frozenset(
    {
        "NUCLEAR",
        "HYDRO_WATER_RESERVOIR",
        "HYDRO_RUN_OF_RIVER_AND_POUNDAGE",
        "HYDRO_PUMPED_STORAGE",
        "FOSSIL_GAS",
        "FOSSIL_OIL",
        "FOSSIL_HARD_COAL",
    }
)

RESIDUAL_COLUMNS = (
    "load_day_ahead_forecast_mw",
    "wind_onshore_day_ahead_forecast_mw",
    "wind_offshore_day_ahead_forecast_mw",
    "solar_day_ahead_forecast_mw",
)

#: ``identifier`` is the stable outage identity; ``message_id`` embeds the
#: version suffix (``..._003``) and must NOT be used for deduplication.
_REQUIRED_INTERVAL_COLUMNS = [
    "publication_date",
    "identifier",
    "event_status",
    "fuel_type",
    "interval_start",
    "interval_end",
    "unavailable_capacity_mw",
]


def unavailable_capacity_hourly(
    intervals: pd.DataFrame,
    hours: pd.DatetimeIndex,
    fuel_groups: dict[str, frozenset[str]],
) -> pd.DataFrame:
    """Hourly unavailable MW per fuel group at D-1-midnight-Paris vintage.

    For each Paris delivery day, the applicable state of every outage
    message is its latest version published strictly before that day's
    local midnight; versions whose status is not ACTIVE cancel the message.
    """
    messages = intervals.loc[:, _REQUIRED_INTERVAL_COLUMNS].dropna(
        subset=["publication_date", "interval_start", "interval_end"]
    )
    all_fuels = frozenset().union(*fuel_groups.values())
    messages = messages[messages["fuel_type"].isin(all_fuels)]
    messages = messages.sort_values("publication_date", kind="stable").reset_index(
        drop=True
    )
    publication_ns = pd.DatetimeIndex(messages["publication_date"]).as_unit("ns").asi8

    # Parquet round-trips can yield ms-resolution indexes; force ns so the
    # asi8 comparisons against the ns-resolution intervals are homogeneous.
    hours = pd.DatetimeIndex(hours).sort_values().as_unit("ns")
    day_start = (
        hours.tz_convert(PARIS_TZ).normalize().tz_convert("UTC")
    )
    out = {name: np.zeros(len(hours)) for name in fuel_groups}
    hour_ns = hours.asi8

    for cutoff in day_start.unique():
        prefix_len = int(np.searchsorted(publication_ns, cutoff.value, side="left"))
        if prefix_len == 0:
            continue
        prefix = messages.iloc[:prefix_len]
        # Latest version per outage; a version may span several interval
        # rows sharing one publication_date, so filter rather than dedup.
        last_publication = prefix.groupby("identifier")["publication_date"].transform(
            "max"
        )
        applicable = prefix[prefix["publication_date"] == last_publication]
        active = applicable[applicable["event_status"] == "ACTIVE"]
        day_mask = day_start == cutoff
        day_first = int(np.argmax(day_mask))
        day_hours = hour_ns[day_mask]
        if not len(day_hours):
            continue
        start = np.maximum(
            pd.DatetimeIndex(active["interval_start"]).as_unit("ns").asi8, day_hours[0]
        )
        end = pd.DatetimeIndex(active["interval_end"]).as_unit("ns").asi8
        lo = np.searchsorted(day_hours, start, side="left")
        hi = np.searchsorted(day_hours, end, side="left")
        mw = active["unavailable_capacity_mw"].to_numpy(dtype=float)
        fuel = active["fuel_type"].to_numpy()
        for name, group in fuel_groups.items():
            acc = np.zeros(len(day_hours) + 1)
            in_group = np.isin(fuel, list(group)) & (hi > lo) & np.isfinite(mw)
            np.add.at(acc, lo[in_group], mw[in_group])
            np.add.at(acc, hi[in_group], -mw[in_group])
            out[name][day_first : day_first + len(day_hours)] += np.cumsum(acc[:-1])

    return pd.DataFrame(out, index=hours)


def day_ahead_thermal_margin_features(
    forecast_frame: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    """Target-time tightness columns for the feature builder's forecast frame.

    Column names deliberately contain ``day_ahead`` so the builder masks
    target hours on the next local delivery day (whose D-1 vintage would
    postdate the origin).
    """
    missing = [c for c in RESIDUAL_COLUMNS if c not in forecast_frame.columns]
    if missing:
        raise ValueError(f"forecast frame lacks residual-demand columns: {missing}")
    index = pd.DatetimeIndex(forecast_frame.index).as_unit("ns")
    load, wind_on, wind_off, solar = (
        forecast_frame[c].astype(float).set_axis(index) for c in RESIDUAL_COLUMNS
    )
    residual = load - wind_on - wind_off - solar

    unavailable = unavailable_capacity_hourly(
        intervals,
        index,
        {"nuclear": NUCLEAR_FUELS, "dispatchable": DISPATCHABLE_FUELS},
    )
    out = pd.DataFrame(index=index)
    out["residual_demand_day_ahead_mw"] = residual
    out["nuclear_unavailable_day_ahead_mw"] = unavailable["nuclear"]
    out["dispatchable_unavailable_day_ahead_mw"] = unavailable["dispatchable"]
    out["thermal_tightness_day_ahead_mw"] = residual + unavailable["nuclear"]
    out["thermal_tightness_wide_day_ahead_mw"] = residual + unavailable["dispatchable"]
    tightness = out["thermal_tightness_day_ahead_mw"]
    out["thermal_tightness_delta_day_ahead_mw"] = tightness - tightness.shift(
        freq="24h"
    ).reindex(out.index)
    return out
