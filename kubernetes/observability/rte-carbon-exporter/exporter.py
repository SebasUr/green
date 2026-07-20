#!/usr/bin/env python3
"""Zero-dependency Prometheus exporter for RTE/eCO2mix carbon intensity.

Makes the French grid's *production-based* carbon intensity (RTE ``taux_co2``)
available as a live Prometheus gauge, so Grafana can show real-time operational
emissions (Kepler watts x live intensity) next to energy.

This is the same ground-truth signal the green_observatory package uses offline
(dataset ``eco2mix-national-tr``, field ``taux_co2``, gCO2eq/kWh, UTC). It is
kept dependency-free (Python standard library only) so it runs on a stock
``python:3.12-slim`` image mounted from a ConfigMap — no image build required.

Exposed metrics (text exposition on ``/metrics``):

    rte_carbon_intensity_gco2eq_per_kwh{zone,basis,source}   latest taux_co2
    rte_carbon_intensity_point_timestamp_seconds{...}        UTC epoch of that point
    rte_carbon_exporter_up                                   1 = fresh, 0 = stale/failed
    rte_carbon_exporter_last_success_timestamp_seconds       last successful fetch
    rte_carbon_exporter_fetch_failures_total                 cumulative fetch failures

Derived in PromQL, not here:
    intensity age  = time() - rte_carbon_intensity_point_timestamp_seconds
    emission rate  = kepler_..._watts * on() group_left() rte_carbon_intensity_gco2eq_per_kwh / 1000  (gCO2eq/h)

Config via environment:
    ODRE_BASE_URL        default https://odre.opendatasoft.com/api/explore/v2.1
    ODRE_DATASET         default eco2mix-national-tr
    ZONE                 default FR   (label only)
    REFRESH_SECONDS      default 300  (ODRE publishes every 15-30 min)
    MAX_AGE_SECONDS      default 5400 (mark `up=0` if the newest point is older)
    HTTP_TIMEOUT_SECONDS default 30
    LISTEN_PORT          default 9110
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_URL = os.getenv("ODRE_BASE_URL", "https://odre.opendatasoft.com/api/explore/v2.1").rstrip("/")
DATASET = os.getenv("ODRE_DATASET", "eco2mix-national-tr")
ZONE = os.getenv("ZONE", "FR")
REFRESH_SECONDS = float(os.getenv("REFRESH_SECONDS", "300"))
MAX_AGE_SECONDS = float(os.getenv("MAX_AGE_SECONDS", "5400"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "9110"))

LABELS = f'zone="{ZONE}",basis="production",source="rte-eco2mix"'


class State:
    """Latest known carbon point plus exporter health, guarded by a lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.intensity: float | None = None
        self.point_epoch: float | None = None
        self.last_success_epoch: float | None = None
        self.failures = 0

    def update(self, intensity: float, point_epoch: float) -> None:
        with self.lock:
            self.intensity = intensity
            self.point_epoch = point_epoch
            self.last_success_epoch = time.time()

    def fail(self) -> None:
        with self.lock:
            self.failures += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "intensity": self.intensity,
                "point_epoch": self.point_epoch,
                "last_success_epoch": self.last_success_epoch,
                "failures": self.failures,
            }


STATE = State()


def fetch_latest() -> tuple[float, float]:
    """Return ``(intensity_gco2eq_per_kwh, point_utc_epoch)`` for the newest point."""
    params = urllib.parse.urlencode({
        "select": "date_heure,taux_co2",
        "where": "taux_co2 IS NOT NULL",
        "order_by": "date_heure DESC",
        "limit": "1",
        "timezone": "UTC",
    })
    url = f"{BASE_URL}/catalog/datasets/{DATASET}/records?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "rte-carbon-exporter/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    results = payload.get("results") or []
    if not results:
        raise ValueError("ODRE returned no populated taux_co2 record")
    record = results[0]
    intensity = float(record["taux_co2"])
    stamp = record["date_heure"].replace("Z", "+00:00")
    point_epoch = datetime.fromisoformat(stamp).astimezone(timezone.utc).timestamp()
    return intensity, point_epoch


def refresh_loop() -> None:
    while True:
        try:
            intensity, point_epoch = fetch_latest()
            STATE.update(intensity, point_epoch)
            print(f"[rte-exporter] {ZONE} taux_co2={intensity} gCO2eq/kWh "
                  f"point={datetime.fromtimestamp(point_epoch, timezone.utc).isoformat()}",
                  flush=True)
        except (urllib.error.URLError, ValueError, KeyError, OSError) as exc:
            STATE.fail()
            print(f"[rte-exporter] fetch failed: {exc}", flush=True)
        time.sleep(REFRESH_SECONDS)


def render_metrics() -> str:
    snap = STATE.snapshot()
    now = time.time()
    fresh = (
        snap["intensity"] is not None
        and snap["point_epoch"] is not None
        and (now - snap["point_epoch"]) <= MAX_AGE_SECONDS
    )
    lines = [
        "# HELP rte_carbon_intensity_gco2eq_per_kwh RTE/eCO2mix production-based carbon intensity.",
        "# TYPE rte_carbon_intensity_gco2eq_per_kwh gauge",
    ]
    if snap["intensity"] is not None:
        lines.append(f'rte_carbon_intensity_gco2eq_per_kwh{{{LABELS}}} {snap["intensity"]}')
    lines += [
        "# HELP rte_carbon_intensity_point_timestamp_seconds UTC time of the latest carbon point.",
        "# TYPE rte_carbon_intensity_point_timestamp_seconds gauge",
    ]
    if snap["point_epoch"] is not None:
        lines.append(f'rte_carbon_intensity_point_timestamp_seconds{{{LABELS}}} {snap["point_epoch"]}')
    lines += [
        "# HELP rte_carbon_exporter_up 1 if the latest carbon point is present and within MAX_AGE_SECONDS.",
        "# TYPE rte_carbon_exporter_up gauge",
        f"rte_carbon_exporter_up {1 if fresh else 0}",
        "# HELP rte_carbon_exporter_last_success_timestamp_seconds Last successful ODRE fetch (UTC epoch).",
        "# TYPE rte_carbon_exporter_last_success_timestamp_seconds gauge",
        f'rte_carbon_exporter_last_success_timestamp_seconds {snap["last_success_epoch"] or 0}',
        "# HELP rte_carbon_exporter_fetch_failures_total Cumulative ODRE fetch failures.",
        "# TYPE rte_carbon_exporter_fetch_failures_total counter",
        f'rte_carbon_exporter_fetch_failures_total {snap["failures"]}',
    ]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.startswith("/metrics"):
            body = render_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/healthz"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args) -> None:  # silence per-request logging
        return


def main() -> None:
    threading.Thread(target=refresh_loop, daemon=True).start()
    server = ThreadingHTTPServer(("", LISTEN_PORT), Handler)
    print(f"[rte-exporter] serving /metrics on :{LISTEN_PORT} "
          f"(dataset={DATASET}, refresh={REFRESH_SECONDS}s, zone={ZONE})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
