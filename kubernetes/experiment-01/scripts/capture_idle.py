#!/usr/bin/env python3
"""Capture a reproducible Kepler/Prometheus idle baseline.

The collector uses only the Python standard library plus kubectl.  It opens a
temporary Prometheus port-forward unless --prometheus-url is supplied, records
Kubernetes state at both ends of the window, downloads raw range-query results,
and produces CSV/JSON summaries without discarding the source samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
DEFAULT_NODE_SELECTOR = "sustainability.cern.ch/hardware=baremetal"


def utc_iso(epoch: float | None = None) -> str:
    value = datetime.fromtimestamp(epoch or time.time(), tz=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def utc_slug(epoch: float | None = None) -> str:
    value = datetime.fromtimestamp(epoch or time.time(), tz=timezone.utc)
    return value.strftime("%Y%m%dT%H%M%SZ")


def parse_duration(value: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600}
    text = value.strip().lower()
    try:
        if text[-1] in units:
            seconds = float(text[:-1]) * units[text[-1]]
        else:
            seconds = float(text)
    except (IndexError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "usa segundos o un sufijo s/m/h, por ejemplo 900 o 15m"
        ) from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("la duración debe ser positiva")
    return int(seconds)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class CommandError(RuntimeError):
    pass


class Kubectl:
    def __init__(self, kubeconfig: str | None) -> None:
        self.prefix = ["kubectl"]
        if kubeconfig:
            self.prefix.extend(["--kubeconfig", os.path.expanduser(kubeconfig)])

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [*self.prefix, *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if check and proc.returncode:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise CommandError(f"kubectl {' '.join(args)}: {detail}")
        return proc

    def json(self, *args: str) -> Any:
        proc = self.run(*args, "-o", "json")
        return json.loads(proc.stdout)


class Prometheus:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"no se pudo consultar Prometheus: {url}: {exc}") from exc
        if path in ("/-/ready", "/-/healthy"):
            return body
        payload = json.loads(body)
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus devolvió un error para {path}: {payload}")
        return payload

    def query(self, expression: str, at: float | None = None) -> Any:
        params: dict[str, Any] = {"query": expression}
        if at is not None:
            params["time"] = f"{at:.3f}"
        return self.get("/api/v1/query", params)

    def query_range(
        self, expression: str, start: float, end: float, step: int
    ) -> Any:
        return self.get(
            "/api/v1/query_range",
            {
                "query": expression,
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "step": str(step),
            },
        )


class PortForward:
    def __init__(
        self,
        kubectl: Kubectl,
        namespace: str,
        service: str,
        remote_port: int,
    ) -> None:
        self.kubectl = kubectl
        self.namespace = namespace
        self.service = service
        self.remote_port = remote_port
        self.local_port = self._free_port()
        self.process: subprocess.Popen[str] | None = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def __enter__(self) -> "PortForward":
        command = [
            *self.kubectl.prefix,
            "-n",
            self.namespace,
            "port-forward",
            f"svc/{self.service}",
            f"{self.local_port}:{self.remote_port}",
        ]
        self.process = subprocess.Popen(
            command,
            # kubectl can emit several verbose transport lines for every HTTP
            # request. If nobody drains a PIPE it eventually fills and stalls
            # the port-forward, so discard transport chatter after relying on
            # the readiness probe below for diagnostics.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        deadline = time.monotonic() + 30
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "port-forward terminó inesperadamente con "
                    f"exitCode={self.process.returncode}"
                )
            try:
                Prometheus(self.url).get("/-/ready")
                return self
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(0.25)
        self.__exit__(None, None, None)
        raise RuntimeError(f"Prometheus no estuvo listo en 30 s: {last_error}")

    def __exit__(self, *_: Any) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def prom_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def discover_node(kubectl: Kubectl, requested: str | None, selector: str) -> dict[str, Any]:
    if requested:
        return kubectl.json("get", "node", requested)
    nodes = kubectl.json("get", "nodes", "-l", selector).get("items", [])
    if len(nodes) != 1:
        names = [node["metadata"]["name"] for node in nodes]
        raise RuntimeError(
            f"se esperaba un nodo para selector {selector!r}; encontrados: {names}. "
            "Usa --node para seleccionarlo explícitamente."
        )
    return nodes[0]


def internal_ip(node: dict[str, Any]) -> str:
    for address in node.get("status", {}).get("addresses", []):
        if address.get("type") == "InternalIP":
            return str(address["address"])
    raise RuntimeError("el nodo no tiene InternalIP")


def result_rows(payload: Any) -> list[dict[str, Any]]:
    return payload.get("data", {}).get("result", [])


def metric_names(prometheus: Prometheus) -> list[str]:
    payload = prometheus.get(
        "/api/v1/label/__name__/values", {"match[]": '{__name__=~"kepler_.*"}'}
    )
    return list(payload.get("data", []))


def discover_zones(prometheus: Prometheus, node: str) -> list[dict[str, str]]:
    expression = (
        "count by(zone,path)(kepler_node_cpu_joules_total"
        f'{{node_name="{prom_escape(node)}"}})'
    )
    zones: list[dict[str, str]] = []
    for item in result_rows(prometheus.query(expression)):
        metric = item.get("metric", {})
        zones.append({"zone": metric.get("zone", ""), "path": metric.get("path", "")})
    return sorted(zones, key=lambda item: (item["zone"], item["path"]))


def snapshot(kubectl: Kubectl, node: str, destination: Path) -> tuple[Any, list[str]]:
    destination.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    node_json = kubectl.json("get", "node", node)
    pods_json = kubectl.json(
        "get", "pods", "-A", "--field-selector", f"spec.nodeName={node}"
    )
    json_dump(destination / "node.json", node_json)
    json_dump(destination / "pods.json", pods_json)

    for pod in pods_json.get("items", []):
        namespace = pod.get("metadata", {}).get("namespace", "")
        pod_name = pod.get("metadata", {}).get("name", "")
        for status in pod.get("status", {}).get("containerStatuses", []) or []:
            waiting = status.get("state", {}).get("waiting", {})
            if waiting.get("reason") != "CrashLoopBackOff":
                continue
            container_name = status.get("name", "")
            proc = kubectl.run(
                "logs",
                "-n",
                namespace,
                pod_name,
                "-c",
                container_name,
                "--previous",
                "--tail=200",
                check=False,
            )
            log_dir = destination / "crashloop-logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{namespace}__{pod_name}__{container_name}.previous.log"
            (log_dir / filename).write_text(proc.stdout, encoding="utf-8")
            if proc.returncode:
                warnings.append(
                    f"kubectl logs {namespace}/{pod_name} {container_name}: "
                    f"{proc.stderr.strip()}"
                )

    commands = {
        "pods.txt": (
            "get",
            "pods",
            "-A",
            "-o",
            "wide",
            "--field-selector",
            f"spec.nodeName={node}",
        ),
        "events.txt": (
            "get",
            "events",
            "-A",
            "--field-selector",
            f"involvedObject.kind=Node,involvedObject.name={node}",
            "--sort-by=.lastTimestamp",
        ),
        "top-node.txt": ("top", "node", node),
        # The metrics.k8s.io API does not support spec.nodeName as a field
        # selector. Save the small cluster-wide table and correlate it with
        # pods.json, which is already restricted to this node.
        "top-pods.txt": ("top", "pods", "-A"),
    }
    for filename, args in commands.items():
        proc = kubectl.run(*args, check=False)
        (destination / filename).write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            warning = f"kubectl {' '.join(args)}: {proc.stderr.strip()}"
            warnings.append(warning)
            (destination / f"{filename}.error").write_text(warning + "\n", encoding="utf-8")
    return pods_json, warnings


def pod_findings(pods_payload: Any) -> dict[str, Any]:
    running: list[str] = []
    blockers: list[str] = []
    crashloops: list[str] = []
    restarts: dict[str, int] = {}
    for pod in pods_payload.get("items", []):
        namespace = pod.get("metadata", {}).get("namespace", "")
        name = pod.get("metadata", {}).get("name", "")
        identity = f"{namespace}/{name}"
        phase = pod.get("status", {}).get("phase", "")
        if phase == "Running":
            running.append(identity)
        restart_count = 0
        for status in pod.get("status", {}).get("containerStatuses", []) or []:
            restart_count += int(status.get("restartCount", 0))
            waiting = status.get("state", {}).get("waiting", {})
            if waiting.get("reason") == "CrashLoopBackOff":
                crashloops.append(identity)
        if restart_count:
            restarts[identity] = restart_count
        if phase == "Running" and namespace == "green-experiment":
            blockers.append(f"workload experimental activo: {identity}")
        if phase == "Running" and name.startswith("node-debugger-"):
            blockers.append(f"sesión de depuración activa: {identity}")
    for identity in sorted(set(crashloops)):
        blockers.append(f"CrashLoopBackOff en el nodo: {identity}")
    return {
        "running_pods": sorted(running),
        "blocking_conditions": blockers,
        "crashloop_pods": sorted(set(crashloops)),
        "container_restarts": dict(sorted(restarts.items())),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def observed_counter_delta(values: list[float]) -> tuple[float | None, int]:
    if len(values) < 2:
        return None, 0
    delta = 0.0
    resets = 0
    for previous, current in zip(values, values[1:]):
        difference = current - previous
        if difference >= 0:
            delta += difference
        else:
            resets += 1
            delta += current
    return delta, resets


def series_statistics(
    query_name: str,
    series: dict[str, Any],
    expected_samples: int,
) -> dict[str, Any]:
    metric = series.get("metric", {})
    samples = series.get("values", [])
    values = [float(sample[1]) for sample in samples if math.isfinite(float(sample[1]))]
    name = metric.get("__name__", query_name)
    is_counter = name.endswith("_total")
    delta, resets = observed_counter_delta(values) if is_counter else (None, 0)
    return {
        "query": query_name,
        "metric": name,
        "labels": metric,
        "sample_count": len(values),
        "sample_coverage_ratio": len(values) / expected_samples if expected_samples else None,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "observed_counter_delta": delta,
        "counter_resets": resets,
    }


def write_samples_csv(path: Path, collected: dict[str, Any]) -> None:
    fields = [
        "query",
        "metric",
        "timestamp_utc",
        "timestamp_unix",
        "value",
        "zone",
        "path",
        "namespace",
        "pod_namespace",
        "pod",
        "pod_name",
        "container",
        "device",
        "mode",
        "cpu",
        "labels_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_name, payload in collected.items():
            for series in result_rows(payload):
                labels = series.get("metric", {})
                for timestamp, value in series.get("values", []):
                    writer.writerow(
                        {
                            "query": query_name,
                            "metric": labels.get("__name__", query_name),
                            "timestamp_utc": utc_iso(float(timestamp)),
                            "timestamp_unix": timestamp,
                            "value": value,
                            "zone": labels.get("zone", ""),
                            "path": labels.get("path", ""),
                            "namespace": labels.get("namespace", ""),
                            "pod_namespace": labels.get("pod_namespace", ""),
                            "pod": labels.get("pod", ""),
                            "pod_name": labels.get("pod_name", ""),
                            "container": labels.get("container", ""),
                            "device": labels.get("device", ""),
                            "mode": labels.get("mode", ""),
                            "cpu": labels.get("cpu", ""),
                            "labels_json": json.dumps(labels, sort_keys=True),
                        }
                    )


def write_series_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query",
        "metric",
        "zone",
        "path",
        "sample_count",
        "sample_coverage_ratio",
        "first",
        "last",
        "min",
        "max",
        "mean",
        "median",
        "p95",
        "observed_counter_delta",
        "counter_resets",
        "labels_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            labels = row["labels"]
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "zone": labels.get("zone", ""),
                    "path": labels.get("path", ""),
                    "labels_json": json.dumps(labels, sort_keys=True),
                }
            )


def primary_summary(rows: list[dict[str, Any]], zone: str) -> dict[str, Any]:
    wanted = {
        "kepler_node_cpu_watts": "total_watts",
        "kepler_node_cpu_active_watts": "active_watts",
        "kepler_node_cpu_idle_watts": "idle_watts",
        "kepler_node_cpu_joules_total": "total_energy",
        "kepler_node_cpu_active_joules_total": "active_energy",
        "kepler_node_cpu_idle_joules_total": "idle_energy",
        "kepler_node_cpu_usage_ratio": "cpu_usage_ratio",
        "node_cpu_utilization_ratio": "node_exporter_cpu_usage_ratio",
        "node_memory_utilization_ratio": "memory_usage_ratio",
        "cadvisor_cpu_usage_cores": "container_cpu_usage_cores",
    }
    output: dict[str, Any] = {}
    for row in rows:
        metric = row["metric"]
        if metric not in wanted:
            continue
        labels = row["labels"]
        if metric.startswith("kepler_node_cpu_") and metric != "kepler_node_cpu_usage_ratio":
            if labels.get("zone") != zone:
                continue
        key = wanted[metric]
        output[key] = {
            "sample_count": row["sample_count"],
            "coverage_ratio": row["sample_coverage_ratio"],
            "mean": row["mean"],
            "median": row["median"],
            "p95": row["p95"],
        }
        if row["observed_counter_delta"] is not None:
            output[key]["observed_delta_joules"] = row["observed_counter_delta"]
            output[key]["counter_resets"] = row["counter_resets"]
    return output


def build_queries(node: str, ip: str, include_processes: bool) -> dict[str, str]:
    escaped_node = prom_escape(node)
    instance = prom_escape(f"{ip}:9100")
    queries = {
        "kepler_node": f'{{__name__=~"kepler_node_.*",node_name="{escaped_node}"}}',
        "kepler_pod": f'{{__name__=~"kepler_pod_.*",node_name="{escaped_node}"}}',
        "kepler_container": (
            f'{{__name__=~"kepler_container_.*",node_name="{escaped_node}"}}'
        ),
        "node_cpu_seconds": (
            f'node_cpu_seconds_total{{instance="{instance}"}}'
        ),
        "node_cpu_utilization_ratio": (
            "1 - avg(rate(node_cpu_seconds_total"
            f'{{instance="{instance}",mode="idle"}}[1m]))'
        ),
        "node_memory": (
            '{__name__=~"node_memory_(MemAvailable|MemFree|MemTotal|Buffers|Cached)_bytes",'
            f'instance="{instance}"}}'
        ),
        "node_memory_utilization_ratio": (
            "1 - (node_memory_MemAvailable_bytes"
            f'{{instance="{instance}"}} / node_memory_MemTotal_bytes{{instance="{instance}"}})'
        ),
        "node_load": (
            f'{{__name__=~"node_load(1|5|15)",instance="{instance}"}}'
        ),
        "node_frequency_temperature_pressure": (
            '{__name__=~"node_(cpu_frequency_hertz|cpu_scaling_frequency_hertz|hwmon_temp_celsius|thermal_zone_temp|pressure_.*_waiting_seconds_total)",'
            f'instance="{instance}"}}'
        ),
        "node_disk": (
            '{__name__=~"node_disk_(read_bytes|written_bytes|reads_completed|writes_completed|io_time_seconds)_total",'
            f'instance="{instance}"}}'
        ),
        "node_network": (
            '{__name__=~"node_network_(receive|transmit)_(bytes|drop|errs)_total",'
            f'instance="{instance}",device!="lo"}}'
        ),
        "node_power_supply": (
            f'{{__name__=~"node_power_supply_.*",instance="{instance}"}}'
        ),
        "cadvisor_cpu_usage_cores": (
            "sum(rate(container_cpu_usage_seconds_total"
            f'{{node="{escaped_node}",container!=""}}[1m]))'
        ),
        # Keep the per-container counters raw.  Prometheus can calculate rates
        # later; evaluating rate()+grouping for every container made collection
        # unnecessarily expensive on the CERN cluster.
        "cadvisor_cpu_counters": (
            "container_cpu_usage_seconds_total"
            f'{{node="{escaped_node}",container!=""}}'
        ),
        "cadvisor_memory": (
            "container_memory_working_set_bytes"
            f'{{node="{escaped_node}",container!=""}}'
        ),
    }
    if include_processes:
        queries["kepler_process"] = (
            f'{{__name__=~"kepler_process_.*",node_name="{escaped_node}"}}'
        )
    return queries


def capture_static_metadata(
    kubectl: Kubectl,
    prometheus: Prometheus,
    output: Path,
    node_name: str,
    node_payload: Any,
) -> list[str]:
    warnings: list[str] = []
    json_dump(output / "cluster" / "node.json", node_payload)
    commands = {
        "kubernetes-version.json": ("version", "-o", "json"),
        "kepler-daemonset.json": ("get", "daemonset", "kepler", "-n", "kepler", "-o", "json"),
        "kepler-servicemonitor.json": ("get", "servicemonitor", "kepler", "-n", "kepler", "-o", "json"),
        "prometheus.json": (
            "get",
            "prometheus",
            "-n",
            "monitoring",
            "-o",
            "json",
        ),
    }
    for filename, args in commands.items():
        proc = kubectl.run(*args, check=False)
        target = output / "cluster" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            warnings.append(f"kubectl {' '.join(args)}: {proc.stderr.strip()}")

    kepler_logs = kubectl.run(
        "logs",
        "-n",
        "kepler",
        "-l",
        "app.kubernetes.io/name=kepler",
        "--tail=200",
        check=False,
    )
    (output / "cluster" / "kepler-logs.txt").write_text(
        kepler_logs.stdout, encoding="utf-8"
    )
    if kepler_logs.returncode:
        warnings.append(f"no se pudieron guardar logs de Kepler: {kepler_logs.stderr.strip()}")

    for filename, path in (
        ("prometheus-buildinfo.json", "/api/v1/status/buildinfo"),
        ("prometheus-runtimeinfo.json", "/api/v1/status/runtimeinfo"),
    ):
        try:
            json_dump(output / "prometheus" / filename, prometheus.get(path))
        except RuntimeError as exc:
            warnings.append(str(exc))

    try:
        targets = prometheus.get("/api/v1/targets")
        data = targets.get("data", {})
        selected = []
        for target in data.get("activeTargets", []):
            labels = target.get("labels", {})
            discovered = target.get("discoveredLabels", {})
            if (
                labels.get("job") == "kepler"
                or labels.get("nodename") == node_name
                or discovered.get("__meta_kubernetes_pod_node_name") == node_name
            ):
                selected.append(target)
        json_dump(
            output / "prometheus" / "targets.json",
            {"status": "success", "data": {"activeTargets": selected}},
        )
    except RuntimeError as exc:
        warnings.append(str(exc))
    return warnings


def wait_window(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    print(f"Capturando ventana idle durante {seconds} s...", flush=True)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if remaining > 60:
            print(f"  quedan aproximadamente {math.ceil(remaining / 60)} min", flush=True)
        time.sleep(min(60, remaining))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Captura automática del baseline idle de Kepler/Prometheus."
    )
    parser.add_argument("--duration", type=parse_duration, default=900, help="15m por defecto")
    parser.add_argument("--step", type=int, default=10, help="paso de query_range en segundos")
    parser.add_argument("--node", help="nodo; por defecto se descubre por label")
    parser.add_argument("--node-selector", default=DEFAULT_NODE_SELECTOR)
    parser.add_argument("--zone", default="package", help="dominio primario para el resumen")
    parser.add_argument("--kubeconfig", help="respeta KUBECONFIG si se omite")
    parser.add_argument("--prometheus-url", help="omite el port-forward y usa esta URL")
    parser.add_argument("--prometheus-namespace", default="monitoring")
    parser.add_argument(
        "--prometheus-service", default="monitoring-kube-prometheus-prometheus"
    )
    parser.add_argument("--prometheus-port", type=int, default=9090)
    parser.add_argument("--output", type=Path, help="directorio de salida")
    parser.add_argument(
        "--include-processes",
        action="store_true",
        help="incluye series Kepler por proceso (alta cardinalidad)",
    )
    return parser


def collect(args: argparse.Namespace, prometheus: Prometheus, kubectl: Kubectl) -> Path:
    experiment_root = Path(__file__).resolve().parent.parent
    output = (args.output or experiment_root / "idle" / utc_slug()).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"el directorio de salida no está vacío: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_dump(
        output / "status.json",
        {
            "status": "in_progress",
            "created_at": utc_iso(),
            "schema_version": SCHEMA_VERSION,
        },
    )

    node_payload = discover_node(kubectl, args.node, args.node_selector)
    node_name = node_payload["metadata"]["name"]
    node_ip = internal_ip(node_payload)
    zones = discover_zones(prometheus, node_name)
    if not zones:
        raise RuntimeError(f"Prometheus no devuelve energía Kepler para {node_name}")
    if args.zone not in {item["zone"] for item in zones}:
        raise RuntimeError(f"zone={args.zone!r} no existe; disponibles: {zones}")

    up_expression = 'up{namespace="kepler"}'
    up_start = prometheus.query(up_expression)
    if not any(float(item["value"][1]) == 1.0 for item in result_rows(up_start)):
        raise RuntimeError("el target de Kepler no está up en Prometheus")

    warnings = capture_static_metadata(
        kubectl, prometheus, output, node_name, node_payload
    )
    catalog = metric_names(prometheus)
    json_dump(output / "prometheus" / "kepler-metric-names.json", catalog)

    pods_start, snapshot_warnings = snapshot(kubectl, node_name, output / "snapshots" / "start")
    warnings.extend(snapshot_warnings)
    start_findings = pod_findings(pods_start)

    started_at = time.time()
    wait_window(args.duration)
    ended_at = time.time()

    queries = build_queries(node_name, node_ip, args.include_processes)
    collected: dict[str, Any] = {}
    raw_dir = output / "prometheus" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expression in queries.items():
        print(f"Descargando {name}...", flush=True)
        try:
            payload = prometheus.query_range(expression, started_at, ended_at, args.step)
        except RuntimeError as exc:
            if name == "kepler_node":
                raise
            payload = {
                "status": "error",
                "errorType": "collection_error",
                "error": str(exc),
                "data": {"resultType": "matrix", "result": []},
            }
            warnings.append(f"no se pudo descargar {name}: {exc}")
        collected[name] = payload
        json_dump(raw_dir / f"{name}.json", payload)
        if not result_rows(payload):
            warnings.append(f"la consulta {name} no devolvió series")

    up_end = prometheus.query(up_expression)
    pods_end, snapshot_warnings = snapshot(kubectl, node_name, output / "snapshots" / "end")
    warnings.extend(snapshot_warnings)
    end_findings = pod_findings(pods_end)

    expected_samples = math.floor((ended_at - started_at) / args.step) + 1
    stats_rows: list[dict[str, Any]] = []
    for query_name, payload in collected.items():
        for series in result_rows(payload):
            stats_rows.append(series_statistics(query_name, series, expected_samples))

    write_samples_csv(output / "metrics.csv", collected)
    write_series_csv(output / "series-summary.csv", stats_rows)
    primary = primary_summary(stats_rows, args.zone)

    up_end_ok = any(float(item["value"][1]) == 1.0 for item in result_rows(up_end))
    blocking = sorted(
        set(start_findings["blocking_conditions"] + end_findings["blocking_conditions"])
    )
    kepler_coverage = [
        row["sample_coverage_ratio"]
        for row in stats_rows
        if row["metric"] == "kepler_node_cpu_watts"
        and row["labels"].get("zone") == args.zone
        and row["sample_coverage_ratio"] is not None
    ]
    if not kepler_coverage or min(kepler_coverage) < 0.8:
        blocking.append("cobertura de kepler_node_cpu_watts inferior al 80 %")
    if not up_end_ok:
        blocking.append("el target Kepler no estaba up al final")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "capture_type": "idle-baseline",
        "node": node_name,
        "node_internal_ip": node_ip,
        "primary_zone": args.zone,
        "available_zones": zones,
        "requested_duration_seconds": args.duration,
        "actual_duration_seconds": ended_at - started_at,
        "query_step_seconds": args.step,
        "started_at": utc_iso(started_at),
        "ended_at": utc_iso(ended_at),
        "timezone": "UTC",
        "prometheus_url_supplied": bool(args.prometheus_url),
        "include_processes": args.include_processes,
        "command": [str(item) for item in sys.argv],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "valid_idle_baseline": not blocking,
        "blocking_conditions": blocking,
        "collection_warnings": sorted(set(filter(None, warnings))),
        "pod_state_start": start_findings,
        "pod_state_end": end_findings,
        "primary_metrics": primary,
        "notes": [
            "Los deltas son diferencias observadas entre muestras, no extrapolaciones PromQL increase().",
            "El resumen usa una sola zone para evitar doble conteo entre core y package.",
            "metrics.csv y prometheus/raw conservan las muestras originales para análisis posterior.",
            "Las métricas Kepler por proceso se excluyen por defecto debido a su alta cardinalidad.",
        ],
    }
    json_dump(output / "metadata.json", metadata)
    json_dump(output / "summary.json", summary)
    json_dump(
        output / "status.json",
        {
            "status": "complete",
            "completed_at": utc_iso(),
            "schema_version": SCHEMA_VERSION,
            "valid_idle_baseline": summary["valid_idle_baseline"],
        },
    )
    print(f"\nCaptura guardada en: {output}", flush=True)
    print(
        "Estado: " + ("VÁLIDA" if summary["valid_idle_baseline"] else "NO VÁLIDA"),
        flush=True,
    )
    for item in blocking:
        print(f"  - {item}", flush=True)
    return output


def main() -> int:
    args = build_parser().parse_args()
    if args.step <= 0:
        raise SystemExit("--step debe ser positivo")
    if not shutil.which("kubectl"):
        raise SystemExit("kubectl no está disponible en PATH")
    kubectl = Kubectl(args.kubeconfig)
    try:
        if args.prometheus_url:
            collect(args, Prometheus(args.prometheus_url), kubectl)
        else:
            with PortForward(
                kubectl,
                args.prometheus_namespace,
                args.prometheus_service,
                args.prometheus_port,
            ) as forward:
                collect(args, Prometheus(forward.url), kubectl)
    except (CommandError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
