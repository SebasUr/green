#!/usr/bin/env python3
"""Run and measure a parametrizable Monte Carlo Kubernetes workload."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import capture_idle as collector


SCHEMA_VERSION = "1.0"
NAMESPACE = "green-experiment"
CONTAINER = "monte-carlo"
DEFAULT_WORKERS = 16
DEFAULT_SAMPLES_PER_WORKER = 3_150_000_000
DEFAULT_BASE_SEED = 20_260_713
DEFAULT_TARGET_SECONDS = 900
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")


class TrialError(RuntimeError):
    pass


def parse_nonnegative_duration(value: str) -> int:
    if value.strip() in {"0", "0s", "0m", "0h"}:
        return 0
    return collector.parse_duration(value)


def parse_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def default_trial_id() -> str:
    return "calibration-" + collector.utc_slug().lower()


def validate_label(value: str, field: str, max_length: int = 63) -> str:
    if len(value) > max_length or not DNS_LABEL.fullmatch(value):
        raise TrialError(
            f"{field} debe ser un label DNS en minúsculas de hasta {max_length} caracteres"
        )
    return value


def render_job(template: Path, replacements: dict[str, str]) -> str:
    rendered = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        if "\n" in value or "\r" in value or '"' in value:
            raise TrialError(f"valor no permitido para {key}: {value!r}")
        rendered = rendered.replace(f"${{{key}}}", value)
    unresolved = sorted(set(re.findall(r"\$\{([A-Z0-9_]+)\}", rendered)))
    if unresolved:
        raise TrialError(f"variables sin resolver en la plantilla: {unresolved}")
    return rendered


def find_pod(
    kubectl: collector.Kubectl,
    trial_id: str,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    selector = f"sustainability.cern.ch/trial={trial_id}"
    while time.monotonic() < deadline:
        pods = kubectl.json("get", "pods", "-n", NAMESPACE, "-l", selector)
        items = pods.get("items", [])
        if items:
            return items[0]
        time.sleep(2)
    raise TrialError(f"el Job no creó un pod en {timeout_seconds} s")


def wait_for_job(
    kubectl: collector.Kubectl,
    job_name: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    next_report = 0.0
    while time.monotonic() < deadline:
        job = kubectl.json("get", "job", job_name, "-n", NAMESPACE)
        status = job.get("status", {})
        conditions = {
            item.get("type"): item
            for item in status.get("conditions", []) or []
            if item.get("status") == "True"
        }
        if "Complete" in conditions or status.get("succeeded", 0) >= 1:
            return job
        if "Failed" in conditions or status.get("failed", 0) >= 1:
            reason = conditions.get("Failed", {}).get("message", "Job failed")
            raise TrialError(reason)
        now = time.monotonic()
        if now >= next_report:
            elapsed = timeout_seconds - max(0, int(deadline - now))
            print(
                f"Job activo: elapsed={elapsed}s active={status.get('active', 0)}",
                flush=True,
            )
            next_report = now + 30
        time.sleep(poll_seconds)
    raise TrialError(f"timeout esperando {job_name} después de {timeout_seconds} s")


def container_termination(pod: dict[str, Any]) -> dict[str, Any]:
    statuses = pod.get("status", {}).get("containerStatuses", []) or []
    for status in statuses:
        if status.get("name") != CONTAINER:
            continue
        terminated = status.get("state", {}).get("terminated")
        if not terminated:
            terminated = status.get("lastState", {}).get("terminated")
        if terminated:
            return terminated
    raise TrialError("no se encontró el estado terminado del contenedor Monte Carlo")


def save_workload_artifacts(
    kubectl: collector.Kubectl,
    output: Path,
    job_name: str,
    pod_name: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    job = kubectl.json("get", "job", job_name, "-n", NAMESPACE)
    pod = kubectl.json("get", "pod", pod_name, "-n", NAMESPACE)
    logs = kubectl.run("logs", pod_name, "-n", NAMESPACE, "-c", CONTAINER)
    collector.json_dump(output / "job.json", job)
    collector.json_dump(output / "pod.json", pod)
    (output / "workload-output.json").write_text(logs.stdout, encoding="utf-8")
    return job, pod, logs.stdout


def parse_workload_output(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrialError(f"el output del workload no es JSON válido: {exc}") from exc


def workload_queries(node: str, pod: str, zone: str) -> dict[str, str]:
    node = collector.prom_escape(node)
    pod = collector.prom_escape(pod)
    labels = (
        f'node_name="{node}",pod_namespace="{NAMESPACE}",pod_name="{pod}"'
    )
    return {
        "workload_kepler_pod": (
            f'{{__name__=~"kepler_pod_.*",{labels}}}'
        ),
        "workload_pod_energy_joules": (
            "sum(max without(state)(kepler_pod_cpu_joules_total"
            f'{{{labels},zone="{collector.prom_escape(zone)}"}}))'
        ),
        "workload_pod_watts": (
            "sum(max without(state)(kepler_pod_cpu_watts"
            f'{{{labels},zone="{collector.prom_escape(zone)}"}}))'
        ),
        "workload_cpu_usage_cores": (
            "sum(rate(container_cpu_usage_seconds_total"
            f'{{namespace="{NAMESPACE}",pod="{pod}",container="{CONTAINER}"}}[1m]))'
        ),
        "workload_memory_bytes": (
            "sum(container_memory_working_set_bytes"
            f'{{namespace="{NAMESPACE}",pod="{pod}",container="{CONTAINER}"}})'
        ),
    }


def collect_queries(
    prometheus: collector.Prometheus,
    queries: dict[str, str],
    start: float,
    end: float,
    step: int,
    output: Path,
) -> tuple[dict[str, Any], list[str]]:
    collected: dict[str, Any] = {}
    warnings: list[str] = []
    raw_dir = output / "prometheus" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expression in queries.items():
        print(f"Descargando {name}...", flush=True)
        try:
            payload = prometheus.query_range(expression, start, end, step)
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
        collector.json_dump(raw_dir / f"{name}.json", payload)
        if not collector.result_rows(payload):
            warnings.append(f"la consulta {name} no devolvió series")
    return collected, warnings


def finite_samples(series: dict[str, Any], start: float, end: float) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for timestamp, value in series.get("values", []):
        timestamp = float(timestamp)
        value = float(value)
        if start <= timestamp <= end and math.isfinite(value):
            samples.append((timestamp, value))
    return samples


def value_statistics(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "sample_count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": collector.percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def one_series_stats(
    payload: Any, start: float, end: float, use_max: bool = False
) -> dict[str, Any] | None:
    values: list[float] = []
    for series in collector.result_rows(payload):
        values.extend(value for _, value in finite_samples(series, start, end))
    stats = value_statistics(values)
    if stats is not None and use_max:
        stats["final_or_max"] = max(values)
    return stats


def counter_delta_for_interval(
    payload: Any,
    metric_name: str,
    zone: str,
    start: float,
    end: float,
) -> dict[str, Any] | None:
    for series in collector.result_rows(payload):
        labels = series.get("metric", {})
        if labels.get("__name__") != metric_name or labels.get("zone") != zone:
            continue
        samples = sorted(
            (float(ts), float(value))
            for ts, value in series.get("values", [])
            if math.isfinite(float(value))
        )
        before = [sample for sample in samples if sample[0] <= start]
        after = [sample for sample in samples if sample[0] >= end]
        if not before or not after:
            return None
        selected = [
            sample for sample in samples if before[-1][0] <= sample[0] <= after[0][0]
        ]
        delta, resets = collector.observed_counter_delta(
            [value for _, value in selected]
        )
        return {
            "observed_delta_joules": delta,
            "counter_resets": resets,
            "first_sample_at": collector.utc_iso(selected[0][0]),
            "last_sample_at": collector.utc_iso(selected[-1][0]),
            "sample_count": len(selected),
        }
    return None


def node_gauge_stats(
    payload: Any,
    metric_name: str,
    zone: str | None,
    start: float,
    end: float,
) -> dict[str, Any] | None:
    for series in collector.result_rows(payload):
        labels = series.get("metric", {})
        if labels.get("__name__") != metric_name:
            continue
        if zone is not None and labels.get("zone") != zone:
            continue
        values = [value for _, value in finite_samples(series, start, end)]
        return value_statistics(values)
    return None


def image_details(pod: dict[str, Any]) -> dict[str, Any]:
    for status in pod.get("status", {}).get("containerStatuses", []) or []:
        if status.get("name") == CONTAINER:
            return {
                "requested": next(
                    container.get("image")
                    for container in pod.get("spec", {}).get("containers", [])
                    if container.get("name") == CONTAINER
                ),
                "image": status.get("image"),
                "image_id": status.get("imageID"),
                "container_id": status.get("containerID"),
            }
    return {}


def build_result(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    workload: dict[str, Any],
    pod: dict[str, Any],
    collected: dict[str, Any],
    workload_start: float,
    workload_end: float,
    preflight_blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    runtime = float(workload.get("elapsed_seconds", workload_end - workload_start))
    pod_energy = one_series_stats(
        collected["workload_pod_energy_joules"],
        workload_start,
        workload_end + args.post_buffer,
        use_max=True,
    )
    energy_joules = pod_energy.get("final_or_max") if pod_energy else None
    scientific_ok = (
        workload.get("workers") == args.workers
        and workload.get("samples_per_worker") == args.samples_per_worker
        and workload.get("total_samples") == args.workers * args.samples_per_worker
        and isinstance(workload.get("pi_estimate"), (int, float))
    )
    blockers = list(preflight_blockers)
    if not scientific_ok:
        blockers.append("el output científico no coincide con los parámetros solicitados")
    if energy_joules is None or energy_joules <= 0:
        blockers.append("Kepler no devolvió energía atribuida positiva para el pod")

    recommended = None
    if runtime > 0:
        recommended = round(args.samples_per_worker * args.target_runtime / runtime)

    return {
        "schema_version": SCHEMA_VERSION,
        "trial_id": args.trial_id,
        "policy": args.policy,
        "valid_trial": not blockers,
        "blocking_conditions": sorted(set(blockers)),
        "collection_warnings": sorted(set(warnings)),
        "parameters": {
            "workers": args.workers,
            "samples_per_worker": args.samples_per_worker,
            "base_seed": args.base_seed,
            "total_samples": args.workers * args.samples_per_worker,
            "memory": args.memory,
            "target_runtime_seconds": args.target_runtime,
        },
        "execution": {
            "workload_started_at": collector.utc_iso(workload_start),
            "workload_finished_at": collector.utc_iso(workload_end),
            "runtime_seconds": runtime,
            "pi_estimate": workload.get("pi_estimate"),
            "scientific_output_matches_parameters": scientific_ok,
            "recommended_samples_per_worker_for_target_runtime": recommended,
            "image": image_details(pod),
        },
        "energy": {
            "primary_zone": args.zone,
            "pod_cpu_energy_joules": energy_joules,
            "pod_cpu_energy_kwh": energy_joules / 3_600_000 if energy_joules else None,
            "average_attributed_pod_power_watts": (
                energy_joules / runtime if energy_joules and runtime > 0 else None
            ),
            "pod_watts": one_series_stats(
                collected["workload_pod_watts"], workload_start, workload_end
            ),
            "node_total_energy": counter_delta_for_interval(
                collected["kepler_node"],
                "kepler_node_cpu_joules_total",
                args.zone,
                workload_start,
                workload_end,
            ),
            "node_active_energy": counter_delta_for_interval(
                collected["kepler_node"],
                "kepler_node_cpu_active_joules_total",
                args.zone,
                workload_start,
                workload_end,
            ),
            "node_idle_energy": counter_delta_for_interval(
                collected["kepler_node"],
                "kepler_node_cpu_idle_joules_total",
                args.zone,
                workload_start,
                workload_end,
            ),
            "node_total_watts": node_gauge_stats(
                collected["kepler_node"],
                "kepler_node_cpu_watts",
                args.zone,
                workload_start,
                workload_end,
            ),
            "node_active_watts": node_gauge_stats(
                collected["kepler_node"],
                "kepler_node_cpu_active_watts",
                args.zone,
                workload_start,
                workload_end,
            ),
            "node_cpu_usage_ratio": node_gauge_stats(
                collected["kepler_node"],
                "kepler_node_cpu_usage_ratio",
                None,
                workload_start,
                workload_end,
            ),
        },
        "metadata": metadata,
        "notes": [
            "La energía del pod es CPU operational energy attributed by Kepler, no energía total del servidor.",
            "Los deltas del nodo usan la muestra anterior al inicio y la primera posterior al final.",
            "El valor por defecto de muestras es una estimación; usa la recomendación para calibrar el siguiente trial.",
        ],
    }


def run_trial(
    args: argparse.Namespace,
    kubectl: collector.Kubectl,
    prometheus: collector.Prometheus,
) -> Path:
    experiment_root = Path(__file__).resolve().parent.parent
    output = (args.output or experiment_root / "trials" / args.trial_id).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrialError(f"el directorio de salida no está vacío: {output}")
    output.mkdir(parents=True, exist_ok=True)
    collector.json_dump(
        output / "status.json",
        {
            "status": "in_progress",
            "created_at": collector.utc_iso(),
            "schema_version": SCHEMA_VERSION,
        },
    )

    job_name = f"mc-{args.trial_id}"
    job_created = False
    try:
        node_payload = collector.discover_node(
            kubectl, args.node, args.node_selector
        )
        node_name = node_payload["metadata"]["name"]
        node_ip = collector.internal_ip(node_payload)
        allocatable_cpu = int(node_payload["status"]["allocatable"]["cpu"])
        if args.workers > allocatable_cpu:
            raise TrialError(
                f"workers={args.workers} supera CPU allocatable={allocatable_cpu}"
            )

        existing = kubectl.run("get", "job", job_name, "-n", NAMESPACE, check=False)
        if existing.returncode == 0:
            raise TrialError(f"el Job {NAMESPACE}/{job_name} ya existe")

        zones = collector.discover_zones(prometheus, node_name)
        if args.zone not in {item["zone"] for item in zones}:
            raise TrialError(f"zone={args.zone!r} no existe; disponibles: {zones}")
        up = prometheus.query('up{namespace="kepler"}')
        if not any(
            float(item["value"][1]) == 1.0 for item in collector.result_rows(up)
        ):
            raise TrialError("el target Kepler no está up")

        warnings = collector.capture_static_metadata(
            kubectl, prometheus, output, node_name, node_payload
        )
        pods_start, snapshot_warnings = collector.snapshot(
            kubectl, node_name, output / "snapshots" / "start"
        )
        warnings.extend(snapshot_warnings)
        preflight = collector.pod_findings(pods_start)
        preflight_blockers = preflight["blocking_conditions"]
        if preflight_blockers and not args.allow_dirty_node:
            raise TrialError(
                "el nodo no está limpio:\n- " + "\n- ".join(preflight_blockers)
            )

        if not args.skip_configmap_apply:
            kubectl.run("apply", "-f", str(args.configmap_manifest.resolve()))

        rendered = render_job(
            args.job_template.resolve(),
            {
                "TRIAL_ID": args.trial_id,
                "POLICY": args.policy,
                "WORKERS": str(args.workers),
                "SAMPLES_PER_WORKER": str(args.samples_per_worker),
                "BASE_SEED": str(args.base_seed),
                "IMAGE": args.image,
                "MEMORY": args.memory,
            },
        )
        rendered_path = output / "job-submitted.yaml"
        rendered_path.write_text(rendered, encoding="utf-8")

        submitted_at = time.time()
        kubectl.run("apply", "-f", str(rendered_path))
        job_created = True
        print(f"Job creado: {NAMESPACE}/{job_name}", flush=True)
        pod = find_pod(kubectl, args.trial_id)
        pod_name = pod["metadata"]["name"]
        print(f"Pod: {NAMESPACE}/{pod_name}", flush=True)
        wait_for_job(kubectl, job_name, args.timeout, args.poll_interval)

        _job, pod, log_text = save_workload_artifacts(
            kubectl, output, job_name, pod_name
        )
        termination = container_termination(pod)
        if termination.get("exitCode") != 0:
            raise TrialError(
                f"el contenedor terminó con exitCode={termination.get('exitCode')}"
            )
        workload_start = parse_timestamp(termination["startedAt"])
        workload_end = parse_timestamp(termination["finishedAt"])
        workload = parse_workload_output(log_text)

        if args.post_buffer:
            print(
                f"Esperando {args.post_buffer}s para el último scrape de Prometheus...",
                flush=True,
            )
            time.sleep(args.post_buffer)
        query_start = submitted_at - args.pre_buffer
        query_end = max(time.time(), workload_end + args.post_buffer)

        queries = collector.build_queries(node_name, node_ip, args.include_processes)
        queries.update(workload_queries(node_name, pod_name, args.zone))
        collected, query_warnings = collect_queries(
            prometheus,
            queries,
            query_start,
            query_end,
            args.step,
            output,
        )
        warnings.extend(query_warnings)

        pods_end, snapshot_warnings = collector.snapshot(
            kubectl, node_name, output / "snapshots" / "end"
        )
        warnings.extend(snapshot_warnings)
        expected_samples = math.floor((query_end - query_start) / args.step) + 1
        rows: list[dict[str, Any]] = []
        for query_name, payload in collected.items():
            for series in collector.result_rows(payload):
                rows.append(
                    collector.series_statistics(query_name, series, expected_samples)
                )
        collector.write_samples_csv(output / "metrics.csv", collected)
        collector.write_series_csv(output / "series-summary.csv", rows)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "capture_type": "monte-carlo-trial",
            "trial_id": args.trial_id,
            "job_name": job_name,
            "pod_name": pod_name,
            "pod_uid": pod["metadata"]["uid"],
            "node": node_name,
            "node_internal_ip": node_ip,
            "available_zones": zones,
            "primary_zone": args.zone,
            "submitted_at": collector.utc_iso(submitted_at),
            "query_started_at": collector.utc_iso(query_start),
            "query_ended_at": collector.utc_iso(query_end),
            "query_step_seconds": args.step,
            "preflight": preflight,
            "command": [str(item) for item in sys.argv],
            "timezone": "UTC",
        }
        result = build_result(
            args,
            metadata,
            workload,
            pod,
            collected,
            workload_start,
            workload_end,
            preflight_blockers,
            warnings,
        )
        collector.json_dump(output / "metadata.json", metadata)
        collector.json_dump(output / "result.json", result)
        collector.json_dump(
            output / "status.json",
            {
                "status": "complete",
                "completed_at": collector.utc_iso(),
                "schema_version": SCHEMA_VERSION,
                "valid_trial": result["valid_trial"],
            },
        )
        if args.cleanup_job:
            kubectl.run("delete", "job", job_name, "-n", NAMESPACE)
            job_created = False
        print(f"\nTrial guardado en: {output}", flush=True)
        print("Estado: " + ("VÁLIDO" if result["valid_trial"] else "NO VÁLIDO"))
        print(f"Runtime: {result['execution']['runtime_seconds']:.3f}s")
        print(f"Pod energy: {result['energy']['pod_cpu_energy_joules']} J")
        print(
            "Siguiente samples/worker para "
            f"{args.target_runtime}s: "
            f"{result['execution']['recommended_samples_per_worker_for_target_runtime']}"
        )
        return output
    except BaseException as exc:
        if job_created:
            failed_job = kubectl.run(
                "get", "job", job_name, "-n", NAMESPACE, "-o", "json", check=False
            )
            if failed_job.returncode == 0:
                (output / "job-failure.json").write_text(
                    failed_job.stdout, encoding="utf-8"
                )
            failed_pods = kubectl.run(
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-l",
                f"sustainability.cern.ch/trial={args.trial_id}",
                "-o",
                "json",
                check=False,
            )
            if failed_pods.returncode == 0:
                (output / "pods-failure.json").write_text(
                    failed_pods.stdout, encoding="utf-8"
                )
                try:
                    items = json.loads(failed_pods.stdout).get("items", [])
                except json.JSONDecodeError:
                    items = []
                if items:
                    failed_pod_name = items[0]["metadata"]["name"]
                    failed_logs = kubectl.run(
                        "logs",
                        failed_pod_name,
                        "-n",
                        NAMESPACE,
                        "-c",
                        CONTAINER,
                        check=False,
                    )
                    (output / "workload-failure.log").write_text(
                        failed_logs.stdout + failed_logs.stderr,
                        encoding="utf-8",
                    )
        if args.cleanup_job and job_created:
            cleanup = kubectl.run(
                "delete",
                "job",
                job_name,
                "-n",
                NAMESPACE,
                "--ignore-not-found=true",
                check=False,
            )
            if cleanup.returncode:
                print(
                    f"WARNING: no se pudo limpiar {job_name}: {cleanup.stderr.strip()}",
                    file=sys.stderr,
                )
        collector.json_dump(
            output / "status.json",
            {
                "status": "failed",
                "failed_at": collector.utc_iso(),
                "schema_version": SCHEMA_VERSION,
                "error": str(exc),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Ejecuta Monte Carlo en Kubernetes y captura métricas Kepler."
    )
    parser.add_argument("--trial-id", default=default_trial_id())
    parser.add_argument("--policy", default="calibration")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--samples-per-worker", type=int, default=DEFAULT_SAMPLES_PER_WORKER
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--target-runtime", type=int, default=DEFAULT_TARGET_SECONDS)
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--memory", default="2Gi")
    parser.add_argument("--zone", default="package")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--pre-buffer", type=parse_nonnegative_duration, default=30)
    parser.add_argument("--post-buffer", type=parse_nonnegative_duration, default=30)
    parser.add_argument("--timeout", type=collector.parse_duration, default=7200)
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--node")
    parser.add_argument("--node-selector", default=collector.DEFAULT_NODE_SELECTOR)
    parser.add_argument("--kubeconfig", help="respeta KUBECONFIG si se omite")
    parser.add_argument("--prometheus-url")
    parser.add_argument("--prometheus-namespace", default="monitoring")
    parser.add_argument(
        "--prometheus-service", default="monitoring-kube-prometheus-prometheus"
    )
    parser.add_argument("--prometheus-port", type=int, default=9090)
    parser.add_argument(
        "--job-template",
        type=Path,
        default=root / "manifests" / "monte-carlo-job-template.yaml",
    )
    parser.add_argument(
        "--configmap-manifest",
        type=Path,
        default=root / "manifests" / "monte-carlo-configmap.yml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-configmap-apply", action="store_true")
    parser.add_argument("--include-processes", action="store_true")
    parser.add_argument(
        "--allow-dirty-node",
        action="store_true",
        help="ejecuta diagnóstico aunque haya debugger, CrashLoop o otro workload",
    )
    parser.add_argument("--cleanup-job", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.trial_id = validate_label(args.trial_id, "trial-id", 50)
    args.policy = validate_label(args.policy, "policy")
    if args.workers <= 0 or args.samples_per_worker <= 0:
        raise TrialError("workers y samples-per-worker deben ser positivos")
    if args.target_runtime <= 0 or args.step <= 0 or args.poll_interval <= 0:
        raise TrialError("target-runtime, step y poll-interval deben ser positivos")
    if not re.fullmatch(r"[1-9][0-9]*(?:Ki|Mi|Gi|Ti)?", args.memory):
        raise TrialError("memory debe ser una cantidad Kubernetes, por ejemplo 2Gi")
    if not args.job_template.is_file():
        raise TrialError(f"no existe la plantilla: {args.job_template}")
    if not args.skip_configmap_apply and not args.configmap_manifest.is_file():
        raise TrialError(f"no existe el ConfigMap: {args.configmap_manifest}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        if not shutil.which("kubectl"):
            raise TrialError("kubectl no está disponible en PATH")
        kubectl = collector.Kubectl(args.kubeconfig)
        if args.prometheus_url:
            run_trial(args, kubectl, collector.Prometheus(args.prometheus_url))
        else:
            with collector.PortForward(
                kubectl,
                args.prometheus_namespace,
                args.prometheus_service,
                args.prometheus_port,
            ) as forward:
                run_trial(args, kubectl, collector.Prometheus(forward.url))
    except (collector.CommandError, TrialError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
