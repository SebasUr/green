"""Read-only Kubernetes access through the user's existing ``kubectl``."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from datetime import datetime
from typing import Any

from green_observatory.observability.models import (
    ContainerProvenance,
    JobProvenance,
    PodExecution,
    PodProvenance,
)


class KubernetesCommandError(RuntimeError):
    pass


def parse_kubernetes_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class KubectlClient:
    """Small read-only adapter that respects ``KUBECONFIG`` by default."""

    def __init__(self, kubeconfig: str | None = None) -> None:
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
            raise KubernetesCommandError(f"kubectl {' '.join(args)}: {detail}")
        return proc

    def json(self, *args: str) -> Any:
        return json.loads(self.run(*args, "-o", "json").stdout)

    def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        return self.json("get", "job", name, "-n", namespace)

    def list_jobs(self, selector: str, namespace: str | None = None) -> list[dict[str, Any]]:
        args = ["get", "jobs"]
        if namespace:
            args.extend(["-n", namespace])
        else:
            args.append("-A")
        if selector:
            args.extend(["-l", selector])
        return self.json(*args).get("items", [])

    def pods_for_job(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        namespace = job["metadata"]["namespace"]
        job_uid = job["metadata"]["uid"]
        pods = self.json("get", "pods", "-n", namespace).get("items", [])
        return [
            pod
            for pod in pods
            if any(
                owner.get("kind") == "Job" and owner.get("uid") == job_uid
                for owner in pod.get("metadata", {}).get("ownerReferences", [])
            )
        ]

    def pod_logs(self, namespace: str, name: str, container: str | None = None) -> str:
        """Return a pod's stdout. Only available while the pod object still exists."""
        args = ["logs", name, "-n", namespace]
        if container:
            args.extend(["-c", container])
        proc = self.run(*args, check=False)
        if proc.returncode:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise KubernetesCommandError(f"kubectl {' '.join(args)}: {detail}")
        return proc.stdout


def job_outcome(job: dict[str, Any]) -> str | None:
    conditions = {
        item.get("type"): item
        for item in job.get("status", {}).get("conditions", []) or []
        if item.get("status") == "True"
    }
    if "Complete" in conditions or job.get("status", {}).get("succeeded", 0):
        return "succeeded"
    if "Failed" in conditions:
        return "failed"
    return None


def pod_execution(pod: dict[str, Any]) -> PodExecution | None:
    """Extract the complete pod lifetime used by the pod-level Kepler metric."""
    status = pod.get("status", {})
    raw_start = status.get("startTime")
    statuses = [
        *(status.get("initContainerStatuses", []) or []),
        *(status.get("containerStatuses", []) or []),
    ]
    terminated = []
    for container in statuses:
        state = container.get("state", {}).get("terminated")
        if state:
            terminated.append(state)
    if not terminated:
        return None
    raw_start = raw_start or min(item["startedAt"] for item in terminated if item.get("startedAt"))
    finished_values = [item["finishedAt"] for item in terminated if item.get("finishedAt")]
    if not raw_start or not finished_values:
        return None
    start = parse_kubernetes_timestamp(raw_start)
    finish = max(parse_kubernetes_timestamp(value) for value in finished_values)
    metadata = pod.get("metadata", {})
    return PodExecution(
        uid=metadata["uid"],
        name=metadata["name"],
        phase=status.get("phase"),
        node_name=pod.get("spec", {}).get("nodeName"),
        started_at=start,
        finished_at=finish,
        duration_seconds=max(0.0, (finish - start).total_seconds()),
        succeeded=status.get("phase") == "Succeeded",
    )


def _container_provenance(spec: dict[str, Any], statuses: dict[str, dict]) -> ContainerProvenance:
    name = spec.get("name", "")
    status = statuses.get(name, {})
    resources = spec.get("resources", {}) or {}
    requests = resources.get("requests", {}) or {}
    limits = resources.get("limits", {}) or {}
    env = {
        item["name"]: str(item.get("value"))
        for item in (spec.get("env", []) or [])
        # valueFrom (secrets/configmaps/fieldRef) is deliberately not resolved.
        if item.get("value") is not None and "name" in item
    }
    return ContainerProvenance(
        name=name,
        image=spec.get("image"),
        image_id=status.get("imageID"),
        command=list(spec.get("command", []) or []),
        args=list(spec.get("args", []) or []),
        env=env,
        cpu_request=requests.get("cpu"),
        cpu_limit=limits.get("cpu"),
        memory_request=requests.get("memory"),
        memory_limit=limits.get("memory"),
    )


def pod_provenance(pod: dict[str, Any]) -> PodProvenance:
    """Extract exactly what ran: resolved image digests, command, resources."""
    metadata = pod.get("metadata", {})
    spec = pod.get("spec", {}) or {}
    status = pod.get("status", {}) or {}
    statuses = {
        item.get("name"): item
        for item in [
            *(status.get("initContainerStatuses", []) or []),
            *(status.get("containerStatuses", []) or []),
        ]
        if item.get("name")
    }
    containers = [
        _container_provenance(item, statuses)
        for item in [
            *(spec.get("initContainers", []) or []),
            *(spec.get("containers", []) or []),
        ]
    ]
    return PodProvenance(
        pod_uid=metadata.get("uid", ""),
        pod_name=metadata.get("name", ""),
        node_name=spec.get("nodeName"),
        node_selector=dict(spec.get("nodeSelector", {}) or {}),
        containers=containers,
    )


def job_provenance(job: dict[str, Any], pods: list[dict[str, Any]]) -> JobProvenance:
    spec = job.get("spec", {}) or {}
    return JobProvenance(
        backoff_limit=spec.get("backoffLimit"),
        completions=spec.get("completions"),
        parallelism=spec.get("parallelism"),
        pods=[pod_provenance(pod) for pod in pods],
    )


class PrometheusPortForward:
    """Own a temporary port-forward for the lifetime of the observer command."""

    def __init__(
        self,
        kubectl: KubectlClient,
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

    def __enter__(self) -> PrometheusPortForward:
        self.process = subprocess.Popen(
            [
                *self.kubectl.prefix,
                "-n",
                self.namespace,
                "port-forward",
                f"svc/{self.service}",
                f"{self.local_port}:{self.remote_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise KubernetesCommandError("Prometheus port-forward terminated unexpectedly")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return self
            time.sleep(0.25)
        self.__exit__(None, None, None)
        raise KubernetesCommandError("Prometheus port-forward was not ready after 30 seconds")

    def __exit__(self, *_: Any) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
