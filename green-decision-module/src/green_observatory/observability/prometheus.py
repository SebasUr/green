"""Prometheus HTTP client and Kepler 0.11 pod-counter queries."""

from __future__ import annotations

from typing import Any

import httpx


class PrometheusError(RuntimeError):
    pass


class PrometheusClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = httpx.get(
                f"{self.base_url}{path}", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PrometheusError(f"Prometheus request failed: {exc}") from exc
        if payload.get("status") != "success":
            raise PrometheusError(f"Prometheus returned an error: {payload}")
        return payload.get("data", {})

    def query_range(
        self,
        expression: str,
        start: float,
        end: float,
        step: int,
    ) -> list[dict[str, Any]]:
        data = self.get(
            "/api/v1/query_range",
            {
                "query": expression,
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "step": str(step),
            },
        )
        return list(data.get("result", []))

    def query(self, expression: str, at: float) -> list[dict[str, Any]]:
        """Instant query evaluated at ``at`` (epoch seconds)."""
        data = self.get("/api/v1/query", {"query": expression, "time": f"{at:.3f}"})
        return list(data.get("result", []))


def kepler_pod_counter_query(pod_uid_pattern: str, zone: str) -> str:
    """Normalize Kepler's duplicated running/terminated state series."""
    return (
        "max by(pod_id,pod_name,pod_namespace,node_name,zone)("
        "kepler_pod_cpu_joules_total{"
        f'pod_id=~"{pod_uid_pattern}",zone="{zone}"'
        "})"
    )


def kepler_node_counter_query(metric: str, node: str, zone: str) -> str:
    """Node-level cumulative energy counter for one RAPL zone."""
    return f'sum(max without(state)({metric}{{node_name="{node}",zone="{zone}"}}))'


def kepler_node_cpu_ratio_query(node: str) -> str:
    return f'kepler_node_cpu_usage_ratio{{node_name="{node}"}}'


def kepler_node_pods_counter_query(node: str, zone: str) -> str:
    """Every pod Kepler attributed energy to on this node (co-tenants included)."""
    return (
        "max by(pod_id,pod_name,pod_namespace)("
        "kepler_pod_cpu_joules_total{"
        f'node_name="{node}",zone="{zone}"'
        "})"
    )


def kepler_up_ratio_query(window_seconds: int) -> str:
    """Fraction of the window during which the Kepler exporter was scrapeable."""
    return f'avg_over_time(up{{job="kepler"}}[{max(1, int(window_seconds))}s])'


def node_container_restarts_query(node: str, window_seconds: int) -> str:
    """Container restarts on the node during the window (needs kube-state-metrics)."""
    window = max(1, int(window_seconds))
    return (
        f"sum(increase(kube_pod_container_status_restarts_total[{window}s])"
        f' * on(namespace,pod) group_left() kube_pod_info{{node="{node}"}})'
    )
