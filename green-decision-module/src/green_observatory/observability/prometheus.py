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


def kepler_pod_counter_query(pod_uid_pattern: str, zone: str) -> str:
    """Normalize Kepler's duplicated running/terminated state series."""
    return (
        "max by(pod_id,pod_name,pod_namespace,node_name,zone)("
        "kepler_pod_cpu_joules_total{"
        f'pod_id=~"{pod_uid_pattern}",zone="{zone}"'
        "})"
    )
