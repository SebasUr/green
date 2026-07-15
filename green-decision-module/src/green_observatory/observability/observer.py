"""Polling observer that writes one JSON report per labelled terminal Job."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from green_observatory.observability.cluster import KubectlClient, job_outcome
from green_observatory.observability.models import JobCarbonReport
from green_observatory.observability.reporter import JobReporter


def report_path(output: Path, report: JobCarbonReport) -> Path:
    identity = report.job
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{identity.namespace}__{identity.name}")
    return output / f"{safe}__{identity.uid}.json"


def write_report(output: Path, report: JobCarbonReport) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    destination = report_path(output, report)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def existing_final_report(output: Path, job_uid: str) -> Path | None:
    for path in output.glob(f"*__{job_uid}.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("quality", {}).get("final") is True:
            return path
    return None


class JobObserver:
    def __init__(
        self,
        kubectl: KubectlClient,
        reporter: JobReporter,
        output: Path,
        *,
        selector: str = "sustainability.cern.ch/track=true",
        namespace: str | None = None,
        max_job_age_seconds: float = 7 * 86400,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.kubectl = kubectl
        self.reporter = reporter
        self.output = output
        self.selector = selector
        self.namespace = namespace
        self.max_job_age_seconds = max_job_age_seconds
        self.emit = emit

    def _recent_enough(self, job: dict) -> bool:
        status = job.get("status", {})
        raw = status.get("completionTime")
        if not raw:
            terminal_conditions = [
                item
                for item in status.get("conditions", []) or []
                if item.get("status") == "True" and item.get("type") in {"Complete", "Failed"}
            ]
            raw = terminal_conditions[-1].get("lastTransitionTime") if terminal_conditions else None
        if not raw:
            return True
        finished = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - finished).total_seconds() <= self.max_job_age_seconds

    def run_once(self) -> list[Path]:
        written: list[Path] = []
        for job in self.kubectl.list_jobs(self.selector, self.namespace):
            if job_outcome(job) is None or not self._recent_enough(job):
                continue
            metadata = job["metadata"]
            uid = metadata["uid"]
            if existing_final_report(self.output, uid):
                continue
            label = f"{metadata['namespace']}/{metadata['name']}"
            try:
                pods = self.kubectl.pods_for_job(job)
                report = self.reporter.build(job, pods)
                path = write_report(self.output, report)
                written.append(path)
                state = "FINAL" if report.quality.final else "PROVISIONAL"
                emissions = report.carbon.emissions_gco2eq
                value = f"{emissions:.6f} gCO2eq" if emissions is not None else "unavailable"
                self.emit(f"[{state}] {label}: {value} -> {path}")
            except Exception as exc:  # keep observing other jobs and retry next poll
                self.emit(f"[RETRY] {label}: {exc}")
        return written

    def run_forever(self, poll_seconds: int = 30) -> None:
        self.emit(
            f"Observing Jobs selector={self.selector!r} "
            f"namespace={self.namespace or 'all'} every {poll_seconds}s"
        )
        while True:
            self.run_once()
            time.sleep(poll_seconds)
