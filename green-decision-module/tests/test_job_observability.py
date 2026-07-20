"""Pure tests for automatic Kepler/RTE per-Job accounting."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd
import pytest

from green_observatory.observability.accounting import account_carbon, measure_counter
from green_observatory.observability.observer import JobObserver
from green_observatory.observability.reporter import JobReporter
from green_observatory.observability.summary import summarize_reports


def epoch(minutes: float = 0) -> float:
    return datetime(2026, 7, 14, tzinfo=timezone.utc).timestamp() + minutes * 60


def test_counter_delta_interpolates_boundaries_and_handles_reset():
    measured = measure_counter(
        [
            (epoch(0), 100),
            (epoch(1), 700),
            (epoch(2), 50),  # reset: contributes the new value
            (epoch(3), 650),
        ],
        epoch(0.5),
        epoch(2.5),
    )
    # Half of 600 + reset interval 50 + half of 600.
    assert measured.energy_joules == pytest.approx(650)
    assert measured.counter_resets == 1
    assert measured.coverage_ratio == 1.0
    assert [(item.start, item.end) for item in measured.increments] == [
        (epoch(0.5), epoch(1)),
        (epoch(1), epoch(2)),
        (epoch(2), epoch(2.5)),
    ]


def test_counter_reports_partial_boundary_coverage():
    measured = measure_counter(
        [(epoch(1), 10), (epoch(2), 70)],
        epoch(0),
        epoch(3),
        boundary_tolerance_seconds=120,
    )
    assert measured.energy_joules == 60
    assert measured.coverage_ratio == pytest.approx(1 / 3)
    assert measured.warnings


def test_new_pod_counter_can_assume_zero_at_creation():
    measured = measure_counter(
        [(epoch(0.2), 120), (epoch(1), 600)],
        epoch(0),
        epoch(1),
        assume_created_at_start=True,
    )
    assert measured.energy_joules == 600
    assert measured.coverage_ratio == 1.0
    assert "assumed to start at zero" in measured.warnings[0]


def test_carbon_is_energy_weighted_not_time_averaged():
    increments = measure_counter(
        [(epoch(0), 0), (epoch(1), 600), (epoch(2), 1800)],
        epoch(0),
        epoch(2),
    ).increments
    carbon = pd.Series(
        [10.0, 40.0],
        index=pd.to_datetime([epoch(0), epoch(1)], unit="s", utc=True),
    )
    result = account_carbon(increments, carbon)
    assert result.accounted_energy_joules == 1800
    assert result.weighted_intensity_gco2eq_per_kwh == pytest.approx(30.0)
    assert result.emissions_gco2eq == pytest.approx(1800 / 3_600_000 * 30)
    assert result.energy_coverage_ratio == 1.0


JOB_POD_SAMPLES = [
    [epoch(0), "0"],
    [epoch(1), "600"],
    [epoch(2), "1800"],
    [epoch(2 + 20 / 60), "2000"],
]


def _series(metric, values):
    return {"metric": metric, "values": values}


class FakePrometheus:
    """Routes by metric so each of the reporter's queries gets a coherent answer."""

    def __init__(self, co_tenant_joules: float = 0.0, restarts: float = 0.0,
                 kepler_up: float = 1.0) -> None:
        self.co_tenant_joules = co_tenant_joules
        self.restarts = restarts
        self.kepler_up = kepler_up

    def _job_pod(self):
        return _series(
            {"pod_id": "pod-uid-1", "pod_name": "example-abc",
             "pod_namespace": "experiments", "node_name": "node-0", "zone": "package"},
            JOB_POD_SAMPLES,
        )

    def query_range(self, expression, start, end, step):
        if "kepler_node_cpu_usage_ratio" in expression:
            return [_series({"node_name": "node-0"}, [[epoch(0), "0.25"], [epoch(2), "0.30"]])]
        if "kepler_node_cpu_active_joules_total" in expression:
            return [_series({"node_name": "node-0", "zone": "package"},
                            [[epoch(0), "0"], [epoch(2), "2000"]])]
        if "kepler_node_cpu_idle_joules_total" in expression:
            return [_series({"node_name": "node-0", "zone": "package"},
                            [[epoch(0), "0"], [epoch(2), "5000"]])]
        if "kepler_node_cpu_joules_total" in expression:
            return [_series({"node_name": "node-0", "zone": "package"},
                            [[epoch(0), "0"], [epoch(2), "7000"]])]
        assert 'zone="package"' in expression
        if 'node_name="node-0"' in expression:  # every pod on the node (co-tenants)
            pods = [self._job_pod()]
            if self.co_tenant_joules:
                pods.append(_series(
                    {"pod_id": "pod-uid-other", "pod_name": "noisy-neighbour",
                     "pod_namespace": "kube-system"},
                    [[epoch(0), "0"], [epoch(2), str(self.co_tenant_joules)]],
                ))
            return pods
        return [self._job_pod()]

    def query(self, expression, at):
        if "restarts" in expression:
            return [{"metric": {}, "value": [at, str(self.restarts)]}]
        if "up{" in expression:
            return [{"metric": {}, "value": [at, str(self.kepler_up)]}]
        return []


class FakeCarbon:
    def load(self, start, end):
        return pd.Series(
            [10.0, 20.0],
            index=pd.to_datetime([epoch(0), epoch(1)], unit="s", utc=True),
            name="carbon_intensity_gco2_kwh",
        )


def test_reporter_builds_final_multi_source_contract():
    job = {
        "metadata": {
            "uid": "job-uid-1",
            "name": "example",
            "namespace": "experiments",
            "labels": {"sustainability.cern.ch/track": "true"},
            "annotations": {
                "sustainability.cern.ch/scheduler": "green-window",
                "kubectl.kubernetes.io/last-applied-configuration": "large payload",
            },
        },
        "status": {
            "succeeded": 1,
            "conditions": [{"type": "Complete", "status": "True"}],
        },
    }
    pod = {
        "metadata": {
            "uid": "pod-uid-1",
            "name": "example-abc",
            "ownerReferences": [{"kind": "Job", "uid": "job-uid-1"}],
        },
        "spec": {"nodeName": "node-0"},
        "status": {
            "phase": "Succeeded",
            "startTime": "2026-07-14T00:00:10Z",
            "containerStatuses": [
                {
                    "state": {
                        "terminated": {
                            "startedAt": "2026-07-14T00:00:10Z",
                            "finishedAt": "2026-07-14T00:02:10Z",
                        }
                    }
                }
            ],
        },
    }
    report = JobReporter(FakePrometheus(), FakeCarbon()).build(job, [pod])
    assert report.quality.final is True
    assert report.quality.valid is True
    assert report.energy.total_joules == pytest.approx(1800)
    assert report.carbon.energy_weighted_intensity_gco2eq_per_kwh == pytest.approx(
        17.2222222222
    )
    assert report.carbon.emissions_gco2eq == pytest.approx(1800 / 3_600_000 * 17.2222222222)
    assert report.energy.scope == "operational_cpu_energy_attributed_by_kepler"
    assert report.carbon.accounting_scope == "operational_cpu_no_pue"
    assert report.job.annotations == {
        "sustainability.cern.ch/scheduler": "green-window"
    }


WORKLOAD_STDOUT = '{"pi_estimate": 3.14159, "total_samples": 1000}\n'


class FakeKubectl:
    def __init__(self, stdout: str = WORKLOAD_STDOUT) -> None:
        self.stdout = stdout

    def pod_logs(self, namespace, name, container=None):
        return self.stdout


def _job_with_spec():
    return {
        "metadata": {"uid": "job-uid-1", "name": "example", "namespace": "experiments",
                     "labels": {"sustainability.cern.ch/track": "true"}},
        "spec": {"backoffLimit": 0, "completions": 1, "parallelism": 1},
        "status": {"succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}]},
    }


def _pod_with_spec():
    return {
        "metadata": {"uid": "pod-uid-1", "name": "example-abc"},
        "spec": {
            "nodeName": "node-0",
            "nodeSelector": {"sustainability.cern.ch/hardware": "baremetal"},
            "containers": [{
                "name": "monte-carlo",
                "image": "python:3.12-slim",
                "command": ["python", "/workload/monte_carlo.py"],
                "env": [
                    {"name": "WORKERS", "value": "16"},
                    # valueFrom must not be resolved into the report.
                    {"name": "TOKEN", "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}},
                ],
                "resources": {"requests": {"cpu": "16", "memory": "2Gi"},
                              "limits": {"cpu": "16", "memory": "2Gi"}},
            }],
        },
        "status": {
            "phase": "Succeeded",
            "startTime": "2026-07-14T00:00:10Z",
            "containerStatuses": [{
                "name": "monte-carlo",
                "imageID": "docker.io/library/python@sha256:deadbeef",
                "state": {"terminated": {"startedAt": "2026-07-14T00:00:10Z",
                                         "finishedAt": "2026-07-14T00:02:10Z"}},
            }],
        },
    }


def test_report_captures_provenance_node_context_and_output_hash():
    report = JobReporter(
        FakePrometheus(), FakeCarbon(), kubectl=FakeKubectl()
    ).build(_job_with_spec(), [_pod_with_spec()])

    container = report.provenance.pods[0].containers[0]
    assert container.image == "python:3.12-slim"
    assert container.image_id == "docker.io/library/python@sha256:deadbeef"
    assert container.command == ["python", "/workload/monte_carlo.py"]
    assert container.cpu_limit == "16"
    # Literal env is recorded; secret references are deliberately not resolved.
    assert container.env == {"WORKERS": "16"}
    assert report.provenance.pods[0].node_selector == {
        "sustainability.cern.ch/hardware": "baremetal"
    }

    assert report.node_context.node_name == "node-0"
    assert report.node_context.active_energy_joules == pytest.approx(1833.333, rel=1e-3)
    assert report.node_context.job_share_of_active_energy == pytest.approx(0.9818, rel=1e-3)
    assert report.node_context.cpu_utilization_mean == pytest.approx(0.275)

    output = report.workload_outputs[0]
    assert output.stdout_sha256 == hashlib.sha256(WORKLOAD_STDOUT.encode()).hexdigest()
    assert output.parsed_json == {"pi_estimate": 3.14159, "total_samples": 1000}
    assert output.truncated is False
    assert report.schema_version == "1.1"


def test_isolation_is_clean_when_the_job_owns_the_node():
    report = JobReporter(FakePrometheus(), FakeCarbon()).build(
        _job_with_spec(), [_pod_with_spec()]
    )
    assert report.isolation.clean_node is True
    assert report.isolation.co_tenant_pods == []
    assert report.isolation.co_tenant_energy_share == pytest.approx(0.0)
    assert report.isolation.kepler_up_ratio == pytest.approx(1.0)
    assert report.isolation.container_restarts == 0


def test_isolation_flags_a_noisy_neighbour_without_invalidating_accounting():
    report = JobReporter(
        FakePrometheus(co_tenant_joules=500), FakeCarbon()
    ).build(_job_with_spec(), [_pod_with_spec()])

    assert report.isolation.clean_node is False
    assert [item.pod_name for item in report.isolation.co_tenant_pods] == ["noisy-neighbour"]
    assert report.isolation.co_tenant_energy_share == pytest.approx(0.2029, rel=1e-2)
    assert any("co-tenant" in item for item in report.isolation.warnings)
    # Contamination is a scientific-comparability signal, not a measurement fault:
    # the accounting stays valid/final so the observer does not retry it forever.
    assert report.quality.valid is True
    assert report.quality.final is True
    assert report.energy.total_joules == pytest.approx(1800)


def test_isolation_flags_restarts_and_kepler_gaps():
    report = JobReporter(
        FakePrometheus(restarts=3, kepler_up=0.5), FakeCarbon()
    ).build(_job_with_spec(), [_pod_with_spec()])
    assert report.isolation.clean_node is False
    assert report.isolation.container_restarts == 3
    assert report.isolation.kepler_up_ratio == pytest.approx(0.5)
    assert any("restart" in item for item in report.isolation.warnings)
    assert any("scrapeable" in item for item in report.isolation.warnings)


def test_energy_intervals_are_persisted_only_when_requested_and_reconcile():
    plain = JobReporter(FakePrometheus(), FakeCarbon()).build(
        _job_with_spec(), [_pod_with_spec()]
    )
    assert plain.energy_intervals is None

    detailed = JobReporter(FakePrometheus(), FakeCarbon(), include_intervals=True).build(
        _job_with_spec(), [_pod_with_spec()]
    )
    # The audit trail must reconstruct the headline totals exactly.
    assert sum(item.joules for item in detailed.energy_intervals) == pytest.approx(
        detailed.energy.total_joules
    )
    assert sum(
        item.emissions_gco2eq for item in detailed.energy_intervals
    ) == pytest.approx(detailed.carbon.emissions_gco2eq)
    assert all(
        item.carbon_intensity_gco2eq_per_kwh in (10.0, 20.0)
        for item in detailed.energy_intervals
    )


def test_context_collection_can_be_disabled():
    report = JobReporter(
        FakePrometheus(), FakeCarbon(), kubectl=FakeKubectl(), collect_context=False,
        capture_logs=False,
    ).build(_job_with_spec(), [_pod_with_spec()])
    assert report.provenance is None
    assert report.node_context is None
    assert report.isolation is None
    assert report.workload_outputs == []


def test_observer_is_idempotent_and_summary_preserves_dimensions(tmp_path):
    report = JobReporter(FakePrometheus(), FakeCarbon()).build(
        {
            "metadata": {
                "uid": "job-uid-1",
                "name": "example",
                "namespace": "experiments",
                "labels": {
                    "sustainability.cern.ch/track": "true",
                    "sustainability.cern.ch/workload": "monte-carlo",
                    "sustainability.cern.ch/policy": "green-window",
                },
            },
            "status": {
                "succeeded": 1,
                "completionTime": "2026-07-14T00:02:10Z",
                "conditions": [{"type": "Complete", "status": "True"}],
            },
        },
        [
            {
                "metadata": {"uid": "pod-uid-1", "name": "example-abc"},
                "spec": {"nodeName": "node-0"},
                "status": {
                    "phase": "Succeeded",
                    "startTime": "2026-07-14T00:00:10Z",
                    "containerStatuses": [
                        {
                            "state": {
                                "terminated": {
                                    "startedAt": "2026-07-14T00:00:10Z",
                                    "finishedAt": "2026-07-14T00:02:10Z",
                                }
                            }
                        }
                    ],
                },
            }
        ],
    )

    class FakeKubectl:
        def list_jobs(self, selector, namespace):
            return [{"metadata": {"uid": report.job.uid}}]

        def pods_for_job(self, job):
            return []

    class StaticReporter:
        def build(self, job, pods):
            return report

    # Supply terminal metadata to the observer while keeping report generation static.
    terminal = {
        "metadata": {
            "uid": report.job.uid,
            "name": report.job.name,
            "namespace": report.job.namespace,
        },
        "status": {
            "succeeded": 1,
            "completionTime": "2026-07-14T00:02:10Z",
            "conditions": [{"type": "Complete", "status": "True"}],
        },
    }
    fake_kubectl = FakeKubectl()
    fake_kubectl.list_jobs = lambda selector, namespace: [terminal]
    observer = JobObserver(
        fake_kubectl,
        StaticReporter(),
        tmp_path,
        max_job_age_seconds=10**9,
        emit=lambda message: None,
    )
    assert len(observer.run_once()) == 1
    assert observer.run_once() == []

    frame = summarize_reports(tmp_path)
    assert len(frame) == 1
    assert frame.iloc[0]["workload"] == "monte-carlo"
    assert frame.iloc[0]["policy"] == "green-window"
    assert frame.iloc[0]["emissions_gco2eq"] == pytest.approx(
        report.carbon.emissions_gco2eq
    )
