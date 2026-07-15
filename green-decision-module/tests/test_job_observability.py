"""Pure tests for automatic Kepler/RTE per-Job accounting."""

from __future__ import annotations

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


class FakePrometheus:
    def query_range(self, expression, start, end, step):
        assert 'zone="package"' in expression
        return [
            {
                "metric": {
                    "pod_id": "pod-uid-1",
                    "pod_name": "example-abc",
                    "pod_namespace": "experiments",
                    "node_name": "node-0",
                    "zone": "package",
                },
                "values": [
                    [epoch(0), "0"],
                    [epoch(1), "600"],
                    [epoch(2), "1800"],
                    [epoch(2 + 20 / 60), "2000"],
                ],
            }
        ]


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
