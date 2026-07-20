#!/usr/bin/env python3
"""Generate the two Job-centric Grafana dashboards.

  jobs-overview.json  — a clickable table of Jobs that ran (from kube-state-metrics)
  job-drilldown.json  — per-Job energy/CO2 curves (Kepler + RTE recording rules)

Provenance, on purpose: these dashboards are for EXPLORATION. Numbers here are
recomputed from Prometheus and are approximate (job-controller window, Riemann
sums). The authoritative per-Job accounting stays `greenctl jobs report`
(pod-precise boundaries, reset-aware counters, per-interval carbon weighting,
quality gates). Both dashboards say so in a text panel.

Grounded on what this cluster actually exposes (verified live):
  kube_job_info / kube_job_status_start_time / kube_job_status_completion_time
  kube_job_labels{label_sustainability_cern_ch_policy|trial|experiment|workload}
    (requires KSM --metric-labels-allowlist; see monitoring-values.yaml)
  kepler_pod_cpu_watts / kepler_pod_cpu_joules_total  (zone=package, dedup state)
  pod_co2_emission_gco2_per_second                    (recording rule)
"""
from __future__ import annotations

import json
import os

DS = {"type": "prometheus", "uid": "${datasource}"}
DRILL_UID = "green-obs-job-drill"
NS = "$namespace"
JOB = "$job"
ZONE = "package"

# Job pods are named "<job>-<suffix>", so this is a reliable job -> pod join.
POD_SEL = f'pod_namespace=~"{NS}",pod_name=~"{JOB}-.*",zone="{ZONE}"'
POD_WATTS = f'sum(max without(state)(kepler_pod_cpu_watts{{{POD_SEL}}}))'
POD_KWH_COUNTER = f'sum(max without(state)(kepler_pod_cpu_joules_total{{{POD_SEL}}}))/3600000'
POD_KWH_RANGE = (
    f'sum(max without(state)(increase(kepler_pod_cpu_joules_total{{{POD_SEL}}}[$__range])))/3600000'
)
POD_CO2_S = f'sum(pod_co2_emission_gco2_per_second{{pod_namespace=~"{NS}",pod_name=~"{JOB}-.*"}})'

_id = [0]


def nid() -> int:
    _id[0] += 1
    return _id[0]


def row(title, y):
    return {"id": nid(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "datasource": DS, "panels": []}


def stat(title, expr, x, y, unit="short", w=6, h=4, decimals=2, desc=""):
    return {
        "id": nid(), "type": "stat", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"refId": "A", "expr": expr, "datasource": DS, "instant": True}],
        "fieldConfig": {"defaults": {
            "unit": unit, "decimals": decimals, "color": {"mode": "thresholds"},
            "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            "mappings": [],
        }, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "orientation": "auto", "colorMode": "value", "graphMode": "area",
                    "textMode": "auto", "justifyMode": "auto"},
    }


def timeseries(title, targets, x, y, unit="watt", w=12, h=8, desc="", fill=15):
    return {
        "id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [dict(t, datasource=DS) for t in targets],
        "fieldConfig": {"defaults": {
            "unit": unit, "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineInterpolation": "smooth", "lineWidth": 2,
                       "fillOpacity": fill, "gradientMode": "opacity", "spanNulls": False,
                       "showPoints": "never", "pointSize": 5,
                       "stacking": {"mode": "none", "group": "A"}, "axisPlacement": "auto"},
            "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
        }, "overrides": []},
        "options": {"tooltip": {"mode": "multi", "sort": "desc"},
                    "legend": {"displayMode": "list", "placement": "bottom",
                               "calcs": ["mean", "max", "lastNotNull"]}},
    }


def text(title, content, x, y, w=24, h=4):
    return {"id": nid(), "type": "text", "title": title,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "options": {"mode": "markdown", "content": content}}


def datasource_var():
    return {"name": "datasource", "type": "datasource", "query": "prometheus",
            "label": "Data source", "current": {}, "hide": 0, "refresh": 1, "regex": ""}


# ===========================================================================
# Dashboard 1 — Jobs overview (clickable list)
# ===========================================================================
panels: list[dict] = []
y = 0
panels.append(row("Jobs that ran (click a row to drill into its energy)", y)); y += 1

LBL = 'label_sustainability_cern_ch_policy,label_sustainability_cern_ch_trial,' \
      'label_sustainability_cern_ch_workload,label_sustainability_cern_ch_experiment'
JOB_FILTER = (
    f'kube_job_labels{{namespace=~"{NS}"'
    ',label_sustainability_cern_ch_policy=~"$policy"'
    ',label_sustainability_cern_ch_trial=~"$trial"}'
)

panels.append({
    "id": nid(), "type": "table",
    "title": "Jobs",
    "description": (
        "From kube-state-metrics. Click the job name to open the drilldown scoped to that Job. "
        "Times are the Job-controller window; greenctl uses pod-precise boundaries, so the "
        "duration here can differ by a few seconds."
    ),
    "datasource": DS,
    "gridPos": {"h": 12, "w": 24, "x": 0, "y": y},
    "targets": [
        # Start (ms) with the experiment dimensions joined on.
        {"refId": "A", "datasource": DS, "format": "table", "instant": True,
         "expr": f'(kube_job_status_start_time{{namespace=~"{NS}"}} * 1000)'
                 f' * on(namespace, job_name) group_left({LBL}) {JOB_FILTER}'},
        # End (ms). Absent while a Job is still running -> empty cell (outer join).
        {"refId": "B", "datasource": DS, "format": "table", "instant": True,
         "expr": f'(kube_job_status_completion_time{{namespace=~"{NS}"}} * 1000)'
                 f' * on(namespace, job_name) group_left() {JOB_FILTER}'},
        # Succeeded / failed counters (x1 from kube_job_labels keeps the same filter).
        {"refId": "C", "datasource": DS, "format": "table", "instant": True,
         "expr": f'kube_job_status_succeeded{{namespace=~"{NS}"}}'
                 f' * on(namespace, job_name) group_left() {JOB_FILTER}'},
        {"refId": "D", "datasource": DS, "format": "table", "instant": True,
         "expr": f'kube_job_status_failed{{namespace=~"{NS}"}}'
                 f' * on(namespace, job_name) group_left() {JOB_FILTER}'},
    ],
    "transformations": [
        {"id": "joinByField", "options": {"byField": "job_name", "mode": "outer"}},
        {"id": "calculateField", "options": {
            "alias": "Duration",
            "mode": "binary",
            "binary": {"left": "Value #B", "operator": "-", "right": "Value #A"},
            "replaceFields": False,
        }},
        {"id": "organize", "options": {
            "renameByName": {
                "job_name": "Job",
                "namespace": "Namespace",
                "label_sustainability_cern_ch_policy": "Policy",
                "label_sustainability_cern_ch_trial": "Trial",
                "label_sustainability_cern_ch_workload": "Workload",
                "label_sustainability_cern_ch_experiment": "Experiment",
                "Value #A": "Start",
                "Value #B": "End",
                "Value #C": "Succeeded",
                "Value #D": "Failed",
            },
            "excludeByName": {
                "Time": True, "Time 1": True, "Time 2": True, "Time 3": True, "Time 4": True,
                "namespace 2": True, "namespace 3": True, "namespace 4": True,
                "__name__": True, "job": True, "instance": True, "container": True,
                "endpoint": True, "service": True, "pod": True, "uid": True,
                "label_sustainability_cern_ch_experiment": True,
            },
            "indexByName": {},
        }},
    ],
    "fieldConfig": {
        "defaults": {"custom": {"align": "auto", "filterable": True}},
        "overrides": [
            # The clickable column: opens the drilldown with the Job's own window.
            {"matcher": {"id": "byName", "options": "Job"}, "properties": [
                {"id": "links", "value": [{
                    "title": "Energy & CO₂ drilldown for this Job",
                    "url": f'/d/{DRILL_UID}/job-drilldown'
                           '?var-datasource=${datasource}'
                           '&var-namespace=${__data.fields["Namespace"]}'
                           '&var-job=${__data.fields["Job"]}'
                           '&from=${__data.fields["Start"]}'
                           '&to=${__data.fields["End"]}',
                    "targetBlank": False,
                }]},
                {"id": "custom.width", "value": 260},
            ]},
            {"matcher": {"id": "byName", "options": "Start"},
             "properties": [{"id": "unit", "value": "dateTimeAsIso"}]},
            {"matcher": {"id": "byName", "options": "End"},
             "properties": [{"id": "unit", "value": "dateTimeAsIso"}]},
            {"matcher": {"id": "byName", "options": "Duration"},
             "properties": [{"id": "unit", "value": "ms"}, {"id": "decimals", "value": 0}]},
            {"matcher": {"id": "byName", "options": "Succeeded"},
             "properties": [{"id": "custom.width", "value": 100}]},
            {"matcher": {"id": "byName", "options": "Failed"},
             "properties": [{"id": "custom.width", "value": 90},
                            {"id": "custom.cellOptions",
                             "value": {"type": "color-background", "mode": "basic"}},
                            {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                                {"color": "green", "value": None}, {"color": "red", "value": 1}]}}]},
        ],
    },
    "options": {"showHeader": True, "sortBy": [{"displayName": "Start", "desc": True}]},
})
y += 12
panels.append(text(
    "How to read this",
    "**Exploration view.** The table lists Kubernetes Jobs currently known to "
    "kube-state-metrics (live objects; deleted Jobs disappear from the list even though their "
    "Kepler series stay in Prometheus for the retention window).\n\n"
    "Click a **Job** name to open the drilldown with the time range preset to that Job's window.\n\n"
    "> For authoritative energy & carbon numbers use `greenctl jobs report <job> -n <ns>` — it uses "
    "pod-precise boundaries, reset-aware counters and per-interval RTE weighting. These dashboards "
    "recompute from Prometheus and are approximate.",
    0, y))

jobs_dashboard = {
    "uid": "green-obs-jobs",
    "title": "Green Observatory — Jobs",
    "tags": ["green-observatory", "kepler", "carbon", "jobs", "cern"],
    "timezone": "utc", "schemaVersion": 39, "version": 1, "editable": True,
    "graphTooltip": 0, "refresh": "30s",
    "time": {"from": "now-7d", "to": "now"},
    "annotations": {"list": [{"builtIn": 1, "type": "dashboard", "name": "Annotations & Alerts",
                              "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                              "enable": True, "hide": True,
                              "iconColor": "rgba(0, 211, 255, 1)"}]},
    "templating": {"list": [
        datasource_var(),
        {"name": "namespace", "type": "query", "datasource": DS, "label": "Namespace",
         "query": "label_values(kube_job_info, namespace)",
         "current": {"text": "All", "value": "$__all"},
         "refresh": 2, "sort": 1, "includeAll": True, "allValue": ".*", "multi": True},
        {"name": "policy", "type": "query", "datasource": DS, "label": "Policy",
         "query": "label_values(kube_job_labels, label_sustainability_cern_ch_policy)",
         "current": {"text": "All", "value": "$__all"},
         "refresh": 2, "sort": 1, "includeAll": True, "allValue": ".*", "multi": True},
        {"name": "trial", "type": "query", "datasource": DS, "label": "Trial",
         "query": "label_values(kube_job_labels, label_sustainability_cern_ch_trial)",
         "current": {"text": "All", "value": "$__all"},
         "refresh": 2, "sort": 1, "includeAll": True, "allValue": ".*", "multi": True},
    ]},
    "panels": panels,
}

# ===========================================================================
# Dashboard 2 — Job drilldown
# ===========================================================================
_id[0] = 0
dpanels: list[dict] = []
y = 0
dpanels.append(row("Job $job — energy & CO₂ (exploration)", y)); y += 1
dpanels.append(stat("Energy (range)", POD_KWH_RANGE, 0, y, unit="kwatth", decimals=6,
                    desc="Kepler pod CPU energy accumulated over the selected range. "
                         "Authoritative value: greenctl jobs report."))
dpanels.append(stat("Average power (range)",
                    f'{POD_KWH_RANGE} * 3600000 / ($__range_s)', 6, y, unit="watt", decimals=2,
                    desc="Range energy / range duration. Meaningful when the range is the Job window."))
dpanels.append(stat("Cumulative CO₂ (range)",
                    f'sum_over_time({POD_CO2_S}[$__range:15s])*15', 12, y, unit="short", decimals=4,
                    desc="gCO2eq accumulated over the range, per-interval weighted via the "
                         "pod_co2_emission recording rule (Riemann, 15s)."))
dpanels.append(stat("Peak power (range)",
                    f'max_over_time(({POD_WATTS})[$__range:15s])', 18, y, unit="watt", decimals=2,
                    desc="Highest attributed pod power seen in the range."))
y += 4
dpanels.append(timeseries("Pod CPU power", [
    {"refId": "A", "expr": POD_WATTS, "legendFormat": "$job"},
], 0, y, unit="watt", desc="Kepler-attributed CPU power for this Job's pod(s)."))
dpanels.append(timeseries("Pod CO₂ emission rate", [
    {"refId": "A", "expr": f'{POD_CO2_S} * 3600', "legendFormat": "$job"},
], 12, y, unit="short", desc="gCO2eq/h = pod watts × live RTE intensity. Needs the recording "
                            "rules + rte-carbon-exporter."))
y += 8
dpanels.append(timeseries("Cumulative energy (Kepler counter)", [
    {"refId": "A", "expr": POD_KWH_COUNTER, "legendFormat": "$job"},
], 0, y, unit="kwatth", desc="The raw cumulative Kepler counter in kWh — should rise "
                            "monotonically while the pod runs and flatten when it ends.", fill=25))
dpanels.append(timeseries("Grid carbon intensity during the run", [
    {"refId": "A", "expr": "max(rte_carbon_intensity_gco2eq_per_kwh)",
     "legendFormat": "RTE FR gCO2eq/kWh"},
], 12, y, unit="short", desc="What the grid was doing while this Job ran.", fill=20))
y += 8
dpanels.append(text(
    "Provenance",
    "These panels are **recomputed from Prometheus** (Kepler + RTE exporter) and are meant for "
    "exploring the shape of a run.\n\n"
    "The **authoritative** energy and carbon for this Job — pod-precise boundaries, reset-aware "
    "counter deltas, per-interval RTE weighting and quality gates — comes from:\n\n"
    "```bash\ngreenctl jobs report $job -n $namespace --output runs/job-reports\n```\n"
    "If a number here disagrees with the JSON report, the report wins.",
    0, y))

drill_dashboard = {
    "uid": DRILL_UID,
    "title": "Green Observatory — Job drilldown",
    "tags": ["green-observatory", "kepler", "carbon", "jobs", "cern"],
    "timezone": "utc", "schemaVersion": 39, "version": 1, "editable": True,
    "graphTooltip": 1, "refresh": "10s",
    "time": {"from": "now-24h", "to": "now"},
    "annotations": {"list": [{"builtIn": 1, "type": "dashboard", "name": "Annotations & Alerts",
                              "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                              "enable": True, "hide": True,
                              "iconColor": "rgba(0, 211, 255, 1)"}]},
    "links": [{"type": "dashboards", "title": "Back to Jobs", "tags": ["jobs"],
               "asDropdown": False, "icon": "external link", "includeVars": False,
               "keepTime": False, "targetBlank": False}],
    "templating": {"list": [
        datasource_var(),
        {"name": "namespace", "type": "query", "datasource": DS, "label": "Namespace",
         "query": "label_values(kube_job_info, namespace)",
         "current": {"text": "green-experiment", "value": "green-experiment"},
         "refresh": 2, "sort": 1, "includeAll": False, "multi": False},
        {"name": "job", "type": "query", "datasource": DS, "label": "Job",
         "query": f'label_values(kube_job_info{{namespace=~"{NS}"}}, job_name)',
         "current": {}, "refresh": 2, "sort": 1, "includeAll": False, "multi": False},
    ]},
    "panels": dpanels,
}

# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "dashboards"))
os.makedirs(OUT, exist_ok=True)
for name, doc in (("jobs-overview.json", jobs_dashboard),
                  ("job-drilldown.json", drill_dashboard)):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    json.load(open(path))  # validate
    n = sum(1 for p in doc["panels"] if p["type"] != "row")
    print(f"wrote {name}: {n} panels + {sum(1 for p in doc['panels'] if p['type']=='row')} rows")
