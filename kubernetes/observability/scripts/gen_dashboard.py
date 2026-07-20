#!/usr/bin/env python3
"""Generate the Green Observatory Grafana dashboard JSON.

Queries here were validated live against the cluster Prometheus:
  - kepler_* metrics expose pod_name / pod_namespace / pod_id / node_name /
    zone(core|package) / state(running|terminated); NOT the sustainability.cern.ch/* labels.
  - real-time carbon (rte_carbon_intensity_gco2eq_per_kwh) is provided by the
    optional rte-carbon-exporter; carbon panels degrade to "No data" without it.
"""
from __future__ import annotations

import json

DS = {"type": "prometheus", "uid": "${datasource}"}
ZONE = '$zone'
NODE = '$node'
NS = '$namespace'
POD = '$pod'

# pod power, deduplicated across Kepler's running/terminated state series
POD_WATTS = (
    f'sum by (pod_name)(max without(state)('
    f'kepler_pod_cpu_watts{{pod_namespace=~"{NS}",pod_name=~"{POD}",zone="{ZONE}"}}))'
)
POD_ENERGY_RANGE = (
    f'sum by (pod_name)(max without(state)('
    f'increase(kepler_pod_cpu_joules_total{{pod_namespace=~"{NS}",pod_name=~"{POD}",zone="{ZONE}"}}[$__range])))'
    f'/3600000'
)
INTENSITY = 'max(rte_carbon_intensity_gco2eq_per_kwh)'

panels: list[dict] = []
_id = [0]


def nid() -> int:
    _id[0] += 1
    return _id[0]


def row(title: str, y: int) -> dict:
    return {
        "id": nid(), "type": "row", "title": title, "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "datasource": DS, "panels": [],
    }


def stat(title, expr, x, y, unit="watt", w=4, h=4, decimals=1, desc="",
         steps=None, reduce="lastNotNull"):
    return {
        "id": nid(), "type": "stat", "title": title, "description": desc,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"refId": "A", "expr": expr, "datasource": DS, "instant": True}],
        "fieldConfig": {
            "defaults": {
                "unit": unit, "decimals": decimals,
                "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute",
                               "steps": steps or [{"color": "green", "value": None}]},
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": [reduce], "fields": "", "values": False},
            "orientation": "auto", "colorMode": "value",
            "graphMode": "area", "textMode": "auto", "justifyMode": "auto",
        },
    }


def timeseries(title, targets, x, y, unit="watt", w=12, h=8, desc="",
               stacking="none", steps=None, fill=10, legend="list"):
    return {
        "id": nid(), "type": "timeseries", "title": title, "description": desc,
        "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [dict(t, datasource=DS) for t in targets],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "palette-classic"},
                "custom": {
                    "drawStyle": "line", "lineInterpolation": "smooth",
                    "lineWidth": 2, "fillOpacity": fill, "gradientMode": "opacity",
                    "spanNulls": False, "showPoints": "never",
                    "pointSize": 5, "stacking": {"mode": stacking, "group": "A"},
                    "axisPlacement": "auto", "axisLabel": "",
                },
                "thresholds": {"mode": "absolute",
                               "steps": steps or [{"color": "green", "value": None}]},
            },
            "overrides": [],
        },
        "options": {
            "tooltip": {"mode": "multi", "sort": "desc"},
            "legend": {"displayMode": legend, "placement": "bottom",
                       "calcs": ["mean", "max", "lastNotNull"]},
        },
    }


CI_STEPS = [
    {"color": "green", "value": None},
    {"color": "yellow", "value": 40},
    {"color": "orange", "value": 80},
    {"color": "red", "value": 150},
]

y = 0
# ---- Row 1: node overview -------------------------------------------------
panels.append(row("Node — baremetal (real-time)", y)); y += 1
panels.append(stat("Total CPU power", f'sum(kepler_node_cpu_watts{{node_name=~"{NODE}",zone="{ZONE}"}})',
                   0, y, unit="watt",
                   desc="RAPL package-domain power for the whole node (active + idle)."))
panels.append(stat("Active CPU power", f'sum(kepler_node_cpu_active_watts{{node_name=~"{NODE}",zone="{ZONE}"}})',
                   4, y, unit="watt",
                   desc="Dynamic power attributable to running work."))
panels.append(stat("Idle CPU power", f'sum(kepler_node_cpu_idle_watts{{node_name=~"{NODE}",zone="{ZONE}"}})',
                   8, y, unit="watt", desc="Static/idle draw of the RAPL domain."))
panels.append(stat("CPU utilization", f'kepler_node_cpu_usage_ratio{{node_name=~"{NODE}"}}*100',
                   12, y, unit="percent",
                   steps=[{"color": "green", "value": None}, {"color": "yellow", "value": 60},
                          {"color": "red", "value": 90}],
                   desc="Node CPU busy ratio as reported by Kepler."))
panels.append(stat("Live grid intensity", INTENSITY, 16, y, unit="short", decimals=0,
                   steps=CI_STEPS,
                   desc="RTE/eCO2mix production-based carbon intensity (gCO2eq/kWh). "
                        "Requires the rte-carbon-exporter; shows No data otherwise."))
panels.append(stat("Node emission rate", f'sum(kepler_node_cpu_active_watts{{node_name=~"{NODE}",zone="{ZONE}"}})*{INTENSITY}/1000',
                   20, y, unit="short", decimals=1,
                   steps=CI_STEPS,
                   desc="Operational CO2 rate = active watts x live intensity / 1000 (gCO2eq/h). "
                        "Requires the exporter."))
y += 4

# ---- Row 2: total CO2 (real-time, from recording rules) -------------------
NODE_CO2_S = f'node_co2_emission_gco2_per_second{{node_name=~"{NODE}",zone="{ZONE}"}}'
NODE_CO2_ACT_S = f'node_co2_emission_active_gco2_per_second{{node_name=~"{NODE}",zone="{ZONE}"}}'
POD_CO2_S = f'pod_co2_emission_gco2_per_second{{pod_namespace=~"{NS}",pod_name=~"{POD}"}}'
panels.append(row("Total CO₂ — what Kubernetes is emitting (real-time)", y)); y += 1
panels.append(stat("Node CO₂ rate (total)", f'sum({NODE_CO2_S})*3600', 0, y, unit="short", w=6, decimals=2,
                   desc="Whole-node CPU emission rate now, gCO2eq/h = node package watts × live RTE intensity. "
                        "From recording rule node_co2_emission_gco2_per_second."))
panels.append(stat("Workload CO₂ rate (active)", f'sum({NODE_CO2_ACT_S})*3600', 6, y, unit="short", w=6, decimals=3,
                   desc="Emission rate attributable to running work (active watts) now, gCO2eq/h."))
panels.append(stat("Cumulative CO₂ (range, total)", f'sum(sum_over_time({NODE_CO2_S}[$__range:15s]))*15', 12, y,
                   unit="short", w=6, decimals=2,
                   desc="Node CPU CO2 accumulated over the selected range (gCO2eq). Per-interval weighted via the "
                        "recording rule (Riemann sum, 15s = eval interval). Live approximation; greenctl jobs is authoritative per Job."))
panels.append(stat("Cumulative CO₂ (range, workload)", f'sum(sum_over_time({NODE_CO2_ACT_S}[$__range:15s]))*15', 18, y,
                   unit="short", w=6, decimals=3,
                   desc="Active/workload CO2 accumulated over the range (gCO2eq)."))
y += 4
panels.append(timeseries("CO₂ emission rate", [
    {"refId": "A", "expr": f'sum({NODE_CO2_S})*3600', "legendFormat": "node total"},
    {"refId": "B", "expr": f'sum({NODE_CO2_ACT_S})*3600', "legendFormat": "workload (active)"},
], 0, y, unit="short", desc="gCO2eq/h over time. Needs the recording rules + RTE exporter.", fill=15))
panels.append({
    "id": nid(), "type": "bargauge",
    "title": "Cumulative CO₂ by pod (range, gCO2eq)",
    "description": "Per-pod CO2 accumulated over the range from the recording rule (Riemann sum).",
    "datasource": DS,
    "gridPos": {"h": 8, "w": 12, "x": 12, "y": y},
    "targets": [{"refId": "A", "datasource": DS, "instant": True,
                 "expr": f'topk(10, sum by (pod_name)(sum_over_time({POD_CO2_S}[$__range:15s])*15))',
                 "legendFormat": "{{pod_name}}"}],
    "fieldConfig": {"defaults": {"unit": "short", "decimals": 4,
                                 "color": {"mode": "continuous-GrYlRd"},
                                 "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}},
                    "overrides": []},
    "options": {"orientation": "horizontal", "displayMode": "gradient",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
})
y += 8

# ---- Row 3: power & carbon over time ---------------------------------------
panels.append(row("Power & carbon over time", y)); y += 1
panels.append(timeseries("Node CPU power (package)", [
    {"refId": "A", "expr": f'sum(kepler_node_cpu_watts{{node_name=~"{NODE}",zone="{ZONE}"}})', "legendFormat": "total"},
    {"refId": "B", "expr": f'sum(kepler_node_cpu_active_watts{{node_name=~"{NODE}",zone="{ZONE}"}})', "legendFormat": "active"},
    {"refId": "C", "expr": f'sum(kepler_node_cpu_idle_watts{{node_name=~"{NODE}",zone="{ZONE}"}})', "legendFormat": "idle"},
], 0, y, unit="watt", desc="Total vs active vs idle package power."))
panels.append(timeseries("Live grid carbon intensity", [
    {"refId": "A", "expr": INTENSITY, "legendFormat": "RTE FR gCO2eq/kWh"},
], 12, y, unit="short", desc="From rte-carbon-exporter. Thresholds mark green/dirty grid.",
    steps=CI_STEPS, fill=20))
y += 8

# ---- Row 3: per-workload --------------------------------------------------
panels.append(row("Per-workload (pods on baremetal)", y)); y += 1
panels.append(timeseries("Per-pod CPU power", [
    {"refId": "A", "expr": POD_WATTS, "legendFormat": "{{pod_name}}"},
], 0, y, unit="watt", desc="Attributed CPU power per pod (state-deduplicated)."))
panels.append(timeseries("Per-pod emission rate", [
    {"refId": "A", "expr": f'({POD_WATTS})*on() group_left() {INTENSITY}/1000', "legendFormat": "{{pod_name}}"},
], 12, y, unit="short", desc="pod watts x live intensity / 1000 (gCO2eq/h). Requires the exporter."))
y += 8
panels.append({
    "id": nid(), "type": "table",
    "title": "Per-pod energy & power over selected range",
    "description": "increase() of the Kepler pod counter over the dashboard time range. "
                   "For audited per-Job energy+carbon use `greenctl jobs report`.",
    "datasource": DS,
    "gridPos": {"h": 8, "w": 24, "x": 0, "y": y},
    "targets": [
        {"refId": "A", "expr": POD_ENERGY_RANGE, "datasource": DS, "format": "table", "instant": True},
        {"refId": "B", "expr": POD_WATTS, "datasource": DS, "format": "table", "instant": True},
    ],
    "transformations": [
        {"id": "merge", "options": {}},
        {"id": "organize", "options": {
            "renameByName": {"pod_name": "Pod", "Value #A": "Energy (kWh)", "Value #B": "Power now (W)"},
            "excludeByName": {"Time": True, "__name__": True, "job": True, "instance": True},
        }},
    ],
    "fieldConfig": {"defaults": {"custom": {"align": "auto", "filterable": True}}, "overrides": [
        {"matcher": {"id": "byName", "options": "Energy (kWh)"},
         "properties": [{"id": "unit", "value": "kwatth"}, {"id": "decimals", "value": 5},
                        {"id": "custom.cellOptions", "value": {"type": "gauge", "mode": "gradient"}}]},
        {"matcher": {"id": "byName", "options": "Power now (W)"},
         "properties": [{"id": "unit", "value": "watt"}, {"id": "decimals", "value": 2}]},
    ]},
    "options": {"showHeader": True, "sortBy": [{"displayName": "Energy (kWh)", "desc": True}]},
})
y += 8

# ---- Row 4: range totals --------------------------------------------------
panels.append(row("Range totals & grid context", y)); y += 1
panels.append(stat("Node active energy (range)",
                   f'sum(increase(kepler_node_cpu_active_joules_total{{node_name=~"{NODE}",zone="{ZONE}"}}[$__range]))/3600000',
                   0, y, unit="kwatth", w=6, decimals=5,
                   desc="Active CPU energy accumulated over the selected range."))
panels.append(stat("Est. operational emissions (range)",
                   f'sum(increase(kepler_node_cpu_active_joules_total{{node_name=~"{NODE}",zone="{ZONE}"}}[$__range]))/3600000*{INTENSITY}',
                   6, y, unit="short", w=6, decimals=3, steps=CI_STEPS,
                   desc="Approximate: range active energy (kWh) x CURRENT intensity (gCO2eq). "
                        "Precise per-interval accounting is done post-hoc by greenctl jobs."))
panels.append({
    "id": nid(), "type": "bargauge",
    "title": "Energy by pod (range, kWh)",
    "datasource": DS,
    "gridPos": {"h": 8, "w": 12, "x": 12, "y": y},
    "targets": [{"refId": "A", "expr": f'topk(10, {POD_ENERGY_RANGE})', "datasource": DS,
                 "legendFormat": "{{pod_name}}", "instant": True}],
    "fieldConfig": {"defaults": {"unit": "kwatth", "decimals": 5,
                                 "color": {"mode": "continuous-GrYlRd"},
                                 "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}},
                    "overrides": []},
    "options": {"orientation": "horizontal", "displayMode": "gradient",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
})
y += 8

dashboard = {
    "uid": "green-observatory-rt",
    "title": "Green Observatory — Real-time Energy & Carbon",
    "tags": ["green-observatory", "kepler", "carbon", "cern", "sustainability"],
    "timezone": "utc",
    "schemaVersion": 39,
    "version": 1,
    "editable": True,
    "graphTooltip": 1,
    "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"},
    "annotations": {"list": [{
        "builtIn": 1, "type": "dashboard", "name": "Annotations & Alerts",
        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
        "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
    }]},
    "templating": {"list": [
        {"name": "datasource", "type": "datasource", "query": "prometheus",
         "label": "Data source", "current": {}, "hide": 0, "refresh": 1, "regex": ""},
        {"name": "node", "type": "query", "datasource": DS,
         "label": "Node", "query": "label_values(kepler_node_cpu_watts, node_name)",
         "regex": "/.*baremetal.*/", "current": {}, "refresh": 2, "sort": 1,
         "includeAll": False, "multi": False},
        {"name": "zone", "type": "custom", "label": "RAPL zone",
         "query": "package,core", "current": {"text": "package", "value": "package"},
         "options": [{"text": "package", "value": "package", "selected": True},
                     {"text": "core", "value": "core", "selected": False}],
         "includeAll": False, "multi": False, "hide": 0},
        {"name": "namespace", "type": "query", "datasource": DS,
         "label": "Namespace", "query": "label_values(kepler_pod_cpu_watts, pod_namespace)",
         "current": {"text": "green-experiment", "value": "green-experiment"},
         "refresh": 2, "sort": 1, "includeAll": True, "allValue": ".*", "multi": True},
        {"name": "pod", "type": "query", "datasource": DS,
         "label": "Pod", "query": f'label_values(kepler_pod_cpu_watts{{pod_namespace=~"{NS}"}}, pod_name)',
         "current": {"text": "All", "value": "$__all"},
         "refresh": 2, "sort": 1, "includeAll": True, "allValue": ".*", "multi": True},
    ]},
    "panels": panels,
}

import os
_here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(_here, "..", "dashboards", "green-observatory-overview.json")
out = os.path.normpath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(dashboard, f, indent=2)
    f.write("\n")
print("wrote", out)
print("panels:", sum(1 for p in panels if p["type"] != "row"), "+ rows:", sum(1 for p in panels if p["type"] == "row"))
print("json valid:", bool(json.load(open(out))))
