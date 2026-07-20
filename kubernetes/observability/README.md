# Green Observatory — real-time observability stack

Real-time energy **and** carbon visualization for workloads measured on the
baremetal node, layered on the existing `kube-prometheus-stack` + Kepler 0.11.4.

Two layers, because they have different data sources:

| Signal | Source | Real-time? |
|---|---|---|
| CPU energy / power / utilization | Kepler → Prometheus (10 s scrape) | **Yes**, natively |
| Grid carbon intensity (RTE `taux_co2`) | ODRE eCO2mix API | Only if the **RTE exporter** below is deployed |
| Per-Job audited energy **+** carbon | `greenctl jobs report` (Kepler ⋈ RTE) | No — post-hoc, authoritative |

Grafana is for **live monitoring**; `greenctl jobs` remains the **authoritative
per-Job accounting** (reset-aware counter deltas, energy-weighted intensity,
quality gates). The dashboard's range-integral emission figure is a live
approximation, not a replacement.

Contents:

```
observability/
├── dashboards/
│   ├── green-observatory-overview.json   # live energy + CO₂ (node & pods), 20 panels
│   ├── jobs-overview.json                # clickable table of Jobs that ran
│   ├── job-drilldown.json                # per-Job energy/CO₂ curves
│   └── dashboard-configmap.yaml          # wraps all three for the Grafana sidecar
├── rules/
│   └── co2-recording-rules.yaml          # Kepler × RTE → CO₂ emission-rate series
├── rte-carbon-exporter/
│   ├── exporter.py                        # zero-dependency RTE→Prometheus exporter
│   ├── configmap.yaml                     # exporter.py mounted into a stock image
│   └── deployment.yaml                    # Deployment + Service + ServiceMonitor
└── scripts/
    ├── gen_dashboard.py                   # builds green-observatory-overview.json
    ├── gen_jobs_dashboards.py             # builds jobs-overview.json + job-drilldown.json
    └── gen_configmaps.py                  # re-wraps sources into the ConfigMaps
```

### The three dashboards

| Dashboard | Answers |
|---|---|
| **Real-time Energy & Carbon** | What is the node/cluster emitting *right now*? Live power, CO₂ rate, cumulative CO₂, per-pod breakdown. |
| **Jobs** | Which Jobs ran? Table from kube-state-metrics with policy/trial/duration; click a row → |
| **Job drilldown** | What did *this* Job's run look like? Pod power, cumulative energy, CO₂ rate, grid intensity — time range preset to the Job's window. |

> **Provenance.** All three **recompute from Prometheus** and are for *exploration*.
> The authoritative per-Job accounting is `greenctl jobs report` (pod-precise
> boundaries, reset-aware counters, per-interval RTE weighting, quality gates).
> The Job-controller window in the Jobs table can differ from the pod window by a
> few seconds; if a number disagrees with the JSON report, the report wins.
> Carbon timeseries only exist from when the RTE exporter was deployed onward —
> Jobs that ran before that show energy but no carbon in Grafana (`greenctl`
> still resolves their carbon, because it fetches historical RTE).

---

## 0. Prerequisite — fix the crashing Grafana sidecars

On this CERN Magnum cluster the two Grafana `k8s-sidecar` containers
(`grafana-sc-dashboard`, `grafana-sc-datasources`) CrashLoop:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
  certificate verify failed: Missing Authority Key Identifier
```

The sidecar's strict OpenSSL rejects the API-server CA (Magnum-issued certs omit
the X.509 *Authority Key Identifier* extension). The main `grafana` container is
healthy, but the pod stays `1/3 Ready`, so the Service has no endpoints and
Grafana is unreachable. Until this is fixed, **no ConfigMap dashboard or
datasource loads**.

**Durable fix** (already applied in
[`experiment-01/manifests/monitoring-values.yaml`](../experiment-01/manifests/monitoring-values.yaml)
under `grafana.sidecar.*.skipTlsVerify: true`):

```bash
helm upgrade --install monitoring \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values kubernetes/experiment-01/manifests/monitoring-values.yaml \
  --wait
```

**Immediate fix without waiting for Helm** (reverted on the next `helm upgrade`,
so keep the values change too):

```bash
export KUBECONFIG=~/config
kubectl -n monitoring set env deployment/monitoring-grafana \
  -c grafana-sc-dashboard   SKIP_TLS_VERIFY=true
kubectl -n monitoring set env deployment/monitoring-grafana \
  -c grafana-sc-datasources SKIP_TLS_VERIFY=true
kubectl -n monitoring rollout status deployment/monitoring-grafana
```

Verify the pod is `3/3 Ready`:

```bash
kubectl -n monitoring get pods -l app.kubernetes.io/name=grafana
```

---

## 1. Load the dashboard

Once the dashboard sidecar is healthy it watches for ConfigMaps labelled
`grafana_dashboard=1` in all namespaces.

```bash
kubectl apply -f kubernetes/observability/dashboards/dashboard-configmap.yaml
```

The dashboard appears in Grafana under the **Green Observatory** folder as
**“Green Observatory — Real-time Energy & Carbon”** within ~30 s.

Regenerate the ConfigMap after editing the JSON:

```bash
kubectl create configmap green-observatory-dashboard \
  --namespace monitoring \
  --from-file=green-observatory-overview.json=kubernetes/observability/dashboards/green-observatory-overview.json \
  --dry-run=client -o yaml \
| kubectl label --local -f - grafana_dashboard=1 -o yaml \
> kubernetes/observability/dashboards/dashboard-configmap.yaml
# then re-add the `grafana_folder` annotation (see the committed file).
```

### Dashboard variables

- **datasource** — Prometheus data source (auto-detected).
- **node** — defaults to the baremetal node (regex `.*baremetal.*`).
- **zone** — RAPL zone, default `package` (use `core` only for cross-checks).
- **namespace / pod** — filter per workload. Kepler exposes only
  `pod_namespace` / `pod_name`, **not** the `sustainability.cern.ch/*` labels,
  so live filtering is by namespace + pod name.

---

## 2. (Optional) Make carbon real-time — RTE exporter

Without this, the carbon/emission panels show **No data** (Kepler has no carbon
signal). The exporter publishes RTE `taux_co2` as
`rte_carbon_intensity_gco2eq_per_kwh` so Grafana can compute a live emission
rate = `kepler_..._watts × intensity / 1000` (gCO2eq/h).

```bash
kubectl apply -f kubernetes/observability/rte-carbon-exporter/configmap.yaml
kubectl apply -f kubernetes/observability/rte-carbon-exporter/deployment.yaml
kubectl -n monitoring rollout status deployment/rte-carbon-exporter
```

Verify it scrapes and that Prometheus discovered it:

```bash
kubectl -n monitoring port-forward deploy/rte-carbon-exporter 9110:9110 &
curl -s localhost:9110/metrics | grep rte_carbon_intensity_gco2eq_per_kwh
# Prometheus target:  up{job="rte-carbon-exporter"}  →  1
```

**Design notes**
- Dependency-free (Python stdlib only); runs on stock `python:3.12-slim` mounted
  from the ConfigMap — no image build or registry push.
- Same ground truth as the offline model: dataset `eco2mix-national-tr`, field
  `taux_co2`, production-based gCO2eq/kWh, UTC.
- Runs on the observability node (`nodeSelector: role=observability`) so it never
  perturbs the baremetal measurement node.
- **Needs egress** to `https://odre.opendatasoft.com`. If the cluster blocks
  internet egress, run `exporter.py` off-cluster and use Prometheus
  `remote_write`/Pushgateway, or scrape it from Prometheus via a static target.
- Config via env: `REFRESH_SECONDS` (default 300), `MAX_AGE_SECONDS` (5400),
  `ZONE`, `LISTEN_PORT` (9110).

---

## 3. CO₂ recording rules

Turn the live Kepler power × RTE intensity into ready-to-use emission-rate series.
Evaluated every 15 s, so a cumulative built from them is **per-interval weighted**
(not multiplied by a single "current" intensity):

```bash
kubectl apply -f kubernetes/observability/rules/co2-recording-rules.yaml
```

Produces `node_co2_emission_gco2_per_second`,
`node_co2_emission_active_gco2_per_second` and `pod_co2_emission_gco2_per_second`.

```text
gCO2eq/h            = series * 3600
gCO2eq over range T = sum_over_time(series[T:15s]) * 15
```

## 4. kube-state-metrics label allowlist (needed by the Jobs dashboard)

Since KSM v2, `kube_*_labels` metrics are **empty** unless labels are explicitly
allow-listed — without this you cannot filter/group by `policy`/`trial`/`workload`.
Durable fix lives in
[`experiment-01/manifests/monitoring-values.yaml`](../experiment-01/manifests/monitoring-values.yaml)
(`kube-state-metrics.metricLabelsAllowlist`). To apply it immediately without Helm:

```bash
LABELS='sustainability.cern.ch/track,sustainability.cern.ch/workload,sustainability.cern.ch/policy,sustainability.cern.ch/experiment,sustainability.cern.ch/trial'
kubectl -n monitoring patch deploy monitoring-kube-state-metrics --type=json \
  -p "[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/0/args/-\",\
\"value\":\"--metric-labels-allowlist=jobs=[${LABELS}],pods=[${LABELS}]\"}]"
```

KSM sanitizes the names: `sustainability.cern.ch/policy` becomes
`kube_job_labels{label_sustainability_cern_ch_policy="..."}`.

## 5. Sanity queries (Prometheus)

```promql
# node power (package zone), baremetal
sum(kepler_node_cpu_watts{node_name=~".*baremetal.*", zone="package"})

# per-pod power, state-deduplicated
sum by (pod_name)(max without(state)(
  kepler_pod_cpu_watts{pod_namespace="green-experiment", zone="package"}))

# live grid intensity (needs the exporter)
rte_carbon_intensity_gco2eq_per_kwh

# live node emission rate (gCO2eq/h)
sum(kepler_node_cpu_active_watts{node_name=~".*baremetal.*", zone="package"})
  * on() group_left() rte_carbon_intensity_gco2eq_per_kwh / 1000
```

See [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) for how this fits the whole
system.
