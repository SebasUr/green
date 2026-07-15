Claro. Aquí va la guía en formato normal, actualizada desde la versión original que compartiste  y adaptada a **Kepler 0.11.4**.

# Experimento 1 — Forecast-driven temporal shifting on bare metal

## Pregunta de investigación

> ¿Ejecutar un workload CPU diferible durante una ventana seleccionada por el Green Window Observatory reduce sus emisiones operacionales frente a ejecutarlo inmediatamente, sin cambiar la cantidad de trabajo realizado ni degradar significativamente su tiempo de ejecución?

La cadena experimental será:

```text
Green Window Observatory
        ↓
RUN_NOW o DEFER + recommended_start_time
        ↓
mismo workload CPU en el nodo bare metal
        ↓
Kepler 0.11.4 + Prometheus
        ↓
energía operacional atribuida × intensidad de carbono realizada
        ↓
comparación run-now vs green-window
```

El Green Window Observatory ya demostró en replay que captura aproximadamente el 73% del potencial de perfect foresight a 24 horas. Este experimento busca verificar si esa ventaja también se observa al incluir el consumo energético real del workload, su duración y la variabilidad del nodo.

---

# 1. Alcance

## Incluido

* Green Window Observatory.
* Horizonte principal de 24 horas.
* Workload CPU determinista y diferible.
* Worker bare metal.
* Kepler 0.11.4.
* Prometheus y Grafana.
* Intensidad de carbono realizada de RTE/ODRÉ.
* Decisión `RUN_NOW` o `DEFER`.
* Comparación de:

  * energía;
  * emisiones operacionales;
  * runtime;
  * waiting time;
  * precisión de la ventana seleccionada.

## Fuera del Experimento 1

* Kueue.
* kube-green.
* Koordinator.
* Joulie.
* Power capping.
* GPU.
* Boavizta como métrica principal.
* Admission controller automático.
* Scheduler plugin.

El objetivo inicial es aislar el efecto del temporal shifting. La automatización con Kueue o un controller puede hacerse después.

---

# 2. Topología experimental

El clúster tiene:

```text
Control plane VM
Worker VM
Worker bare metal
```

Distribución recomendada:

```text
Worker VM:
  Prometheus
  Grafana
  kube-state-metrics
  Green Window Observatory

Worker bare metal:
  Kepler 0.11.4
  workload experimental

Control plane:
  componentes de control de Kubernetes
```

Esto evita que Prometheus, Grafana o el modelo introduzcan ruido adicional sobre el nodo donde se mide el workload.

## Etiquetar los nodos

```bash
export BM_NODE=sebas-green-baremetal-jpmdwiscskyy-node-0
export VM_NODE=sebas-green-cyl76pnkgqu3-node-0

kubectl label node "$BM_NODE" \
  sustainability.cern.ch/hardware=baremetal \
  sustainability.cern.ch/experiment-node=true \
  --overwrite

kubectl label node "$VM_NODE" \
  sustainability.cern.ch/hardware=vm \
  sustainability.cern.ch/role=observability \
  --overwrite

kubectl get nodes \
  -L sustainability.cern.ch/hardware \
  -L sustainability.cern.ch/role \
  -L sustainability.cern.ch/experiment-node
```

## Guardar la caracterización inicial

```bash
mkdir -p experiment-01/cluster

kubectl get nodes -o wide \
  > experiment-01/cluster/nodes.txt

kubectl describe node "$BM_NODE" \
  > experiment-01/cluster/baremetal-node.txt

kubectl get node "$BM_NODE" -o yaml \
  > experiment-01/cluster/baremetal-node.yaml

kubectl get pods -A -o wide \
  --field-selector spec.nodeName="$BM_NODE" \
  > experiment-01/cluster/baremetal-existing-pods.txt
```

Guardar recursos disponibles:

```bash
kubectl get node "$BM_NODE" \
  -o jsonpath='CPU capacity: {.status.capacity.cpu}{"\n"}CPU allocatable: {.status.allocatable.cpu}{"\n"}Memory allocatable: {.status.allocatable.memory}{"\n"}Architecture: {.status.nodeInfo.architecture}{"\n"}Kernel: {.status.nodeInfo.kernelVersion}{"\n"}OS: {.status.nodeInfo.osImage}{"\n"}'
```

---

# 3. Prometheus y Grafana

Si `kube-prometheus-stack` todavía no está instalado:

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts

helm repo update
```

Crear `monitoring-values.yaml`:

```yaml
grafana:
  nodeSelector:
    sustainability.cern.ch/role: observability

prometheusOperator:
  nodeSelector:
    sustainability.cern.ch/role: observability

prometheus:
  prometheusSpec:
    nodeSelector:
      sustainability.cern.ch/role: observability

    retention: 30d
    retentionSize: 40GB

    scrapeInterval: 15s
    evaluationInterval: 15s

alertmanager:
  alertmanagerSpec:
    nodeSelector:
      sustainability.cern.ch/role: observability

kube-state-metrics:
  nodeSelector:
    sustainability.cern.ch/role: observability
```

Instalar:

```bash
helm upgrade --install monitoring \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring-values.yaml \
  --wait
```

Verificar:

```bash
kubectl get pods -n monitoring -o wide
kubectl get prometheus -n monitoring
kubectl get servicemonitors -A
```

---

# 4. Kepler 0.11.4

Kepler 0.11.4 usa la arquitectura nueva de Kepler. No necesita las variables legacy usadas por versiones anteriores, como:

```text
ENABLE_EBPF_CGROUPID
EXPOSE_HW_COUNTER_METRICS
EXPOSE_CGROUP_METRICS
CGROUP_METRICS
ENABLE_PROCESS_METRICS
```

Tu archivo actual es correcto:

```yaml
daemonset:
  nodeSelector:
    sustainability.cern.ch/hardware: baremetal

  resources:
    requests:
      cpu: 50m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi

serviceMonitor:
  enabled: true
  interval: 10s
  path: /metrics
  labels:
    release: monitoring
```

## Instalación

```bash
helm upgrade --install kepler \
  oci://quay.io/sustainable_computing_io/charts/kepler \
  --version 0.11.4 \
  --namespace kepler \
  --create-namespace \
  --values kepler-values.yaml \
  --atomic \
  --timeout 5m
```

## Verificación

```bash
helm list -n kepler
helm status kepler -n kepler
helm get values kepler -n kepler --all

kubectl get daemonset -n kepler
kubectl get pods -n kepler -o wide
kubectl get service -n kepler
kubectl get servicemonitor -n kepler
```

Debe existir un solo pod de Kepler ejecutándose sobre el nodo bare metal:

```bash
kubectl get pods -n kepler \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase'
```

Comprobar el selector:

```bash
kubectl get daemonset kepler -n kepler \
  -o jsonpath='{.spec.template.spec.nodeSelector}{"\n"}'
```

Resultado esperado:

```text
map[sustainability.cern.ch/hardware:baremetal]
```

Revisar logs:

```bash
kubectl logs -n kepler \
  -l app.kubernetes.io/name=kepler \
  --tail=200
```

---

# 5. Validar el endpoint de Kepler

Kepler 0.11.4 expone las métricas en el puerto `28282`.

```bash
kubectl port-forward -n kepler \
  svc/kepler 28282:28282
```

En otra terminal:

```bash
curl -s http://localhost:28282/metrics \
  | grep '^kepler_build_info'
```

Listar las métricas energéticas disponibles:

```bash
curl -s http://localhost:28282/metrics \
  | grep -E '^kepler_(node|pod|container)_cpu_(watts|joules_total)' \
  | head -100
```

Las métricas principales esperadas son:

```text
kepler_node_cpu_watts
kepler_node_cpu_joules_total

kepler_node_cpu_active_watts
kepler_node_cpu_active_joules_total

kepler_node_cpu_idle_watts
kepler_node_cpu_idle_joules_total

kepler_pod_cpu_watts
kepler_pod_cpu_joules_total

kepler_container_cpu_watts
kepler_container_cpu_joules_total
```

Las métricas `*_watts` representan potencia instantánea.

Las métricas `*_joules_total` son contadores acumulativos de energía.

---

# 6. Integración con Prometheus

Abrir Prometheus:

```bash
kubectl -n monitoring get svc | grep prometheus
```

Después:

```bash
kubectl -n monitoring port-forward \
  svc/monitoring-kube-prometheus-prometheus \
  9090:9090
```

Verificar que Prometheus descubrió Kepler:

```promql
up{namespace="kepler"}
```

También:

```bash
curl -sG http://localhost:9090/api/v1/query \
  --data-urlencode 'query=up{namespace="kepler"}' \
  | jq
```

Comprobar frecuencia de scraping:

```promql
count_over_time(
  kepler_node_cpu_watts{
    node_name="sebas-green-baremetal-jpmdwiscskyy-node-0"
  }[1m]
)
```

Con un intervalo de 10 segundos deberían aparecer aproximadamente seis muestras por minuto para cada serie.

---

# 7. Inspeccionar los dominios RAPL

Antes de construir las consultas definitivas debes comprobar qué labels y dominios aparecen realmente en tu nodo.

```promql
count by (zone, path) (
  kepler_node_cpu_joules_total{
    node_name="sebas-green-baremetal-jpmdwiscskyy-node-0"
  }
)
```

También puedes ver las series directamente:

```bash
curl -s http://localhost:28282/metrics \
  | grep '^kepler_node_cpu_joules_total'
```

Los dominios podrían incluir elementos como:

```text
package
core
uncore
dram
psys
```

Depende del procesador y de lo que exponga el kernel.

No debes sumar todas las zonas automáticamente. Algunas pueden representar componentes incluidos dentro de otros dominios y producir doble conteo.

Primero registra:

```text
zone
path
socket o package
componente físico representado
```

Luego elige un dominio estable para usarlo en todos los trials.

Por ejemplo:

```text
PRIMARY_ZONE=package
```

El valor real debe corresponder a lo que aparezca en tus métricas.

---

# 8. Qué mide Kepler

Kepler obtiene energía del hardware mediante RAPL y atribuye la parte activa a los workloads usando su consumo de CPU time.

Conceptualmente:

```text
workload energy attribution =
    workload CPU time
    ───────────────── × measured active CPU energy
    total CPU time
```

Por tanto:

```text
kepler_pod_cpu_joules_total
```

no es un medidor físico independiente dentro del pod. Es energía medida a nivel hardware y atribuida al pod.

La forma correcta de describirla es:

> CPU operational energy attributed to the workload.

No debes describirla inicialmente como:

> Total server energy consumption.

RAPL no garantiza incluir todos los componentes físicos, como ventiladores, NIC, discos, PSU o cooling.

---

# 9. Workload del Experimento 1

Se utilizará un Monte Carlo CPU determinista con:

* número fijo de muestras;
* semilla fija;
* número fijo de procesos;
* misma imagen;
* mismos requests y limits;
* mismo nodo bare metal;
* misma cantidad de trabajo científico.

No conviene utilizar un workload definido únicamente por duración, porque dos ejecuciones de 30 minutos podrían completar cantidades diferentes de trabajo.

## ConfigMap del workload

Crear `monte-carlo-configmap.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: green-experiment
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: monte-carlo-code
  namespace: green-experiment
data:
  monte_carlo.py: |
    import json
    import multiprocessing as mp
    import os
    import random
    import socket
    import time
    from datetime import datetime, timezone

    workers = int(os.getenv("WORKERS", "16"))
    samples_per_worker = int(
        os.getenv("SAMPLES_PER_WORKER", "50000000")
    )
    base_seed = int(os.getenv("BASE_SEED", "20260713"))

    def simulate(args):
        worker_id, samples, seed = args
        rng = random.Random(seed)
        inside = 0

        for _ in range(samples):
            x = rng.random()
            y = rng.random()

            if x * x + y * y <= 1.0:
                inside += 1

        return inside

    if __name__ == "__main__":
        started = datetime.now(timezone.utc)
        t0 = time.monotonic()

        work = [
            (i, samples_per_worker, base_seed + i)
            for i in range(workers)
        ]

        with mp.Pool(processes=workers) as pool:
            results = pool.map(simulate, work)

        total_samples = workers * samples_per_worker
        total_inside = sum(results)
        pi_estimate = 4.0 * total_inside / total_samples

        elapsed = time.monotonic() - t0
        finished = datetime.now(timezone.utc)

        print(json.dumps({
            "hostname": socket.gethostname(),
            "workers": workers,
            "samples_per_worker": samples_per_worker,
            "total_samples": total_samples,
            "pi_estimate": pi_estimate,
            "elapsed_seconds": elapsed,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat()
        }, indent=2))
```

Aplicar:

```bash
kubectl apply -f monte-carlo-configmap.yaml
```

---

# 10. Plantilla del Job

Crear `monte-carlo-job-template.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mc-${TRIAL_ID}
  namespace: green-experiment
  labels:
    sustainability.cern.ch/experiment: "green-window-01"
    sustainability.cern.ch/trial: "${TRIAL_ID}"
    sustainability.cern.ch/policy: "${POLICY}"
spec:
  backoffLimit: 0

  template:
    metadata:
      labels:
        sustainability.cern.ch/experiment: "green-window-01"
        sustainability.cern.ch/trial: "${TRIAL_ID}"
        sustainability.cern.ch/policy: "${POLICY}"

    spec:
      restartPolicy: Never

      nodeSelector:
        sustainability.cern.ch/hardware: baremetal

      containers:
        - name: monte-carlo
          image: "${IMAGE}"
          imagePullPolicy: IfNotPresent

          command:
            - python
            - /workload/monte_carlo.py

          env:
            - name: WORKERS
              value: "${WORKERS}"

            - name: SAMPLES_PER_WORKER
              value: "${SAMPLES_PER_WORKER}"

            - name: BASE_SEED
              value: "${BASE_SEED}"

          resources:
            requests:
              cpu: "${WORKERS}"
              memory: "${MEMORY}"

            limits:
              cpu: "${WORKERS}"
              memory: "${MEMORY}"

          volumeMounts:
            - name: workload
              mountPath: /workload
              readOnly: true

      volumes:
        - name: workload
          configMap:
            name: monte-carlo-code
```

Para los trials formales conviene reemplazar:

```text
python:3.12-slim
```

por una imagen fijada con digest:

```text
python:3.12-slim@sha256:<DIGEST>
```

## Ejecutar un trial

```bash
export TRIAL_ID=calibration-01
export POLICY=calibration
export WORKERS=16
export SAMPLES_PER_WORKER=50000000
export BASE_SEED=20260713
export IMAGE=python:3.12-slim
export MEMORY=2Gi

envsubst < monte-carlo-job-template.yaml \
  | kubectl apply -f -
```

Esperar:

```bash
kubectl wait \
  --for=condition=complete \
  job/mc-"$TRIAL_ID" \
  -n green-experiment \
  --timeout=2h
```

Consultar logs:

```bash
kubectl logs \
  job/mc-"$TRIAL_ID" \
  -n green-experiment
```

Comprobar el nodo:

```bash
kubectl get pods -n green-experiment -o wide
```

## Ejecución y captura automatizadas

`scripts/run_monte_carlo.py` reemplaza el procedimiento manual anterior. El
runner aplica el ConfigMap, renderiza y crea el Job, espera su finalización y
consulta automáticamente Prometheus y Kepler para el intervalo del workload.

La ejecución aproximada de 15 minutos usa por defecto:

```text
workers:             16
samples por worker:  3,150,000,000
total samples:       50,400,000,000
base seed:           20260713
CPU request/limit:   16
memory request/limit: 2Gi
primary RAPL zone:   package
target runtime:      900 s
```

Es una estimación inicial. El runtime real depende del procesador y de la carga
de fondo. Al terminar, `result.json` incluye
`recommended_samples_per_worker_for_target_runtime`, calculado con:

```text
new_samples = current_samples × 900 / measured_runtime
```

El default se calibró en el nodo bare metal el 14 de julio de 2026. Una
ejecución con `100,000,000` muestras por worker tardó `28.436 s` y recomendó
`3,165,042,128`; se redondeó a `3,150,000,000`, equivalente a unos `896 s` bajo
las mismas condiciones.

### Ejecución con los valores por defecto

```bash
cd kubernetes/experiment-01
export KUBECONFIG=~/config

python3 scripts/run_monte_carlo.py \
  --trial-id calibration-01
```

Si se omite `--trial-id`, el script genera uno con timestamp UTC.

### Smoke test corto

Antes del trial de 15 minutos se puede validar el pipeline con menos trabajo:

```bash
python3 scripts/run_monte_carlo.py \
  --trial-id smoke-01 \
  --samples-per-worker 100000000 \
  --cleanup-job
```

En la calibración observada este smoke test dura unos 28 segundos, suficiente
para obtener varias muestras con el scrape de Kepler cada 10 segundos.

### Parámetros principales

```text
--workers N                 procesos y CPUs solicitadas
--samples-per-worker N      trabajo realizado por cada proceso
--base-seed N               semilla determinista
--target-runtime N          objetivo usado para recomendar la calibración
--image IMAGE               imagen del workload
--memory 2Gi                request y limit de memoria
--policy calibration        label de política experimental
--zone package              único dominio RAPL usado en el resumen
--timeout 2h                timeout del Job
--pre-buffer 30s            contexto anterior al Job en Prometheus
--post-buffer 30s           espera para capturar el último scrape
--cleanup-job               borra Job/Pod después de guardar los artefactos
```

Para trials formales se debe fijar la imagen por digest:

```bash
python3 scripts/run_monte_carlo.py \
  --trial-id calibration-02 \
  --image 'python:3.12-slim@sha256:<DIGEST>'
```

El runner aborta antes de crear el Job si detecta otro workload experimental,
un `node-debugger` o un pod en `CrashLoopBackOff` sobre el bare metal. La opción
`--allow-dirty-node` permite ejecutar un diagnóstico, pero el resultado queda
marcado como no válido para el experimento formal.

### Artefactos producidos

```text
trials/<trial-id>/
├── status.json
├── metadata.json
├── result.json
├── job-submitted.yaml
├── job.json
├── pod.json
├── workload-output.json
├── metrics.csv
├── series-summary.csv
├── cluster/
├── prometheus/
│   └── raw/
└── snapshots/
    ├── start/
    └── end/
```

`result.json` contiene runtime, salida científica, imagen y digest reales,
energía CPU atribuida al pod, potencia media, energía total/activa/idle del nodo,
utilización de CPU y la recomendación para la siguiente calibración. Los JSON
crudos y `metrics.csv` permiten recalcular el análisis sin volver a consultar
Prometheus.

## Notebook de análisis y comparación

El notebook `analysis/experiment_analysis.ipynb` descubre automáticamente todas
las carpetas de `idle/` y `trials/`. No es necesario añadir cada trial a mano.
Las capturas fallidas o incompletas aparecen en el inventario de calidad, pero
se excluyen de las comparaciones científicas.

Preparar el entorno existente:

```bash
conda activate green-observatory
pip install -r kubernetes/experiment-01/analysis/requirements.txt
```

Desde VS Code, abrir el notebook y seleccionar el kernel
`green-observatory`. Alternativamente:

```bash
jupyter lab \
  kubernetes/experiment-01/analysis/experiment_analysis.ipynb
```

Para ejecutarlo sin interfaz:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  kubernetes/experiment-01/analysis/experiment_analysis.ipynb
```

El notebook produce:

* inventario de todas las capturas y sus quality gates;
* tablas normalizadas de idle y workloads;
* cobertura de todas las familias métricas capturadas;
* comparación de potencia y energía entre idle y cada trial válido;
* series de potencia alineadas al inicio del workload;
* validación de energía integrando watts frente a contadores acumulativos;
* escalado de runtime y energía con la cantidad de muestras;
* CV de runtime y energía para calibraciones idénticas repetidas;
* plantilla explícita para pares `run-now` contra `green-window`;
* exploración parametrizable de cualquier métrica cruda.

Las tablas y figuras se exportan a `analysis/generated/` cada vez que se ejecuta.
La lógica reutilizable vive en `analysis/experiment_analysis.py`, por lo que se
puede importar también desde otros scripts o tests.

---

# 11. Identificar el pod

```bash
POD=$(
  kubectl get pods -n green-experiment \
    -l sustainability.cern.ch/trial="$TRIAL_ID" \
    -o jsonpath='{.items[0].metadata.name}'
)

POD_UID=$(
  kubectl get pod "$POD" \
    -n green-experiment \
    -o jsonpath='{.metadata.uid}'
)

echo "POD=$POD"
echo "POD_UID=$POD_UID"
```

Guardar objetos y resultados:

```bash
mkdir -p experiment-01/trials/"$TRIAL_ID"

kubectl get pod "$POD" \
  -n green-experiment \
  -o json \
  > experiment-01/trials/"$TRIAL_ID"/pod.json

kubectl get job mc-"$TRIAL_ID" \
  -n green-experiment \
  -o yaml \
  > experiment-01/trials/"$TRIAL_ID"/job.yaml

kubectl logs "$POD" \
  -n green-experiment \
  > experiment-01/trials/"$TRIAL_ID"/workload-output.json
```

---

# 12. Métricas de pod en Kepler 0.11.4

Kepler 0.11.4 expone métricas de pod con labels como:

```text
pod_id
pod_name
pod_namespace
state
zone
node_name
```

Primero confirma los labels reales en tu instalación:

```bash
curl -s http://localhost:28282/metrics \
  | grep '^kepler_pod_cpu_joules_total' \
  | head
```

## Potencia instantánea del pod

```promql
sum by (pod_name, pod_namespace) (
  kepler_pod_cpu_watts{
    pod_namespace="green-experiment",
    pod_name="<POD_NAME>"
  }
)
```

Si observas varias series por `state`, puedes eliminar ese label:

```promql
sum(
  max without (state) (
    kepler_pod_cpu_watts{
      pod_namespace="green-experiment",
      pod_name="<POD_NAME>"
    }
  )
)
```

## Energía atribuida al pod

Después de que el Job termine:

```promql
sum(
  max without (state) (
    max_over_time(
      kepler_pod_cpu_joules_total{
        pod_namespace="green-experiment",
        pod_name="<POD_NAME>",
        zone="<PRIMARY_ZONE>"
      }[2h]
    )
  )
)
```

El rango `[2h]` debe ser mayor que la duración del Job.

Debes inspeccionar primero si la métrica del pod tiene realmente el label `zone` en tu despliegue. Si no aparece, elimina ese filtro y documenta cómo está exponiendo las series Kepler 0.11.4 en tu nodo.

Consulta mediante API:

```bash
QUERY='sum(max without(state)(max_over_time(kepler_pod_cpu_joules_total{pod_namespace="green-experiment",pod_name="'"$POD"'"}[2h])))'

curl -sG http://localhost:9090/api/v1/query \
  --data-urlencode "query=$QUERY" \
  | jq
```

---

# 13. Métricas secundarias a nivel de nodo

La métrica primaria será:

```text
kepler_pod_cpu_joules_total
```

Como validación secundaria se utilizará la energía activa del nodo:

```promql
sum(
  increase(
    kepler_node_cpu_active_joules_total{
      node_name="sebas-green-baremetal-jpmdwiscskyy-node-0",
      zone="<PRIMARY_ZONE>"
    }[45m]
  )
)
```

También puedes observar potencia activa:

```promql
sum(
  kepler_node_cpu_active_watts{
    node_name="sebas-green-baremetal-jpmdwiscskyy-node-0",
    zone="<PRIMARY_ZONE>"
  }
)
```

Y potencia total del dominio:

```promql
sum(
  kepler_node_cpu_watts{
    node_name="sebas-green-baremetal-jpmdwiscskyy-node-0",
    zone="<PRIMARY_ZONE>"
  }
)
```

Interpretación:

```text
Pod energy:
  energía activa atribuida al workload.

Node active energy:
  energía activa total del nodo durante el intervalo.

Node total energy:
  energía total del dominio RAPL durante el intervalo,
  incluyendo la parte idle.
```

---

# 14. Baseline idle

Antes de ejecutar Monte Carlo, mide al menos 15 minutos de idle.

```promql
kepler_node_cpu_watts{
  node_name="sebas-green-baremetal-jpmdwiscskyy-node-0",
  zone="<PRIMARY_ZONE>"
}
```

```promql
kepler_node_cpu_active_watts{
  node_name="sebas-green-baremetal-jpmdwiscskyy-node-0",
  zone="<PRIMARY_ZONE>"
}
```

```promql
kepler_node_cpu_idle_watts{
  node_name="sebas-green-baremetal-jpmdwiscskyy-node-0",
  zone="<PRIMARY_ZONE>"
}
```

Registrar:

```text
median total watts
p95 total watts
median active watts
median idle watts
CPU utilization
pods presentes en el nodo
fecha y hora UTC
```

Comprobar workloads presentes:

```bash
kubectl get pods -A -o wide \
  --field-selector spec.nodeName="$BM_NODE"
```

Kepler debe permanecer activo durante baseline y workload.

## Captura automatizada

El script `scripts/capture_idle.py` realiza la medición completa sin usar la
interfaz de Prometheus. Respeta `KUBECONFIG`, descubre el nodo bare metal por su
label, abre un `port-forward` temporal a Prometheus y lo cierra al terminar.

En este clúster se observaron los dominios siguientes:

```text
zone=core     path=aggregated-core
zone=package  path=aggregated-package
```

Se conservan ambos dominios en los datos crudos, pero se usa únicamente
`zone=package` en el resumen para evitar doble conteo.

Ejecutar una captura formal de 15 minutos:

```bash
cd kubernetes/experiment-01
export KUBECONFIG=~/config

python3 scripts/capture_idle.py \
  --duration 15m \
  --step 10 \
  --zone package
```

No hace falta abrir Prometheus o Grafana en otra terminal. Para una prueba de
funcionamiento corta se puede usar `--duration 1m`; esa prueba no sustituye el
baseline formal.

El recolector guarda:

```text
idle/<UTC timestamp>/
├── status.json
├── metadata.json
├── summary.json
├── metrics.csv
├── series-summary.csv
├── cluster/
├── prometheus/
│   ├── kepler-metric-names.json
│   ├── targets.json
│   └── raw/
└── snapshots/
    ├── start/
    └── end/
```

`status.json` permite distinguir una ejecución completa de otra interrumpida.
`summary.json` contiene las medianas, p95, cobertura de muestras y deltas de
energía del dominio primario. `metrics.csv` y `prometheus/raw/` conservan todas
las muestras descargadas para que el análisis pueda repetirse sin volver a
consultar Prometheus.

También se capturan, cuando están disponibles:

* potencia, energía total, activa e idle de Kepler;
* utilización de CPU reportada por Kepler;
* energía y potencia atribuida a pods y contenedores;
* CPU, memoria, load, frecuencia, temperatura y pressure de node-exporter;
* I/O de disco y red;
* CPU y memoria por contenedor desde cAdvisor;
* pods, eventos y `kubectl top` al principio y al final;
* logs anteriores de cualquier contenedor en `CrashLoopBackOff`;
* versiones, targets y objetos de configuración relevantes.

Las métricas Kepler por proceso se omiten por defecto debido a su alta
cardinalidad. Para incluirlas explícitamente:

```bash
python3 scripts/capture_idle.py \
  --duration 15m \
  --zone package \
  --include-processes
```

Antes de aceptar una captura, comprobar:

```bash
jq '{valid_idle_baseline, blocking_conditions, primary_metrics}' \
  idle/<UTC timestamp>/summary.json
```

Una captura marcada como `valid_idle_baseline: false` se conserva como
diagnóstico, pero no debe entrar en la estadística formal. El script invalida
la ventana si detecta un workload en `green-experiment`, un `node-debugger`, un
pod en `CrashLoopBackOff`, Kepler caído o menos del 80 % de las muestras de
potencia esperadas.

### Estado observado antes del baseline formal

La captura diagnóstica del 13 de julio de 2026 encontró dos bloqueos sobre el
nodo bare metal:

```text
default/node-debugger-sebas-green-baremetal-jpmdwiscskyy-node-0-jcbg6
kube-system/cern-magnum-openstack-manila-csi-nodeplugin-th8wh
```

El primer pod es una sesión `kubectl debug` que sigue abierta. Cuando ya no se
necesite, cerrarla antes del baseline:

```bash
kubectl delete pod -n default \
  node-debugger-sebas-green-baremetal-jpmdwiscskyy-node-0-jcbg6
```

El segundo pertenece al DaemonSet administrado por Magnum
`cern-magnum-openstack-manila-csi-nodeplugin`. El contenedor
`cephfs-registrar` falla porque el plugin intenta consultar el instance ID en el
endpoint de metadata OpenStack `169.254.169.254`, que no responde en el nodo
bare metal. Se reinicia aproximadamente cada cinco minutos.

No basta con borrar el pod: el DaemonSet lo recreará. Antes de la medición
formal hay que coordinar una de estas soluciones con quien gestione el clúster:

* excluir el nodo bare metal del DaemonSet Manila si ese CSI no se necesita allí;
* configurar el plugin para identificar correctamente el bare metal;
* corregir el acceso a metadata si el diseño del clúster exige Manila en ese nodo.

Hasta entonces, las capturas sirven para validar el pipeline y caracterizar el
ruido de fondo, pero no como baseline idle formal.

---

# 15. Calibración energética

Antes de ejecutar ventanas verdes:

1. Ejecuta el mismo workload cinco veces.
2. Mantén la misma semilla.
3. Mantén los mismos recursos.
4. Ajusta el número de muestras para obtener entre 30 y 45 minutos.
5. Mide runtime y energía.

Para ajustar muestras:

```text
new_samples =
    current_samples × target_runtime / measured_runtime
```

Ejemplo:

```text
50 M × 40 / 10 = 200 M por worker
```

Registrar:

```text
runtime
pod CPU energy
node active CPU energy
average pod power
pi estimate
total samples
pod UID
image digest
background utilization
```

Potencia media:

```text
average_power_watts =
    energy_joules / runtime_seconds
```

Criterio de estabilidad:

```text
runtime coefficient of variation < 10%
pod-energy coefficient of variation < 10%
all scientific outputs identical
Kepler clearly follows workload start and completion
```

---

# 16. Run-now contra green-window

## Definición del workload

```text
estimated runtime:
  p90 de la calibración

deadline:
  decision time + 24 h

interruptible:
  false

deferrable:
  true
```

## Ejemplo de salida del modelo

```json
{
  "decision_time": "2026-07-13T10:00:00Z",
  "horizon_hours": 24,
  "required_duration_minutes": 45,
  "decision": "DEFER",
  "recommended_start_time": "2026-07-14T02:00:00Z",
  "recommended_end_time": "2026-07-14T02:45:00Z",
  "current_predicted_intensity": 16.4,
  "window_predicted_intensity": 12.8,
  "expected_reduction_percent": 21.9
}
```

Política inicial:

```text
if a valid window exists before the deadline
and expected reduction >= 5%:
    DEFER
else:
    RUN_NOW
```

Debes guardar siempre el forecast emitido originalmente. No se debe reconstruir posteriormente usando datos reales.

---

# 17. Diseño pareado

Para cada origen de decisión se crean dos ejecuciones.

## Control

```text
Policy:
  run-now

Start:
  inmediatamente después de la decisión
```

## Tratamiento

```text
Policy:
  green-window

Start:
  ventana recomendada por el modelo
```

Mantener:

```text
mismo número de muestras
misma semilla
mismo número de CPUs
misma memoria
misma imagen y digest
mismo nodo
mismo código
mismos requests y limits
```

Repeticiones:

```text
5 pares:
  piloto

10 pares:
  resultado principal inicial
```

También debes registrar los casos donde el modelo recomienda `RUN_NOW`.

---

# 18. Cálculo de energía

Para el pod:

```text
energy_joules =
    valor final de kepler_pod_cpu_joules_total
```

Convertir:

```text
energy_kWh =
    energy_joules / 3,600,000
```

Potencia media:

```text
average_power_W =
    energy_joules / runtime_seconds
```

---

# 19. Cálculo de carbono

Si el workload cae dentro de un solo intervalo:

```text
operational_carbon_g =
    energy_kWh
    × realized_carbon_intensity_gCO2_per_kWh
```

Si atraviesa varios intervalos:

```text
operational_carbon_g =
    Σ energy_kWh(interval)
      × carbon_intensity(interval)
```

La señal debe corresponder a:

```text
RTE taux CO₂
production-based
France
UTC
```

No mezcles estos resultados con una intensidad `consumption-based`.

Sin un PUE documentado, reporta:

```text
IT operational carbon
```

Con un PUE oficial:

```text
facility_adjusted_carbon_g =
    energy_kWh
    × PUE
    × realized_carbon_intensity
```

---

# 20. Perfect foresight

Perfect foresight no interviene en la decisión live.

Se calcula después usando la intensidad realizada.

Para un workload con duración `D`, el oracle debe seleccionar la ventana continua de duración `D` con menor carbono dentro del horizonte.

```text
Cperfect_foresight =
    mínimo carbono realizable
    dentro de las siguientes 24 h
    respetando duración y deadline
```

No debes seleccionar solamente la hora individual más verde si el workload dura varias horas.

---

# 21. Métricas finales

## Métrica primaria

```text
paired carbon saving (%) =

100 ×
(carbon_run_now - carbon_green)
───────────────────────────────
        carbon_run_now
```

## Métricas secundarias

```text
pod_energy_green / pod_energy_run_now

node_active_energy_green /
node_active_energy_run_now

runtime_green / runtime_run_now

waiting time

forecasted carbon reduction

realized carbon reduction

forecast error

regret against perfect foresight

deadline violations

job failures

scientific output equality
```

Fracción del ahorro alcanzable:

```text
captured savings (%) =

(Cnow - Cgreen)
─────────────── × 100
  (Cnow - CPF)
```

---

# 22. Hipótesis

## H1 — Carbon reduction

```text
La ejecución en ventanas seleccionadas por el modelo
reduce las emisiones operacionales respecto a run-now.
```

## H2 — Energy stability

```text
El mismo workload consume aproximadamente la misma
energía cuando se ejecuta en diferentes horas.
```

## H3 — Performance stability

```text
El runtime no cambia sustancialmente entre run-now
y green-window.
```

---

# 23. Criterio inicial de éxito

```text
median carbon reduction > 5%

median pod-energy difference < 10%

median runtime difference < 10%

zero deadline violations

zero failed jobs

identical workload outputs
```

Un ahorro pequeño también es válido. Puede indicar que la intensidad de carbono francesa permaneció estable o que no existieron suficientes oportunidades de temporal shifting.

---

# 24. Estructura del repositorio

```text
experiment-01/
├── README.md
├── cluster/
│   ├── nodes.txt
│   ├── baremetal-node.txt
│   ├── baremetal-node.yaml
│   ├── baremetal-existing-pods.txt
│   ├── kubernetes-version.yaml
│   ├── helm-version.txt
│   ├── helm-releases.txt
│   ├── kepler-values-all.yaml
│   └── software-versions.txt
├── manifests/
│   ├── monitoring-values.yaml
│   ├── kepler-values.yaml
│   ├── monte-carlo-configmap.yaml
│   └── monte-carlo-job-template.yaml
├── scripts/
│   ├── capture_idle.py
│   └── run_monte_carlo.py
├── idle/
│   └── <UTC timestamp>/
│       ├── status.json
│       ├── metadata.json
│       ├── summary.json
│       ├── metrics.csv
│       ├── series-summary.csv
│       ├── cluster/
│       ├── prometheus/
│       └── snapshots/
├── trials/
│   └── <trial-id>/
│       ├── forecast.json
│       ├── decision.json
│       ├── job.yaml
│       ├── pod.json
│       ├── workload-output.json
│       ├── prometheus-energy.json
│       ├── energy.csv
│       ├── carbon.csv
│       └── result.json
└── analysis/
    ├── analyse.py
    ├── results.csv
    └── figures/
```

---

# 25. Congelar versiones

```bash
kubectl version -o yaml \
  > experiment-01/cluster/kubernetes-version.yaml

helm version \
  > experiment-01/cluster/helm-version.txt

helm list -A \
  > experiment-01/cluster/helm-releases.txt

helm get values kepler -n kepler --all \
  > experiment-01/cluster/kepler-values-all.yaml

helm get manifest kepler -n kepler \
  > experiment-01/cluster/kepler-manifest.yaml

kubectl get pods -A -o json \
  > experiment-01/cluster/all-pods.json
```

Guardar la imagen real de Kepler:

```bash
kubectl get daemonset kepler -n kepler \
  -o jsonpath='{.spec.template.spec.containers[*].image}{"\n"}'
```

Guardar build info:

```bash
curl -s http://localhost:28282/metrics \
  | grep '^kepler_build_info' \
  > experiment-01/cluster/kepler-build-info.txt
```

También registrar:

```text
Green Window Observatory Git commit
model artifact hash
training cutoff
forecast creation timestamp
Monte Carlo source commit
workload image digest
Kepler chart version 0.11.4
Kepler image digest
Prometheus chart version
primary RAPL zone
scrape interval
timezone UTC
```

---

# Entregable final

> A reproducible end-to-end evaluation of forecast-driven temporal shifting for a deterministic CPU Monte Carlo workload on a Kubernetes bare-metal node, using Kepler 0.11.4 CPU-energy attribution and realized French-grid carbon intensity.

Tabla principal:

| Trial | Run-now CI | Green CI | Run-now energy | Green energy | Carbon saving | Delay |
| ----- | ---------: | -------: | -------------: | -----------: | ------------: | ----: |
| 01    |          … |        … |              … |            … |             … |     … |
| 02    |          … |        … |              … |            … |             … |     … |

Figuras principales:

```text
Run-now carbon
Green-window carbon
Perfect-foresight carbon
```

y:

```text
Kepler pod watts
Kepler node active watts
CPU utilization
workload start/end
```

## Orden inmediato

```text
1. Confirmar etiquetas de nodos.
2. Confirmar que Kepler 0.11.4 solo corre en bare metal.
3. Verificar el ServiceMonitor.
4. Confirmar que Prometheus scrapea Kepler.
5. Listar los dominios y labels RAPL reales.
6. Seleccionar PRIMARY_ZONE.
7. Medir 15 minutos de idle.
8. Ejecutar un Monte Carlo corto.
9. Verificar kepler_pod_cpu_watts.
10. Verificar kepler_pod_cpu_joules_total.
11. Ejecutar cinco calibraciones.
12. Fijar el workload definitivo.
13. Iniciar pares run-now vs green-window.
```
