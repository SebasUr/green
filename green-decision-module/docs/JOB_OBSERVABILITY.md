# Observabilidad energética y de carbono por Kubernetes Job

El observador genera automáticamente un JSON por cada Job etiquetado cuando
este termina. El reporte combina:

- energía CPU atribuida al pod por Kepler 0.11 (`zone="package"`);
- intensidad de carbono realizada de RTE/eCO2mix;
- emisiones operacionales sin PUE;
- cobertura, resets, muestras ausentes y otras señales de calidad.

La fórmula se aplica por intervalo y no con una intensidad media global:

```text
emissions_gco2eq =
    sum(delta_kepler_joules_interval / 3_600_000
        * rte_carbon_intensity_gco2eq_per_kwh_interval)
```

## Preparación

Usa el entorno y la instalación editable del proyecto:

```bash
conda activate green-observatory
cd green-decision-module
pip install -e .
export KUBECONFIG=~/config
```

El comando usa `kubectl` para leer Jobs y Pods. Si no se proporciona
`--prometheus-url`, abre y mantiene automáticamente un port-forward al servicio
Prometheus configurado.

## Etiquetar un Job

La etiqueta se añade al objeto `Job`; no es necesario modificar la imagen ni
añadir un sidecar:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: workload-example
  namespace: green-experiment
  labels:
    sustainability.cern.ch/track: "true"
    sustainability.cern.ch/workload: "example"
    sustainability.cern.ch/policy: "run-now"
spec:
  # ...
```

Conviene etiquetar también el pod template para consultas Grafana, aunque el
observador selecciona el Job por su propia etiqueta.

También se puede etiquetar un Job ya creado:

```bash
kubectl label job -n green-experiment workload-example \
  sustainability.cern.ch/track=true
```

## Observar continuamente

Desde `green-decision-module/`:

```bash
greenctl jobs observe \
  --namespace green-experiment \
  --output runs/job-reports
```

Para todos los namespaces, omite `--namespace`. Para validar sin dejar un
proceso activo:

```bash
greenctl jobs observe --once --output runs/job-reports
```

Si Prometheus ya está accesible:

```bash
greenctl jobs observe \
  --prometheus-url http://127.0.0.1:9090 \
  --output runs/job-reports
```

El selector es configurable:

```bash
greenctl jobs observe \
  --selector 'sustainability.cern.ch/track=true'
```

## Reportar un Job concreto

La etiqueta no es necesaria para este modo:

```bash
greenctl jobs report mc-calibration-01 \
  --namespace green-experiment \
  --output runs/job-reports
```

## Estados del reporte

El observador escribe el JSON aunque aún no sea científicamente utilizable:

- `quality.final=true`: energía e intensidad cumplen las coberturas mínimas;
- `quality.final=false`: reporte provisional; se vuelve a intentar en el
  siguiente poll;
- `quality.valid=false`: revisar `quality.warnings` antes de comparar.

RTE puede publicar el último punto después de que el Job termine. Durante ese
intervalo es normal ver un reporte provisional.

## Contenido del JSON

```json
{
  "job": {
    "uid": "...",
    "namespace": "green-experiment",
    "name": "mc-example",
    "labels": {
      "sustainability.cern.ch/workload": "monte-carlo",
      "sustainability.cern.ch/policy": "run-now"
    }
  },
  "energy": {
    "total_joules": 42413.38,
    "total_kwh": 0.011781,
    "average_power_watts": 45.64,
    "pods": []
  },
  "carbon": {
    "energy_weighted_intensity_gco2eq_per_kwh": 18.4,
    "emissions_gco2eq": 0.2168
  },
  "quality": {
    "valid": true,
    "final": true,
    "energy_coverage_ratio": 0.99,
    "carbon_energy_coverage_ratio": 1.0,
    "warnings": []
  }
}
```

Los labels `workload`, `policy`, `scheduler`, `experiment` y cualquier otro
label del Job se preservan. Así los reportes se pueden agrupar después sin
escribir lógica específica para Monte Carlo.

## Construir el CSV comparativo

Todos los reportes finales se pueden aplanar en una sola tabla:

```bash
greenctl jobs summarize \
  --reports runs/job-reports \
  --output runs/job-reports/summary.csv
```

La tabla contiene energía, potencia media, intensidad ponderada, emisiones,
coberturas y las dimensiones `workload`, `policy`, `scheduler`, `experiment` y
`trial`. Usa `--include-provisional` solamente para diagnosticar capturas aún no
finalizadas.

## Enriquecimiento de reproducibilidad (schema 1.1)

Además de energía y carbono, el reporte captura **post-hoc y en solo lectura**
todo lo necesario para que un run sea reproducible y comparable. Ningún bloque
es obligatorio: si una consulta falla, el bloque queda vacío con un warning y la
contabilidad no se pierde.

| Bloque | Contiene | Para qué |
|---|---|---|
| `provenance` | imagen + **imageID (digest real)**, command/args, env literal, requests/limits, `nodeSelector` | saber exactamente qué corrió |
| `node_context` | energía total/activa/idle del nodo, CPU ratio, `job_share_of_active_energy` | validación secundaria |
| `isolation` | **post-flight**: co-tenants y su energía, restarts, `kepler_up_ratio`, `clean_node` | ¿la medición es comparable? |
| `workload_outputs` | stdout + **`stdout_sha256`** (+ `parsed_json` si es JSON) | igualdad de salida científica |
| `energy_intervals` | traza por intervalo de scrape: J + intensidad + gCO2eq | auditoría; reemplaza a `metrics.csv` |

```bash
greenctl jobs report mc-calibration-01 -n green-experiment --include-intervals
greenctl jobs report mc-calibration-01 -n green-experiment --no-context --no-capture-logs
```

Flags: `--include-intervals` (traza por intervalo, off por defecto),
`--no-context` (omite provenance/node_context/isolation), `--no-capture-logs`
(omite stdout). Aplican igual a `jobs report` y a `jobs observe`.

### El post-flight es más fuerte que un pre-flight

Un pre-flight solo prueba que el nodo estaba limpio en `t0`. `isolation` se
reconstruye desde Prometheus y cubre **toda la ventana**: si otro pod arrancó a
mitad del run, aparece. `clean_node` es `false` cuando los co-tenants superan el
5 % de la energía atribuida, hubo restarts en el nodo, o Kepler no fue
scrapeable ≥99 % de la ventana.

**`clean_node` NO afecta a `quality.valid`/`final`**, y es deliberado: la
contabilidad sigue siendo correcta aunque el nodo estuviera sucio. Son dos
preguntas distintas — *«¿la medición es buena?»* (`quality`) y *«¿este run es
comparable con otro?»* (`isolation`). Mezclarlas haría que un run contaminado se
reintentara para siempre.

### `stdout_sha256`: igualdad de salida sin lógica por workload

Dos ejecuciones deterministas con los mismos parámetros deben dar el **mismo
hash**. Eso cubre el criterio *«identical scientific outputs»* para **cualquier**
workload, sin parsear π ni nada específico del Monte Carlo. En el CSV de
`summarize` sale como columna `stdout_sha256`.

## Límites operativos

- El alcance actual es energía operacional CPU atribuida por Kepler; no aplica
  PUE ni estima el consumo total del nodo.
- `node_context` es **contexto, no atribución**: si algo más corrió en el nodo,
  esas cifras lo incluyen. Lo atribuible al Job es `energy.total_joules`.
- El Job y sus pods deben seguir visibles hasta que el observador los lea. No
  uses un `ttlSecondsAfterFinished` inferior al intervalo de polling. Esto es
  especialmente cierto para `workload_outputs`: los logs viven con el pod, así
  que si el GC se lo lleva, el stdout y su hash se pierden (la energía y el
  carbono no: siguen en Prometheus).
- `provenance.env` solo recoge variables con `value` literal; las que vienen de
  `valueFrom` (secrets, configmaps) **no se resuelven** a propósito.
- `container_restarts` necesita kube-state-metrics (`kube_pod_info`); sin él el
  campo queda a `null` con un warning.
- Prometheus debe retener las series Kepler durante el tiempo suficiente.
- Los Jobs sin serie Kepler o sin cobertura RTE permanecen provisionales y
  conservan las causas en `quality.warnings`.
- El total incluye pods fallidos y reintentos porque también consumen energía.
