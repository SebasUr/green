# Green Window Observatory — Reporte de Metodología (V1.0, módulo de carbono)

Documento para quitar la "caja negra": qué datos entran, cómo se tratan, qué
modelos elegí y por qué, cómo se predicen las ventanas, cómo se evalúa y qué
significan *exactamente* los números. Todos los resultados salen de los datos
reales del repo (backtest sobre 348 orígenes, feb–abr 2026).

---

## 0. Antes que nada: el "60%" NO es una moneda

El número que te chocó — el modelo captura **61.8%** — **no es una probabilidad de
acierto**. Es **"% del potencial del oráculo"**, definido así:

```
% potencial = (carbono_si_corres_AHORA  −  carbono_de_la_ventana_que_elige_el_modelo)
              ───────────────────────────────────────────────────────────────────────
              (carbono_si_corres_AHORA  −  carbono_de_la_ventana_PERFECTA (oráculo))
```

- **0%** = no mejora nada frente a "correr ahora mismo".
- **100%** = tan bueno como un oráculo que *ve el futuro real*.

En números absolutos del backtest: correr ahora = **14.91**, el modelo te lleva a
**12.96**, y lo perfecto sería **11.76** gCO₂/kWh. El modelo se queda con
`(14.91−12.96)/(14.91−11.76) = 61.9%` del ahorro máximo posible → **~13% menos
intensidad** que correr sin pensar.

**¿Qué saca una moneda?** Elegir una hora al azar entre las candidatas cae, en
promedio, en la media → **~0%** del potencial. De hecho la *persistencia* (que no
sabe rankear) saca **−4.1%**, o sea *peor* que correr ahora. Y en "acertar la hora
más verde exacta", el azar acierta 1 de 6 = **16.7%**; el modelo acierta **35.1%**
→ **2.1× mejor que la moneda**. Así que 62% es señal real, no ruido.

**El matiz honesto** que sí debes saber: en feb–abr 2026 el grid francés estuvo
*uniformemente* muy limpio (mayormente 5–30 gCO₂/kWh). El propio oráculo solo
ahorra 3.15 gCO₂ frente a "correr ahora". Cuando el grid está *plano y verde*,
**no hay mucho que ganar** — el 62% mide la *habilidad* del modelo; el *beneficio
absoluto* depende del régimen y en 2026 fue chico.

---

## 1. Qué pregunta responde (y cuál no)

Responde: **"¿cuándo es más verde correr?"** en el grid francés. NO responde nada
a nivel de workload (si un job debe dormir, cuánta energía gasta un pod, etc.) —
eso es V1.1. Es una **capa de inteligencia temporal** de la señal de carbono.

---

## 2. Qué datos recibo

### 2.1 Fuente primaria — ODRE / eCO2mix (RTE), sin autenticación

Formato de cada fila: `time, plugin_instance, value` → lo normalizo a un *frame
canónico* horario. Campos que uso (dataset `eco2mix-national-cons-def` histórico
+ `eco2mix-national-tr` tiempo real):

| Campo ODRE | Columna canónica | Qué es |
|---|---|---|
| `taux_co2` | `carbon_intensity_gco2_kwh` | **verdad de terreno**, gCO₂/kWh, *production-based* (mix generado en FR) |
| `consommation` | `consumption_mw` | consumo (MW) |
| `nucleaire, gaz, charbon, fioul` | `nuclear/gas/coal/fuel_oil_mw` | generación por fuente |
| `eolien, solaire, hydraulique` | `wind/solar/hydro_mw` | renovables |
| `bioenergies, pompage` | `bioenergy/pumped_storage_mw` | bioenergía, bombeo |
| `ech_physiques` | `physical_exchange_mw` | intercambios (**negativo = exportación**) |

Hechos verificados contra la API real:
- **`taux_co2` es la verdad oficial de RTE**, disponible 2011→2026, a **cadencia
  30 min** (`:00`/`:30`); los slots `:15`/`:45` y la cola no consolidada vienen
  `null` → se descartan.
- **`date_heure` es UTC real** (verificado: la solar pica ~12 UTC = mediodía local).
- Snapshot de trabajo: **46 697 filas horarias, 2021-01-01 → 2026-04-30**, media
  29.9, min 5, max 110 gCO₂/kWh.

### 2.2 Fuente comparativa — Electricity Maps (opcional, con tu API key)

- Intensidad **consumption-based** (reparte imports/exports), factor *lifecycle*.
- Da: `latest`, `forecast` (**solo ~24h adelante**, horario), `history` (24h reales).
- Tu tier **no** da el mix de generación (`power-breakdown/forecast` → 401).

---

## 3. Cómo trato los datos (pipeline)

```
descarga ODRE (export/json, por año)
  → mapeo de campos a nombres canónicos
  → parseo de date_heure a UTC (tz-aware, obligatorio)
  → descarto filas sin taux_co2 (slots :15/:45 y cola no consolidada)
  → resampleo 30min → horario (media): taux_co2 horario = media de los 2 puntos
  → frame canónico: índice UTC ordenado, sin duplicados, columnas fijas
  → snapshot .parquet  (modo "replay" = reproducible, sin depender de la red)
```

**Invariante central: sin fuga temporal (no look-ahead).** Todo lo que el modelo
ve en el instante de decisión `t0` proviene de datos ≤ `t0`. El valor futuro solo
se usa como *etiqueta* de entrenamiento, nunca como feature. Hay un test que
verifica que las features en `t0` son idénticas calculadas sobre toda la serie o
solo sobre el pasado ≤ `t0`.

---

## 4. Qué modelos escogí y por qué

No hay *un* modelo: hay una **escalera de 5**, del más tonto al más listo. Cada
peldaño existe para justificar (o descartar) el siguiente y como vara de medir.

| # | Modelo | Qué hace | Por qué está |
|---|---|---|---|
| 1 | **Persistencia** | "el futuro = el valor actual" | piso mínimo absoluto |
| 2 | **Climatología** | mediana histórica por (mes, día-semana, hora) en hora local Paris | patrón recurrente |
| 3 | **Climatología corregida** | climatología + residuo reciente (EWMA que decae con el horizonte) | ajusta el nivel del día de hoy |
| 4 | **SARIMAX** | ARIMA(2,1,1) + estacionalidad de Fourier (diaria+semanal) | el "modelo estadístico clásico" |
| 5 | **Modelo del proyecto** | gradient boosting sobre 34 features tabulares | captura no-linealidades + info del mix |
| — | **Oráculo** | usa el futuro real conocido | cota superior (no desplegable) |

### 4.1 Por qué gradient boosting y no "modelos convencionales"

- **vs ARIMA/SARIMA clásico:** lineal, univariante, mal con estacionalidades
  múltiples (diaria+semanal+anual) y con exógenas. (Lo probé — ver §7; empata a
  nuestra climatología corregida.)
- **vs deep learning (LSTM/Transformer):** necesita mucho dato, caro, difícil de
  explicar, sobreajusta. Con ~45k puntos tabulares no suele ganarle al boosting.
  El plan pide *reproducible y explicable*.
- **vs regresión lineal:** demasiado rígida para hora × estación × mix.
- **Por qué árboles (HistGradientBoosting):** el problema es tabular con efectos
  de calendario e interacciones no lineales; los árboles las capturan solos,
  aceptan tipos mixtos, no requieren escalado, manejan **NaN de forma nativa**
  (importante: los lags tienen NaN al arranque), son rápidos y explicables.

### 4.2 Cómo funciona el modelo del proyecto por dentro

**Regresión directa multi-horizonte**: **6 estimadores independientes**
(`HistGradientBoostingRegressor`), uno por horizonte `{1,3,6,12,24,48}h`. Cada uno
usa **34 features**, en 3 grupos, todas conocidas *en t0* o deterministas:

- **Calendario del target (9):** hora, día-semana, mes, finde, festivo FR, y
  seno/coseno de hora y día-del-año. Es el *cuándo* — legítimo, el calendario del
  futuro se conoce hoy.
- **Señal reciente (11):** valor actual, lags (1,2,3,24,168h), medias móviles
  (3,6,24h), pendiente 6h, y el residuo respecto a la climatología.
- **Estado del sistema en t0 (14):** nuclear, gas, carbón, eólica, solar, hidro,
  consumo, intercambios + ratios (cuota renovable, cuota nuclear, exportación).

**Entrenamiento:** para cada horizonte se arma `(X = features en t0, y = carbono
real en t0+h)` sobre los datos anteriores al periodo de test, y se ajusta el
estimador. **Predicción:** en `t0` se construye la fila de features y cada
estimador da su número. (El "recursivo" —predecir t+1 y realimentar— se descarta
porque necesitaría pronosticar también el mix futuro, que no tenemos.)

---

## 5. Cómo predice las ventanas verdes

Dos pasos, ambos sobre el forecast (o sobre datos reales, para ventanas históricas):

1. **`green_score` ∈ [0,1], mayor = más verde.** Se normaliza por *rango
   empírico* dentro del horizonte: `green_score(x) = 1 − F(x)`, donde `F` es la
   fracción de horas del horizonte con intensidad ≤ x. Una hora en el p10 (muy
   limpia) puntúa ~0.90; en el p90, ~0.10. Es robusto a la asimetría del carbono.

2. **Detección de ventana = bloque contiguo bajo el p25 del horizonte.** Se toman
   las horas por debajo del percentil 25 de la intensidad del horizonte, se unen
   bloques separados por huecos ≤ `merge_gap`, se filtran por duración
   (min/max), y se rankean por `green_score` medio. Cada ventana lleva
   `carbon_score`, intensidad media, confianza y una razón textual.

Ejemplo real (`greenctl windows analyze`): en un horizonte de 48h de finales de
abril (todo ~6 gCO₂/kWh) detecta 3 ventanas contiguas de score ~0.79.

---

## 6. Cómo lo evalúo y qué significa cada métrica

**Protocolo:** *backtest de origen rodante (rolling-origin), sin fuga.* Se entrena
con datos `< 2026-02-01` (44 563 filas). Se avanza un "origen" cada 6h por el
periodo de test (feb–abr 2026 → **348 orígenes**) y en cada uno se pronostican
los 6 horizontes. Nada usa datos posteriores al origen.

**Métricas de punto:** `MAE`, `RMSE`, `bias` (pred−real) por horizonte.

**Métricas de ventana** (la decisión real "¿en cuál de las horas pronosticadas
corro?"): cada estrategia elige la hora que *predice* más verde; se mide el
carbono **real** en esa hora contra dos referencias — correr-ahora y el oráculo:

- `mean_realized_gco2` — intensidad real media a la que correrías siguiéndolo.
- `mean_regret` — cuánto por encima del oráculo (0 = perfecto).
- `pct_oracle_potential` — el % explicado en §0.
- `spearman` — correlación de ranking entre lo predicho y lo real (1 = perfecto).
- `top1_accuracy` — fracción de veces que eligió la hora *realmente* más verde.

---

## 7. Resultados

### 7.1 Error de punto — MAE (gCO₂/kWh) por horizonte

| modelo | 1h | 3h | 6h | 12h | 24h | 48h |
|---|---|---|---|---|---|---|
| persistencia | 0.79 | 1.75 | 3.99 | 3.30 | 3.21 | 3.76 |
| climatología | 17.31 | 17.31 | 17.44 | 17.50 | 17.57 | 17.68 |
| corregida | 5.13 | 5.06 | 6.04 | 7.35 | 10.08 | 13.65 |
| sarimax | 0.88 | 2.10 | 3.51 | **3.28** | **3.74** | **4.63** |
| project (ML) | 0.81 | 1.56 | 2.91 | 3.61 | 4.24 | 5.43 |

La climatología va **+17 sesgada**: 2026 fue atípicamente bajo vs 2021-2025.
SARIMAX (con `d=1`) rastrea mejor el *nivel* a largo plazo y le gana al ML en MAE
a 24/48h. Persistencia es fortísima a corto por la autocorrelación del grid.

**En porcentaje (WAPE = MAE / nivel real medio; el nivel real del test es
≈14.4 gCO₂/kWh, *calculado*, no supuesto):** el ML se equivoca ~5% a 1h, ~20% a
6h y ~38% a 2 días → **≈21% global**; persistencia ≈19% global. Se prefiere WAPE
sobre MAPE porque el MAPE se infla al dividir por horas de muy bajo carbono. Ojo:
el WAPE mide *precisión del número*, no *utilidad de la decisión* — persistencia
tiene buen WAPE pero es **inútil para elegir ventana** (§7.2).

### 7.2 Selección de ventana verde — lo que importa

| estrategia | realized gCO₂ | regret | % oráculo | spearman | top‑1 |
|---|---|---|---|---|---|
| run-now | 14.91 | 3.16 | 0% | — | — |
| persistencia | 15.04 | 3.29 | −4.1% | — | 0.24 |
| climatología | 13.85 | 2.09 | 33.7% | 0.05 | 0.20 |
| corregida | 13.36 | 1.61 | 49.1% | 0.05 | 0.26 |
| sarimax | 13.36 | 1.60 | 49.2% | 0.16 | 0.25 |
| **project (ML)** | **12.96** | **1.21** | **61.8%** | **0.23** | **0.35** |
| oráculo | 11.76 | 0.00 | 100% | 1.00 | 1.00 |

**Lecciones:** (1) persistencia tiene gran MAE pero es **inútil para rankear**
(predice plano). (2) SARIMAX ≡ climatología corregida en ventanas (~49%): el
clásico estadístico *es* nuestro baseline hecho a mano. (3) El ML gana en ranking
(62%) porque **usa el mix de generación**, que un modelo puramente temporal no
tiene.

### 7.3 Ensemble (tu pregunta)

Promediar el ML + SARIMAX (diversos y buenos):

| | MAE global | % oráculo (ventanas) |
|---|---|---|
| sarimax | 3.02 | 49.2% |
| project (ML) | 3.09 | 61.8% |
| **ensemble ML+SARIMAX** | **2.80** ✅ | 58.3% |

El ensemble **gana en MAE** (errores diversos se cancelan) pero **no en selección
de ventana** (mezclar el peor rankeador diluye al mejor). Moraleja: ensamblar,
pero para el objetivo correcto.

### 7.4 Electricity Maps — comparación en vivo (`greenctl carbon compare-live`)

EM no puede entrar al backtest histórico (no expone forecasts pasados y es
*consumption-based*). Lo comparable, en vivo:

- **Brecha de base** (23h recientes): RTE production = 26.2, EM consumption =
  **33.7** (diff **+7.4**), pero **correlación 0.98** → niveles distintos, misma
  forma. (EM suma el carbono de la electricidad importada; RTE no.)
- **Acuerdo de forecast** (próximas 25h): **Spearman 0.67** — coinciden bastante
  en qué horas son verdes. Discrepan en la *más* verde: nosotros 14:00Z (pico
  solar), EM 02:00Z (baja demanda). Divergencia real y discutible.

---

## 8. Limitaciones honestas

- **Régimen 2026 plano-verde** → poco margen absoluto que ganar (§0).
- El ML **sobre-predice ~3.5 gCO₂ a 48h** (los árboles no extrapolan; regresan al
  nivel histórico más alto). Mejorable pesando lo reciente o con dense-horizons.
- El **Spearman de ranking (0.23) es modesto**; la calidad de *decisión* (% oráculo)
  es buena, pero el orden fino de horas es mejorable.
- Ventanas densas usan la climatología corregida (el ML solo entrena 6 horizontes).
- EM: distinta base y sin scoring vs realidad todavía (ventana futura).

## 9. Qué NO hace (alcance V1.0)

No decide nada por workload (dormir/diferir/power-cap), no atribuye CO₂ por pod,
no muta Kubernetes. Todo CDC/facility (M3+), simulación (M6), API (M8) están
andamiados pero sin implementar. Eso es V1.1+.

## 10. Cómo reproducir todo

```bash
conda activate green-observatory
greenctl carbon import   --output data/cache/carbon_fr_hourly.parquet   # datos ODRE
greenctl carbon train    --test-start 2026-02-01                        # entrena el ML
greenctl carbon compare  --test-start 2026-02-01                        # tabla MAE + ventanas + oráculo
greenctl carbon compare-live                                            # vs Electricity Maps (con API key)
greenctl carbon forecast --horizon-hours 48                             # ventanas verdes predichas
greenctl windows analyze --horizon-hours 48                             # ventanas sobre datos reales
```

SARIMAX y el ensemble se reproducen por script (documentados arriba); statsmodels
es dependencia opcional (`pip install -e '.[stats]'`).
