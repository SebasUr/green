# Green Window Observatory — Reporte único del modelo de CO₂ (Francia)

Documento único y autoritativo del track de pronóstico de intensidad de carbono
de la red francesa. Reemplaza a los borradores anteriores (`REPORT.md`,
`REPORTE.md`, `REPORTE_C.md`, `REPORT_FRANCE24.md`, `own_report.md`,
`runs/daily_refit_2026/MEJORA_MODELO_2026.md`). Fecha de corte: 20 de julio de 2026.

---

## 1. Resumen ejecutivo

Se pronostica la **intensidad de carbono de la producción eléctrica francesa**
(gCO₂/kWh, `taux_co2` de RTE) para las próximas 24 horas, y se rankean las
ventanas más limpias para agendar cargas diferibles.

**Modelos vigentes (configuración de producción):**

| Uso | Modelo | Métrica | Periodo/protocolo |
|---|---|---|---|
| **Nivel de CO₂ (operacional)** | `physical_alpha2` + calibración causal 14d | **13,06 % MAPE** | mayo–jul 2026, reentreno diario, target provisional en vivo |
| **Ranking de ventanas verdes** | `share_lgbm` (físico por participaciones) | **57,8 % del oráculo**, regret 0,925 gCO₂/kWh | mar–abr 2026, timestamp-causal |
| **Referencia académica (nivel)** | gate físico/Direct + calibración | **11,83 % MAPE**, MAE 1,80 g | mar–abr 2026, target consolidado |

**Contexto:** Francia es el peor caso para MAPE (mitad del tiempo por debajo de
15 gCO₂/kWh; error dominado por rampas de gas poco frecuentes). Referencias
publicadas para redes comparables bajo protocolos más favorables: Alemania
11,6 %, España 11,4 % (EnsembleCI). El resultado está cerca de ese rango y el
cuello de botella es informativo, no de modelo (ver §6).

---

## 2. Qué se predice: tres objetos distintos

1. **Target consolidado/definitivo** (`taux_co2` histórico corregido por RTE):
   estable, para benchmark académico. Disponible hasta el último día consolidado.
2. **Target provisional publicado en vivo:** lo que ve la app en tiempo real
   antes de la consolidación. Otra definición numérica (aplicar la fórmula
   provisional al consolidado da 38 % de MAPE — las señales están correlacionadas
   0,9999 pero no son intercambiables).
3. **Surrogate histórico del provisional:** reconstrucción ex-post de la fórmula
   provisional con componentes del histórico definitivo, para entrenar el
   pipeline operacional.

**Regla de oro:** nunca mezclar un MAPE contra consolidado con uno contra
provisional. Los 11,83 % (académico) y 13,06 % (operacional) miden objetos
distintos y no son comparables entre sí.

### Fórmula provisional de RTE (primera publicación)

```text
(986*coal + 777*fuel_oil + 429*gas + 494*bioenergy)
----------------------------------------------------------------
 nuclear + gas + coal + fuel + wind + solar + hydro + bioenergy
```

Reconstruye el `taux_co2` provisional publicado con MAPE 0,755 %, correlación
0,99993. El error del modelo está casi todo en anticipar los MW de gas, no en la
reconstrucción de CO₂.

---

## 3. Arquitectura de producción

### 3.1 Nivel de CO₂: `PhysicalProxyMoE` (alpha2) + calibración

No predice CO₂ directamente. Pronostica componentes físicos y aplica la fórmula
fija de RTE:

- **Gas:** tres expertos LightGBM por régimen (baseload `<500 MW`, CCG
  `500–2.499 MW`, peak `≥2.500 MW`), mezclados con las probabilidades de un
  clasificador de régimen. La variante `alpha2` eleva las probabilidades a `p²`
  y renormaliza (separación de régimen más dura sin llegar a gate one-hot).
- **Carbón, fuel, bioenergía, generación total:** regresores pooled.
- **Calibración causal de 14 días:** por cada origen, corrige el sesgo reciente
  por bloques horarios (h1–7, h8–15, h16–21, h22–24) usando solo errores con
  `target_time < origin`, aplicando el 25 % de la corrección (shrink hacia 1).

**Reentreno diario expandente**, origen 00:00 UTC, features closed-hour (estado
hasta t−1h, la fila `t` se enmascara porque resume `[t, t+1h)`).

### 3.2 Ventanas verdes: `ConsolidatedShareRegressor` (`share_lgbm`)

Predice directamente las siete fracciones emisoras sobre la generación
doméstica y aplica factores positivos aprendidos solo en entrenamiento. Evita
que un error del denominador amplifique todos los combustibles. Ordena las
horas mucho mejor que el modelo de nivel (top-1 50 %, Spearman 0,521).

### 3.3 Separación señal/decisión

El MAPE evalúa la señal de nivel; el potencial de oráculo y el regret evalúan la
decisión de elegir ventanas. `physical_alpha2` gana en nivel; `share_lgbm` gana
en ranking. Se conservan como dos salidas modulares porque optimizan objetivos
distintos.

---

## 4. Datos y features

### 4.1 Histórico y tiempo real

- **Consolidado:** `carbon_fr_hourly_enriched.parquet` — 42.113 h, jul-2021 a
  30-abr-2026, 34 columnas (agregados + subtipos de gas/fuel/hidro/bioenergía +
  intercambios + batería).
- **Tiempo real (descubrimiento clave):** el dataset `eco2mix-national-tr` de
  ODRE **retiene `taux_co2` + mix completo desde el último día consolidado**, no
  una ventana de ~30 días. Descargado completo 30-abr → 19-jul (1.917 h, sin
  huecos): `carbon_fr_realtime_2026_full.parquet`. Coincide 99,86 % con el
  holdout archivado en el solape. **Se purga al consolidar → re-archivar
  periódicamente.**
- **Puente mayo–junio:** forecasts de mix y precio se cerraron con Energy Charts
  (`mix_day_ahead_fr_hourly_bridged.parquet`,
  `day_ahead_price_fr_hourly_bridged.parquet`), continuos 2021 → hoy.

### 4.2 Señales causales prospectivas

Todas causales por timestamp; las opt-in filtran por vintage de publicación.

- Forecast day-ahead de demanda, eólica on/offshore y solar (Energy Charts).
- Precio day-ahead francés.
- Indisponibilidades RTE versionadas por activo (`updated_date <= origin`).
- Estado físico de la última hora cerrada; lags alineados a cada target D-1/D-7.
- Calendario, hora, horizonte; demanda residual, rampas, resúmenes de curva 24h.

### 4.3 Features de sistema añadidas (julio 2026) — promovidas

| Feature | Fuente | Correlación con gas | Estado |
|---|---|---|---|
| Programa D-1 de intercambios | RTE Exchange Schedule v2 (FMS/OAuth) | nivel 0,62; Δ vs D-1 **0,11** | opt-in `--exchange-schedule` |
| Margen térmico implícito | demanda residual + averías con vintage | nivel 0,79; Δ vs D-1 **0,56** | opt-in `--thermal-margin` |
| Generación total programada D-1 | ENTSO-E A71 vía File Library | nivel 0,61; con gen. real **0,92** | opt-in `--entsoe-a71` |

El margen térmico y el A71 (paquete "sys") son la mejor dieta de datos
prospectiva conseguida. El programa de intercambios aporta contexto marginal.

---

## 5. Resultados

### 5.1 Evaluación rolling 2026 (mayo–julio, reentreno diario, 1.896 h)

Comparación pareada contra target provisional en vivo:

| Configuración | MAPE global | Mayo | Junio | Julio |
|---|---:|---:|---:|---:|
| `physical_alpha2` baseline (solo TR extendido) | 13,50 % | 8,64 | 13,47 | 21,89 |
| **`physical_alpha2` + features sys** | **13,17 %** | 8,22 | 12,89 | 22,11 |
| **`physical_alpha2` + sys + calibración 14d (producción)** | **13,06 %** | — | — | — |
| Direct baseline | 14,45 % | 10,70 | 13,62 | 22,26 |
| Direct + sys | 14,15 % | 10,51 | 13,44 | 21,58 |
| **Direct + sys + intercambios (ctx3)** | **14,10 %** | 10,72 | 13,18 | 21,45 |

Julio es objetivamente duro (persistencia D-1 = 30 %); no es degradación del
modelo. Con la ventana larga, **el gate/blend compuesto ya no supera al físico
calibrado** — la cabeza física es el caballo de batalla y el gate operacional
puede retirarse.

### 5.2 La única mejora de feature con IC limpio

**Direct + los tres paquetes de contexto (ctx3) vs baseline:** 14,448 → 14,102 %
(−0,346 pp), IC bootstrap de bloques de 7 días **[−0,697, −0,033]** — no cruza
cero. Muerde donde importa: cuartil de rampas de gas 24,39 → 23,39 %, julio
−0,81. Es la primera feature de toda la campaña que sobrevive al estándar de
promoción estricto.

En el físico las features de sistema mejoran q1–q3 pero el cuartil de rampas
(q4) queda intocado (25,0 → 25,2): a las cabezas de MW de gas el contexto no les
llega; al modelo de intensidad (Direct), sí.

---

## 6. Diagnóstico del cuello de botella

- La intensidad correlaciona **0,942 con la generación de gas** (solar −0,22,
  eólica −0,12). En una red con fósil <4 %, el gas explica casi toda la varianza
  del error.
- **El 25 % de horas con mayor cambio de gas frente a D-1 concentra el 42,3 %
  del error MAPE** (20,03 % en ese cuartil vs 6,8–7,8 % cuando el gas es estable).
- **El modelo ya "ve" las rampas:** acierta su dirección el 86 % de las veces,
  pendiente real/predicha 1,05 (calibrado), R² 0,59 — más que una lectura lineal
  de demanda residual + precio (0,42). No queda heurística que exprimir.
- **Techo de la información:** si se conociera el gas exacto de mañana, el oráculo
  daría **4,02 % global / 4,57 % en el cuartil de rampas** (vs 13–25 % actuales).
  Esos ~9 puntos no son techo del modelo: son información que RTE tiene en su
  programa de despacho térmico y no publica.

**Conclusión:** es un problema de información, no de arquitectura. Ninguna familia
de modelo nueva sobre las mismas features va a mover ese 41 % de varianza faltante.

---

## 7. Todo lo probado y rechazado (ledger)

Con protocolo timestamp-causal cuando aplica; las cifras pre-fix son screening.

| Experimento | Resultado | Veredicto |
|---|---|---|
| Ranker LambdaMART (ventanas) | 42,2 % oráculo vs 57,8 % de shares | Rechazado |
| Extremely Randomized Trees | 15,62 % MAPE (+2,18 vs Direct, IC [1,07, 3,41]) | Rechazado |
| LSTM | WAPE ~2× peor, 63,9 % oráculo | Rechazado |
| SARIMAX | ≈ climatología corregida; pierde vs GBM | Comparación |
| Stacking / EnsembleCI denso | no transfiere entre estaciones | Rechazado |
| CCG-MoE (régimen para gas pooled) | sobreajuste estacional (12,18 dev → 13,35 primavera) | Rechazado |
| Forecast solar RTE | mejora física mínima, empeora ranking | Rechazado |
| Reconciliación dura de gas total | físico 12,63 → 15,49 % | Rechazado |
| Lag D-2 | mejora dev 0,008 pp, se invierte en primavera | Rechazado |
| Recencia (half-life 730d) | pierde 0,32 pp en mar–abr | Rechazado |
| PCA, multiescala, temperatura, precios vecinos, CatBoost, 24-por-horizonte | pre-fix, sin transferencia | Screening |
| Intercambios programados D-1 (solos) | físico −0,09 (IC cruza 0); Direct −0,18 (IC roza 0) | Marginal, opt-in |

**Modelos que sí se promueven:** `physical_alpha2`+calibración (nivel),
`share_lgbm` (ventanas), y el paquete de features sys/ctx3 sobre el Direct.

---

## 8. Estado de causalidad y honestidad metodológica

- **Causal por timestamp:** ninguna observación tiene hora posterior al origen.
  El código lo hace estricto (tests fijan la regla de hora cerrada y los lags).
- **Causal por vintage (pendiente parcial):** los estados/labels del histórico
  definitivo pudieron corregirse ex-post. Las indisponibilidades RTE y el
  programa de intercambios sí están versionados. Para cerrar la brecha por
  completo se necesita archivar snapshots provisionales diariamente.
- La arquitectura física se motivó tras inspeccionar mar–abr; **agosto es la
  primera revalidación prospectiva con vintage limpio** de todo el pipeline.
- El A71 histórico: la migración de plataforma 2025 re-selló los `UpdateTime`
  previos; se verificó empíricamente que los valores siguen siendo forecasts D-1
  genuinos (MAE 1,7–4,4 GW/año vs realizado) y se aplica una política de confianza
  documentada (`A71_TRUSTED_STAMPS_FROM = 2025-10-01`).

---

## 9. Próximos pasos (por valor esperado)

1. **Regenerar `full_models_all_history` con el builder closed-hour** para
   desbloquear `rank_consensus` (63,5 % de potencial de oráculo, hoy en
   cuarentena por depender de predicciones legacy). Es la mejora más grande al
   alcance sin datos nuevos, y ataca la métrica real (decisión de ventana).
   Ahora barato con `--parallel-origins`.
2. **Archivo diario del snapshot TR + provisional** (cron) para cerrar el vintage
   y asegurar la revalidación de agosto.
3. **ENTSO-E RESTful token** (correo con asunto `Restful API access` a
   transparency@entsoe.eu): habilita queries finas de A71/A69/A65 para la
   operación diaria incremental.
4. **Feature de clean spark spread** (precio − gas TTF − EUA): la economía del
   despacho, única veta con física detrás sin perforar.
5. **Cuantiles para la capa de decisión:** no baja el MAPE pero puede mejorar el
   regret de `share_lgbm` decidiendo bajo incertidumbre explícita.
6. **Revalidación prospectiva de agosto** y walk-forward de 12 meses regenerado
   con el builder limpio (cubre otoño-invierno, nunca visto bajo este protocolo).

---

## 10. Reproducción (comandos vigentes)

Desde la raíz `green-decision-module`, entorno conda `green-observatory`.

**Producción — físico con features de sistema, reentreno diario (paralelo):**

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m green_observatory.carbon.realtime_proxy_live_refit \
  --model physical --pooled-gas \
  --carbon-live data/cache/carbon_fr_realtime_2026_full.parquet \
  --mix-forecast data/cache/mix_day_ahead_fr_hourly_bridged.parquet \
  --price-forecast data/cache/day_ahead_price_fr_hourly_bridged.parquet \
  --thermal-margin --entsoe-a71 data/cache/entsoe_a71_generation_forecast_fr.parquet \
  --eval-start 2026-05-01 --eval-end 2026-07-18 \
  --parallel-origins 5 --model-threads 1 \
  --output-dir runs/daily_refit_2026/realtime_proxy_daily_refit_tr_extended_sys
```

**Referencia académica consolidada (11,83 %):** el código y los artefactos del
track consolidado (`consolidated_physical*`, `consolidated_share*`, checkpoints
closed-hour mar–abr) viven en la rama **`snapshot2007`**, igual que todos los
experimentos rechazados del ledger (§7). La rama `main` conserva solo el stack
de producción: físico + Direct + gate + ventanas.

**Datos que solo se pueden re-obtener con cuenta/OAuth (no regenerables sin ella):**
`carbon_fr_realtime_2026_full.parquet` (TR, se purga en origen),
`rte_exchange_schedule_da.parquet`, `entsoe_a71_generation_forecast_fr.parquet`.

**Runs conservados en `runs/daily_refit_2026/`:** producción física
(`realtime_proxy_daily_refit_tr_extended_sys`), calibración
(`causal_operational_gate_tr_extended_sys`), Direct de referencia
(`live_direct_daily_refit_tr_extended_ctx3`) y el backtest de invierno
(`live_direct_winter_2025_26`).
