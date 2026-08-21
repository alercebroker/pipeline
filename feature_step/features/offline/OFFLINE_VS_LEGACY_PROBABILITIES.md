# Probabilidades offline contra las almacenadas en legacy

**2026-08-21** · 46.616 objetos · rama `fix/ztf-feature-parser-extra-fields`

Reemplaza el estudio de 100 objetos de `OFFLINE_VS_LEGACY_VALIDATION.md` §5, que
recalculaba el pipeline por objeto y por eso no podía crecer. Este compara dos
tablas ya escritas.

## Datos

- **Nuestro lado:** `multisurvey_ztf.probability`, del probe de 126 workers
  (63.000 oids aleatorios de `n_det >= 2`, semilla 20260822) más 199 del smoke.
  Las 126 unidades cerraron con `db_prob_rows == prob_rows`.
- **Legacy:** `alerce.probability`, `classifier_version = '2.1.0'`, los cinco
  `lc_classifier_BHRF_forced_phot*`.

Traducciones: `class_id` contra `multisurvey_ztf.taxonomy` (la autoridad, no el
fixture), oid por `idmapper`. El rank 1 se deriva de las probabilidades y no de
la columna `ranking` — la escribieron códigos distintos con años de diferencia.
Empates rotos por `class_name` en ambos lados.

## 1. Legacy cubre el 13% de la población

```
clasificados por nosotros : 46.616
con probabilidad legacy   :  6.047   (13,0%)
solo legacy               :      0
```

No hay un solo objeto que legacy tenga y nosotros no. La corrida no reproduce un
catálogo: lo llena.

## 2. Coincidencia de clase, sobre esos 6.047

| cabeza | coinciden | tasa |
|---|---:|---:|
| flat | 5.164 | **85,4%** |
| top | 5.643 | 93,3% |
| transient | 4.755 | 78,6% |
| stochastic | 5.555 | 91,9% |
| periodic | 5.349 | 88,5% |

Ese 85,4% está medido sobre el 13% mejor observado. Ver §5.

## 3. El desacuerdo tiene estructura

De los 883 desacuerdos del clasificador plano, **439 (la mitad) son legacy
diciendo CV/Nova contra una clase periódica nuestra**: RSCVn 118, YSO 109,
Periodic-Other 98, EA 62, LPV 26, DSCT 15, RRLab 11.

Mismo patrón que sobre 13 objetos en el estudio anterior, con 60× más datos.
Consistente con la asimetría de features documentada: legacy tiene NaN donde
nosotros tenemos valor —PS1 incluido— y sin la información estrella-vs-galaxia
el modelo cae a CV/Nova.

## 4. Casi todos los desacuerdos son empates

Margen entre la primera y la segunda clase:

| | legacy p25 | p50 | p75 | nuestro p50 |
|---|---:|---:|---:|---:|
| coinciden | 0,053 | 0,122 | 0,297 | 0,132 |
| **difieren** | 0,011 | **0,032** | 0,064 | **0,031** |

Dónde estaba nuestra clase en el ranking de legacy: **rank 2 en el 62,2%**, rank
3 en el 21,5%, rank 5 o peor en el 10,6%. Nunca dejó de puntuarla.

Cuántos desacuerdos son dudosos:

| margen | legacy | nosotros | alguno |
|---|---:|---:|---:|
| < 0,05 | 66,1% | 65,1% | 86,7% |
| < 0,10 | 88,4% | 83,7% | 96,8% |
| < 0,20 | **98,9%** | 94,0% | 99,8% |

Criterio combinado —nuestra clase era la segunda de legacy **y** legacy dudaba
con margen < 0,20— cubre 544 de 883, el **61,6%**.

Los CV/Nova → periódica siguen el patrón: margen legacy p50 0,046, rank 2 en el
61,0%, y legacy le daba a nuestra clase una mediana de 0,118.

**No es que legacy dudara y nosotros no.** Nuestro margen es igual de angosto
(0,031 contra 0,032). Ambos lados están indecisos sobre los mismos objetos.

## 5. Apertura por `n_det`

**El cohorte comparable no se parece a la corrida.** Mediana de `n_det`: **67**
entre comparables, **3** entre el resto.

| `n_det` | clasificados | con legacy | cobertura | coincidencia |
|---|---:|---:|---:|---:|
| 2-5 | 30.477 | 13 | **0,0%** | (61,5%) |
| 6-10 | 7.203 | 252 | 3,5% | 79,4% |
| 11-20 | 2.953 | 878 | 29,7% | 81,4% |
| 21-50 | 2.182 | 1.417 | 64,9% | 78,8% |
| 51-100 | 1.276 | 1.070 | 83,9% | 83,7% |
| 101-300 | 1.512 | 1.423 | 94,1% | 89,4% |
| >300 | 1.013 | 994 | 98,1% | **96,3%** |

Legacy clasificó 13 de los 30.477 objetos con 2-5 detecciones, que son el 65% de
la corrida. Reponderado a la población real, el 85,4% de §2 queda **entre 69% y
80%** — el extremo bajo asume que el bin 2-5 se comporta como su muestra de 13,
el alto que se comporta como el bin 6-10. La conclusión útil no es el número:
para el 65% de la corrida no hay con qué medirlo.

**El margen angosto sobrevive al control y se fortalece:**

| `n_det` | margen p50 donde coinciden | donde difieren | rank 2 |
|---|---:|---:|---:|
| 6-10 | 0,066 | 0,028 | 55,8% |
| 11-20 | 0,077 | 0,037 | 58,3% |
| 21-50 | 0,086 | 0,032 | 61,5% |
| 51-100 | 0,104 | 0,027 | 61,5% |
| 101-300 | 0,162 | 0,031 | 68,9% |
| >300 | **0,373** | **0,027** | 73,0% |

El margen de los desacuerdos es plano en ~0,03; el de los aciertos sube de 0,066
a 0,373. Si el desacuerdo fuera falta de información subiría igual que el otro.
No sube: hay objetos que el modelo no resuelve por más detecciones que reciba.
(Contrasta con el análisis de NaN del estudio anterior, donde el efecto sí se
disolvía al controlar por `n_det`.)

**El modo CV/Nova → periódica sí es de baja información:** 15,4% del bin en 2-5
detecciones, 10,5% en 11-20, 4,8% en 101-300, **1,4%** arriba de 300.

## Conclusiones

**Sostenido.** El desacuerdo sustantivo es menor que el 14,6% nominal: ocurre casi
enteramente entre clases casi empatadas, con cada lado teniendo a la clase del
otro como segunda opción. No son contradicciones sino la misma distribución con
los dos primeros lugares invertidos. Y no se explica por falta de detecciones.

**No sostenido.**

- Las dos versiones no son equivalentes: el cohorte comparable es el 13% mejor
  observado, y reponderado la coincidencia cae a 69-80%.
- `classifier_version = '2.1.0'` en ambos lados no garantiza la misma revisión de
  código. Parte del desacuerdo puede ser versión, no pipeline.
- Bajo margen no es inofensivo para quien lee la clase rank 1 sin la
  probabilidad.

**Abierto.**

- El 65% de la corrida (`n_det` 2-5) no tiene validación posible contra legacy.
  Requiere otra vía: consistencia interna, catálogos externos o inspección.
- Si el modelo fue entrenado con los huecos de NaN presentes, alimentarlo con
  vectores más completos lo saca de su distribución. Depende del set de
  entrenamiento, no de la base.

## Reproducir

```bash
poetry run python scripts/offline_compare_stored_vs_legacy.py \
    --oid-file <oids de la corrida> --json-out agreement.parquet
```

Los oids salen de `multisurvey_ztf.feature` (`SELECT DISTINCT oid`, 0,6 s sobre
322 MB) mientras esa tabla solo contenga escrituras nuestras. Buscarlos en
`multisurvey_ztf.probability` costaría un scan de 278 GB: 1.417 millones de filas
y ningún índice por `classifier_id` ni `classifier_version`.

El análisis de margen usa `borderline_report()` de
`features/offline/probability_compare.py`, que necesita las distribuciones
completas: el margen no se recupera del ganador.
