# Probabilidades offline contra las almacenadas en legacy

**Fecha:** 2026-08-21 · **Cohorte:** 46.616 objetos · **Rama:** `fix/ztf-feature-parser-extra-fields`

Reemplaza al estudio de 100 objetos de `OFFLINE_VS_LEGACY_VALIDATION.md` §5.
Aquel comparaba recalculando el pipeline por objeto, y por eso no podía crecer.
Este compara dos tablas ya escritas, así que cuesta lo mismo comparar 46.616 que
comparar uno.

## De dónde salen los datos

**Lado nuestro:** `multisurvey_ztf.probability`, escrito por el probe de 126
workers del 2026-08-21 (`--load-db`, 63.000 oids aleatorios del corte
`n_det >= 2`, semilla 20260822) más los 199 del smoke inicial. Las 126 unidades
cerraron con `db_prob_rows == prob_rows`, así que la base tiene exactamente lo
que la corrida calculó.

**Lado legacy:** `alerce.probability`, `classifier_version = '2.1.0'`, los cinco
clasificadores `lc_classifier_BHRF_forced_phot*`.

Las tablas no comparten vocabulario: la nuestra va por oid bigint y
`classifier_id`/`class_id` smallint, legacy por oid string y nombres de texto.
Las dos traducciones ocurren en el script — `class_id` contra
`multisurvey_ztf.taxonomy`, que es la autoridad y no el fixture, y el oid por
`idmapper`.

**El rank 1 se deriva de las probabilidades, no de la columna `ranking`.** Las
dos tablas las escribieron códigos distintos con años de diferencia; una
convención que difiera entre ellas decidiría la comparación en silencio. Los
empates se rompen por `class_name` para que el mismo empate caiga igual de los
dos lados.

## Hallazgo 1: legacy solo cubre el 13% de la población

```
clasificados por nosotros : 46.616
con probabilidad legacy   :  6.047   (13,0%)
solo legacy               :      0
```

40.569 objetos del corte nunca fueron clasificados por ALeRCE, y **no hay un solo
objeto que legacy tenga y nosotros no**. La corrida no está reproduciendo un
catálogo existente: lo está llenando. La comparación solo es posible sobre un
séptimo de la población.

Esto condiciona todo lo que sigue. Los 6.047 no son una muestra aleatoria de los
46.616 — son los que legacy alcanzó a clasificar, y legacy clasificó lo que
estaba bien observado. El estudio anterior ya había mostrado que la coincidencia
depende fuerte de `n_det` (66,7% entre 1-20 detecciones, 100% arriba de 300), así
que las tasas de abajo son probablemente **optimistas** respecto de lo que
pasaría si legacy hubiera clasificado todo. Queda por medir.

## Hallazgo 2: la coincidencia de clase

| cabeza | comparados | coinciden | tasa |
|---|---:|---:|---:|
| flat | 6.047 | 5.164 | **85,4%** |
| top | 6.047 | 5.643 | 93,3% |
| transient | 6.047 | 4.755 | 78,6% |
| stochastic | 6.047 | 5.555 | 91,9% |
| periodic | 6.047 | 5.349 | 88,5% |

## Hallazgo 3: el desacuerdo tiene una estructura, no es ruido

De los 883 desacuerdos del clasificador plano, **439 — la mitad exacta — son
legacy diciendo CV/Nova contra una clase periódica nuestra**:

```
 118  legacy CV/Nova         -> RSCVn
 109  legacy CV/Nova         -> YSO
  98  legacy CV/Nova         -> Periodic-Other
  62  legacy CV/Nova         -> EA
  26  legacy CV/Nova         -> LPV
  15  legacy CV/Nova         -> DSCT
  11  legacy CV/Nova         -> RRLab
```

Es el mismo patrón que habíamos encontrado a mano sobre 13 objetos, ahora con 60×
más datos. Es consistente con la asimetría de features ya documentada: legacy
tiene NaN donde nosotros tenemos valor —PS1 entre ellos— y sin la información
estrella-vs-galaxia el modelo cae a CV/Nova.

## Hallazgo 4: casi todos los desacuerdos son empates, no contradicciones

Esta es la parte que cambia la lectura de los tres hallazgos anteriores. Un
cambio de rank 1 entre dos clases separadas por 0,002 no es el mismo evento que
uno entre 0,9 y 0,05, y contar ambos como "desacuerdo" esconde cuál ocurrió.

**Margen entre la primera y la segunda clase:**

| | legacy p25 | p50 | p75 | nuestro p25 | p50 | p75 |
|---|---:|---:|---:|---:|---:|---:|
| coinciden | 0,053 | 0,122 | 0,297 | 0,054 | 0,132 | 0,346 |
| **difieren** | 0,011 | **0,032** | 0,064 | 0,011 | **0,031** | 0,071 |

Donde los dos lados coinciden, el ganador saca ~0,12 de ventaja. Donde difieren,
saca 0,03 — **cuatro veces menos**. Los desacuerdos se concentran casi
exclusivamente en objetos que ninguno de los dos lados podía decidir.

**Dónde estaba nuestra clase en el ranking de legacy** (883 desacuerdos):

```
rank 2 de legacy       549   62,2%
rank 3 de legacy       190   21,5%
rank 4 de legacy        50    5,7%
rank 5 o peor           94   10,6%
nunca la puntuo          0    0,0%
```

**Cuántos desacuerdos son dudosos:**

| umbral | legacy dudaba | nosotros dudábamos | alguno de los dos |
|---|---:|---:|---:|
| margen < 0,05 | 66,1% | 65,1% | 86,7% |
| margen < 0,10 | 88,4% | 83,7% | 96,8% |
| margen < 0,20 | **98,9%** | 94,0% | 99,8% |
| margen < 0,30 | 99,9% | 96,1% | 100,0% |

Y el criterio combinado —nuestra clase era la segunda de legacy **y** legacy
dudaba con margen < 0,20— cubre **544 de 883, el 61,6%**.

**Los CV/Nova → periódica, que son la mitad del desacuerdo, siguen el mismo
patrón:** margen legacy p50 = 0,046, nuestra clase era su rank 2 en el 61,0% de
los casos, y la probabilidad que legacy le daba a nuestra clase tenía mediana
0,118 contra un ganador que apenas superaba eso.

## Qué concluir, y qué no

**Sí se puede concluir** que el 14,6% de desacuerdo del clasificador plano
sobrestima el desacuerdo sustantivo. Prácticamente todo ocurre en objetos donde
ambos modelos están indecisos entre dos clases casi empatadas, y en la mayoría de
los casos cada lado tenía a la clase del otro como segunda opción. No son
contradicciones; son la misma distribución de probabilidad con el orden de los
dos primeros invertido.

**No se puede concluir** que las dos versiones sean equivalentes:

- El cohorte comparable es el 13% mejor observado. Falta medir el sesgo por
  `n_det`.
- `classifier_version = '2.1.0'` en ambos lados no garantiza la misma revisión de
  código. Parte del desacuerdo puede ser diferencia de versión, no de pipeline.
- Que el desacuerdo sea de bajo margen no lo vuelve inofensivo para un usuario
  que lee solo la clase rank 1 sin mirar la probabilidad.

**Queda abierto** si el modelo fue entrenado con los huecos de NaN presentes. Si
lo fue, alimentarlo con vectores más completos lo saca de la distribución en la
que aprendió — y nuestros vectores son sistemáticamente más completos.

## Reproducir

```bash
poetry run python scripts/offline_compare_stored_vs_legacy.py \
    --oid-file <oids de la corrida> --json-out agreement.parquet
```

Los oids de una corrida se sacan de `multisurvey_ztf.feature` (`SELECT DISTINCT
oid`, 0,6 s sobre 322 MB) mientras esa tabla solo contenga escrituras nuestras.
Buscarlos en `multisurvey_ztf.probability` costaría un scan de 278 GB: esa tabla
tiene 1.417 millones de filas y ningún índice por `classifier_id` ni por
`classifier_version`.

El análisis de margen usa `borderline_report()` de
`features/offline/probability_compare.py`, que necesita las distribuciones
completas —no solo el rank 1— porque el margen no se puede recuperar del ganador.
