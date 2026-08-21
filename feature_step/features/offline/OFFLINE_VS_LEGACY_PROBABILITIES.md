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
estaba bien observado. El sesgo está medido en el hallazgo 5 y es severo: la
mediana de `n_det` es 67 entre los comparables y 3 entre el resto.

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

## Hallazgo 5: apertura por `n_det`

**El cohorte comparable no se parece a la población de la corrida.**

| `n_det` | clasificados | con legacy | cobertura |
|---|---:|---:|---:|
| 2-5 | 30.477 | 13 | **0,0%** |
| 6-10 | 7.203 | 252 | 3,5% |
| 11-20 | 2.953 | 878 | 29,7% |
| 21-50 | 2.182 | 1.417 | 64,9% |
| 51-100 | 1.276 | 1.070 | 83,9% |
| 101-300 | 1.512 | 1.423 | 94,1% |
| >300 | 1.013 | 994 | 98,1% |

Mediana de `n_det`: **67** entre los comparables, **3** entre el resto. Legacy
clasificó el 98% de los objetos bien observados y 13 de los 30.477 que tienen
entre 2 y 5 detecciones — y esos 30.477 son el 65% de la corrida.

**La coincidencia sube con `n_det`:**

| `n_det` | comparados | coinciden | tasa |
|---|---:|---:|---:|
| 2-5 | 13 | 8 | 61,5% |
| 6-10 | 252 | 200 | 79,4% |
| 11-20 | 878 | 715 | 81,4% |
| 21-50 | 1.417 | 1.116 | 78,8% |
| 51-100 | 1.070 | 896 | 83,7% |
| 101-300 | 1.423 | 1.272 | 89,4% |
| >300 | 994 | 957 | **96,3%** |

Reponderando las tasas por bin a la población real de la corrida, el 85,4% del
hallazgo 2 cae a algo **entre 69% y 80%**: el extremo bajo asume que los objetos
de 2-5 detecciones se comportan como su propia muestra de 13, el alto que se
comportan como el bin de 6-10. La conclusión útil no es el número sino que para
el 65% de la corrida no hay con qué medirlo.

**El margen angosto sobrevive al control por `n_det` — y se fortalece:**

| `n_det` | margen p50 legacy donde coinciden | donde difieren | nuestra clase era su rank 2 |
|---|---:|---:|---:|
| 6-10 | 0,066 | 0,028 | 55,8% |
| 11-20 | 0,077 | 0,037 | 58,3% |
| 21-50 | 0,086 | 0,032 | 61,5% |
| 51-100 | 0,104 | 0,027 | 61,5% |
| 101-300 | 0,162 | 0,031 | 68,9% |
| >300 | **0,373** | **0,027** | 73,0% |

El margen de los desacuerdos es plano en ~0,03 en todos los bins. Lo que crece
con `n_det` es la confianza donde los dos lados coinciden: de 0,066 a 0,373. En
objetos con más de 300 detecciones, los aciertos salen con 13,8 veces más margen
que los desacuerdos.

Esto descarta la explicación simple. Si el desacuerdo fuera solo falta de
información, su margen debería subir con `n_det` igual que el de los aciertos. No
sube. Hay un subconjunto de objetos que el modelo no resuelve por más
detecciones que reciba, y ahí es donde las dos versiones se cruzan. (Contrasta
con el análisis de NaN del estudio anterior, donde el efecto sí se disolvía al
controlar por `n_det`.)

**El modo de falla CV/Nova → periódica sí es de baja información:**

| `n_det` | casos | % del bin |
|---|---:|---:|
| 2-5 | 2 | 15,4% |
| 6-10 | 26 | 10,3% |
| 11-20 | 92 | 10,5% |
| 21-50 | 181 | 12,8% |
| 51-100 | 95 | 8,9% |
| 101-300 | 69 | 4,8% |
| >300 | 14 | **1,4%** |

Se desvanece cuando hay datos: del 15,4% del bin más pobre al 1,4% del más rico.

## Qué concluir, y qué no

**Sí se puede concluir** que el 14,6% de desacuerdo del clasificador plano
sobrestima el desacuerdo sustantivo. Prácticamente todo ocurre en objetos donde
ambos modelos están indecisos entre dos clases casi empatadas, y en la mayoría de
los casos cada lado tenía a la clase del otro como segunda opción. No son
contradicciones; son la misma distribución de probabilidad con el orden de los
dos primeros invertido.

**No se puede concluir** que las dos versiones sean equivalentes:

- El cohorte comparable es el 13% mejor observado (hallazgo 5). Reponderada a
  la población real, la coincidencia del clasificador plano está entre 69% y
  80%, y para el 65% de la corrida no hay con qué medirla.
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
