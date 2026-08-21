# Cuántas filas de `feature` va a insertar el run completo

Medido sobre lo que ya está en `multisurvey_ztf.feature` (`sid = 0`, `version = 1`),
extrapolado al catálogo.

## La cuenta directa

```
filas           3,442,717
oids               46,616
filas / oid         73.85
x 27,000,000 -> 1,994,022,632   (1.99e9)
```

## Los 27M son los objetos con `n_det >= 2`, no el catálogo

`multisurvey_ztf.object` (sid=0) tiene **130,451,381** objetos. El **79.87 %**
(104,189,227) tiene `n_det < 2`: no pasan `MIN_DETECTIONS_FEATURES = 2` y no
escriben **ninguna** fila. Quedan **26,262,154** — que es de dónde sale el
"26.3M" del `SERVER_RUNBOOK.md`.

| `n_det` | objetos del catálogo | % |
|---|---|---|
| < 2 | 104,189,227 | 79.87 |
| 2–4 | 17,009,649 | 13.04 |
| 5–9 | 5,003,537 | 3.84 |
| 10–19 | 1,712,242 | 1.31 |
| 20–49 | 1,022,666 | 0.78 |
| 50–99 | 521,354 | 0.40 |
| 100+ | 992,706 | 0.76 |

## Corregido por composición

Las filas por objeto dependen fuerte de `n_det` (mismo efecto que en `BY_NDET.md`),
y nuestra muestra no tiene la misma mezcla que el catálogo: sobre-representa los
objetos largos.

| bin | % muestra | % catálogo (≥2) | filas/oid | filas proyectadas |
|---|---|---|---|---|
| 2–4 | 57.95 | 64.77 | 48.86 | 831,091,450 |
| 5–9 | 21.23 | 19.05 | 73.45 | 367,509,793 |
| 10–19 | 7.61 | 6.52 | 101.40 | 173,621,339 |
| 20–49 | 4.96 | 3.89 | 140.31 | 143,490,266 |
| 50–99 | 2.81 | 1.99 | 172.08 | 89,714,596 |
| 100+ | 5.44 | 3.78 | 191.64 | 190,242,178 |

| estimación | filas |
|---|---|
| plano, 73.85 × 27.0M | 1.99e9 |
| plano, 73.85 × 26.26M | 1.94e9 |
| **ponderado por la `n_det` del catálogo** | **1.80e9** (68.37 filas/oid) |

La corrección es chica (−10 %) porque la muestra ya está dominada por objetos
cortos. **1.8–2.0e9 filas** es el rango razonable.

## El runbook está alto por más del doble

`SERVER_RUNBOOK.md:356` proyecta `feature: 4.14e9 rows` sobre 26.3M oids — eso es
**157 filas/oid**, y el texto de la §8 dice "~193 feature rows per object". Contra
la tabla de arriba, 157–193 es el rango de objetos de **50+ detecciones**, o sea
el 0.4–0.8 % del catálogo. Esa proyección salió de un probe cuyos oids no tienen
la composición del catálogo. Los números medidos acá la corrigen a la mitad.

Consecuencia práctica: el sizing de disco del runbook (~68 GB de parquet) también
está al doble; con `--no-shards` no aplica.

## Advertencias

- `object.n_det` es el snapshot de hoy, no el que vio el step (ver `BY_NDET.md`).
  Los objetos que sumaron detecciones desde entonces caen en bins más bajos, lo
  que empuja la estimación **hacia abajo**.
- Los 46,616 son objetos del stream reciente, no una muestra aleatoria del
  catálogo. La reponderación arregla la composición por `n_det`, pero no un
  eventual sesgo *dentro* de cada bin (p. ej. cobertura de xmatch).
- Es solo `feature`. `probability` es aparte: 45 filas/oid según el runbook.
