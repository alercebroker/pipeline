# % de faltantes por feature, abierto por `n_det`

Misma cuenta que `README.md` (faltante = fila ausente en `multisurvey_ztf.feature`),
con los 46,616 objetos partidos por número de detecciones.

Dos ejes, porque no responden lo mismo:

- **`--axis object`** → `object.n_det`, total de todas las bandas.
  CSV: `missing_by_ndet_object.csv` (215 pares × 6 bins).
- **`--axis band`** → `magstat.n_det` de la banda de la feature, solo band 1/2.
  Es la compuerta que de verdad ven turbo-FATS y MHPS.
  CSV: `missing_by_ndet_band.csv` (176 pares × 6 bins).

```bash
<venv>/bin/python features/offline/missing_ztf/missing_by_ndet.py --axis object
<venv>/bin/python features/offline/missing_ztf/missing_by_ndet.py --axis band
```

## Titular

El 65.65 % global es un promedio sobre una población dominada por objetos cortos:
**27,013 de 46,616 (58 %) tienen 2–4 detecciones**, mediana 4. Abierto por `n_det`
el faltante colapsa.

| bin `n_det` | objetos | media missing% (215 pares) | pares > 75 % |
|---|---|---|---|
| 2–4 | 27,013 | **77.27** | 147 / 215 |
| 5–9 | 9,896 | 65.84 | 123 / 215 |
| 10–19 | 3,546 | 52.84 | 80 / 215 |
| 20–49 | 2,314 | 34.74 | 10 / 215 |
| 50–99 | 1,309 | 19.96 | 10 / 215 |
| 100+ | 2,538 | **10.87** | 10 / 215 |

Con 100+ detecciones solo 10 de 215 pares siguen sobre 75 % faltante — y son los
diez de la sección "al revés", que no son un problema de estadística.

## Familias representativas (eje `object.n_det`)

| familia (band g) | 2-4 | 5-9 | 10-19 | 20-49 | 50-99 | 100+ |
|---|---|---|---|---|---|---|
| MHPS (8) | 99.97 | 98.88 | 91.14 | 65.86 | 37.59 | **9.81** |
| turbo-FATS (26) | 100.00 | 98.83 | 82.60 | 41.27 | 19.56 | **6.90** |
| GP_DRW (2) | 100.00 | 96.34 | 73.10 | 35.57 | 15.74 | 5.79 |
| Psi_CS / Psi_eta | 99.61 | 92.60 | 71.26 | 39.50 | 16.20 | 6.11 |
| Harmonics (14) | 94.82 | 76.77 | 53.67 | 28.22 | 11.08 | 3.15 |
| Period_band, delta_period | 94.91 | 77.47 | 54.88 | 29.43 | 12.76 | 3.62 |
| período multibanda / Power_rate (gr) | 93.68 | 73.09 | 47.86 | 21.31 | 5.50 | **0.83** |
| fleet (5) | 79.42 | 62.44 | 50.08 | 39.46 | 25.13 | 18.36 |
| colores g-r (gr) | 79.90 | 52.64 | 38.64 | 27.74 | 16.27 | 7.72 |
| ulens (5) | 66.09 | 33.66 | 19.40 | 10.11 | 6.34 | 2.44 |
| SPM (6) | 22.09 | 7.95 | 4.88 | 3.07 | 2.29 | 0.83 |
| colores WISE puros (–) | 24.95 | 10.04 | 14.58 | 17.93 | 18.03 | 15.37 |
| coordenadas, `Timespan`, `SPM_mjd_ref` | 0 | 0 | 0 | 0 | 0 | 0 |

Banda r se comporta igual (±3 pt en todos los bins); el detalle está en el CSV.

## Las dos compuertas duras

**turbo-FATS: ≥ 6 detecciones en la banda.** En el eje por banda da **100.00 %
exacto** para `n_det ≤ 4`, y el mínimo `magstat.n_det` observado con la feature
presente es **6**. Está en el código: `turbofats/FeatureSpace.py:38` devuelve NaN
para todas las 26 features si `len(observations) <= 5` (tras `drop_duplicates("mjd")`
y `brightness.notna()`). Dos caminos independientes, mismo número.

| turbo-FATS, eje `magstat.n_det` | 0 | 1-4 | 5-9 | 10-19 | 20-49 | 50+ |
|---|---|---|---|---|---|---|
| band g | 100.00 | 100.00 | 84.71 | 37.30 | 13.05 | **1.84** |
| band r | 100.00 | 100.00 | 89.46 | 43.06 | 12.22 | **1.75** |

**MHPS: no es umbral de conteo, pero es el que más tarda en llenarse.**
`MHPSExtractor` solo saltea con `len(band_detections) == 0`, así que el NaN viene
de adentro de `mhps.statistics`. Sigue en **91.9 %** con 5–9 detecciones en la
banda y no baja de 5 % hasta 50+.

| MHPS, eje `magstat.n_det` | 0 | 1-4 | 5-9 | 10-19 | 20-49 | 50+ |
|---|---|---|---|---|---|---|
| band g | 100.00 | 99.86 | 91.95 | 68.36 | 32.86 | **4.03** |
| band r | 100.00 | 99.81 | 91.54 | 67.57 | 34.03 | **3.49** |

**No se explica por el baseline temporal.** Abriendo por `object.deltamjd`:
< 10 d → 99.4 % faltante, > 365 d → 88.8 %, y hay MHPS presente con `deltamjd`
de 0.0014 d. La ventana temporal no es la compuerta; es la cadencia dentro de la
banda. Si hace falta el número exacto, hay que mirar `mhps.statistics`.

## Las diez que van al revés

Estas **empeoran** con más detecciones, en los dos ejes:

| feature (g) | 2-4 | 5-9 | 10-19 | 20-49 | 50-99 | 100+ |
|---|---|---|---|---|---|---|
| `last_brightness_before_band` | 77.07 | 87.61 | 90.27 | 93.22 | 96.26 | **99.53** |
| `max_brightness_before_band` | 77.07 | 87.61 | 90.27 | 93.22 | 96.26 | **99.53** |
| `median_brightness_before_band` | 77.07 | 87.61 | 90.27 | 93.22 | 96.26 | **99.53** |
| `dbrightness_first_det_band` | 87.13 | 91.71 | 93.15 | 94.73 | 96.87 | **99.57** |
| `dbrightness_forced_phot_band` | 87.13 | 91.71 | 93.15 | 94.73 | 96.87 | **99.57** |

Son las de fotometría forzada **antes** de la primera detección. Un objeto con
100+ detecciones lleva años en el survey: su primera detección queda fuera de la
ventana de forced photometry disponible, así que no hay "antes" que medir. No es
un bug de cobertura — es la definición de la feature. Son justamente los 10 pares
que quedan sobre 75 % en el bin 100+.

Los colores WISE también suben un poco con `n_det` (10.0 % en 5–9 → 15.4 % en
100+): los objetos de historia larga caen más en zonas donde AllWISE no aporta
contraparte. Efecto chico comparado con lo anterior.

## Advertencias sobre el eje

- **`object.n_det` es un snapshot posterior al cómputo, no el que vio el step.**
  `feature.updated_date` = 2026-08-20/21, pero `object.updated_date` va de
  2026-06-08 a 2026-08-14. El `n_det` de la DB puede quedar **por debajo** del que
  tenía el mensaje de magstats, lo que empuja objetos a bins más bajos. La
  monotonía es tan fuerte que el sesgo no cambia la lectura, pero los números de
  un bin no son exactos al decimal.
- **`magstat` está incompleta respecto de lo que vio el step**: 4,620 de 39,626
  objetos (11.7 %) tienen features en band g sin fila en `magstat` para g. Por eso
  el bin `0` del eje por banda no da 100 % faltante en todas las familias
  (p. ej. Harmonics queda en 94 %, no en 100 %).
