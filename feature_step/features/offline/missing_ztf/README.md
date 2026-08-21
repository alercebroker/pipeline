# % de valores faltantes por feature — `multisurvey_ztf.feature`

Run de features del step ZTF: **46,616 objetos**, `version = 1` (`27.5.7a31`),
`sid = 0`, 3,442,717 filas, **215 pares (feature, band)**.

## Cómo se cuenta

En este schema los NaN/inf **no se guardan**: `prepare_ao_features_for_db`
(`features/utils/parsers.py:406`) filtra `value.notna()` antes del upsert.
Verificado en la tabla: `count(*) where value is null` = **0** sobre 3,442,717
filas. Entonces *faltante* = **la fila `(oid, feature_id, band)` no existe**.

> Ojo: es al revés que en `alerce.feature`, donde el NaN sí se persiste como
> `NULL` y el cálculo es `value IS NULL / count(*)` (ver `../nan_distribution/`).

**Denominador = 46,616**, los oids que tienen `Coordinate_x/y/z`
(`feature_id` 124/125/126, band 0). Se emiten para todo objeto que llega a
escribirse sin depender de la fotometría, y de hecho los 46,616 oids distintos
de la tabla tienen las tres. `SPM_mjd_ref` y `Timespan` también dan 0% —
confirman el censo por un camino independiente.

```
missing%(feature, band) = 100 * (46616 - count(distinct oid)) / 46616
```

**Los ~60k objetos procesados no son el denominador.** Un objeto que no pasa
`MIN_DETECTIONS_FEATURES = 2` no escribe *ninguna* fila, así que no aparece acá.
Estos porcentajes son *entre los objetos que sí quedaron en la tabla*.

## Resultado

`missing_per_feature.csv` — las 215 filas, ordenadas de mayor a menor.
Regenerar con `missing_per_feature.py` (ver abajo).

| corte | valor |
|---|---|
| media sobre los 215 pares | **65.65 %** |
| mediana | 77.63 % |
| media sobre las 199 features del modelo BHRF 2.1.0 | **67.10 %** |
| pares con > 50 % faltante | 151 / 215 |
| pares con > 75 % faltante | 123 / 215 |

Por banda:

| band | pares | media missing% |
|---|---|---|
| 0 (sin banda) | 22 | 21.18 |
| 1 (g) | 88 | 71.86 |
| 2 (r) | 88 | 72.22 |
| 12 (g,r) | 17 | 57.03 |

Las features sin banda —colores de catálogo, cross-match, coordenadas— son las
más completas; todo lo que necesita la curva de luz de una banda está sobre 70 %.

## Por familia

Las features caen en grupos de porcentaje idéntico: comparten la misma compuerta
del extractor, así que o salen todas o no sale ninguna.

| missing% | n_obj | features (banda) |
|---|---|---|
| 0.00 | 46,616 | `Coordinate_x/y/z`, `SPM_mjd_ref`, `Timespan` (–) |
| 0.15 | 46,546 | `sgscore1`, `distpsnr1` (–) |
| 3.26–6.79 | 43.4k–45.1k | `ps_i-z`, `ps_r-i`, `ps_g-r` (–), `mean_distnr`, `mean_chinr`, `mean_sharpnr` (gr) |
| 15.12 / 15.94 | 39.6k / 39.2k | SPM (6 c/u): `SPM_A`, `_beta`, `_gamma`, `_t0`, `_tau_fall`, `_tau_rise` (g / r) |
| 19.93 | 37,324 | colores WISE `W1-W2`, `W2-W3`, `W3-W4` (–) |
| 20.13 | 37,230 | `sigma_distnr` (gr) |
| 29.56 / 34.01 | 32.8k / 30.8k | `positive_fraction` (g / r) |
| 30.34 / 31.76 | 32.5k / 31.8k | `TDE_decay`, `TDE_mag0`, `TDE_mjd_ref` (r / g) |
| 38.17 / 38.84 | 28.8k / 28.5k | `n_forced_phot_band_before/after` (r / g) |
| 40.64 / 46.96 | 27.7k / 24.7k | `g-W1..W4` / `r-W1..W4` (–) |
| 47.73 / 48.23 | 24.4k / 24.1k | ulens (5 c/u): `ulens_u0`, `_tE`, `_fs`, `_t0`, `_mag0` (g / r) |
| 48.90 / 49.79 | 23.8k / 23.4k | `max_/median_brightness_after_band` (r / g) |
| 50.23–52.02 | 22.4k–23.2k | `ulens_chi`, `SPM_chi` (g y r) |
| 62.67 / 63.87 | 17.4k / 16.8k | `g-r_mean/max` / `g-r_mean_corr/max_corr` (gr) |
| 66.75 / 68.04 | 15.5k / 14.9k | fleet (5 c/u): `fleet_a`, `_w`, `_m0`, `_t0`, `_mjd_ref` (g / r) |
| 70.93 / 72.21 | 13.6k / 13.0k | `fleet_chi` (g / r) |
| 74.70 | 11,793 | período multibanda: `Multiband_period`, `PPE`, `Power_rate_*` (gr) |
| 77.21 / 78.27 | 10.6k / 10.1k | Harmonics (14 c/u): `Harmonics_mag_1..7`, `_phase_2..7`, `_mse` (g / r) |
| 77.63 / 78.73 | 10.4k / 9.9k | `Period_band`, `delta_period` (g / r) |
| 78.18 / 79.33 | 10.2k / 9.6k | `Harmonics_chi` (g / r) |
| 79.12 | 9,734 | `color_variation` (gr) |
| 79.30 / 81.80 | 9.7k / 8.5k | `TDE_decay_chi` (r / g) |
| 82.29 / 82.87 | 8.3k / 8.0k | `last_/max_/median_brightness_before_band` (r / g) |
| 85.55–86.65 | 6.2k–6.7k | `Psi_CS`, `Psi_eta`, `GP_DRW_sigma`, `GP_DRW_tau` (g y r) |
| **88.18 / 88.30** | 5.5k / 5.5k | **turbo-FATS, 26 c/u**: `Amplitude`, `Mean`, `Std`, `Skew`, `Rcs`, `StetsonK`, `Beyond1Std`, `MedianBRP`, `Q31`, `Pvar`, `ExcessVar`, `IAR_phi`, `LinearTrend`, `SF_ML_amplitude/gamma`, etc. (g / r) |
| 89.89 / 90.38 | 4.7k / 4.5k | `dbrightness_first_det_band`, `dbrightness_forced_phot_band` (g / r) |
| **90.21–90.71** | 4.3k–4.6k | **MHPS, 8 c/u**: `MHPS_ratio`, `_low`, `_high`, `_non_zero`, `_PN_flag`, `_ratio_365_30`, `_low_365`, `_high_30` (g / r) |

## Lo que salta a la vista

- **Las 199 features del modelo BHRF tienen todas al menos una fila.** Ninguna
  quedó 100 % faltante — a diferencia del run `dev1` de `nan_distribution`, donde
  los colores WISE estaban en 100 % por correr sin xmatch. Acá el xmatch anduvo:
  WISE queda en 19.9 % (colores puros) y 40.6/47.0 % (los cruzados con g/r, que
  además necesitan magnitud media de la banda).
- **El grueso del faltante es por pocas detecciones, no por fallas.** Los dos
  bloques más vacíos —turbo-FATS (88 %) y MHPS (90 %)— son justo los que exigen
  una curva de luz con varias épocas en la banda. Solo ~5.5k de 46.6k objetos
  (12 %) llegan a tener estadística por banda, y ~4.3k (9 %) a tener MHPS.
- **g y r están parejas** (71.9 % vs 72.2 % de media): no hay una banda rota.
- **El piso de completitud lo ponen las features de catálogo**: coordenadas,
  `sgscore1`/`distpsnr1` (99.85 %) y los colores Pan-STARRS (93–97 %), que salen
  del cross-match y no dependen de cuántas veces se vio el objeto.

## Regenerar

```bash
cd feature_step
/Users/panchoandrades/Library/Caches/pypoetry/virtualenvs/feature-step-gEc3Lb8--py3.10/bin/python \
  features/offline/missing_ztf/missing_per_feature.py
```

(`poetry run` se rompe por el `.python-version` huérfano en `Desktop/repos/`;
por eso el intérprete directo.) Toma credenciales de
`features/offline/credentials.json` y escribe `missing_per_feature.csv`.
La consulta es un solo `GROUP BY` sobre 3.4M filas — segundos, no minutos.
