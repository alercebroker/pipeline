# ColorFeatureExtractor

Computes inter-band color indices from light-curve photometry, operating in either flux or magnitude mode depending on the `just_flux` constructor flag.

- **Source:** `lc_classifier/lc_classifier/features/extractors/color_feature_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `lc_classifier.utils` (`flux2mag`, `flux_err_2_mag_err`), `numpy`

## Purpose / Meaning

Color is the difference in brightness between two photometric bands measured simultaneously (or quasi-simultaneously) on the same object. It serves as a proxy for spectral shape and is one of the cheapest discriminators among transient and variable-star classes. This extractor produces one color per adjacent band pair (e.g. `g-r`, `r-i`) from the available detections, using either corrected magnitudes and differential magnitudes (when `just_flux=False`) or differential fluxes (when `just_flux=True`).

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Ordered list of band identifiers, e.g. `["g", "r"]` or `["u", "g", "r", "i", "z", "Y"]`. Colors are computed for each adjacent pair `bands[i]` vs `bands[i+1]`. |
| `just_flux` | `bool` | — (required) | If `True`, operate purely on `diff_flux` rows and emit flux-ratio features. If `False`, operate on both `magnitude` and `diff_flux` rows and emit magnitude-difference features. |

### `AstroObject` fields read

- `detections` — all rows in `astro_object.detections`; columns used: `brightness`, `e_brightness`, `unit`, `fid`, `sid`.
- `features` — read only to concatenate the new rows onto the existing `features` DataFrame.

`forced_photometry` is **not** read by this extractor.

### Pre-filtering applied

**When `just_flux=True`** (`preprocess_detections_just_flux`):
- Drop rows where `brightness` is `NaN`.
- Keep only rows where `unit == "diff_flux"`.

**When `just_flux=False`** (`preprocess_detections_just_magnitude`):
- Drop rows where `brightness` is `NaN`.
- Split into two sub-DataFrames:
  - `corrected_mags`: rows where `unit == "magnitude"` **and** `e_brightness < 1.0`.
  - `diff_magnitudes`: rows where `unit == "diff_flux"`, converted in-place to magnitudes (see *Underlying library / math* below); the converted rows are labeled `unit = "diff_magnitude"` after transformation.

No minimum-length cutoff is enforced. A band with zero surviving detections produces `np.nan` for that band's statistic.

### Valid `unit` values

- `"magnitude"` — used as-is for the `corrected_mags` path (only when `just_flux=False`).
- `"diff_flux"` — used as-is for the `just_flux=True` path; converted to magnitudes for the `diff_magnitudes` path.

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`.

- `fid` is set to `",".join(self.bands)` for **all** features from this extractor (e.g. `"g,r"` for ZTF, `"u,g,r,i,z,Y"` for ELAsTiCC).
- `sid` is set to `",".join(sorted(detections["sid"].unique()))`.
- `version` is `"1.0.0"`.

### Features when `just_flux=True`

For each adjacent band pair `(bands[i], bands[i+1])`:

| Feature `name` | `fid` scope | Meaning |
|---|---|---|
| `{bands[i]}-{bands[i+1]}` | all bands joined | 90th-percentile absolute flux of band `i` divided by (90th-percentile absolute flux of band `i+1` + 1). |

The `+ 1` term in the denominator prevents division-by-zero when the next band has zero or very small flux. The denominator bias is hardcoded (see *Hardcoded values*).

### Features when `just_flux=False`

For each adjacent band pair `(bands[i], bands[i+1])`, four features are emitted — two from the `diff_magnitudes` path and two from the `corrected_mags` path:

| Feature `name` | `fid` scope | Meaning |
|---|---|---|
| `{bands[i]}-{bands[i+1]}_mean` | all bands joined | Mean magnitude of band `i` minus mean magnitude of band `i+1`, computed on the `diff_magnitude` sub-set. |
| `{bands[i]}-{bands[i+1]}_max` | all bands joined | Minimum magnitude (= maximum brightness) of band `i` minus minimum magnitude of band `i+1`, computed on the `diff_magnitude` sub-set. |
| `{bands[i]}-{bands[i+1]}_mean_corr` | all bands joined | Same as `_mean` but computed on the `corrected_mags` sub-set. |
| `{bands[i]}-{bands[i+1]}_max_corr` | all bands joined | Same as `_max` but computed on the `corrected_mags` sub-set. |

**Sentinel:** if any band in a pair has zero detections after filtering, the corresponding statistic (mean or max) is `np.nan`, which propagates into the feature value as `np.nan` via normal arithmetic. No explicit `np.nan` guard is added; the arithmetic (`np.nan - float` or `float - np.nan`) produces `np.nan` naturally.

No short-circuit path exists; the extractor always appends rows even when all values are `np.nan`.

## Underlying library / math

### `flux2mag` — `lc_classifier.utils`

```python
def flux2mag(flux):
    """flux in uJy to AB magnitude"""
    return -2.5 * np.log10(flux) + 23.9
```

Applied to the absolute value of `diff_flux` brightness (`np.abs(diff_fluxes["brightness"])`) before computing magnitude statistics. The zero-point `23.9` corresponds to the standard AB system calibrated to microjansky (`m_AB = -2.5 log10(f_uJy) + 23.9`). This value is hardcoded in `utils.py`.

### `flux_err_2_mag_err` — `lc_classifier.utils`

```python
def flux_err_2_mag_err(flux_err, flux):
    return (2.5 * flux_err) / (np.log(10.0) * flux)
```

Standard Gaussian error propagation for the logarithmic magnitude conversion. Applied to `e_brightness` before the magnitude conversion. Note: this is evaluated on the **absolute-valued** flux (`np.abs(diff_fluxes["brightness"])`) only for the magnitude conversion, but the `flux` argument passed inside `preprocess_detections_just_magnitude` is the already-abs-valued array (after `diff_fluxes["brightness"] = np.abs(...)`), so the propagation is consistent.

### `numpy.percentile` (90th percentile, flux path)

`np.percentile(band_flux_abs, 90)` — the 90th percentile of the absolute flux per band. No interpolation method is specified, so NumPy's default (`linear`) is used.

### `numpy.mean` / `numpy.min` (magnitude path)

- `np.mean(band_detections["brightness"])` — arithmetic mean of magnitude values.
- `np.min(band_detections["brightness"])` — minimum value in the magnitude column, which corresponds to the **maximum brightness** (since lower magnitude = brighter). The code comment confirms this interpretation: `# max brightness, min magnitude`.

## Hardcoded values

| Value | Location | Tunable? | Effect |
|---|---|---|---|
| `90` (percentile) | `_diff_flux_colors`, `np.percentile(..., 90)` | Baked in | Controls the flux-color statistic; p90 of absolute flux per band. |
| `1` (denominator bias) | `_diff_flux_colors`, `band_p90_list[i+1] + 1` | Baked in | Prevents division-by-zero in the flux-ratio color; adds 1 uJy to the denominator regardless of the actual flux scale. This is a numerical guard, not a physically motivated additive constant. |
| `1.0` (error threshold) | `preprocess_detections_just_magnitude`, `e_brightness < 1.0` | Baked in | Rejects detections with photometric uncertainty >= 1 magnitude from the `corrected_mags` path. No equivalent cutoff on the `diff_flux` path. |
| `"diff_magnitude"` (unit label) | `preprocess_detections_just_magnitude` | N/A | Internal label assigned after flux-to-magnitude conversion; not propagated to the features table. |

## Important considerations

- **`just_flux=True` vs `just_flux=False` produce incompatible feature schemas.** `just_flux=True` emits N−1 features (one flux ratio per adjacent pair); `just_flux=False` emits 4(N−1) features (mean/max × diff/corr). Downstream models must be trained with the same flag.
- **`fid` column is multi-band joined string, not a single band identifier.** All features from this extractor share the same composite `fid` value (e.g. `"g,r"`). Consumers that filter by a single band `fid` will not find these rows.
- **No forced photometry.** Unlike several other extractors in the same composite, `ColorFeatureExtractor` reads only `astro_object.detections`, ignoring `forced_photometry`.
- **Absolute value taken before log in the flux path.** `diff_flux` can be negative (negative difference-image flux). The extractor takes `np.abs` before the 90th-percentile and before `flux2mag`. This means physically negative flux events (source fainter than reference) contribute positively to the color statistic and are converted to a positive magnitude-like quantity.
- **`e_brightness < 1.0` filter applies only to `corrected_mags`.** Differential-flux detections converted to `diff_magnitude` are not subject to any error cut. If forced photometry data with poor uncertainties were merged into `detections` upstream, they would pass through the diff-magnitude path unchecked.
- **Empty band produces `np.nan` via propagation, not an explicit sentinel.** `np.nan` is appended to `band_means`/`band_maxima`/`band_p90_list` when a band has no detections; subsequent arithmetic (`np.nan - x` or `x / (np.nan + 1)`) propagates `np.nan` naturally into the feature value. No exception is raised and the row is always written.
- **In-place modification of intermediate DataFrame.** Inside `preprocess_detections_just_magnitude`, `diff_fluxes` is a `.copy()` of the filtered detections, so the original `astro_object.detections` is not mutated. The `just_flux` path does not copy; however, the filtered result is not written back to `astro_object`, so the original is also safe.
- **`astro_object.features` is mutated in place** via `pd.concat` + reassignment at the end of `compute_features_single_object`.

## Cross-references

- **Composites that include this extractor:**
  - `ZTFFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/ztf.py`) — `bands=["g","r"]`, `just_flux=False`.
  - `LSSTFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/lsst.py`) — `bands=["u","g","r","i","z","y"]`, `just_flux=False`.
  - `ElasticcFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/elasticc.py`) — `bands=["u","g","r","i","z","Y"]`, `just_flux=True`.
- **Other extractors reading the same `detections` fields:** `TurboFatsExtractor`, `MHPSExtractor`, `HarmonicsExtractor`, `PeriodExtractor`, `TDETailExtractor`, and others — all read `brightness`, `e_brightness`, `fid`, `unit` from `astro_object.detections`.
- **Consumers of emitted feature names:** No other extractor in the repo reads the feature names emitted here (e.g. `g-r_mean`, `g-r_max_corr`). These names are consumed by downstream classifier models (not in this repo) that ingest the full `astro_object.features` table.
