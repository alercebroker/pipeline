# AllwiseColorsFeatureExtractor

Computes magnitude color indices between consecutive AllWISE infrared bands and between each optical survey band and each AllWISE band.

- **Source:** `lc_classifier/lc_classifier/features/extractors/allwise_colors_feature_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `numpy`, `pandas` (stdlib-level; no third-party scientific library)

## Purpose / Meaning

AllWISE infrared colors (W1−W2, W2−W3, W3−W4) and cross-survey colors (optical band − AllWISE band) encode information about the spectral energy distribution of a source. These color indices help distinguish stellar types and extragalactic objects (e.g. AGN, quasars) from transients, which typically have flat or blue infrared SEDs, making them useful discriminators in multi-class light-curve classifiers.

## Input

### Constructor arguments

| Name    | Type        | Default | Meaning |
|---------|-------------|---------|---------|
| `bands` | `List[str]` | —       | Ordered list of optical band identifiers (e.g. `["g", "r"]` for ZTF, `["u","g","r","i","z","y"]` for LSST). Determines which cross-color features are produced. No default; required. |

`self.allwise_bands` is hardcoded to `["W1", "W2", "W3", "W4"]` and is not a constructor argument.

### `AstroObject` fields read

- `detections` — columns used: `fid`, `brightness`, `e_brightness`, `unit`, `sid`. The mean brightness per band is computed from filtered detections.
- `metadata` — a two-column DataFrame with `name` and `value` columns. The extractor looks up rows where `name` is `"W1"`, `"W2"`, `"W3"`, `"W4"` and reads the corresponding `value` as the AllWISE magnitude for that band. If a band is absent from `metadata`, `np.nan` is used.
- `features` — read and extended in-place via `pd.concat`.

`forced_photometry`, `non_detections`, `xmatch`, and `reference` are not read.

### Pre-filtering applied

The `preprocess_detections` method applies these filters to `detections` before any computation:

1. Keep only rows where `unit == "magnitude"` — diff-flux detections are excluded entirely.
2. Drop rows where `brightness` is `NaN`.
3. Drop rows where `e_brightness >= 1.0` — a hardcoded uncertainty ceiling.

The filtered DataFrame may be empty after these steps.

### Valid `unit` values

Only `"magnitude"` passes the filter. Any other unit (e.g. `"diff_flux"`) causes all optical band means to be `np.nan` because no detection survives.

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`. All rows from this extractor have `fid = None`.

Feature names are produced by `_feature_names()` (cached with `@lru_cache(1)`):

**AllWISE consecutive-band colors** (3 features, independent of `bands`):

| Feature `name` | `fid` | Meaning |
|----------------|-------|---------|
| `W1-W2`        | `None` | AllWISE W1 magnitude minus AllWISE W2 magnitude |
| `W2-W3`        | `None` | AllWISE W2 magnitude minus AllWISE W3 magnitude |
| `W3-W4`        | `None` | AllWISE W3 magnitude minus AllWISE W4 magnitude |

**Cross-survey optical-minus-AllWISE colors** (`len(bands) × 4` features):

| Feature `name`    | `fid` | Meaning |
|-------------------|-------|---------|
| `<band>-W1`       | `None` | Mean optical `brightness` in `<band>` minus AllWISE W1 magnitude |
| `<band>-W2`       | `None` | Mean optical `brightness` in `<band>` minus AllWISE W2 magnitude |
| `<band>-W3`       | `None` | Mean optical `brightness` in `<band>` minus AllWISE W3 magnitude |
| `<band>-W4`       | `None` | Mean optical `brightness` in `<band>` minus AllWISE W4 magnitude |

For ZTF (`bands=["g","r"]`) this yields 8 cross-colors: `g-W1`, `g-W2`, `g-W3`, `g-W4`, `r-W1`, `r-W2`, `r-W3`, `r-W4`.  
For LSST (`bands=["u","g","r","i","z","y"]`) this yields 24 cross-colors.

**Total features:** 3 + `len(bands) × 4`.

**`sid` field:** a comma-joined sorted string of all unique survey IDs present in the filtered detections (e.g. `"ZTF"` or `"LSST,ZTF"`). All feature rows for this extractor share the same `sid` value.

**Sentinel value:** `np.nan` is emitted under the following conditions:
- An AllWISE band is absent from `astro_object.metadata`: that band's value is `np.nan`, propagating to all colors that involve it.
- An optical band has no detections surviving `preprocess_detections`: that band's mean is `np.nan`, propagating to all `<band>-W*` colors for that band.
- An AllWISE consecutive-band delta (`W1-W2`, etc.) is `np.nan` when `len(detections) == 0` after preprocessing, regardless of whether AllWISE metadata is available (see *Important considerations*).

## Underlying library / math

No third-party scientific library is invoked. All computation uses:

- `np.mean` — arithmetic mean of filtered `brightness` values per band.
- `np.stack` / `pd.DataFrame` / `pd.concat` — feature assembly.
- `np.sort` / `str.join` — `sid` string construction.

The color is always computed as `a - b` where `a` and `b` are magnitudes (smaller magnitude = brighter). A positive color value means the source is brighter at the shorter wavelength.

## Hardcoded values

| Location | Value | Tunable? | Effect |
|----------|-------|----------|--------|
| `self.allwise_bands` | `["W1", "W2", "W3", "W4"]` | No | Determines which metadata rows are looked up and the set of infrared bands for color computation. |
| `preprocess_detections` | `unit == "magnitude"` | No | Excludes all non-magnitude detections before computing optical means. |
| `preprocess_detections` | `e_brightness < 1.0` | No | Rejects detections with uncertainty ≥ 1 magnitude. |
| `_feature_names` | `@lru_cache(1)` | N/A | Feature name list is computed once per instance and cached. Safe only if `self.bands` and `self.allwise_bands` are never mutated after construction. |

## Important considerations

- **AllWISE delta guard is asymmetric.** The three consecutive AllWISE colors (`W1-W2`, `W2-W3`, `W3-W4`) are set to `np.nan` when `len(detections) == 0` (line `if len(detections) > 0`), but the cross-survey colors (`<band>-W*`) are computed unconditionally — they will be `np.nan` only if the optical band mean or an AllWISE value is `np.nan`. This means that if detections is empty, the AllWISE-only colors are forced to `np.nan` even when all four AllWISE magnitudes are available in metadata. This appears to be an unintended coupling: the AllWISE-only color should not logically depend on whether any optical detections exist.

- **In-place mutation.** `compute_features_single_object` modifies `astro_object.features` in-place by replacing it with a `pd.concat` result. It does not reset the index, so the features DataFrame may have a non-contiguous index after multiple extractors run.

- **`fid` is always `None`.** All features are emitted with `fid=None`, meaning they are not band-specific. Downstream consumers must not filter by `fid` when reading these features.

- **`metadata` structure assumed.** The extractor assumes `astro_object.metadata` is a DataFrame with columns `"name"` and `"value"`. If `metadata` does not contain any W-band rows (e.g. when xmatch data is unavailable), all AllWISE values will be `np.nan` and all 3 + `len(bands)×4` features will be `np.nan`.

- **How AllWISE magnitudes reach `metadata`.** In `lc_classifier/lc_classifier/utils.py` (~line 342), when an xmatch record is present, `W1`, `W2`, `W3`, `W4` are inserted as `["name", "value"]` rows into the metadata DataFrame. If xmatch is `None`, no W-band rows are added and every AllWISE feature will be `np.nan`.

- **`value` dtype.** The `value` column is cast to `np.float64` after constructing the DataFrame from a mixed `np.stack` of name strings and float values. If a value cannot be cast (e.g. an unexpected string in metadata), a `ValueError` will propagate uncaught.

- **No `min_length` guard.** Unlike most other extractors, there is no minimum number of detections required. A single surviving detection in a band is sufficient to compute a mean (equal to that detection's brightness).

- **`_feature_names` cache is instance-level.** `@lru_cache(1)` on a method uses `self` as part of the cache key, so each instance has its own cached list. This is correct behavior but means the names are fixed at first call time.

## Cross-references

- **Composites that include this extractor:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `ZTFFeatureExtractor`, instantiated as `AllwiseColorsFeatureExtractor(["g", "r"])`.
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — `LSSTFeatureExtractor`, instantiated as `AllwiseColorsFeatureExtractor(["u","g","r","i","z","y"])`.

- **Other extractors that read the same `AstroObject` fields:**
  - `detections` with `unit == "magnitude"` is also consumed by `ColorFeatureExtractor`, `FoldedKimExtractor`, `HarmonicsExtractor`, and `TurboFatsExtractor`.
  - `metadata` W-band rows are populated by the xmatch utility in `lc_classifier/lc_classifier/utils.py`.

- **Consumers of the emitted feature names:** No other extractor in the repo reads `W1-W2`, `W2-W3`, `W3-W4`, or `<band>-W*` feature names. These are terminal outputs consumed by downstream classifiers (models), not by other feature extractors.
