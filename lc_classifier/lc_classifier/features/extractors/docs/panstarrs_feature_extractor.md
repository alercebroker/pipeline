# PanStarrsFeatureExtractor

Extracts static cross-match features from PanSTARRS-1 photometric catalogue data stored in an object's metadata: star-galaxy score, nearest-source distance, and three optical colours.

- **Source:** `lc_classifier/lc_classifier/features/extractors/panstarrs_feature_extractor.py`
- **Version:** `1.0.1`
- **Base class:** `FeatureExtractor` (abstract, from `lc_classifier.features.core.base`)
- **External libs:** none — pure NumPy / pandas arithmetic

## Purpose / Meaning

PanSTARRS-1 provides pre-computed photometry for static sky sources cross-matched against the transient. The star-galaxy score (`sgscore1`) and angular separation (`distpsnr1`) help distinguish between stellar and extended host-galaxy counterparts. The three colours (`ps_g-r`, `ps_r-i`, `ps_i-z`) characterise the spectral energy distribution of the nearest static source, which is informative for host-galaxy typing and for distinguishing nuclear transients from variable stars.

## Input

### Constructor arguments

None. The constructor only populates `self.required_metadata`; there are no user-tunable parameters.

### `AstroObject` fields read

- `metadata` — a `pd.DataFrame` with columns `name` and `value`. The extractor looks up these six rows by `name`:

  | `metadata` row `name` | Meaning |
  |-----------------------|---------|
  | `sgscore1`            | Star-galaxy score for the nearest PanSTARRS source (0 = galaxy, 1 = star) |
  | `distpsnr1`           | Angular separation to the nearest PanSTARRS source (arcsec) |
  | `sgmag1`              | PanSTARRS g-band PSF magnitude of the nearest source |
  | `srmag1`              | PanSTARRS r-band PSF magnitude of the nearest source |
  | `simag1`              | PanSTARRS i-band PSF magnitude of the nearest source |
  | `szmag1`              | PanSTARRS z-band PSF magnitude of the nearest source |

- `detections` — not read for any numerical computation, but its length is checked to decide whether to emit `np.nan` for scores and colours.
- `features` — read and concatenated with the new rows (mutated in-place via `pd.concat`).

### Pre-filtering applied

1. **Metadata completeness check:** if any of the six required metadata names is absent from `metadata["name"]`, all five output features are emitted as `np.nan` immediately.
2. **Validity guards on scores:** if `sgscore1 < 0` or `distpsnr1 < 0`, both are replaced with `np.nan`.
3. **Empty-detection guard (scores):** if `len(astro_object.detections) == 0`, `sgscore1` and `distpsnr1` are set to `np.nan`.
4. **Colour sentinel guard:** each colour is set to `np.nan` if either of its two constituent magnitudes is `< -30.0`.
5. **Empty-detection guard (colours):** if `len(astro_object.detections) == 0`, all three colours are set to `np.nan`.

### Valid `unit` values

This extractor does not read photometric brightness or use a `unit` parameter. It only reads catalogue metadata values.

## Output

All rows are appended to `astro_object.features` (mutated in-place via `pd.concat`). The `fid` column is always `None` (features are band-agnostic). The `sid` column is always `"panstarrs"`.

| Feature `name` | `fid`  | `sid`       | Meaning |
|----------------|--------|-------------|---------|
| `sgscore1`     | `None` | `panstarrs` | Star-galaxy classifier score for the nearest PanSTARRS-1 source; range ~[0, 1] |
| `distpsnr1`    | `None` | `panstarrs` | Angular separation in arcsec to the nearest PanSTARRS-1 source |
| `ps_g-r`       | `None` | `panstarrs` | g − r colour (magnitudes) of the nearest PanSTARRS-1 source |
| `ps_r-i`       | `None` | `panstarrs` | r − i colour (magnitudes) of the nearest PanSTARRS-1 source |
| `ps_i-z`       | `None` | `panstarrs` | i − z colour (magnitudes) of the nearest PanSTARRS-1 source |

**Sentinel:** `np.nan` is emitted for any feature whose input values fail the guards listed above.

**Short-circuit condition:** if the six required metadata fields are not all present, all five features are emitted as `np.nan` and the colour computation is skipped entirely.

## Underlying library / math

No third-party scientific library is invoked. All computation is direct arithmetic on scalar values read from `astro_object.metadata`:

- `ps_g-r = sgmag1 - srmag1`
- `ps_r-i = srmag1 - simag1`
- `ps_i-z = simag1 - szmag1`

NumPy is imported but used only for `np.nan`. pandas is used for DataFrame construction and row filtering.

## Hardcoded values

| Value | Location | Tunable? | Meaning |
|-------|----------|----------|---------|
| `< 0` threshold on `sgscore1` and `distpsnr1` | line 37 | No | Treats negative catalogue values as invalid sentinels; replaces with `np.nan` |
| `< -30.0` threshold on each band magnitude | lines 49–62 | No | Treats magnitudes below −30 as missing/invalid (common ZTF/PanSTARRS sentinel for non-detections); replaces the corresponding colour with `np.nan` |
| `"panstarrs"` as `sid` | line 76 | No | Survey identifier hard-wired into every output row |

## Important considerations

- **No detections needed for features, but emptiness is checked:** the extractor does not read any photometric time-series, yet it gates all outputs on `len(astro_object.detections) == 0`. If there are zero detections, every feature is `np.nan` even if the metadata is fully populated and valid.
- **Metadata format dependency:** the extractor accesses `metadata` as a long-format DataFrame with `name` and `value` columns. A wide-format or differently named metadata table will cause a `KeyError` or return `np.nan` via the completeness check.
- **`[0]` indexing is unsafe:** `metadata[metadata["name"] == "sgscore1"]["value"].values[0]` will raise `IndexError` if a required name appears zero times in `metadata`. The completeness check (`field_intersection`) guards this only if the set of present names is a strict subset; if a name appears but its row has no `value`, a downstream error is still possible.
- **Multiple rows with the same `name`:** if a metadata field appears more than once, `values[0]` silently uses the first occurrence with no warning.
- **Mutation:** `astro_object.features` is replaced in-place via `pd.concat`. If the original `features` DataFrame was referenced externally, the external reference is now stale.
- **`fid = None`:** downstream classifiers that filter features by `fid` must account for `None` (not `np.nan` or an integer band code) in these rows.
- **`lru_cache` import is unused:** `functools.lru_cache` is imported but never applied in this file.

## Cross-references

- **Composite that includes this extractor:** `ZTFFeatureExtractor` in `lc_classifier/lc_classifier/features/composites/ztf.py` — `PanStarrsFeatureExtractor()` is instantiated with no arguments and run as part of the full ZTF feature pipeline.
- **Metadata source:** `lc_classifier/lc_classifier/utils.py` populates the six PanSTARRS metadata fields (`sgscore1`, `distpsnr1`, `sgmag1`, `srmag1`, `simag1`, `szmag1`) from a cross-match result (`xmatch`) when constructing an `AstroObject`.
- **Other extractors reading `metadata`:** `AllwiseColorsFeatureExtractor` and `CoordinateExtractor` also read `astro_object.metadata` for catalogue cross-match or position fields.
- **Downstream consumers of feature names:** `lc_classifier/lc_classifier/utils.py` references `sgscore1` and `distpsnr1` as cross-match input fields, not as downstream consumers of the extracted features. No other file in the repo was found consuming `ps_g-r`, `ps_r-i`, or `ps_i-z` by name — these are expected to feed the final classifier model directly.
