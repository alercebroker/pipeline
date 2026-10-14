# SNExtractor

Computes supernova-oriented flux features per photometric band, characterising the brightness level before, at, and after the first detection using forced-photometry baselines.

- **Source:** `lc_classifier/lc_classifier/features/extractors/sn_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor` (abstract, from `lc_classifier.features.core.base`)
- **External libs:** `numpy`, `pandas` (no third-party scientific library)

## Purpose / Meaning

Supernovae and similar transient events exhibit a characteristic rise in differential flux starting from a quiescent pre-explosion baseline. This extractor quantifies the pre- and post-detection forced-photometry context — how many baseline epochs exist, the brightness jump at first detection, and the extreme/central flux values surrounding the event — to give classifiers a direct view of transient onset and evolution.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Photometric band identifiers to iterate over (e.g. `["g","r"]`, `list("ugrizY")`). |
| `unit` | `str` | — (required) | Physical unit of the `brightness` column. Must be `"diff_flux"`; any other value raises `ValueError` at construction time. |
| `use_forced_photo` | `bool` | — (required) | Whether to use `astro_object.forced_photometry`. If `False`, all forced-photometry features are emitted as `np.nan`. |

### `AstroObject` fields read

- `detections` — columns used: `mjd`, `brightness`, `fid`, `unit`, `sid`.
- `forced_photometry` — columns used: `mjd`, `brightness`, `fid`, `unit`. Read only when `use_forced_photo=True` and the field is not `None`.

### Pre-filtering applied

1. `detections` is filtered to rows where `unit == self.unit`.
2. `detections` is sorted ascending by `mjd` before the first-detection timestamp and first brightness are extracted.
3. Per band, `detections_band` is re-sorted by `mjd` before reading `iloc[0]`.
4. `forced_photometry` is filtered to `unit == self.unit`, then to the current band (`fid == band`), then split at `first_detection_mjd` into before/after sub-tables, each sorted by `mjd`.
5. No global `min_length` trim beyond `detections_min_len = 1` for `positive_fraction`.

### Valid `unit` values

Only `"diff_flux"` is accepted; the constructor raises `ValueError` for anything else. Downstream, `brightness` values from `detections` and `forced_photometry` are treated as signed flux differences (positive = brightening above template).

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`.

`sid` is derived from the unique `sid` values present in the (unit-filtered) detections, sorted alphabetically and joined with a comma. `version` is `"1.0.0"`.

For each band in `self.bands`, the following ten features are appended:

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `positive_fraction` | per band | Fraction of in-band detections with `brightness > 0` (i.e. above the difference-image template). |
| `n_forced_phot_band_before` | per band | Number of forced-photometry epochs **before** `first_detection_mjd` in this band. |
| `dbrightness_first_det_band` | per band | `brightness(first_detection_in_band) - last_forced_phot_brightness_before`. Brightness step at onset. |
| `dbrightness_forced_phot_band` | per band | `brightness(first_detection_in_band) - median(forced_phot_before_in_band)`. Brightness jump relative to median baseline. |
| `last_brightness_before_band` | per band | Brightness of the chronologically last forced-photometry epoch before first detection. |
| `max_brightness_before_band` | per band | Maximum brightness among all pre-detection forced-photometry epochs. |
| `median_brightness_before_band` | per band | Median brightness among all pre-detection forced-photometry epochs. |
| `n_forced_phot_band_after` | per band | Number of forced-photometry epochs **after** `first_detection_mjd` in this band. |
| `max_brightness_after_band` | per band | Maximum brightness among all post-detection forced-photometry epochs. |
| `median_brightness_after_band` | per band | Median brightness among all post-detection forced-photometry epochs. |

**Sentinel value:** `np.nan`. The exact conditions that produce `np.nan` for each feature are listed under *Important considerations*.

## Underlying library / math

No third-party scientific library is invoked. All arithmetic is native `numpy`:

- `np.mean(array > 0)` — fraction of positive elements.
- `np.median(array)` — median brightness.
- `np.max(array)` — maximum brightness.
- Subtraction for delta-brightness features.

`pandas` is used only for filtering, sorting, and `DataFrame` construction.

## Hardcoded values

| Literal | Location | Tunable? | Meaning |
|---------|----------|----------|---------|
| `self.detections_min_len = 1` | `__init__` | No | Minimum number of in-band detections required before `positive_fraction` is computed instead of emitting `np.nan`. |
| `valid_units = ["diff_flux"]` | `__init__` | No | The sole permitted unit; raises `ValueError` at construction for any other string. |
| `"1.0.0"` | `self.version` | No | Written to every emitted feature row. |

No windowing, period, frequency, or threshold constants are present.

## Important considerations

**When `np.nan` is emitted:**

- `positive_fraction`: emitted as `np.nan` if the band has fewer than `detections_min_len` (= 1) detections after unit filtering.
- All nine remaining features: emitted as `np.nan` when `use_forced_photo=False` or `forced_photometry is None` or `len(forced_photometry_band) == 0`.
- `n_forced_phot_band_before`, `last_brightness_before_band`, `dbrightness_first_det_band`, `dbrightness_forced_phot_band`, `max_brightness_before_band`, `median_brightness_before_band`: emitted as `np.nan` (except `n_forced_phot_band_before`, which is set to `0`) when no forced-photometry epochs precede `first_detection_mjd` in the band.
- `max_brightness_after_band`, `median_brightness_after_band`: emitted as `np.nan` when `n_forced_phot_band_after == 0`.
- `first_detection_mjd`: set to `np.nan` when `detections` is empty after unit filtering; in that case all temporal splits of forced photometry will yield empty sub-tables (since `mjd < np.nan` and `mjd > np.nan` are always `False`), so most features will be `np.nan`.

**`dbrightness_first_det_band` vs `dbrightness_forced_phot_band`:**
Both measure a brightness rise at onset, but against different references: `dbrightness_first_det_band` uses the immediately preceding forced-photometry point (chronological last before detection), while `dbrightness_forced_phot_band` uses the median of the entire pre-detection baseline. The two will differ when the baseline is not flat.

**`first_detection_mjd` is global across bands:**
The split between "before" and "after" forced photometry is always relative to the first detection across all bands combined, not the first detection within the current band. A band with an earlier first detection than other bands may therefore have no "before" epochs at all.

**Forced photometry is re-filtered inside the band loop:**
The `forced_photometry` variable is re-assigned to the unit-filtered slice on every iteration of the band loop (line 59: `forced_photometry = forced_photometry[forced_photometry["unit"] == self.unit]`). This is safe because pandas slice assignment creates a new object, but it means the filter is applied redundantly on every band after the first.

**`sid` field encoding:**
All unique `sid` values from the unit-filtered `detections` are sorted and joined into a single comma-separated string (e.g. `"LSST,ZTF"`). Every feature row for the object carries this same string.

**In-place mutation:**
`compute_features_single_object` mutates `astro_object.features` in place via `pd.concat`, consistent with the `FeatureExtractor` contract (`"""This method is inplace"""`).

**No exception handling:**
The extractor does not catch any exceptions internally. A `KeyError` (missing column), `TypeError`, or `IndexError` in the input DataFrames will propagate to the caller.

## Cross-references

**Composites that include this extractor:**

- `lc_classifier/lc_classifier/features/composites/ztf.py` — instantiated as `SNExtractor(bands, unit="diff_flux", use_forced_photo=True)` with `bands = ["g", "r"]`.
- `lc_classifier/lc_classifier/features/composites/lsst.py` — instantiated as `SNExtractor(bands, unit="diff_flux", use_forced_photo=True)` with `bands = ["u","g","r","i","z","y"]`.
- `lc_classifier/lc_classifier/features/composites/elasticc.py` — instantiated as `SNExtractor(bands, unit, use_forced_photo=True)` with `bands = list("ugrizY")` and `unit = "diff_flux"`.

**Tests:**

- `lc_classifier/tests/features/test_sn_extractor.py` — exercises with an ELAsTiCC example object and asserts that `unit="magnitude"` raises `ValueError`.

**Other extractors sharing the same `AstroObject` fields:**

- Any extractor that reads `detections["brightness"]` or `forced_photometry` in `diff_flux` units (e.g. `MHPSExtractor`, `GPDRWExtractor`, `SNParametricModelExtractor`) operates on the same underlying data; no output features from `SNExtractor` are consumed as inputs by other extractors in this repository.
