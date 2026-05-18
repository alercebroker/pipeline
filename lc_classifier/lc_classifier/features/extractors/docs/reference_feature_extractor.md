# ReferenceFeatureExtractor

Computes statistics describing the proximity and morphology of the photometric reference source associated with each ZTF detection: mean/sigma of source–reference angular separation (`distnr`), and observation-count-weighted mean sharpness (`sharpnr`) and chi-squared (`chinr`) of the reference.

- **Source:** `lc_classifier/lc_classifier/features/extractors/reference_feature_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** none (only `numpy`, `pandas`)

## Purpose / Meaning

ZTF difference-image photometry subtracts a reference image from each science exposure. The spatial offset between the transient position and the nearest reference source (`distnr`, in arcsec), along with the point-source morphology of that reference (`sharpnr`, `chinr`), are diagnostics that help distinguish nuclear transients from off-nucleus events and artefacts. A training-set constraint (the ZTF forced-photometry service used a 5 arcsec cone) is baked into the pre-filtering step.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Band identifiers used to filter observations and form the `fid` label of every emitted feature (e.g. `["g", "r"]`). |

The class-level constant `unit = "diff_flux"` is not a constructor argument; it cannot be overridden at instantiation without subclassing.

### `AstroObject` fields read

- `detections` — mandatory. Columns read: `unit`, `brightness`, `e_brightness`, `fid`, `distnr`, `rfid`.
- `forced_photometry` — optional (`None` is handled). When present, concatenated with `detections` before filtering. Same columns as above must be present.
- `reference` — optional (`None` is handled). Expected columns: `rfid` (integer reference-image ID), `sharpnr` (PSF sharpness of the reference source), `chinr` (chi-squared of the PSF fit of the reference source).
- `features` — existing feature rows; the extractor appends to them in place.

`detections` must contain `sid` (used only to populate the output `sid` column); the mandatory columns enforced by `AstroObject.__post_init__` are `oid`, `sid`, `fid`.

### Pre-filtering applied

1. Merge `detections` and `forced_photometry` (if not `None`) via `pd.concat`.
2. Keep rows where `unit == "diff_flux"`.
3. Keep rows where `brightness` is not `NaN`.
4. Keep rows where `e_brightness > 0.0`.
5. Keep rows where `fid` is in `self.bands`.
6. Keep rows where `0.0 <= distnr <= 5.0` and `distnr` is not `NaN` — **hardcoded 5 arcsec upper bound** (see *Hardcoded values*).

After the `distnr` statistics are computed, an additional filter is applied for the `sharpnr`/`chinr` path: rows where `rfid` is `NaN` are dropped and `rfid` is cast to `int`.

### Valid `unit` values

Only `"diff_flux"` passes the unit filter. Observations with any other unit (e.g. `"magnitude"`) are silently discarded before any computation.

## Output

All four features share the same `fid` value: `",".join(self.bands)` (e.g. `"g,r"` for ZTF). The `sid` value is `",".join(sorted(astro_object.detections["sid"].unique()))`.

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `mean_distnr` | all bands joined | Arithmetic mean of `distnr` over all post-filter observations (arcsec). |
| `sigma_distnr` | all bands joined | Standard deviation (`pandas` default: ddof=1) of `distnr` over all post-filter observations. `NaN` if fewer than 2 observations pass the filter (pandas behaviour). |
| `mean_sharpnr` | all bands joined | Observation-count-weighted mean of `sharpnr` across reference images. |
| `mean_chinr` | all bands joined | Observation-count-weighted mean of `chinr` across reference images. |

**Sentinel value:** `np.nan`. Emitted for `mean_distnr` and `sigma_distnr` when no observations survive pre-filtering. Emitted for `mean_sharpnr` and `mean_chinr` when: (a) no observations with a valid `rfid` exist after filtering, (b) `astro_object.reference` is `None` or an empty `DataFrame`, or (c) no observations match any `rfid` in the reference table (`n_obs_with_ref == 0`).

The four feature rows are always appended regardless of whether they are `NaN`; the extractor never short-circuits before building the output `DataFrame`.

Downstream schemas rename these features by appending the band suffix, e.g. `mean_distnr` with `fid="g,r"` → `mean_distnr_12` in the ZTF Avro schema (see *Cross-references*).

## Underlying library / math

No third-party scientific libraries are called. All computation uses standard `pandas` and `numpy`:

- `observations["distnr"].mean()` — arithmetic mean (ignores NaN by default).
- `observations["distnr"].std()` — sample standard deviation with `ddof=1` (pandas default).
- Weighted mean of `sharpnr`/`chinr`: explicit Python loop over reference rows, accumulating `sharpnr_row * n_obs_for_that_rfid`; divides by total matched observation count. Equivalent to weighting each reference's morphology statistics by how many observations in the current object used that reference image.

## Hardcoded values

| Value | Location | Tunable? | Meaning |
|-------|----------|----------|---------|
| `"diff_flux"` | class attribute `unit` | No (without subclassing) | Only `diff_flux` observations are processed. |
| `0.0` (lower bound on `distnr`) | `get_observations`, line 30 | No | Rejects nonsensical negative distances (likely flagged/bad values). |
| `5.0` (upper bound on `distnr`) | `get_observations`, line 31 | No | 5 arcsec cone match — derived from the ZTF forced-photometry training service limit. Comment in source: *"5 arcsec limit is because of training set limitations (it used the ZTF forced photometry service)"*. |

## Important considerations

- **`sigma_distnr` is `NaN` for single-observation objects.** `pandas.Series.std()` with `ddof=1` returns `NaN` when the series has length 1.
- **`astro_object.reference` is expected to have one row per reference image, not one row per observation.** The loop iterates over reference rows and uses `rfid` to count how many observations map to each reference. If `reference` contains duplicate `rfid` values the weighted average double-counts those entries.
- **`rfid` cast to `int` before join.** If `rfid` in `observations` contains float representations of integers (e.g. `1.0`), the cast succeeds. If it contains non-integer floats the cast raises `ValueError` and the extractor will propagate the exception rather than returning NaNs — there is no exception guard in this path.
- **In-place mutation.** `compute_features_single_object` appends to `astro_object.features` directly. Calling the extractor twice on the same object will duplicate all four feature rows.
- **`distnr` sourced from merged detections + forced photometry.** The `distnr` statistics therefore reflect both detection and forced-photometry epochs when `forced_photometry` is provided.
- **`sid` is taken from `detections` only**, not from the merged observations DataFrame. Forced-photometry rows may carry different `sid` values that are not reflected in the output `sid` column.
- **No `min_length` guard.** Unlike many other extractors in this codebase, `ReferenceFeatureExtractor` does not enforce a minimum number of observations; it returns computed values (or NaNs) regardless of how few observations survive filtering.
- **ZTF-specific columns.** `distnr`, `rfid`, `sharpnr`, and `chinr` are ZTF-pipeline columns. This extractor is not meaningful for surveys that do not populate these fields.

## Cross-references

- **Composite that includes this extractor:** `ZTFFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/ztf.py`, line 63). It is instantiated as `ReferenceFeatureExtractor(bands)` with `bands = list("gr")`.
- **Downstream schema consumers:** `lc_classification_step/tests/mockdata/features_ztf.py`, `lc_anomaly_step/tests/mockdata/features_ztf.py`, and `alerce_classifiers/tests/mockdata/schemas.py` all expect the renamed forms `mean_chinr_12`, `mean_distnr_12`, `mean_sharpnr_12`, `sigma_distnr_12`. The rename from `fid="g,r"` to the `_12` suffix happens in the training/ingestion utilities (`training/lc_classifier_ztf/ATAT_ALeRCE/data/utils.py`, lines 161–164).
- **Other extractors reading the same `AstroObject` fields:** No other extractor in the ZTF composite reads `astro_object.reference`. `detections` and `forced_photometry` are consumed by most other extractors in `ZTFFeatureExtractor`.
