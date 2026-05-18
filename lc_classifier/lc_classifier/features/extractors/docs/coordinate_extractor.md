# CoordinateExtractor

Converts a sky object's mean equatorial coordinates (RA/Dec) into unit-sphere
Cartesian coordinates and appends them as three scalar features.

- **Source:** `lc_classifier/lc_classifier/features/extractors/coordinate_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `numpy`, `pandas` (standard; no domain-specific scientific library)

## Purpose / Meaning

Positional features on the unit sphere (`x`, `y`, `z`) give a machine-learning
model a smooth, continuous representation of sky position that avoids the
discontinuity at RA = 0°/360° and the coordinate singularity at the poles.
They are directly usable as regressor inputs without further trigonometric
preprocessing.

## Input

### Constructor arguments

None. `CoordinateExtractor` has no constructor and therefore no tunable
parameters.

### `AstroObject` fields read

- `detections` — columns `ra` and `dec` (decimal degrees, equatorial J2000
  assumed). The mean across all rows is taken.
- `detections` — column `sid` (survey identifier string), used only to build
  the `sid` label for the output rows.

### Pre-filtering applied

None. Every row in `detections` contributes equally to the arithmetic mean of
`ra` and `dec`. No NaN-dropping, band filtering, or minimum-length guard is
applied before the mean.

### Valid `unit` values

Not applicable. `CoordinateExtractor` does not read the `unit` column and
performs no photometric computation.

## Output

Three rows appended to `astro_object.features` (columns `name`, `value`, `fid`,
`sid`, `version`).

| Feature `name`  | `fid` scope | Meaning |
|-----------------|-------------|---------|
| `Coordinate_x`  | `None` (all-band) | `cos(ra) * cos(dec)` on the unit sphere |
| `Coordinate_y`  | `None` (all-band) | `sin(ra) * cos(dec)` on the unit sphere |
| `Coordinate_z`  | `None` (all-band) | `sin(dec)` on the unit sphere |

- All three values are dimensionless and lie in `[-1, 1]`.
- `fid` is set to `None` for every row (position is not band-specific).
- `sid` is set to a comma-separated, sorted string of all unique `sid` values
  found in `detections` (e.g. `"ZTF"` or `"LSST,ZTF"`).
- No sentinel value path exists: the extractor will raise a `KeyError` or
  propagate a `NaN` from `pandas.DataFrame.mean` if `ra`/`dec` are missing or
  entirely NaN — it does not guard against these cases.

## Underlying library / math

No domain-specific scientific library is called. The only numerical operations
are:

1. `detections[["ra","dec"]].mean()` — arithmetic mean across all detections.
2. Degree-to-radian conversion: `ra_rad = ra_deg / 180.0 * np.pi`.
3. Standard spherical-to-Cartesian projection (right-handed, ISO physics
   convention with polar axis along z):

   ```
   x = cos(ra_rad) * cos(dec_rad)
   y = sin(ra_rad) * cos(dec_rad)
   z = sin(dec_rad)
   ```

   This uses the astronomical convention where Dec (not colatitude) is measured
   from the equatorial plane, so `z = sin(dec)` rather than `cos(theta)`.

No library defaults are inherited that affect numerical output.

## Hardcoded values

| Literal | Location | Tunable? | Effect |
|---------|----------|----------|--------|
| `180.0` | degree-to-radian conversion | no | exact; no approximation |
| `np.pi` | degree-to-radian conversion | no | numpy's `float64` pi |

No magic thresholds, band lists, or minimum-sample guards.

## Important considerations

- **Averaging RA across the 0°/360° wrap boundary** is numerically incorrect
  for objects near RA = 0°. The in-source comment acknowledges this
  ("conversion → mean would be better, but this is cheaper"). For objects whose
  detections span the wrap, the mean RA will be wrong, and so will all three
  Cartesian coordinates.
- **No NaN guard.** If `ra` or `dec` contains any NaN values, `pandas.mean`
  silently skips them (NaN-aware mean). If *all* values are NaN, the result is
  `NaN`, which propagates to the three features without raising an exception.
- **No minimum-length guard.** A single detection is sufficient; the extractor
  does not fail on one-point light curves.
- **`fid` is always `None`**, so downstream code that filters features by `fid`
  must handle `None` explicitly.
- **`sid` encoding.** When an `AstroObject` carries detections from multiple
  surveys, `sid` becomes a comma-joined string like `"LSST,ZTF"`. Consumers
  must not assume `sid` is a scalar survey identifier for these features.
- **In-place mutation.** `astro_object.features` is replaced by a new
  `pd.concat` result. Any reference held to the old DataFrame before the call
  becomes stale.
- **No version bump mechanism.** `version = "1.0.0"` is a class-level constant.
  All three features carry the same version string.

## Cross-references

- **Composites that include this extractor:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `ZTFFeatureExtractor`
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — LSST composite
  - `lc_classifier/lc_classifier/features/composites/elasticc.py` — ELAsTiCC composite
- **Tests:** `lc_classifier/tests/features/test_coordinate_extractor.py`
- **Other extractors reading the same fields:** any extractor that reads
  `detections["ra"]` or `detections["dec"]` (e.g. positional cross-match
  extractors), but no extractor was found to *consume* `Coordinate_x`,
  `Coordinate_y`, or `Coordinate_z` as inputs.
- **Downstream consumers of the feature names:** no extractor in the repo reads
  `Coordinate_x/y/z` from `astro_object.features`; these features are intended
  as direct model inputs.
