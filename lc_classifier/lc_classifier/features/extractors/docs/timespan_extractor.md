# TimespanExtractor

Computes the total time baseline of all detections for an astronomical object, collapsed across all bands and surveys into a single scalar feature.

- **Source:** `lc_classifier/lc_classifier/features/extractors/timespan_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** none (only `pandas` and `numpy`, both standard pipeline dependencies)

## Purpose / Meaning

`Timespan` is the elapsed time (in days, MJD difference) between the first and last detection of any kind. It reflects how long the object has been observed regardless of band or survey, which is useful for distinguishing transient events (short timespans) from long-period or persistent variables (long timespans). It also serves as a normalisation denominator for rate-based features computed by other extractors.

## Input

### Constructor arguments

This extractor takes no constructor arguments.

### `AstroObject` fields read

- `detections` — columns: `mjd` (float, Modified Julian Date), `sid` (string survey identifier). No other columns are accessed.

### Pre-filtering applied

None. The extractor reads `astro_object.detections` directly without applying any row filter, NaN drop, band selection, or minimum-length guard.

### Valid `unit` values

Not applicable. The extractor reads only `mjd` and `sid`; it does not branch on any `unit` column.

## Output

One row is appended to `astro_object.features`.

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `Timespan`     | `None`      | `max(mjd) - min(mjd)` across all detections, in days. |

The `sid` field of the output row is a comma-joined, lexicographically sorted string of every unique `sid` value present in `detections` (e.g. `"LSST"` or `"ZTF,LSST"`).

**Sentinel:** no sentinel or `np.nan` is emitted. If `detections` has exactly one row, `Timespan` will be `0.0`. If `detections` is empty, `pandas` will return `NaN` for both `max` and `min`, so the output value will be `NaN` — but the extractor does not guard against an empty DataFrame explicitly.

## Underlying library / math

No third-party scientific library is called. The computation is entirely:

```python
timespan = detections["mjd"].max() - detections["mjd"].min()
```

This is a straightforward range (peak-to-peak) operation on the `mjd` column using `pandas.Series.max` and `pandas.Series.min`.

## Hardcoded values

There are no numeric literals in the extractor body. All behaviour is fully determined by the content of `detections["mjd"]`.

## Important considerations

- **No minimum-length guard.** A single detection produces `Timespan = 0.0`. The extractor will not short-circuit or warn.
- **Empty detections DataFrame.** `pandas` `max()` and `min()` on an empty Series return `NaN` (with a `RuntimeWarning` in some pandas versions), so `Timespan` would be `NaN`. There is no explicit guard.
- **NaN propagation in `mjd`.** If any `mjd` value is `NaN`, `pandas` `max`/`min` silently skip NaNs by default (`skipna=True`), so the result is computed from the remaining valid values with no error or warning to the caller.
- **Multi-survey `sid` encoding.** The `sid` column of the output row is a single comma-joined string (e.g. `"LSST,ZTF"`), not a list. Consumers that split on `","` must handle single-survey objects (no comma present) as a special case.
- **Mutates `astro_object.features` in place.** The extractor concatenates via `pd.concat` and reassigns `astro_object.features`; the original DataFrame reference held by the caller is replaced.
- **`fid` is `None`.** Unlike per-band extractors, `Timespan` uses `fid=None` because it aggregates across all bands. Consumers must not assume a non-null `fid` for this feature.
- **No forced photometry.** `astro_object.forced_photometry` is never read. The timespan reflects only the formal detections, not forced-photometry epochs, regardless of how the object was constructed.
- **Survey ordering.** `sids` are sorted with `np.sort` (lexicographic on strings) before joining, so the `sid` string is deterministic regardless of detection ordering.

## Cross-references

- **Composites that include this extractor:**
  - `ZTFFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/ztf.py`)
  - `LSSTFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/lsst.py`)
  - `ElasticcFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/elasticc.py`)
- **Other extractors reading the same fields:** any extractor that reads `detections["mjd"]` (e.g. period, variability, and color extractors).
- **Consumers of `Timespan`:** not detected via grep in the extractor or composite layer; downstream model code outside `lc_classifier` may use the feature by name.
