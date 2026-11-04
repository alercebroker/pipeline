# DummyExtractor

A no-op stub extractor that prints a message and performs no computation.

- **Source:** `lc_classifier/lc_classifier/features/extractors/dummy_extractor.py`
- **Version:** none — `self.version` is not defined
- **Base class:** `FeatureExtractor` (abstract base from `lc_classifier.features.core.base`)
- **External libs:** none

## Purpose / Meaning

`DummyExtractor` exists solely as a development scaffold. It satisfies the
`FeatureExtractor` interface by implementing `compute_features_single_object`,
but the body does nothing beyond printing a diagnostic string to stdout. It
emits no features and mutates nothing on the `AstroObject`. It is not included
in any composite extractor's production pipeline, though it is imported by
`LSSTFeatureExtractor` (see Cross-references).

## Input

### Constructor arguments

None. `DummyExtractor` defines no `__init__`, so it inherits the default
`object.__init__` with no parameters.

### `AstroObject` fields read

None. The implementation ignores the `astro_object` argument entirely.

### Pre-filtering applied

None.

### Valid `unit` values

Not applicable — no photometric data is consumed.

## Output

`DummyExtractor` appends nothing to `astro_object.features`. The method
returns `None` explicitly, which is consistent with the in-place contract
documented in `FeatureExtractor.compute_features_single_object` ("This method
is inplace"), but no in-place mutation occurs either.

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| *(none)*       | —           | No features are produced. |

**Sentinel behavior:** not applicable — no output path exists to short-circuit.

## Underlying library / math

No third-party scientific libraries are used. The only dependency is the
internal `FeatureExtractor` and `AstroObject` from
`lc_classifier.features.core.base`.

## Hardcoded values

- The string `" Dummy Extractor called "` — printed to stdout on every
  call. Baked in, not configurable.

## Important considerations

- **No `self.version`:** Unlike every production extractor, `DummyExtractor`
  does not set `self.version`. Any downstream code that reads
  `extractor.version` will raise `AttributeError`.
- **Side effect — stdout:** Each call prints to stdout unconditionally. In
  batch processing (`compute_features_batch` iterates over all objects), this
  produces one print per object with no way to suppress it short of redirecting
  stdout externally.
- **Returns `None`, not in-place mutation:** The docstring on
  `FeatureExtractor.compute_features_single_object` says the method is
  in-place. `DummyExtractor` returns `None` and leaves `astro_object.features`
  as the empty `DataFrame` initialised by `AstroObject.__post_init__`.
- **Not registered in the production pipeline:** `LSSTFeatureExtractor`
  imports `DummyExtractor` but does not instantiate it inside
  `_instantiate_extractors`. It is dead import code in that composite.
- **`AstroObject` contract not verified:** Because no field is accessed,
  the mandatory column checks (`oid`, `sid`, `fid` in `detections`) enforced
  by `AstroObject.__post_init__` are the only guard rails; the extractor
  itself adds none.

## Cross-references

- **Composites that import (but do not use) this extractor:**
  `lc_classifier/lc_classifier/features/composites/lsst.py` —
  `LSSTFeatureExtractor` imports `DummyExtractor` at the module level but
  does not include it in the list returned by `_instantiate_extractors`.
- **Other extractors that read the same `AstroObject` fields:** not applicable
  (no fields are read).
- **Consumers of emitted feature names:** none (no features are emitted).
