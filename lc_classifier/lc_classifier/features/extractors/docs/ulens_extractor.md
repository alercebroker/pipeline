# MicroLensExtractor

Fits the five-parameter Paczynski microlensing model to each photometric band of a light curve and returns the best-fit parameters plus the reduced chi-squared goodness-of-fit.

- **Source:** `lc_classifier/lc_classifier/features/extractors/ulens_extractor.py`
- **Version:** `1.0.2`
- **Base class:** `FeatureExtractor` (abstract, from `lc_classifier.features.core.base`)
- **External libs:** `scipy.optimize.curve_fit`, `jax`, `jax.numpy`, `numpy`, `pandas`

## Purpose / Meaning

Microlensing events produce a smooth, achromatic, time-symmetric brightening that is well described by the Paczynski (1986) model. Fitting this model and inspecting the residuals gives classifiers a compact, physically motivated representation of whether an object's light curve is consistent with gravitational microlensing, which is otherwise difficult to distinguish from other single-peak transients (novae, AGN flares, etc.) using only statistical features.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Photometric band identifiers to iterate over (e.g. `["g", "r"]`). One independent fit is performed per band. |

### `AstroObject` fields read

- `detections` — columns used: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`, `sid`.
- `forced_photometry` — when not `None`, concatenated with `detections` before filtering. Columns used: same as `detections`.

### Pre-filtering applied

1. `detections` and `forced_photometry` (if present) are concatenated along `axis=0`.
2. Rows where `unit != "magnitude"` are dropped (`self.unit = "magnitude"` is hardcoded).
3. Rows where `brightness` is `NaN` are dropped.
4. Rows where `e_brightness >= 1.0` are dropped.
5. The `mjd` column is shifted so that `min(mjd) == 0` across the combined observation table.
6. Per band, the band subset is used directly (no sort; order is inherited from the concatenated DataFrame).
7. If fewer than 4 observations remain in a band, all six features for that band are emitted as `np.nan` and the band is skipped.

### Valid `unit` values

Only `"magnitude"` is accepted. This is a class-level constant (`unit = "magnitude"`), not a constructor argument. Any observation with a different `unit` value is silently dropped during pre-filtering; no exception is raised.

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`.

`sid` is derived from the unique `sid` values in `astro_object.detections` (the original detections, not the merged+filtered table), sorted and joined with a comma. `version` is `"1.0.2"`.

For each band in `self.bands`, six features are appended:

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `ulens_u0` | per band | Fitted impact parameter `u0` — the minimum projected lens-source separation in units of the Einstein radius. Bounded `[0, +inf)`. |
| `ulens_tE` | per band | Fitted Einstein crossing time `tE` in days. Bounded `[0, +inf)`. |
| `ulens_fs` | per band | Fitted source flux fraction `fs` — the fraction of total baseline flux contributed by the lensed source. Bounded `[0, 1]`. |
| `ulens_chi` | per band | Reduced chi-squared of the fit: `sum((model - y)^2 / sigma^2) / (N - 4)`. |
| `ulens_t0` | per band | Fitted time of closest approach `t0` in days (in the shifted `mjd` frame where `min(mjd) = 0`). Unbounded. |
| `ulens_mag0` | per band | Fitted baseline magnitude `mag_0`. Unbounded. |

**Sentinel value:** `np.nan`. Emitted for all six features in a band when either (a) fewer than 4 observations are available after filtering, or (b) `scipy.optimize.curve_fit` raises `RuntimeError` (optimizer did not converge or exceeded `max_nfev`).

**`ulens_chi` special case:** if `len(observations_in_band) - 4 < 1` (i.e. exactly 4 observations), the denominator of the reduced chi-squared is less than 1, and `ulens_chi` is emitted as `np.nan` even when the fit succeeded. Specifically, `chi_den = len(model_prediction) - 4`, and `ulens_chi` is `np.nan` when `chi_den < 1`.

## Underlying library / math

### Microlensing model (inline)

The Paczynski point-source/point-lens magnification model is implemented directly in the extractor as `ulens_model_jax`:

```
t' = t - t0
u  = sqrt(u0^2 + (t' / tE)^2)
A  = (u^2 + 2) / (u * sqrt(u^2 + 4))
m(t) = -2.5 * log10(fs * (A - 1) + 1) + mag_0
```

- `u` is the dimensionless lens-source separation as a function of time.
- `A` is the standard Paczynski point-source magnification (diverges as `u -> 0`).
- `fs` blends the magnified source with an unlensed component: when `fs = 1` the model is pure microlensing; when `fs < 1` additional constant (blended) flux is present.
- The model is in magnitude space, with `mag_0` as the baseline magnitude.

The JAX-compiled function `ulens_model_jax` is decorated with `@jax.jit` (imported as `jax_jit`). The wrapper `ulens_model` pads the time array to a multiple of 25 before passing it to the JIT-compiled function, then trims the output back to the original length. This padding is required to avoid JAX recompilation for every distinct array length.

`jax.config.update("jax_enable_x64", True)` is called at module import time, enabling 64-bit floating point throughout JAX computation.

### `scipy.optimize.curve_fit`

- **Function:** `scipy.optimize._minpack_py.curve_fit(f, xdata, ydata, p0, sigma, bounds, max_nfev)`
- **Algorithm:** Levenberg-Marquardt or Trust Region Reflective nonlinear least-squares (TRF is selected automatically when `bounds` are finite, which is the case here for `u0`, `tE`, and `fs`).
- **`sigma` treatment:** `absolute_sigma` is not set (defaults to `False`), meaning `sigma` values are treated as relative weights only. The returned covariance matrix (discarded by the extractor via `_`) is scaled to match sample variance. The `sigma` passed is `e_brightness + 1e-2`, so the floor ensures no zero-weight observations even when `e_brightness = 0`.
- **`max_nfev=1000`:** overrides the library default. With bounds active (TRF method), the library default would be `100 * (N + 1)` where `N = 5` (number of parameters), giving 600. The extractor explicitly sets `max_nfev=1000`, described in a comment as "twice default value" — this comment reflects the `leastsq` default (`200 * (N + 1) = 1200`) rather than the TRF default; the actual ratio depends on the solver selected.
- **`RuntimeError`:** raised by `curve_fit` when the optimizer fails to converge within `max_nfev` evaluations. Caught explicitly; all six features are emitted as `np.nan`.
- **Covariance output:** the covariance matrix `_` is discarded. No parameter uncertainties are stored as features.

## Hardcoded values

| Literal | Location | Tunable? | Meaning |
|---------|----------|----------|---------|
| `self.unit = "magnitude"` | class body | No | Only magnitude-unit observations are accepted. |
| `min_band_length = 4` | `compute_features_single_object` | No | Minimum number of per-band observations required to attempt a fit. |
| `e_brightness` floor `1e-2` | `get_observations` | No | Added to every `e_brightness` value before passing as `sigma` to `curve_fit`, preventing zero-weight observations. |
| `p0 = [0.6, 20.0, 0.5, mjd_max_flux, np.median(y)]` | `compute_features_single_object` | No | Initial parameter guess: `u0=0.6`, `tE=20 days`, `fs=0.5`, `t0` at brightest observed epoch, `mag_0` at median magnitude. |
| `bounds = ([0, 0, 0, -inf, -inf], [inf, inf, 1, inf, inf])` | `compute_features_single_object` | No | Physical bounds: `u0 >= 0`, `tE >= 0`, `0 <= fs <= 1`; `t0` and `mag_0` are unconstrained. |
| `max_nfev = 1000` | `compute_features_single_object` | No | Maximum optimizer function evaluations. Comment says "twice default value". |
| `chi_den = len(model_prediction) - 4` | `compute_features_single_object` | No | Degrees of freedom for reduced chi-squared. Subtracts 4, not 5 (the number of free parameters), which may be an off-by-one — intent is unclear from the source. |
| `pad_length` multiple of 25 | `pad` | No | Array padding to stabilise JAX JIT trace cache: input length is rounded up to next multiple of 25. |
| `jax_enable_x64 = True` | module level | No | Forces 64-bit floats in JAX; set globally at import time, affecting all JAX computations in the process. |

## Important considerations

**Off-by-one in degrees of freedom:** `chi_den = len(model_prediction) - 4` subtracts 4 from the number of observations, but the model has 5 free parameters (`u0`, `tE`, `fs`, `t0`, `mag_0`). The standard reduced chi-squared denominator for 5 parameters would be `N - 5`. Whether the subtraction of 4 is intentional is not documented in the source.

**`t0` is in the shifted MJD frame:** `get_observations` shifts `mjd` by subtracting the minimum across the merged (detections + forced photometry) table. The fitted `ulens_t0` is therefore not in absolute MJD; a consumer that needs an absolute time must add back the original `min(mjd)`, which is not stored anywhere in `astro_object`.

**`mjd_max_flux` initial guess:** the initial `t0` guess is derived as the `mjd` of the observation with the smallest `brightness` value (since `sort_values("brightness").iloc[0]` gives the minimum, which in magnitude scale is the brightest point). This is correct for the initial guess but assumes brightness is in magnitudes (lower = brighter), consistent with `self.unit = "magnitude"`.

**JAX global state mutation:** `jax.config.update("jax_enable_x64", True)` runs at module import time. Importing `ulens_extractor` therefore changes the JAX float precision for the entire Python process, which may affect other JAX code running in the same environment.

**JAX JIT recompilation boundary:** the `pad` function pads arrays to the next multiple of 25. Arrays whose lengths map to different multiples of 25 will each trigger a separate JIT compilation. For short light curves (common in classification pipelines), the padded length is likely 25 for all bands, minimising recompilation.

**No sorting by `mjd`:** unlike many other extractors, `get_observations` does not sort by `mjd`. The time array passed to `curve_fit` follows the concatenation order of `detections` and `forced_photometry`. `curve_fit` with TRF does not require sorted inputs, so this is not a correctness issue, but it means `mjd_max_flux` (derived via `sort_values("brightness").iloc[0]["mjd"]`) is the MJD of the globally brightest point, which may differ from the chronological peak if there are repeated brightness values.

**Only `RuntimeError` is caught:** `curve_fit` may also raise `ValueError` (e.g. if residuals are not finite after JAX computation produces `inf` or `nan` at extreme parameter values). A `ValueError` from `curve_fit` would propagate uncaught to the caller.

**In-place mutation:** `compute_features_single_object` appends to `astro_object.features` via `pd.concat`, consistent with the `FeatureExtractor` contract (`"""This method is inplace"""`).

**`sid` source:** `sid` is read from `astro_object.detections["sid"].unique()`, not from the merged+filtered observation table. If all detections are filtered out by the `unit` filter but `forced_photometry` contributes observations, `sid` will still be populated from the original detections.

## Cross-references

**Composites that include this extractor:**

- `lc_classifier/lc_classifier/features/composites/ztf.py` — instantiated as `MicroLensExtractor(bands)` with `bands = ["g", "r"]`.
- `lc_classifier/lc_classifier/features/composites/lsst.py` — instantiated as `MicroLensExtractor(bands)` with `bands = list("ugrizy")` (exact band list depends on the composite constructor argument).

**Tests:**

- `lc_classifier/tests/features/test_ulens_extractor.py` — exercises with ZTF forced-photometry training examples, `bands=list("gr")`.

**Other extractors sharing the same `AstroObject` fields:**

- Any extractor reading `detections["brightness"]` in magnitude units shares the same underlying data. No output features from `MicroLensExtractor` are consumed as inputs by other extractors in this repository.
