# TurboFatsExtractor

Computes 26 single-band statistical and variability features per photometric band using the in-repo `turbofats` library.

- **Source:** `lc_classifier/lc_classifier/features/extractors/turbofats_extractor.py`
- **Version:** `1.1.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `turbofats` (in-repo at `lc_classifier/lc_classifier/features/turbofats/`), `scipy`, `statsmodels`, `numba`

## Purpose / Meaning

Extracts a broad suite of time-domain statistics (scatter, variability, shape, and time-correlation features) from a single photometric band at a time. The features characterise the brightness distribution, autocorrelation structure, intrinsic variability, and temporal trends of a light curve and are widely used as discriminating inputs to light-curve classifiers.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Band identifiers to iterate over; one feature row set is emitted per band. |
| `unit` | `str` | — (required) | Photometric unit. Must be `"magnitude"` or `"diff_flux"`. Raises `ValueError` otherwise. |

### `AstroObject` fields read

- `detections` — columns: `unit`, `brightness`, `mjd`, `fid`, `sid`, `e_brightness`.

### Pre-filtering applied

1. Rows where `detections["unit"] != self.unit` are dropped.
2. Rows where `detections["brightness"]` is `NaN` are dropped.
3. Duplicate `mjd` values are dropped (first occurrence kept by `drop_duplicates`).
4. Per band: rows matching `fid == band` are extracted and sorted ascending by `mjd`.
5. `FeatureSpace.calculate_features` returns all-`NaN` sentinels if the resulting band lightcurve has **≤ 5** observations.

### Valid `unit` values

- `"magnitude"` — raw magnitude values; feature names are emitted as-is.
- `"diff_flux"` — difference-image flux; feature names are suffixed with `"_flux"` in the output (e.g. `Amplitude_flux`).

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`.

`fid` is the band identifier (one set of rows per band in `self.bands`).  
`sid` is a comma-joined, sorted string of all unique survey identifiers (`sid`) present across the filtered detections.

All features return `np.nan` if the band has ≤ 5 detections (enforced in `FeatureSpace.calculate_features`). Any exception raised inside a single feature's `fit()` call is caught, a warning is issued, and the value is set to `np.nan`.

| Feature `name` | Meaning |
|----------------|---------|
| `Amplitude` | Half the difference between the median of the top 5 % and the median of the bottom 5 % of sorted brightness values. |
| `AndersonDarling` | Anderson-Darling normality test statistic, passed through a sigmoid: `1 / (1 + exp(-10 * (A - 0.3)))`. Bounded in (0, 1); higher = less Gaussian. |
| `Autocor_length` | Lag (integer count) at which the sample autocorrelation function first drops below `exp(-1) ≈ 0.368`. Computed with `statsmodels.tsa.stattools.acf` starting at `nlags=100`, extending by 100 each time the threshold is not reached. Returns the light-curve length if `nlags` exceeds the number of observations. |
| `Beyond1Std` | Fraction of points that lie more than one standard deviation from the error-weighted mean. Standard deviation is computed relative to the weighted mean, not the sample mean. |
| `Con` | Fraction of consecutive triplets of measurements all lying outside 2σ of the mean. σ and mean are unweighted. Denominator is `N - 2`. Returns 0 if `N < 3`. |
| `Eta_e` | Time-weighted version of the von Neumann η statistic: `(1/σ²) × (Σ wᵢ Δmᵢ²) / (Σ wᵢ)`, where weights are `wᵢ = 1/(tᵢ₊₁ - tᵢ)²`. Small values indicate smooth time evolution. |
| `Gskew` | Median-based skewness: `median(m ≤ p3) + median(m ≥ p97) − 2 × median(m)`, where `p3` and `p97` are the 3rd and 97th percentiles. |
| `MaxSlope` | Maximum absolute first difference in brightness divided by the corresponding time interval (units: brightness unit per day). Computed on time-sorted data. |
| `Mean` | Arithmetic mean of brightness values. |
| `Meanvariance` | Ratio of standard deviation to mean: `std / mean`. Dimensionless variability index. Note: undefined / numerically unstable if `mean ≈ 0` (relevant for `diff_flux` unit). |
| `MedianAbsDev` | Median absolute deviation from the median: `median(|m − median(m)|)`. |
| `MedianBRP` | Median buffer range percentage: fraction of points within `(max − min) / 10` of the median. |
| `PairSlopeTrend` | For the last 30 (time-sorted) observations: `(#increasing differences − #non-increasing differences) / 30`. Bounded in [−1, 1]. If fewer than 30 points exist, the last `N` points are used (Python slice `[-30:]`), but the denominator remains 30. |
| `PercentAmplitude` | Maximum absolute deviation from the median divided by the median: `max(|m − median|) / median`. |
| `Q31` | Interquartile range: 75th percentile minus 25th percentile. |
| `Rcs` | Range of cumulative sum: `max(S) − min(S)` where `S[k] = Σᵢ₌₀ᵏ (mᵢ − m̄) / (N × σ)`. |
| `Skew` | Fisher skewness of brightness values via `scipy.stats.skew`. |
| `SmallKurtosis` | Small-sample excess kurtosis (bias-corrected formula for `n < 300`). |
| `Std` | Sample standard deviation of brightness values (`numpy.std`, ddof=0). |
| `StetsonK` | Robust kurtosis measure: `(1/√N) × Σ|δᵢ| / √(Σδᵢ²)`, where `δᵢ = √(N/(N−1)) × (mᵢ − m̄_w) / σᵢ` and `m̄_w` is the error-weighted mean. |
| `Pvar` | Probability of intrinsic variability: chi-squared CDF `Fχ²(χ², N−1)` where `χ² = Σ (mᵢ − m̄)² / σᵢ²`. |
| `ExcessVar` | Excess variance: `Σ((mᵢ − m̄)² − σᵢ²) / (N × m̄²)`. Measures intrinsic variability amplitude. |
| `SF_ML_amplitude` | Amplitude parameter `A` of the structure function model `SF(τ) = A × τ^γ`, fitted by log-linear regression on binned SF values in the range `0.01 ≤ τ/365 ≤ 0.5` days. Sentinel −0.5 if fit cannot be computed. Clipped to [0.005, 15]; set to 0 if below 0.005. |
| `SF_ML_gamma` | Power-law index `γ` from the same structure function fit. Retrieved from `shared_data` set by `SF_ML_amplitude`. Clipped to [−0.5, 3.0]. Sentinel −0.5 if fit cannot be computed. |
| `IAR_phi` | Autoregressive coefficient φ ∈ (0, 1) of an Irregular AutoRegressive (IAR) model, fitted by maximum likelihood via a Kalman filter using `scipy.optimize.minimize_scalar` with `method="bounded"` and `xatol=1e-12`, `maxiter=50000`. Magnitude is standardised before fitting. |
| `LinearTrend` | Ordinary least-squares slope of brightness on `mjd` via `scipy.stats.linregress`. Units: brightness unit per day. |

### Flux-unit name suffix

When `unit == "diff_flux"`, every feature name listed above has `"_flux"` appended (e.g. `Amplitude_flux`, `IAR_phi_flux`).

## Underlying library / math

All feature classes live in `lc_classifier/lc_classifier/features/turbofats/FeatureFunctionLib.py` and three sub-modules under `features/`. They are instantiated once at constructor time and reused across all `compute_features_single_object` calls.

### `FeatureSpace` (in-repo, `turbofats/FeatureSpace.py`)

- Accepts the feature name list; instantiates the corresponding classes from `FeatureFunctionLib` via `getattr`.
- Passes a single `shared_data` dict to every feature instance; features like `SF_ML_gamma` and `IAR_phi` rely on values deposited in `shared_data` by a previously-run feature (`SF_ML_amplitude`). **The feature execution order is therefore load-bearing.**
- Input array layout: `data[0] = brightness`, `data[1] = mjd`, `data[2] = e_brightness` (extracted by `FeatureSpace.__lightcurve_to_array` from columns `["brightness", "mjd", "e_brightness"]`).
- `shared_data` is cleared at the start of every `calculate_features` call.

### `IAR_phi` (`features/irregular_autoregressive.py`)

- **Algorithm:** Irregular AutoRegressive (IAR) model with Kalman filter (Elorrieta et al.). Log-likelihood is evaluated in a `@jit(nopython=True)` kernel `iar_phi_kalman_numba`.
- **Optimisation:** `scipy.optimize.minimize_scalar` bounded on `(0, 1)`.
- **Hardcoded options:** `xatol=1e-12`, `maxiter=50000`.
- If `|phi| >= 1` inside the Kalman loop, the likelihood is pinned to `1e10`.

### `SF_ML_amplitude` / `SF_ML_gamma` (`features/structure_function.py`)

- **Algorithm:** Fits `SF(τ) = A × τ^γ` by log-linear regression (`numpy.polyfit`) on structure function values binned in log-τ space (bin width 0.1 dex, range 5–2000 days). Only τ values between 0.01 and 0.5 years enter the regression.
- All pairs `(i, j)` are computed via a `@jit(nopython=True)` loop. This is O(N²) in the number of observations.
- `SF_ML_gamma` reads its result from `shared_data["g_sf"]`; raises an `Exception` (not swallowed) if `SF_ML_amplitude` has not run first.

### `AndersonDarling`

- Calls `scipy.stats.anderson(magnitude)` and takes only the test statistic (index 0), ignoring the critical-value table.
- Applies a fixed sigmoid with centre 0.3 and scale 10; these constants are baked in.

### `Autocor_length`

- Uses `statsmodels.tsa.stattools.acf` with `fft=False` (direct method).
- Initial `nlags=100`, doubled each iteration if threshold not met; cap is `len(magnitude)`.

### `CAR_sigma` / `CAR_tau` / `CAR_mean` (`features/conditional_autoregressive.py`)

These classes are imported by `FeatureFunctionLib.py` but **none of them appear in the 26-feature list** used by `TurboFatsExtractor`. They are present in the codebase but not activated here.

## Hardcoded values

| Value | Location | Tunable? |
|-------|----------|----------|
| `> 5` observations required | `FeatureSpace.calculate_features` | No |
| `nlags = 100` initial lag for `Autocor_length` | `FeatureFunctionLib.Autocor_length.__init__` | Via `extra_arguments` in `FeatureSpace.__init__`, but not exposed by `TurboFatsExtractor` |
| `consecutiveStar = 3` for `Con` | `FeatureFunctionLib.Con.__init__` | Via `extra_arguments`; not exposed |
| `2σ` threshold for `Con` | `FeatureFunctionLib.Con.fit` | No |
| Top/bottom `5 %` for `Amplitude` | `FeatureFunctionLib.Amplitude.fit` | No |
| Sigmoid centre `0.3`, scale `10` for `AndersonDarling` | `FeatureFunctionLib.AndersonDarling.fit` | No |
| Last `30` observations for `PairSlopeTrend` (denominator always 30) | `FeatureFunctionLib.PairSlopeTrend.fit` | No |
| Percentiles `3`, `97` for `Gskew` | `FeatureFunctionLib.Gskew.fit` | No |
| SF τ range: 5–2000 days, bin width 0.1 dex | `SF_ML_amplitude.bincalc` | No |
| SF fit τ window: 0.01–0.5 years | `SF_ML_amplitude.fit` | No |
| SF sentinel: `a = −0.5`, `gamma = −0.5` | `SF_ML_amplitude.fit` | No |
| SF amplitude clip: [0.005, 15] | `SF_ML_amplitude.fit` | No |
| SF gamma clip: [−0.5, 3.0] | `SF_ML_amplitude.fit` | No |
| IAR bounds: `(0, 1)` | `IAR_phi.fit` | No |
| IAR options: `xatol=1e-12`, `maxiter=50000` | `IAR_phi.fit` | No |
| `shared_data` key ordering (execution order determines dependency resolution) | `FeatureSpace.calculate_features` | No |

## Important considerations

- **Minimum length:** `FeatureSpace.calculate_features` returns all-`NaN` for any band with ≤ 5 detections after filtering. No other short-circuit exists in the extractor itself.
- **Execution order dependency:** `SF_ML_gamma` requires `SF_ML_amplitude` to run first (uses `shared_data["g_sf"]`). The 26-feature list in the constructor preserves this order (`"SF_ML_amplitude"` before `"SF_ML_gamma"`). Reordering the list would silently break `SF_ML_gamma`.
- **Exception swallowing:** Any `Exception` raised inside a feature's `fit()` is caught by `FeatureSpace.calculate_features`, a `warnings.warn` is issued, and the value is set to `np.nan`. Failures are not propagated and may be invisible in production.
- **In-place mutation:** `astro_object.features` is replaced with a new concatenated `DataFrame`. Empty band DataFrames are silently skipped.
- **`Meanvariance` near zero:** For `unit="diff_flux"` the mean flux can be near zero, making `std / mean` numerically unstable or very large. No guard is present.
- **`PairSlopeTrend` denominator is always 30**, even when fewer than 30 observations are present (Python slice `[-30:]` silently clips). The returned value is not bounded to [−1, 1] in that case but is bounded if exactly 30 points are used.
- **`sid` encoding:** `sid` is produced by joining all unique survey IDs present in the filtered (all-band) detections with `","` — not restricted to the current band. Bands with zero detections are still processed and will emit `NaN` rows, though their `sid` reflects the global detection pool.
- **`numba` JIT compilation:** `iar_phi_kalman_numba`, `SFarray`, and `is_sorted` are decorated with `@jit(nopython=True)`. First-call compilation overhead occurs at import or first invocation. dtype mismatches can raise `numba.core.errors.TypingError` at runtime (swallowed to `np.nan` by the outer catch).
- **No `e_brightness` guard:** `StetsonK`, `Beyond1Std`, `Pvar`, `ExcessVar`, `IAR_phi`, `SF_ML_amplitude`, and `SF_ML_gamma` all consume `e_brightness`. If all error values are zero, `StetsonK` will divide by zero; `IAR_phi` explicitly handles this by substituting a zero array. Other features do not guard against zero errors.
- **Stateful `shared_data`:** The same `shared_data` dict is reused across objects; `FeatureSpace.calculate_features` calls `self.shared_data.clear()` at the start of each call, preventing cross-object contamination.

## Cross-references

- **Composites that include this extractor:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `TurboFatsExtractor(bands=["g","r"], unit="magnitude")`
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — `TurboFatsExtractor(bands=["u","g","r","i","z","y"], unit="magnitude")`
  - `lc_classifier/lc_classifier/features/composites/elasticc.py` — `TurboFatsExtractor(bands=list("ugrizY"), unit="diff_flux")` (output names gain `_flux` suffix)
- **Other extractors reading the same `detections` fields:** Most extractors in the same composites (`MHPSExtractor`, `FoldedKimExtractor`, `HarmonicsExtractor`, etc.) also read `brightness`, `mjd`, `e_brightness`, `fid`, and `unit` from `detections`.
- **Consumers of emitted feature names:** No other extractor was found reading the turbofats feature names from `astro_object.features`. They are consumed downstream by the classifier models.
