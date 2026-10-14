# HarmonicsExtractor

Fits a truncated Fourier series to a folded light curve and extracts per-band harmonic amplitudes, relative phases, fit MSE, and reduced chi-squared.

- **Source:** `lc_classifier/lc_classifier/features/extractors/harmonics_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `numpy`, `pandas` (no additional third-party scientific library; the Fourier fit is implemented inline)

## Purpose / Meaning

A periodic light curve can be represented as a sum of sinusoids at harmonics of the fundamental frequency. The amplitudes (`Harmonics_mag_*`) and relative phases (`Harmonics_phase_*`) of those harmonics characterise the light-curve shape independently of its brightness. These shape descriptors are powerful discriminants between variable-star classes (e.g. RR Lyrae vs. eclipsing binaries vs. Cepheids). The goodness-of-fit statistics (`Harmonics_mse`, `Harmonics_chi`) indicate how well the periodic model describes the data and flag cases where the period is unreliable or the light curve is noisy.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Photometric bands to process (e.g. `["g","r"]`). |
| `unit` | `str` | — (required) | Observation unit; must be `"magnitude"` or `"diff_flux"`. Controls `error_tol`. |
| `use_forced_photo` | `bool` | — (required) | If `True`, concatenates `forced_photometry` detections with `detections` before processing. |
| `n_harmonics` | `int` | `7` | Number of Fourier harmonics to fit. Determines the model degrees of freedom (`2*n_harmonics + 1`) and the number of output features. |

### `AstroObject` fields read

- `detections` — columns: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`, `sid`. The `sid` column is read to build the composite survey-ID string attached to every output feature row.
- `forced_photometry` — when `use_forced_photo=True`, concatenated with `detections` along `axis=0`. Expected to carry the same columns.
- `features` — the pre-existing feature table is read to retrieve `Multiband_period` (the fundamental period in days), which must already be present. If it is absent, `compute_features_single_object` raises an `Exception` immediately.

### Pre-filtering applied

1. If `use_forced_photo=True` and `astro_object.forced_photometry is not None`, forced-photometry rows are appended to detections.
2. Rows whose `unit` column does not match `self.unit` are dropped.
3. Rows where `brightness` is `NaN` are dropped.
4. Rows where `e_brightness <= 0.0` are dropped.
5. The remaining observations are split per band; no explicit `mjd` sort is applied (order is preserved from the input DataFrames).

### Valid `unit` values

| Value | Effect |
|-------|--------|
| `"magnitude"` | Sets `error_tol = 1e-2`. Raises `ValueError` at construction for any other string. |
| `"diff_flux"` | Sets `error_tol = 1e-3`. |

`error_tol` is added to each per-observation error in the chi-squared denominator to prevent division by very small numbers (see *Hardcoded values*).

## Output

One set of feature rows is appended per band in `self.bands`. The `fid` column is set to the band string; `sid` is the sorted, comma-joined list of unique survey IDs found in `detections`.

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `Harmonics_mag_1` | per band | Amplitude of the fundamental harmonic: `sqrt(a_1^2 + b_1^2)`. |
| `Harmonics_mag_2` … `Harmonics_mag_<n_harmonics>` | per band | Amplitudes of the 2nd through N-th harmonics. |
| `Harmonics_phase_2` … `Harmonics_phase_<n_harmonics>` | per band | Relative phase of harmonic k with respect to the fundamental: `(phi_k - k*phi_1) mod 2pi` (radians). No `Harmonics_phase_1` is emitted. |
| `Harmonics_mse` | per band | Mean squared error between the fitted model and the observed brightness values. |
| `Harmonics_chi` | per band | Reduced chi-squared: `sum((fit - obs)^2 / (err + error_tol)^2) / (N - (1 + 2*n_harmonics))`. |

**Total rows per band:** `2 + n_harmonics + (n_harmonics - 1)` = `2*n_harmonics + 1`. With the default `n_harmonics=7`: 15 rows per band.

**Sentinel value:** `np.nan` is written for all features of a given band when either:
- `Multiband_period` is `np.nan`, or
- the number of band observations after filtering is strictly less than `self.degrees_of_freedom` (`2*n_harmonics + 1 = 15` by default).

`Harmonics_chi` is additionally set to `np.nan` when `len(fitted_magnitude) - (1 + 2*n_harmonics) < 1`, i.e. when the denominator of the reduced chi-squared would be zero or negative.

## Underlying library / math

No dedicated periodogram library is called. The fit is implemented entirely with `numpy`:

**Design matrix construction**

```
omega[:, 0]        = 1                                  (DC / mean term)
omega[:, 1..N]     = cos(2*pi*k*f0*t),  k = 1..n_harmonics
omega[:, N+1..2N]  = sin(2*pi*k*f0*t),  k = 1..n_harmonics
```

where `f0 = 1 / Multiband_period` and `t` are the observation times in MJD.

**Weighted least-squares solution**

Each row of the design matrix and the brightness vector is scaled by `1/error` (inverse-error weighting), then solved via `numpy.linalg.pinv`:

```python
w_a = (1/error).reshape(-1,1) * omega   # shape (N_obs, 1 + 2*n_harmonics)
w_b = (brightness / error).reshape(-1,1)
coeffs = pinv(w_a) @ w_b                # shape (1 + 2*n_harmonics,)
```

`numpy.linalg.pinv` uses the Moore-Penrose pseudoinverse (SVD-based). This is numerically safe when `w_a` is rank-deficient but may be slow for large `N_obs`.

**Amplitude and phase extraction**

```python
coef_mag[k] = sqrt(a_k^2 + b_k^2)                        # k = 1..n_harmonics
coef_phi[k] = arctan2(b_k, a_k)                           # k = 1..n_harmonics
# relative phase (phi_k - k*phi_1), then mod 2pi for k >= 2
coef_phi = (coef_phi - coef_phi[0] * arange(1, n+1))[1:] % (2*pi)
```

The phase formula subtracts `k * phi_1` from each harmonic's absolute phase, expressing all phases relative to the fundamental. The modulo wraps into `[0, 2*pi)`.

## Hardcoded values

| Value | Location | Tunable? | Effect |
|-------|----------|----------|--------|
| `error_tol = 1e-2` | constructor, `unit == "magnitude"` | No (unit-derived) | Added to each `e_brightness` in chi-squared denominator to prevent near-zero division. |
| `error_tol = 1e-3` | constructor, `unit == "diff_flux"` | No (unit-derived) | Same purpose, smaller floor for flux-space errors. |
| `degrees_of_freedom = 2*n_harmonics + 1` | constructor | Derived from `n_harmonics` | Minimum observations required before fitting; also the denominator offset in reduced chi-squared. |
| `10**-2` (= `0.01`) | `compute_features_single_object` line 65 | No | Added to every `e_brightness` value *before* computing `inverr` (`1/error`). This is separate from `error_tol` and unconditionally inflates measurement errors regardless of `unit`. |
| `2 * np.pi` | design matrix construction | No | Period-to-angular-frequency conversion constant. |
| `n_harmonics = 7` | constructor default | Yes (constructor arg) | Controls model complexity and number of output features. |

**Note on the double error floor:** there are two independent error-inflation mechanisms. `10**-2` (line 65) is always added to raw `e_brightness` before inversion (`inverr = 1/error`), affecting both the weighted fit and the chi-squared numerator. `error_tol` is only added in the chi-squared denominator. For `unit == "magnitude"` both are `0.01`; for `unit == "diff_flux"` the denominator uses `0.001` but the numerator/fit still uses the `0.01` floor. This asymmetry is undocumented in the source.

## Important considerations

- **Ordering dependency.** `HarmonicsExtractor` must run *after* `PeriodExtractor` in any composite pipeline. If `Multiband_period` is absent from `astro_object.features`, `compute_features_single_object` raises an `Exception` (not returns NaN). The composites `ZTFFeatureExtractor`, `LSSTFeatureExtractor`, and `ElasticcFeatureExtractor` all satisfy this ordering constraint.

- **Period read bug risk.** The period is retrieved via `period["value"][0]`, where `0` is the integer *index label*, not a positional index. If the DataFrame index does not contain `0` (e.g. after a concat), this will raise a `KeyError`. It should be `.iloc[0]`.

- **No `mjd` sort.** The extractor does not sort by `mjd`. The Fourier design matrix is built in the order rows arrive. This is safe for the algebraic fit but worth noting for any future sequential processing layered on top.

- **In-place mutation.** `compute_features_single_object` appends rows directly to `astro_object.features` via `pd.concat` and reassigns the attribute. If the extractor raises mid-band (e.g. on the KeyError above), previously appended bands are already committed.

- **Bands with no data.** If a band has no observations after filtering, `len(band_observations) < self.degrees_of_freedom` is `True` (since 0 < 15), so NaN sentinels are written without error.

- **Forced photometry `None` guard.** If `use_forced_photo=True` but `astro_object.forced_photometry is None`, the extractor silently proceeds with detections only. No warning is emitted.

- **`sid` construction.** `sid` is assembled from `detections["sid"].unique()` only — forced-photometry survey IDs are not included even when `use_forced_photo=True`. Whether this is intentional is unclear from the source.

- **Phase interpretation.** `Harmonics_phase_k` is `(phi_k - k*phi_1) mod 2pi` in radians, with `k` starting at 2. There is no `Harmonics_phase_1` feature. Consumers should be aware that the phase wraps; a value near `0` and a value near `2*pi` represent nearly the same phase relationship.

## Cross-references

- **Composites that include this extractor:**
  - `ZTFFeatureExtractor` (`composites/ztf.py`) — `unit="magnitude"`, `use_forced_photo=True`, bands `["g","r"]`
  - `LSSTFeatureExtractor` (`composites/lsst.py`) — `unit="magnitude"`, `use_forced_photo=True`, bands `["u","g","r","i","z","y"]`
  - `ElasticcFeatureExtractor` (`composites/elasticc.py`) — `unit="diff_flux"`, `use_forced_photo=True`, bands `["u","g","r","i","z","Y"]`
- **Upstream dependency:** `PeriodExtractor` (`extractors/period_extractor.py`) — emits `Multiband_period`, which `HarmonicsExtractor` reads as its fundamental frequency.
- **Parallel dependent:** `FoldedKimExtractor` (`extractors/folded_kim_extractor.py`) — also reads `Multiband_period` from `astro_object.features` using the same pattern.
- **Tests:** `tests/features/test_harmonics_feature_extractor.py`, `tests/features/test_ztf_preprocessor.py`
