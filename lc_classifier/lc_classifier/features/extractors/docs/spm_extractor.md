# SPMExtractor

Fits a six-parameter Supernova Parametric Model (SPM) to a multi-band differential-flux light curve using JAX-JIT-compiled objective with analytic gradients supplied to `scipy.optimize.minimize` (TNC), and optionally corrects for Milky Way extinction and cosmological distance before fitting.

- **Source:** `lc_classifier/lc_classifier/features/extractors/spm_extractor.py`
- **Version:** `1.0.1`
- **Base class:** `FeatureExtractor` (abstract, from `lc_classifier.features.core.base`)
- **External libs:** `jax`, `numba`, `scipy.optimize`, `extinction` (v0.4.7), `astropy.cosmology.WMAP5`

## Purpose / Meaning

The SPM is a physically-motivated analytic model for supernova-like transient light curves. Fitting it per band yields interpretable shape parameters (amplitude, rise time, fall time, plateau length, plateau slope, time of onset) that serve as compact descriptors for transient classification. The reduced chi-squared of the fit provides a quality-of-fit metric. Together the parameters form a compact representation of the light curve morphology that is robust to irregular cadence and missing bands.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Photometric band identifiers. Each must be in `{"u","g","r","i","z","y"}`; otherwise `ValueError` is raised. |
| `unit` | `str` | — (required) | Physical unit of `brightness`. Only `"diff_flux"` is accepted; anything else raises `ValueError`. |
| `redshift` | `Optional[str]` | `None` | Name of the `metadata` field holding the host-galaxy redshift. If `None`, no redshift correction is applied. |
| `extinction_color_excess` | `Optional[str]` | `None` | Name of the `metadata` field holding the Milky Way color excess `E(B-V)`. If `None`, no dust correction is applied. |
| `forced_phot_prelude` | `Optional[float]` | `None` | Number of days before the first detection to keep forced-photometry epochs. Epochs earlier than `first_detection_mjd - forced_phot_prelude` are discarded. If `None`, all pre-detection forced photometry is retained. |

### `AstroObject` fields read

- `detections` — columns: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`, `sid`.
- `forced_photometry` — columns: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`. Appended to detections when `astro_object.forced_photometry is not None`.
- `metadata` — queried for the redshift value (by `metadata["name"] == self.redshift_name`) and for `E(B-V)` (by `metadata["name"] == self.extinction_color_excess_name`) when the respective constructor arguments are not `None`.

### Pre-filtering applied

1. `detections` is copied; `astro_object.forced_photometry` is concatenated when present.
2. Rows where `unit != self.unit` are dropped.
3. Rows where `brightness` is `NaN` are dropped.
4. If `forced_phot_prelude` is set, rows with `mjd <= first_detection_mjd - forced_phot_prelude` are dropped.
5. `mjd` is shifted so that `min(mjd) == 0` (i.e. time is relative to the first surviving epoch).
6. All `brightness` and `e_brightness` values are multiplied by `0.001` (mJy → milli-unit conversion note; see *Hardcoded values*).
7. If `mwebv` (extinction) and `zhost` (redshift) are available, per-band flux and error are multiplied by a combined de-attenuation factor before fitting (see *Underlying library / math*).

### Valid `unit` values

Only `"diff_flux"` is accepted. The constructor enforces this at construction time. Downstream code assumes `brightness` values are signed differential fluxes.

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`.

`sid` is derived from the unique `sid` values present in `astro_object.detections` (before any filtering), sorted and joined with a comma. `version` is `"1.0.1"`.

For each band in `self.bands`, seven features are appended:

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `SPM_A` | per band | Amplitude parameter of the SPM model (in the scaled flux unit after the `× 0.001` rescaling). |
| `SPM_t0` | per band | Onset time (days relative to first surviving epoch). The model rises steeply after this time. |
| `SPM_gamma` | per band | Plateau duration in days. The transition between linear plateau and exponential decay occurs at `t1 = t0 + gamma`. |
| `SPM_beta` | per band | Plateau slope parameter in `[0, 1]`. `beta=0` means flat plateau; `beta=1` means maximum linear decline during the plateau phase. |
| `SPM_tau_rise` | per band | Characteristic rise timescale in days. Controls how rapidly flux increases from zero to peak near `t0`. |
| `SPM_tau_fall` | per band | Characteristic fall timescale in days. Controls the exponential decay rate after the plateau ends. |
| `SPM_chi` | per band | Reduced chi-squared of the best-fit model on the band's detection epochs: `sum((model - flux)^2 / (error + 1e-3)^2) / (N - 6)`. |

**Sentinel value:** `np.nan`. Conditions that produce `np.nan`:

- The observations DataFrame is empty after all filtering: all six SPM parameters and `SPM_chi` are `np.nan` for every band.
- A band in `self.bands` has no observations (i.e. is not in `available_bands` from the data): all six SPM parameters and `SPM_chi` are `np.nan` for that band.
- A band has 6 or fewer observations (`N - 6 < 1`): `SPM_chi` is `np.nan` for that band (denominator of chi-squared would be non-positive).

## Underlying library / math

### SPM model formula

The model flux at time `t` for a band with parameters `(A, t0, gamma, beta, tau_rise, tau_fall)` is:

```
t1 = t0 + gamma

rise_factor(t)  = sigmoid((t - t0) / tau_rise)        # 1 / (1 + exp(-(t-t0)/tau_rise))
trans_factor(t) = sigmoid(0.5 * (t - t1))             # transition at t1

plateau(t) = 1 - beta * (t - t0) / gamma              # linearly declining plateau
decay(t)   = (1 - beta) * exp(-(t - t1) / tau_fall)   # exponential decay after t1

model(t) = A * rise_factor(t) * [decay(t) * trans_factor(t) + plateau(t) * (1 - trans_factor(t))]
```

`trans_factor` smoothly blends between the plateau phase (for `t < t1`) and the exponential decay phase (for `t > t1`). `rise_factor` suppresses flux before `t0`.

When `tau_fall < tau_rise` the model can diverge for early times; the code forces `trans_factor = 0` for `t < t1` in that case.

Numerical clipping is applied throughout to prevent overflow:
- `sigmoid_exp_arg` for the transition is clipped to `[-10, 10]`.
- `fall_exp_arg` is clipped to `[-20, 20]`.
- `den_exp_arg` for the rise denominator is clipped; values `< -20` map to zero output.

### Objective function — JAX + `grad`

**Function:** `objective_function_jax` (module-level, decorated with `@jax_jit`)

The objective is the sum of weighted squared residuals across all bands, plus a regularization term:

```
loss = sum_over_bands( dot(band_sqerr, ignore_negative_fluxes) ) + regularization
```

where:
- `band_sqerr[i] = ((model(t_i) - flux_i) / (error_i + smooth_error))^2`
- `ignore_negative_fluxes[i] = exp( -((flux_i + error_i) / (error_i + 1e-3))^2 )` when `flux_i + error_i < 0`, else `1.0`. This soft-downweights epochs where both flux and error are negative (likely bogus detections).
- `smooth_error = 0.5 * percentile(obs_errors, 10)` — adds a floor to the per-point error, preventing over-fitting to very small errors.

**Regularization:**

```
params_var = [var(A) + 1, var(t0) + 0.05, var(gamma) + 0.05, var(beta) + 0.005, var(tau_rise) + 0.05, var(tau_fall) + 0.05]
lambdas    = [0.0, 1.0, 0.1, 20.0, 0.7, 0.01]   # per-parameter weights
regularization = dot(lambdas, sqrt(params_var))
```

Regularization couples multi-band fits: the variance in each parameter across bands is penalized, pushing solutions toward shared shape across bands. `beta` is penalized most strongly (`lambda=20`). `A` (amplitude) is not regularized (`lambda=0`), because amplitudes are expected to differ between bands.

**Analytic gradient:** `grad_objective_function_jax = jax_jit(grad(objective_function_jax))` — JAX auto-diff, JIT-compiled. This exact function is passed as `jac=` to `scipy.optimize.minimize`.

**Note:** JAX is configured with `jax.config.update("jax_enable_x64", True)` at module import time (module-level side effect), enabling 64-bit float throughout.

### Optimizer — `scipy.optimize.minimize` (TNC)

**Function:** `scipy.optimize.minimize(..., method="TNC", bounds=bounds, options={"maxfun": 1000})`

- **Algorithm:** Truncated Newton Conjugate-gradient (TNC) — a bound-constrained quasi-Newton method. Uses the analytic gradient. Handles per-parameter box bounds.
- **`maxfun`:** Maximum 1000 function evaluations per fit (baked in; not tunable via constructor).
- **Bounds:** Per-band parameter bounds are computed from data; shared `t0` bounds are `[-50, max(times)]`.

### Model inference for chi-squared — `model_inference_stable` (Numba)

**Function:** `model_inference_stable(times, A, t0, gamma, beta, t_rise, t_fall)` decorated with `@jit(nopython=True)`.

Implements the same SPM formula as the JAX objective but using `numpy` arithmetic and explicit clipping. Used only for computing chi-squared after the optimization, not during fitting. `numba.set_num_threads(1)` is set in `SNModel.__init__` to avoid thread contention.

### Extinction correction — `extinction.odonnell94`

**Function:** `extinction.odonnell94(wavelengths, av, rv)` — compiled C extension (`extinction` v0.4.7).

- **Algorithm:** O'Donnell (1994) parameterization of the interstellar extinction curve. Returns extinction in magnitudes at each input wavelength.
- **`rv`:** Hardcoded to `3.1` (standard diffuse ISM value).
- **`av`:** Computed as `rv * mwebv` where `mwebv = E(B-V)` from metadata.
- **Usage:** `extinction.odonnell94(np.array([cws[band]]), av, rv)[0]` — single wavelength per call, in Angstroms.
- **De-attenuation factor:** `10 ** (A_lambda / 2.5)` — converts from magnitudes of extinction to a linear flux multiplier.

**Central wavelengths (`self.cws`) used as inputs to `odonnell94`:**

| Band | Wavelength (Angstrom) |
|------|----------------------|
| `u`  | 3671.0 |
| `g`  | 4827.0 |
| `r`  | 6223.0 |
| `i`  | 7546.0 |
| `z`  | 8691.0 |
| `y`  | 9712.0 |

These are baked into `self.cws` and are not tunable.

### Cosmological distance correction — `astropy.cosmology.WMAP5.distmod`

**Function:** `WMAP5.distmod(z)` — returns the distance modulus `mu(z)` in magnitudes.

- **WMAP5 parameters (from `astropy/cosmology/data/WMAP5.ecsv`):** `H0=70.2 km/s/Mpc`, `Om0=0.277`, `Tcmb0=2.725 K`, `Neff=3.04`, flat ΛCDM. Reference: Komatsu et al. 2009, ApJS 180, 330.
- **Correction applied:** A redshift de-attenuation factor `zdeatt = 10^(-(distmod(0.3) - distmod(zhost)) / 2.5)` is combined multiplicatively with the dust correction. The factor maps observed flux to the flux the source would have at the reference redshift `z=0.3`.
- **Reference redshift `0.3`:** Hardcoded; not tunable via constructor.
- **Threshold:** If `zhost < 0.003` or `zhost is None`, `zdeatt = 1.0` (no correction). The threshold `0.003` is hardcoded.

**Note:** The time-axis redshift correction `times /= (1 + zhost)` is present in the source but commented out. Time dilation is therefore not applied.

## Hardcoded values

| Literal | Location | Tunable? | Meaning |
|---------|----------|----------|---------|
| `0.001` | `get_observations` | No | Multiplies all `brightness` and `e_brightness` values. The comment reads "old SPM used milli Jansky, not uJy", indicating this converts from µJy (current pipeline unit) to mJy for backwards compatibility with the model's implicit scale. |
| `self.cws` (6 wavelengths) | `__init__` | No | Effective central wavelengths per band for extinction computation (Angstroms). |
| `rv = 3.1` | `_deattenuation_factor` | No | Standard ratio of total-to-selective extinction for diffuse ISM. |
| `zhost_threshold = 0.003` | `_correct_lightcurve` | No | Minimum redshift below which the cosmological correction is skipped. |
| `reference_z = 0.3` | `_correct_lightcurve` | No | Pivot redshift for the distance-modulus amplitude correction. |
| `pad_multiple = 250` | `pad()` | No | Input arrays are padded to the next multiple of 250 before being passed to the JAX function. Required for JAX JIT to avoid recompiling for every distinct array length. |
| `t0_bounds = [-50.0, max(times)]` | `SNModel.fit` | No | Lower bound on `t0`: onset may be up to 50 days before the first observation. |
| `gamma_guess = 14.0` | `SNModel.fit` | No | Initial guess for plateau duration (days). |
| `beta_guess = 0.5` | `SNModel.fit` | No | Initial guess for plateau slope. |
| `trise_guess = 7.0` | `SNModel.fit` | No | Initial guess for rise timescale (days). |
| `tfall_guess = 28.0` | `SNModel.fit` | No | Initial guess for fall timescale (days). |
| `t0_guess_offset = -10.0` | `SNModel.fit` | No | `t0` initial guess is `argmax_flux_time - 10` days. |
| `A_guess_factor = 1.2` | `SNModel.fit` | No | `A` initial guess is `1.2 * max_flux_in_band`. |
| `A_bounds = [|max_flux|/10, |max_flux|*10]` | `SNModel.fit` | No | Amplitude bounds are data-driven but anchored to the observed maximum. |
| `gamma_bounds = [1.0, 120.0]` | `SNModel.fit` | No | Plateau duration allowed range (days). |
| `beta_bounds = [0.0, 1.0]` | `SNModel.fit` | No | Plateau slope bounded to unit interval. |
| `trise_bounds = [1.0, 100.0]` | `SNModel.fit` | No | Rise timescale allowed range (days). |
| `tfall_bounds = [1.0, 180.0]` | `SNModel.fit` | No | Fall timescale allowed range (days). |
| `smooth_error = 0.5 * percentile(obs_errors, 10)` | `SNModel.fit` | No | Error floor added to each point's uncertainty in the objective. |
| `ignore_negative threshold` | `SNModel.fit` | No | Negative observations: weight `exp(-((flux+err)/(err+1e-3))^2)` when `flux + err < 0`. |
| `maxfun = 1000` | `SNModel.fit` | No | Maximum TNC function evaluations. |
| `sigmoid_factor = 0.5` | `objective_function_jax` | No | Steepness of the plateau-to-decay transition sigmoid. |
| `lambda_regularization` | `objective_function_jax` | No | Per-parameter regularization weights: `[0.0, 1.0, 0.1, 20.0, 0.7, 0.01]` for `[A, t0, gamma, beta, tau_rise, tau_fall]`. |
| `prefered_order = "irzygu"` | `SNModel.fit` | No | Band priority for selecting the best available band for `t0` initial guess (redder bands preferred). |
| `band_mapper = dict(zip("grizyu", range(1,7)))` | `SNModel.fit` | No | Maps band characters to integer IDs for use inside the JAX JIT function. |
| `chi_den_floor = 1` | `SNModel.fit` | No | `SPM_chi` is `np.nan` when `N_obs - 6 < 1`. |
| `1e-3` | `SNModel.fit` (chi computation) | No | Error floor added to `band_errors` in the chi-squared denominator: `(error + 1e-3)^2`. |

## Important considerations

**Unit rescaling trap:** All fluxes are multiplied by `0.001` inside `get_observations`. The SPM parameters `SPM_A` and the chi-squared are therefore expressed in the rescaled unit (mJy if the pipeline delivers µJy). Consumers must be aware of this when interpreting `SPM_A` in physical units.

**JAX module-level side effect:** `jax.config.update("jax_enable_x64", True)` is executed at import time. This affects all JAX computations in the same Python process, not just those in this extractor. Importing `spm_extractor` may silently change the default dtype behaviour of unrelated JAX code.

**JAX JIT recompilation:** The `objective_function_jax` and `grad_objective_function_jax` are `@jax_jit` compiled. JAX traces on the first call and recompiles whenever the input shapes change. Input arrays are padded to the next multiple of 250 to limit the number of distinct shapes seen by JAX (reducing recompilation). The class docstring notes: `"the firsts calls are really expensive because of jax compilations"`.

**Multi-band coupling via regularization:** The optimization is global across all available bands simultaneously. Per-band parameters are regularized against each other through the cross-band variance terms in the objective. Fitting a single-band light curve still works (variance across bands is zero), but the `+ offset` terms in `params_var` ensure regularization is never zero.

**`SNModel` is stateful and reused:** A single `SNModel` instance is created in `SPMExtractor.__init__` and reused across all calls to `compute_features_single_object`. `self.parameters` and `self.chis` are overwritten on each call. This is not thread-safe.

**`numba.set_num_threads(1)`:** Set once in `SNModel.__init__`. This is a process-wide Numba setting that persists after the call and may affect other Numba-compiled functions in the same process.

**Band availability:** The optimizer only runs on `available_bands` (bands that actually appear in the filtered observations). Bands listed in `self.bands` but absent from the data receive `np.nan` for all seven features. The band ordering for `available_bands` (and thus for the parameter block layout in `res.x`) is determined by `np.unique`, which returns bands in sorted lexicographic order.

**`t0` is shared across bands:** Only one `t0_bounds` and `t0_guess` is computed (from the preferred reference band), but each band in `available_bands` gets its own independent `t0` parameter in the optimization. The shared initial guess and bounds are used for all bands.

**Optimizer may not converge:** `res.success` is extracted from the TNC result but is never checked; the best-available `res.x` is used regardless of convergence status. There is no fallback or warning when `success == False`.

**No exception handling in `compute_features_single_object`:** Errors during metadata lookup (e.g., the redshift or `mwebv` key not present in `metadata`), optimizer errors, or Numba errors will propagate to the caller without being caught.

**`_correct_lightcurve` mutates arrays in place:** The `flux` and `e_flux` arrays extracted from the observations DataFrame are modified by multiplying the de-attenuation factor. These are `.values` numpy arrays, so the mutation does not affect the source DataFrame.

**Cosmological correction reference redshift `0.3`:** The correction normalizes flux to the distance modulus difference `distmod(0.3) - distmod(zhost)`. This means at `zhost=0.3` the factor is `1.0`, and at higher (lower) redshifts flux is boosted (reduced) to match the `z=0.3` scale. This is an unusual choice that will produce systematically different `SPM_A` values for objects at different redshifts. The choice is baked in and not documented in source comments.

**`extinction` is a compiled C extension:** `extinction.odonnell94` is implemented in Cython/C (`extinction.cpython-310-x86_64-linux-gnu.so`). No readable Python source is available in the installed package. The function signature inferred from usage: `odonnell94(wave: np.ndarray[float64], a_v: float, r_v: float) -> np.ndarray[float64]` returning extinction in magnitudes.

## Cross-references

**Composites that include this extractor:**

- `lc_classifier/lc_classifier/features/composites/ztf.py` — instantiated as `SPMExtractor(["g","r"], unit="diff_flux", redshift=None, extinction_color_excess=None, forced_phot_prelude=30.0)`.
- `lc_classifier/lc_classifier/features/composites/lsst.py` — instantiated as `SPMExtractor(["u","g","r","i","z","y"], unit="diff_flux", redshift=None, extinction_color_excess=None, forced_phot_prelude=30.0)`.
- `lc_classifier/lc_classifier/features/composites/elasticc.py` — instantiated as `SPMExtractor(list("ugrizY"), unit="diff_flux", redshift="REDSHIFT_HELIO", extinction_color_excess="MWEBV", forced_phot_prelude=30.0)`. This is the only composite that activates both extinction and redshift corrections.

**Other extractors that share the same `AstroObject` fields:**

- `SNExtractor` (`sn_extractor.py`) — also reads `detections` and `forced_photometry` in `diff_flux`; operates on the same flux columns.
- `MHPSExtractor`, `GPDRWExtractor` — also consume `diff_flux` brightness from `detections`/`forced_photometry`.

**No downstream extractor in this repository reads SPM output feature names** as input to further computation; the features are consumed directly by downstream classifiers.
