# TDETailExtractor / FleetExtractor / ColorVariationExtractor

Three co-located extractors in one file that characterise tidal-disruption-event (TDE) light curves: a log-linear tail decay fit (`TDETailExtractor`), a physically-motivated rise-and-plateau model fit (`FleetExtractor`), and a measure of inter-band colour variability (`ColorVariationExtractor`).

- **Source:** `lc_classifier/lc_classifier/features/extractors/tde_extractor.py`
- **Versions:** `TDETailExtractor` `1.0.1` / `FleetExtractor` `1.0.2` / `ColorVariationExtractor` `1.0.1`
- **Base class:** `FeatureExtractor`
- **External libs:** `scipy.optimize.curve_fit`, `jax` (JIT-compiled model evaluation), `numpy`

---

## Purpose / Meaning

All three extractors are designed to capture morphological signatures of TDE light curves in difference-flux photometry:

- `TDETailExtractor` fits a weighted linear model to the post-peak magnitude decay, expressing it as a log-linear function of time elapsed since peak. TDEs typically show a power-law flux decline, which maps to a linear slope in log-time vs. magnitude space.
- `FleetExtractor` fits a parametric FLEET-inspired model to the full light curve in magnitude space, capturing both the rise (exponential growth) and the post-peak plateau/decline through a combined `exp + linear` functional form.
- `ColorVariationExtractor` measures the standard deviation of window-averaged colour (magnitude difference between two bands) as a proxy for chromatic evolution — TDEs tend to show blue, slowly-evolving colour.

---

## Input

### Constructor arguments

#### `TDETailExtractor`

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Band identifiers to process, e.g. `["g", "r"]` |

#### `FleetExtractor`

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — (required) | Band identifiers to process |

#### `ColorVariationExtractor`

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `window_len` | `float` | — (required) | Time window length in days for binning observations when computing per-window colour |
| `band_1` | `str` | — (required) | First (bluer) band identifier |
| `band_2` | `str` | — (required) | Second (redder) band identifier; colour = `band_1 mag - band_2 mag` |

### `AstroObject` fields read

- `detections` — columns used: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`, `sid`
- `forced_photometry` — concatenated with `detections` when not `None` (all three extractors do this)
- `features` — existing feature rows; new rows are appended

### Pre-filtering applied

#### `TDETailExtractor` and `ColorVariationExtractor`
1. Concatenate `detections` and `forced_photometry` (if present).
2. Keep only rows with `unit == "diff_flux"`.
3. Drop rows where `brightness` is `NaN`.
4. Drop rows where `e_brightness <= 0`.
5. Convert `brightness` to absolute value, then convert to AB magnitude using `flux2mag` (μJy → AB mag via `-2.5 * log10(flux) + 23.9`).
6. Convert `e_brightness` to magnitude error using `flux_err_2_mag_err` (`2.5 * flux_err / (ln(10) * flux)`).
7. Apply `e_brightness < 1.0` and `brightness < 30.0` magnitude-space cuts.

#### `FleetExtractor`
1. Concatenate `detections` and `forced_photometry` (if present).
2. Keep only rows with `unit == "diff_flux"`.
3. Drop rows where `brightness` is `NaN`.
4. Keep only rows where `brightness > 1` (at least 1 μJy positive signal; operates in flux space).
5. Drop rows where `e_brightness <= 0`.
6. Magnitudes are computed inside `compute_features_single_object` (not inside `get_observations`).

### Valid `unit` values

All three extractors require `unit == "diff_flux"` (difference flux in μJy). Rows with any other unit are silently discarded. No magnitude-unit path exists.

---

## Output

Features are appended to `astro_object.features` as rows with columns `name`, `value`, `fid`, `sid`, `version`. `sid` is derived from the sorted unique `sid` values in `astro_object.detections`, joined with commas.

### `TDETailExtractor` features

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `TDE_decay` | per band | Slope coefficient `coeffs[1]` of the weighted linear fit `mag = coeffs[0] + coeffs[1] * 2.5*log10(dt + 40)`. Negative values indicate declining magnitude (brightening flux). |
| `TDE_decay_chi` | per band | Reduced chi-squared of the linear fit: `sum((fitted - y)^2 / y_err^2) / (N - 2)`. Lower values indicate a better fit. |
| `TDE_mag0` | per band | Intercept `coeffs[0]` of the fit; an estimate of the effective magnitude at the reference log-time zero. |

Sentinel: `np.nan` for all three features when `len(band_observations) < 2`, or when there are no observations in the window `(t_d, t_d + 200]` days.

### `FleetExtractor` features

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `fleet_a` | per band | Amplitude of the linear (plateau-decay) term in the FLEET model. |
| `fleet_w` | per band | Exponential rate parameter (must be `<= 0` by bounds); controls the decay speed. |
| `fleet_chi` | per band | Reduced chi-squared of the model fit: `sum((model - y)^2 / y_err^2) / (N - 4)`. |
| `fleet_m0` | per band | Magnitude baseline offset (roughly the plateau magnitude). |
| `fleet_t0` | per band | Time offset parameter (days relative to first observation), shifting the model origin. |

Sentinel: `np.nan` for all five features when `len(band_observations) < 4`, or when `scipy.optimize.curve_fit` raises `RuntimeError` (fit did not converge within `max_nfev=800` evaluations).

### `ColorVariationExtractor` features

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `color_variation` | `"band_1,band_2"` string | Sample standard deviation (`ddof=1`) of per-window mean colour (`band_1 mag - band_2 mag`) across time windows. |

Sentinel: `np.nan` when fewer than 2 valid windows have colour estimates, or when no window has `>= 3` observations in each of the two bands. The `fid` field is the literal string `"g,r"` (or whichever pair was configured), not a single-band identifier.

---

## Underlying library / math

### `TDETailExtractor` — weighted linear regression (NumPy only)

No third-party scientific library is called. The fit is a closed-form weighted least-squares solution via `numpy.linalg.pinv`.

**Model:**

```
mag(t) = coeffs[0] + coeffs[1] * 2.5 * log10(dt + 40)
```

where `dt = mjd - t_d` and `t_d` is the MJD of the brightest (minimum magnitude) observation. Only observations in the window `(t_d, t_d + 200]` days are used. The `+ 40` offset prevents `log10(0)` and effectively sets the log-time reference at `dt = -40` days before peak.

**Fit method:** Weighted pseudoinverse (`numpy.linalg.pinv`) of the design matrix `Omega = [1, x]` weighted by `1 / y_err`. The regularisation is implicit in the pseudoinverse but no explicit ridge term is added.

**Chi-squared:** Standard reduced chi-squared with `dof = N - 2`. Returns `np.nan` if `N < 3` (i.e. `chi_den < 1`).

### `FleetExtractor` — `scipy.optimize.curve_fit` + JAX-JIT model

**Function:** `scipy.optimize.curve_fit(fleet_model, x, y, sigma=y_err, p0=..., bounds=..., max_nfev=800)`

`curve_fit` implements nonlinear least-squares using the Levenberg-Marquardt algorithm (or Trust Region Reflective when bounds are provided, which is the case here). It minimises `sum(((f(x, *params) - y) / sigma)^2)`.

**FLEET model** (evaluated by `fleet_model_jax`, JIT-compiled with `jax.jit`):

```
mag(t) = exp(w * (t - t0)) - a * w * (t - t0) + m_0
```

where:
- `t` is time in days relative to the first observation (`mjd - first_mjd`)
- `a` — linear amplitude (bounds: `[0, 10]`)
- `w` — exponential rate, constrained `<= 0` (bounds: `[-100, 0]`); a value of `0` collapses the exponential to 1
- `m_0` — magnitude baseline (bounds: `[0, 30]`)
- `t0` — time offset in days (bounds: `[-50, 10000]`)

The model captures TDE rise (exponential growth when `w < 0` going backward in time) and subsequent plateau/decline through the interplay of the `exp` and linear terms.

**JAX usage:** `jax.config.update("jax_enable_x64", True)` is called at module import, enabling 64-bit floating point throughout JAX operations. `fleet_model_jax` is decorated with `@jax.jit` for compiled evaluation. The outer `fleet_model` wrapper pads the input array to a multiple of 25 before calling the JIT function (to avoid recompilation on every new input length), then slices back to the original length.

**Initial parameter guess (`p0`):** `[0.6, -0.05, mean(y), 0]` for `[a, w, m_0, t0]`.

**Chi-squared:** Standard reduced chi-squared with `dof = N - 4`. Returns `np.nan` if `N < 5` (i.e. `chi_den < 1`).

### `ColorVariationExtractor` — windowed colour statistics (NumPy / pandas only)

No third-party scientific library is called. Colour per window is the difference of per-band mean magnitudes. The standard deviation uses `numpy.std(..., ddof=1)` (sample std). A window is only included if both bands have `>= 3` observations in that window.

---

## Hardcoded values

### `TDETailExtractor`
- `200` days — maximum look-ahead window after peak (`t_d + 200`); baked in.
- `+ 40` days — log-time offset in `2.5 * log10(dt + 40)` to avoid `log(0)`; baked in.
- `1e-2` — floor added to `y_err` before the fit (`y_err = e_brightness + 1e-2`); baked in.
- `e_brightness < 1.0` — magnitude error cut; baked in.
- `brightness < 30.0` — faint-end magnitude cut; baked in.

### `FleetExtractor`
- `brightness > 1` μJy — minimum flux cut before magnitude conversion; baked in.
- `1e-2` — floor added to `y_err` (`y_err = flux_err_2_mag_err(...) + 1e-2`); baked in.
- `p0 = [0.6, -0.05, mean(y), 0]` — initial parameter guess; baked in.
- `bounds = ([0.0, -100.0, 0, -50], [10, 0, 30, 10000])` — parameter bounds; baked in.
- `max_nfev = 800` — maximum function evaluations (comment in source: "twice default value"); baked in.
- `25` — padding block size in `pad()` for JAX recompilation avoidance; baked in.
- `min_length = 4` — minimum observations per band to attempt fit; baked in.
- `jax_enable_x64 = True` — set globally at module import time; affects all JAX operations in the process.

### `ColorVariationExtractor`
- `window_len` — tunable via constructor.
- `3` — minimum observations per band per window to compute a colour; baked in.
- `ddof=1` — sample standard deviation; baked in.
- `e_brightness < 1.0` — magnitude error cut; baked in.
- `brightness < 30.0` — faint-end magnitude cut; baked in.

---

## Important considerations

- **Flux sign / absolute value:** `TDETailExtractor` (and `ColorVariationExtractor`) take `np.abs(brightness)` before converting to magnitude. This means negative-flux difference epochs are treated as positive detections of the same magnitude. This may conflate pre-peak non-detections or host subtraction artefacts with genuine source flux.

- **Peak definition in `TDETailExtractor`:** The brightest (smallest magnitude after conversion) observation is used to define `t_d`. Because the conversion is `flux2mag(|flux|)`, the peak is the observation with the largest `|diff_flux|`, not necessarily the true photometric peak (host contamination or outliers can shift `t_d`).

- **JAX global state:** The call `jax.config.update("jax_enable_x64", True)` executes at module import. Importing `tde_extractor` anywhere in a process silently switches JAX to 64-bit mode, which affects all other JAX users in that process.

- **JAX recompilation trap:** `fleet_model_jax` is JIT-compiled. The `pad()` function pads to the nearest multiple of 25 to reduce the number of distinct input shapes JAX traces, but does not eliminate recompilation entirely if `N % 25` takes many distinct values across objects.

- **`curve_fit` failure mode:** `RuntimeError` (non-convergence) is caught and results in `np.nan` for all five `fleet_*` features. No other exceptions (e.g. `ValueError` from bad inputs, `LinAlgError`) are caught, so those would propagate.

- **`chi_den < 1` guard:** Both `TDETailExtractor` and `FleetExtractor` check `chi_den >= 1` before dividing; if false they emit `np.nan` for the chi-squared feature. For `TDETailExtractor` this requires `N_after_t_d >= 3`; for `FleetExtractor` it requires `N >= 5`.

- **`fid` encoding for `color_variation`:** The `fid` column is set to the string `"band_1,band_2"` (e.g. `"g,r"`), not a single-band identifier. Downstream consumers must handle this composite string; it does not match the single-character `fid` convention used by the other two extractors.

- **`sid` encoding:** `sid` is constructed by sorting and joining all unique `sid` values from `astro_object.detections` with commas. For multi-survey objects this produces a composite string.

- **No sorting by `mjd`:** None of the three extractors sort observations by `mjd` before processing. `TDETailExtractor` uses `.sort_values("brightness")` only to find the peak. Unsorted input does not cause incorrect results because only scalar statistics and sorted-independent numpy operations are used.

- **In-place mutation:** All three extractors concatenate new rows onto `astro_object.features` in place. Calling the same extractor twice on the same object will duplicate feature rows.

- **Flux unit assumption:** The `flux2mag` function assumes input flux is in μJy: `mag = -2.5 * log10(flux) + 23.9`. If `brightness` is in nJy (LSST native), conversion must happen upstream before these extractors are called. The LSST composite uses `psfFlux / 1000` (nJy → μJy) in `create_astro_object_lsst` before populating `diff_flux` rows.

---

## Cross-references

- **Composites that include these extractors:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `ZTFFeatureExtractor` includes `TDETailExtractor(["g","r"])`, `FleetExtractor(["g","r"])`, `ColorVariationExtractor(window_len=10, band_1="g", band_2="r")`.
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — `LSSTFeatureExtractor` includes `TDETailExtractor(["u","g","r","i","z","y"])`, `FleetExtractor(["u","g","r","i","z","y"])`, and five `ColorVariationExtractor` instances for adjacent LSST band pairs (`u-g`, `g-r`, `r-i`, `i-z`, `z-y`), all with `window_len=10`.

- **Consumers of emitted feature names:**
  - `training/lc_classifier_ztf/feature_computation/compute_features.py` — references `TDE_decay`, `fleet_a`, `fleet_w`, `fleet_chi`, `fleet_m0`, `fleet_t0`, `color_variation` in training feature sets.
  - `training/lc_classifier_ztf/ATAT_ALeRCE/data/utils.py` — reads these feature names for model input construction.
  - `lc_anomaly_step/tests/mockdata/features_ztf.py` and `lc_classification_step/tests/mockdata/features_ztf.py` — include mock values for all TDE feature names.
  - `alerce_classifiers/alerce_classifiers/anomaly/utils.py` — references `color_variation` and FLEET features.
  - `libs/db-plugins-multisurvey/` — `_initial_data.py` and `_initial_data_pipeline.py` include these feature names in database schema/seed data.

- **Other extractors reading the same `AstroObject` fields:**
  - `SPMExtractor`, `SNExtractor`, `GPDRWExtractor`, `MHPSExtractor` — all read `detections` and `forced_photometry` with `unit == "diff_flux"`.
