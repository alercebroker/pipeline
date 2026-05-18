# GPDRWExtractor

Fits a Damped Random Walk (DRW) Gaussian Process model per photometric band and returns the inferred amplitude (`GP_DRW_sigma`) and characteristic timescale (`GP_DRW_tau`).

- **Source:** `lc_classifier/lc_classifier/features/extractors/gp_drw_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `celerite2`, `scipy.optimize`

## Purpose / Meaning

The DRW (Ornstein-Uhlenbeck) process is a standard stochastic model for AGN variability and is widely used to characterise the smoothness and amplitude of aperiodic light-curve variation. Fitting this model per band yields two physical parameters — the process variance (`GP_DRW_sigma`) and the correlation decay rate (from which the timescale `GP_DRW_tau` is derived) — that help classifiers distinguish AGN from other variable types.

## Input

### Constructor arguments

| Name    | Type        | Default | Meaning                                                          |
|---------|-------------|---------|------------------------------------------------------------------|
| `bands` | `List[str]` | —       | Photometric bands to process (e.g. `["g", "r"]` for ZTF).       |
| `unit`  | `str`       | —       | Brightness unit; must be `"magnitude"` or `"diff_flux"`. Raises `ValueError` otherwise. |

### `AstroObject` fields read

- `detections` — columns: `brightness`, `e_brightness`, `fid`, `mjd`, `unit`, `sid`.
- `forced_photometry` — **not read**; this extractor uses `astro_object.detections` only.
- `metadata` — not read.

### Pre-filtering applied

1. Rows where `brightness` is `NaN` are dropped.
2. Rows whose `unit` column does not match `self.unit` are dropped.
3. Per band, rows are selected by `fid == band`.
4. If the resulting per-band detection count is less than `detections_min_len` (= 5), the band is skipped and sentinels are emitted.
5. `mjd` is shifted so that `min(mjd) == 0`.
6. Detections are sorted by `mjd` (ascending) — required by celerite2, which raises `ValueError` on unsorted input.
7. `brightness` is mean-subtracted (zero-mean model assumed).

### Valid `unit` values

`"magnitude"` or `"diff_flux"`. Both composites that use this extractor (`ZTFFeatureExtractor`, `LSSTFeatureExtractor`) pass `unit="diff_flux"`. The `unit` filter is applied before the GP fit, so any detection row in `detections` with a different unit value is silently excluded.

## Output

Every row appended to `astro_object.features`. Columns: `name`, `value`, `fid`, `sid`, `version`.

| Feature `name`  | `fid` scope | Meaning                                                                                  |
|-----------------|-------------|------------------------------------------------------------------------------------------|
| `GP_DRW_sigma`  | per band    | Fitted amplitude `a` of the DRW kernel, in units of the input `brightness`. Equals `exp(sol.x[0])` after L-BFGS-B optimisation. |
| `GP_DRW_tau`    | per band    | Fitted characteristic timescale in days. Equals `1 / exp(sol.x[1])`, i.e. `1 / c` from `RealTerm(a, c)`. |

**Sentinel value:** `np.nan` for both features when the per-band detection count is below `detections_min_len` (5). No sentinel is explicitly emitted when the L-BFGS-B optimiser fails to converge; in that case `sol.x` will contain the best iterates reached and the returned values will be non-NaN but potentially unreliable.

`sid` is computed as a comma-joined, sorted string of all unique `sid` values present in the (already unit-filtered) detections — not scoped per band.

## Underlying library / math

### `celerite2.terms.RealTerm`

- **Function:** `celerite2.terms.RealTerm(a=..., c=...)`
- **Algorithm:** Defines the simplest celerite covariance kernel:

  k(τ) = a · exp(−c · τ)

  This is the Ornstein-Uhlenbeck (DRW) kernel. The parameter `a` is the amplitude (variance at zero lag) and `c` is the inverse timescale (decay rate). `c > 0` is required for a valid covariance function; the library's own docstring warns that this term "will generally behave poorly" and that `SHOTerm` should be preferred for most use cases — it is used here deliberately to implement the DRW model.
- **Coefficients returned by `get_coefficients()`:** `(array([a]), array([c]), empty, empty, empty, empty)` — a single real term, no complex component.

### `celerite2.GaussianProcess.compute`

- **Function:** `celerite2.numpy.GaussianProcess.compute(t, *, diag=None, quiet=False)`
- **Algorithm:** Builds the celerite matrices `(c, a, U, V)` from the kernel, then performs a Cholesky factorisation of the `N×N` GP covariance matrix using the semi-separable (rank-1 for `RealTerm`) structure. Complexity is O(N) in both time and memory for a rank-J kernel.
- **`quiet=True`:** When the Cholesky factorisation fails (e.g. due to numerical non-positive-definiteness), instead of raising `LinAlgError`, the log-determinant is set to `-inf` and `_norm` is set to `inf`. This silences the error — `log_likelihood` will then return `-inf`, which drives the optimiser away from that parameter region.
- **Dtype requirement:** All arrays are cast to `float64` C-contiguous arrays internally (`_as_tensor` calls `np.ascontiguousarray(..., dtype=np.float64)`).
- **Sort requirement:** `_check_sorted` raises `ValueError` if `np.any(np.diff(t) < 0)`. The extractor satisfies this by calling `sort_values("mjd")` before passing times to the GP.

### `celerite2.GaussianProcess.log_likelihood`

- **Function:** `gp.log_likelihood(y)`
- **Returns:** The marginal log-likelihood of the GP model:

  log p(y) = −0.5 · (y − μ)ᵀ K⁻¹ (y − μ) − 0.5 · log|K| − (N/2) · log(2π)

  where `K = kernel_matrix + diag(e_brightness²)`, `μ = 0` (mean-subtracted brightness), and the solve uses the pre-factored Cholesky. Returns a scalar `float64`.

### `scipy.optimize.minimize` (L-BFGS-B)

- **Function:** `scipy.optimize.minimize(neg_log_like, x0, method="L-BFGS-B", bounds=..., args=...)`
- **Algorithm:** Limited-memory BFGS with box constraints. Minimises the negative log-likelihood over two parameters in log-space: `params = [log(a), log(c)]`.
- **Returns:** `OptimizeResult`; `sol.x` contains the optimal log-space parameters. `np.exp(sol.x)` gives `[a_opt, c_opt]`.
- **No convergence check:** The extractor does not inspect `sol.success` or `sol.fun`. If the optimiser terminates without converging, the last `sol.x` is used silently.

## Hardcoded values

| Value | Location | Tunable? | Meaning |
|-------|----------|----------|---------|
| `detections_min_len = 5` | `__init__` | No (attribute, but no constructor arg) | Minimum per-band detections required to attempt a GP fit. |
| `a=1.0, c=10.0` | `kernel = terms.RealTerm(a=1.0, c=10.0)` | No | Initial kernel used only to instantiate `GaussianProcess`; overwritten at every optimiser call before `compute`. Has no effect on results. |
| `mean=0.0` | `celerite2.GaussianProcess(kernel, mean=0.0)` | No | GP mean function fixed at zero. The extractor mean-subtracts brightness before calling the GP, so this is consistent. |
| `initial_params = np.zeros((2,))` | `minimize` call | No | Initial log-space parameters: `log(a)=0 → a=1`, `log(c)=0 → c=1`. |
| `bounds=[[-10.0, 19.0], [-6.0, 6.0]]` | `minimize` call | No | Log-space bounds for `a` and `c`. In linear space: `a ∈ [exp(−10), exp(19)] ≈ [4.5e-5, 1.8e8]`; `c ∈ [exp(−6), exp(6)] ≈ [0.0025, 403]` (timescale `τ = 1/c ∈ [0.0025, 403]` days). |
| `method="L-BFGS-B"` | `minimize` call | No | Optimisation algorithm. |

## Important considerations

- **No convergence guard:** `sol.success` is never checked. Silent non-convergence will produce non-NaN outputs that may be physically meaningless. Downstream consumers cannot distinguish converged from non-converged fits.
- **`quiet=True` in `gp.compute`:** A Cholesky failure during the optimisation does not raise an exception; it sets the log-likelihood to `-inf`. This is intentional — it pushes the optimiser away from ill-conditioned parameter regions — but it means all numerical errors during the fit are silently absorbed.
- **Stateful `gp` object inside the closure:** The `gp` object is created once per band and mutated on every objective function evaluation inside `neg_log_like` (kernel and mean are reassigned). This is safe for single-threaded use but not thread-safe.
- **Mean subtraction is irreversible:** `detections_band["brightness"] -= np.mean(...)` operates on a `.copy()`, so the original `astro_object.detections` is not mutated.
- **`sid` is multi-band:** The `sid` column in the output is derived from all (unit-filtered) detections across all bands, not from the per-band subset. All feature rows for an object share the same `sid` string.
- **`GP_DRW_tau` interpretation:** The timescale is `1 / c` (inverse of the kernel decay rate), in the same time unit as `mjd` (days). It is *not* related to the `tau` parameter of `SHOTerm` or `SHOTerm`'s reparameterisation.
- **`RealTerm` positive-definiteness:** The celerite2 documentation notes that `RealTerm` should be used only by advanced users who keep both `a` and `c` strictly positive. The optimisation bounds keep both in positive territory (log-space lower bounds ensure the exponentiated values are > 0), but the optimiser is not constrained to strictly positive values in the interior.
- **Input dtype:** `detections_band["mjd"].values` and related arrays are passed as NumPy arrays; celerite2 casts them to `float64` internally. If the source DataFrame has `float32` columns, a silent copy/cast occurs.

## Cross-references

- **Composites that include this extractor:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `ZTFFeatureExtractor`, instantiated with `bands=["g","r"]` and `unit="diff_flux"`.
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — `LSSTFeatureExtractor`, instantiated with `bands=["u","g","r","i","z","y"]` and `unit="diff_flux"`.
- **Consumers of `GP_DRW_sigma` / `GP_DRW_tau`:**
  - `lc_anomaly_step/tests/mockdata/features_elasticc.py`, `features_ztf.py` — mock feature schemas include these names.
  - `lc_classification_step/tests/mockdata/features_ztf.py`, `features_elasticc.py` — same.
  - `alerce_classifiers/alerce_classifiers/messi/utils.py`, `anomaly/utils.py` — classifier utilities that reference these feature names.
  - `libs/db-plugins-multisurvey/db_plugins/db/sql/_initial_data_pipeline.py`, `_initial_data.py` — database schema definitions listing these feature names.
- **Tests:** `lc_classifier/tests/features/test_gp_drw_extractor.py`
