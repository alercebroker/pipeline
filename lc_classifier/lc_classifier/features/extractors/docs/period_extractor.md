# PeriodExtractor

Computes a multi-band periodogram and derives a best-fit period, per-band periods, a significance estimate (PPE), and optional power-rate features for a light curve.

- **Source:** `lc_classifier/lc_classifier/features/extractors/period_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `P4J` (v1.2.0), `lc_classifier.utils.is_sorted` (via `numba`)

## Purpose / Meaning

Periodicity is one of the strongest discriminants between variable-star classes (e.g. RR Lyrae, Cepheids, eclipsing binaries) and non-periodic transients. `PeriodExtractor` fits a multi-harmonic Analysis of Variance (MHAOV) periodogram across all available bands simultaneously using `P4J.MultiBandPeriodogram`, returning the best global period and—when the lightcurve is long enough—a set of secondary statistics that measure how clearly the period stands out above the noise.

## Input

### Constructor arguments

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `bands` | `List[str]` | — | Ordered list of band identifiers to process (e.g. `["g", "r"]`). Determines which `fid` values are considered and the `fid` string on multiband features. |
| `unit` | `str` | — | Photometric unit; must be `"magnitude"` or `"diff_flux"`. Checked at construction; `ValueError` raised otherwise. |
| `smallest_period` | `float` | — | Lower bound on searched periods (days). Converted to `largest_frequency = 1/smallest_period` inside P4J. |
| `largest_period` | `float` | — | Upper bound on searched periods (days). Converted to `smallest_frequency = 1/largest_period` inside P4J. |
| `trim_lightcurve_to_n_days` | `float \| None` | — | If not `None`, the lightcurve is trimmed to the densest contiguous window of this width in days before computing the periodogram. |
| `min_length` | `int` | — | Minimum number of observations required *per band* for that band to be included; also a global minimum on the total post-filter count. |
| `use_forced_photo` | `bool` | — | If `True`, `astro_object.forced_photometry` is concatenated with `astro_object.detections` before filtering. |
| `return_power_rates` | `bool` | — | If `True`, six `Power_rate_*` features are appended. |
| `shift` | `float` | `0.1` | Controls the coarse frequency-grid density: `grid_size = ceil(f_range * lc_time_length / (2 * shift))`. Smaller values → finer grid → slower computation. |

### `AstroObject` fields read

- `detections` — columns used: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`, `sid`. `sid` is read only for feature-row labelling; `oid` is not required by this extractor.
- `forced_photometry` — same schema as `detections`; concatenated when `use_forced_photo=True` and the field is not `None`.
- `metadata` — row with `name == "aid"` is read for error-log messages only; the `aid` value is not used in computation.

### Pre-filtering applied

1. If `use_forced_photo=True` and `forced_photometry` is not `None`, forced-photometry rows are appended to detections.
2. Rows where `unit != self.unit` are dropped.
3. Rows where `brightness` is `NaN` are dropped.
4. Remaining rows are sorted ascending by `mjd`.
5. `_trim_lightcurve` is applied: a sliding-window O(n) algorithm finds the densest subsequence whose timespan is ≤ `trim_lightcurve_to_n_days` (no-op when `trim_lightcurve_to_n_days is None` or the lightcurve is empty).
6. For each band in `self.bands`, if the band has fewer than `min_length` observations it is excluded from `useful_bands` and therefore from the periodogram input.
7. Observations whose `fid` is not in `useful_bands` are dropped.
8. If the remaining total count is < `min_length`, computation is skipped and all features are emitted as `np.nan`.

### Valid `unit` values

- `"magnitude"` — the extractor mean-subtracts per band inside P4J using a robust weighted median (`robust_loc`); magnitude values are cast to `float32` internally.
- `"diff_flux"` — treated identically numerically; the difference is only in the physical interpretation of the brightness column.

Both units must have non-negative, finite `e_brightness` values for the MHAOV weighting (`w_i = e_i^{-2}`) to be well-defined.

## Output

Every row appended to `astro_object.features` carries `sid` (joined unique survey IDs from `astro_object.detections["sid"]`, sorted and comma-separated) and `version = "1.0.0"`.

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `Multiband_period` | all bands joined (e.g. `"g,r"`) | Best global period (days): `1 / best_freq[0]` where `best_freq[0]` is the top-ranked frequency after coarse + fine-tune grid search. `np.nan` if computation is skipped. |
| `PPE` | all bands joined | Period significance (Pseudo-Peak Entropy): `1 - H / log(100)` where H is the Shannon entropy of the top-100 periodogram values (plus a `1e-2` floor) normalised to sum to 1. Range approximately [0, 1]; higher means more significant. `np.nan` if skipped. |
| `Period_band` | per band (`fid = band`) | Best single-band period (days): `1 / get_best_frequency(band)`, i.e. the coarse-grid index with highest per-band MHAOV power. `np.nan` for any band that did not reach `min_length`. |
| `delta_period` | per band (`fid = band`) | Absolute difference `|Multiband_period - Period_band|` in days. `np.nan` when either parent is `np.nan`. |
| `Power_rate_1_4` | all bands joined | Ratio of periodogram power at period `Multiband_period × 0.25` to power at `Multiband_period`. Only present when `return_power_rates=True`. `np.nan` if the periodogram step returned no valid result. |
| `Power_rate_1_3` | all bands joined | Same ratio at factor `1/3`. |
| `Power_rate_1_2` | all bands joined | Same ratio at factor `0.5`. |
| `Power_rate_2` | all bands joined | Same ratio at factor `2.0`. |
| `Power_rate_3` | all bands joined | Same ratio at factor `3.0`. |
| `Power_rate_4` | all bands joined | Same ratio at factor `4.0`. |

**Sentinel value:** `np.nan` for all features when the lightcurve is too short or a `TypeError` is raised inside P4J. When `return_power_rates=True` but computation failed, all six `Power_rate_*` features are also `np.nan`.

## Underlying library / math

### `P4J.MultiBandPeriodogram` — coarse grid

- **Function:** `MultiBandPeriodogram.set_data(mjds, mags, errs, fids)`
- **Algorithm:** For each band `f`, the class constructs a Cython `MHAOV` object (multi-harmonic AoV with `Nharmonics=1`, `mode=0`). Before passing magnitudes to MHAOV, it subtracts a robust per-band location (error-weighted median via `robust_loc`). All arrays are cast to `float32`.
- **Frequency grid:** `optimal_frequency_grid_evaluation(smallest_period, largest_period, shift)` computes `grid_size = ceil((f_max - f_min) * T / (2 * shift))` where `T` is the lightcurve time span (max − min MJD). `grid_size` is floored at `1_000`. The grid is linearly spaced in frequency between `1/largest_period` and `1/smallest_period`.
- **Multi-band combination:** `_compute_periodogram` evaluates each band's MHAOV separately, then pools them as:

  ```
  per_multiband[k] = (d2_sum * per_sum[k]) / (d1 * max(wvar_sum - per_sum[k], 1e-9))
  ```

  where `d1 = 2 * Nharmonics = 2`, `d2 = N_band - 2*Nharmonics - 1` per band (floored at 0), and `wvar` is the per-band weighted variance. This is an F-statistic formulation of AoV.
- **Returns:** `self.per` (multiband F-statistic array over the frequency grid) and `self.per_single_band` (dict of per-band arrays).

### `P4J.MultiBandPeriodogram` — fine-tune

- **Function:** `optimal_finetune_best_frequencies(times_finer=10.0, n_local_optima=10)`
- **Algorithm:** Finds up to 10 local maxima of `self.per` (simple 3-point comparison), then for each one evaluates the periodogram on a finer grid of width = one coarse step, at resolution `freq_step_coarse / 10`. Updates frequencies in place if the finer grid yields a higher power. Stores the top local optima indices in `self.best_local_optima`, sorted descending by power.
- **`times_finer=10.0` and `n_local_optima=10` are baked in** at the call site in `compute_features_single_object`; they are not constructor arguments.

### `P4J.BasePeriodogram.get_best_frequencies()`

Returns `(self.freq[self.best_local_optima], self.per[self.best_local_optima])` — a pair of arrays of length ≤ `n_local_optima`, sorted by descending multiband power. `best_freq[0]` is the global winner.

### `P4J.BasePeriodogram.get_best_frequency(fid)`

Returns the single frequency index with maximum power in `self.per_single_band[fid]` (argmax over all coarse+fine-tuned frequencies). This is the per-band best frequency, not the multiband one.

### MHAOV algorithm (Cython, compiled)

The `MHAOV` class in `P4J/algorithms/multiharmonic_aov.cpython-310-x86_64-linux-gnu.so` is a Cython extension; no Python source is present in the installed package. Based on the P4J `__init__.py` docstring and the `periodogram` class docstring, MHAOV is described as the *orthogonal multiharmonic AoV periodogram* (reference [7] in the package docstring). For each trial frequency `f`, it fits a truncated Fourier model with `Nharmonics` harmonics to the phased light curve and computes an F-statistic measuring the reduction in weighted variance. With `Nharmonics=1` (the only value used here), this is a 2-parameter sinusoidal fit.

### `lc_classifier.utils.is_sorted`

JIT-compiled via `numba`. Returns `True` if array `a` is non-decreasing. Used in `compute_power_rates` to sort the frequency array before `np.searchsorted` if needed.

## Hardcoded values

| Value | Location | Tunable? | Meaning |
|-------|----------|----------|---------|
| `Nharmonics = 1` | `MultiBandPeriodogram.__init__` (P4J library default) | No — not exposed by `PeriodExtractor` | Number of Fourier harmonics in the MHAOV fit. |
| `mode = 0` | `set_data` in `MultiBandPeriodogram` (P4J) | No | Internal MHAOV mode flag; meaning not documented in available source. |
| `1e-9` | `_compute_periodogram` denominator clamp (P4J) | No | Numerical floor preventing division by zero in F-statistic. |
| `times_finer = 10.0` | `compute_features_single_object` call to `optimal_finetune_best_frequencies` | No | Fine-grid is 10× denser than coarse grid around each local optimum. |
| `n_local_optima = 10` | `compute_features_single_object` call to `optimal_finetune_best_frequencies` | No | Number of local maxima to fine-tune. |
| `entropy_best_n = 100` | `compute_features_single_object` | No | Number of top periodogram values used to compute PPE entropy. |
| `1e-2` | Entropy normalization floor | No | Added to each of the 100 top values before normalising, preventing `log(0)`. This shifts PPE values slightly and caps the maximum possible entropy. |
| `self.factors = [0.25, 1/3, 0.5, 2.0, 3.0, 4.0]` | Constructor | No | Period ratios for `Power_rate_*` features. Correspond to harmonics at ×4, ×3, ×2 and sub-harmonics at ÷2, ÷3, ÷4 of the best period. |
| `grid_size` floor `1_000` | `optimal_frequency_grid_evaluation` (P4J) | No (controlled indirectly by `shift`) | Minimum number of frequency grid points regardless of lightcurve length. |
| `shift = 0.1` | Constructor default | Yes — constructor argument | Nyquist-like oversampling factor for the coarse frequency grid. |

## Important considerations

- **Statefulness.** `self.periodogram_computer` is a single `MultiBandPeriodogram` instance shared across all calls to `compute_features_single_object`. It is fully overwritten by `set_data` on each call, so there is no cross-object contamination, but it is **not thread-safe**. Parallel processing of multiple `AstroObject`s with one `PeriodExtractor` instance will cause race conditions.

- **`float32` dtype.** P4J `set_data` casts `mjds`, `mags`, and `errs` to `float32` unconditionally. MJD values for LSST/ZTF (≈ 58000–70000) lose sub-second precision in `float32` (≈ 4 s resolution at those magnitudes). This can affect period resolution for very short periods.

- **`TypeError` swallowing.** Any `TypeError` raised inside `optimal_frequency_grid_evaluation` or `optimal_finetune_best_frequencies` is caught, logged, and results in all-`NaN` output for that object. Other exception types propagate unhandled.

- **Empty `get_best_frequencies` result.** If `best_freq` is empty (checked with `len(best_freq) == 0`), the extractor logs an error and emits `NaN`s. This can occur if `find_local_maxima` finds zero local optima (e.g. monotone periodogram).

- **Per-band vs. multi-band best frequency.** `Multiband_period` uses the global multi-band F-statistic winner. `Period_band` uses the per-band argmax across the full (coarse + fine-tuned) array, which may select a different frequency index because single-band and multi-band power arrays can disagree on the best peak.

- **`Power_rate_*` interpolation.** `_get_power_ratio` uses `np.searchsorted` with nearest-neighbour rounding (not linear interpolation) to find the power at the desired harmonic frequency. When the desired frequency falls exactly at the grid boundary (< `frequencies[0]` → `i=0`, > `frequencies[-1]` → `i=last`), the boundary value is used silently.

- **`delta_period` can be large.** Because `Multiband_period` uses the multi-band argmax and `Period_band` uses the per-band argmax, `delta_period` is not bounded; large values indicate aliasing or incoherent variability across bands.

- **PPE formula.** `PPE = 1 - H / log(entropy_best_n)` where `H` is the entropy of the normalised top-100 periodogram values. The `1e-2` floor on each value means even a perfectly uniform top-100 distribution gives a slightly sub-zero PPE, and a perfect spike gives a value slightly below 1. The scale is not strictly probabilistic.

- **`trim_lightcurve_to_n_days` algorithm.** The sliding-window implementation counts observations (not unique MJDs), so duplicate MJDs from forced photometry are counted separately. The trimmed window is selected by maximum *count*, not maximum time coverage.

- **Feature naming collision.** Both `Period_band` and `delta_period` are emitted once per band with the same `name` column value, distinguished only by `fid`. Downstream consumers must filter by `fid` to disambiguate.

- **`min_length` is applied twice:** once per-band (to decide which bands join `useful_bands`) and once globally after band filtering. A band with fewer than `min_length` observations is excluded, but it still receives `np.nan` entries for `Period_band` and `delta_period`.

## Cross-references

- **Composites that include this extractor:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `ZTFFeatureExtractor` instantiates `PeriodExtractor` with `unit="magnitude"`, `smallest_period=0.045`, `largest_period=100.0`, `trim_lightcurve_to_n_days=1000.0`, `min_length=15`, `use_forced_photo=True`, `return_power_rates=True`, `shift=0.1`.
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — same parameters as ZTF.
  - `lc_classifier/lc_classifier/features/composites/elasticc.py` — `unit` is a constructor argument (not fixed to `"magnitude"`), `largest_period=50.0`, `trim_lightcurve_to_n_days=500.0`, no explicit `shift` (uses default `0.1`).

- **Downstream consumers of `Multiband_period`:**
  - `lc_classifier/lc_classifier/features/extractors/harmonics_extractor.py` — reads `Multiband_period` from `astro_object.features` to phase-fold the lightcurve for harmonic fitting.
  - `lc_classifier/lc_classifier/features/extractors/folded_kim_extractor.py` — reads `Multiband_period` to phase-fold and compute Kim et al. morphological features.

- **Tests:** `lc_classifier/tests/features/test_period_feature_extractor.py`.
