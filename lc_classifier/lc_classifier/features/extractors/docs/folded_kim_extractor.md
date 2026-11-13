# FoldedKimExtractor

Computes two period-folded variability statistics, `Psi_CS` and `Psi_eta`, per photometric band using a period value that must already be present in `astro_object.features`.

- **Source:** `lc_classifier/lc_classifier/features/extractors/folded_kim_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `numpy`, `pandas` (no additional third-party scientific library; the statistics are computed inline)

## Purpose / Meaning

`Psi_CS` and `Psi_eta` characterise how well the light curve flux varies coherently when folded at the best-fit period. A light curve folded on the correct period will show an ordered, smooth brightness sequence; these two statistics quantify that order. `Psi_CS` is a cumulative-sum range statistic (large when folded brightness drifts monotonically), and `Psi_eta` is the ratio of consecutive-difference variance to total variance (small when folded points change smoothly). Both are useful discriminators between periodic and non-periodic variables and originate in Kim et al. (2014) — see the turbofats implementations `Psi_CS_v2` and `Psi_eta_v2` in `lc_classifier/lc_classifier/features/turbofats/FeatureFunctionLib.py` for the original per-band formulations this extractor is based on.

## Input

### Constructor arguments

| Name    | Type        | Default | Meaning |
|---------|-------------|---------|---------|
| `bands` | `List[str]` | —       | Photometric band identifiers to compute features for (e.g. `["g", "r"]`). Required. |
| `unit`  | `str`       | —       | Unit of the `brightness` column to accept. Must be `"magnitude"` or `"diff_flux"`. Required. Raises `ValueError` otherwise. |

### `AstroObject` fields read

- `features` — must already contain a row with `name == "Multiband_period"`. The value at `features[features["name"] == "Multiband_period"]["value"][0]` is used as the period. An `Exception` is raised if this row is absent.
- `detections` — columns consumed: `unit`, `brightness`, `fid`, `mjd`, `sid`.

### Pre-filtering applied

1. Rows in `detections` where `unit != self.unit` are dropped.
2. Rows where `brightness` is `NaN` are dropped.
3. Per band, if `len(band_detections) <= 2` or `period` is `NaN`, sentinel values are emitted rather than computing.

### Valid `unit` values

`"magnitude"` and `"diff_flux"`. Any other string causes the constructor to raise `ValueError` before any computation occurs.

## Output

Every row appended to `astro_object.features`. Columns: `name`, `value`, `fid`, `sid`, `version`.

| Feature `name` | `fid` scope | Meaning |
|----------------|-------------|---------|
| `Psi_CS`       | per band    | Range of the normalised cumulative sum of mean-subtracted brightness values sorted by folded phase. Defined as `max(s) - min(s)` where `s = cumsum(m_i - mean) / (N * sigma)`. Larger values indicate stronger periodic signal. |
| `Psi_eta`      | per band    | Consecutive-difference variance ratio on the phase-folded, sorted brightness sequence: `sum((m_{i+1} - m_i)^2) / ((N-1) * sigma^2)`. Smaller values indicate a smoother periodic signal. |

**Sentinel value:** `np.nan` is emitted for both features when:
- the band has 2 or fewer detections after filtering, or
- `period` is `np.nan`, or
- `sigma == 0.0` (all brightness values in the band are identical after folding).

## Underlying library / math

No third-party scientific library beyond `numpy` is called. All computation is inline.

**Phase folding:**

```
folded_time = mod(time, 2 * period) / (2 * period)
```

The light curve is folded over a window of `2 * period` (two full cycles), not one. This doubles the effective phase window compared to a standard single-period fold. The reason for this choice is not documented in the source; it differs from the turbofats `Psi_CS_v2` / `Psi_eta_v2` implementations (which use `new_time_v2` computed by `PeriodLS_v2`) and from the original Kim et al. formulation that folds over a single period.

**`Psi_CS` formula:**

```
s_i = cumsum(m_i - mean(m)) / (N * std(m))
Psi_CS = max(s) - min(s)
```

**`Psi_eta` formula:**

```
Psi_eta = sum((m_{i+1} - m_i)^2) / ((N - 1) * std(m)^2)
```

where indices run over the brightness array sorted by `folded_time` and `std` uses NumPy's default `ddof=0`.

**Note on `std` vs `var`:** `Psi_CS` normalises by `std(m)` while `Psi_eta` normalises by `std(m)^2`. The turbofats `Psi_eta_v2` uses `np.var` (`ddof=0`) directly. Both are equivalent; the extractor computes `sigma = np.std(...)` once and reuses `sigma**2` for the `Psi_eta` denominator.

## Hardcoded values

- `2 * period` — the fold window. Baked in; not tunable via the constructor.
- `ddof=0` — `np.std` default; affects normalisation. Not tunable.
- `period` is read from `astro_object.features` at index `[0]` of the filtered series — if there are multiple `Multiband_period` rows, only the first is used (no check for uniqueness).

## Important considerations

- **Execution order dependency:** `FoldedKimExtractor` must run **after** a `PeriodExtractor` (or any extractor that writes `Multiband_period` to `astro_object.features`). In all known composites (`ZTFFeatureExtractor`, `ELAsTiCCFeatureExtractor`, `LSSTFeatureExtractor`) it is placed after `PeriodExtractor` in the `extractors` list. If run standalone or out of order, it raises a bare `Exception` with the message `"Folded Kim extractor was not provided with period data"`.
- **Fold window is `2 * period`, not `period`:** The standard Kim et al. definition folds over one period. The doubled window means the phase axis spans `[0, 1)` but covers two physical cycles. This is an intentional or accidental deviation from the turbofats reference implementation; downstream consumers should be aware that the feature values are not directly comparable to turbofats `Psi_CS_v2` / `Psi_eta_v2`.
- **`sigma == 0` guard:** When all brightness values are equal after filtering, both features are set to `np.nan` rather than dividing by zero.
- **No error handling:** There is no `try/except` block. Any unexpected runtime error (e.g. malformed `detections` DataFrame, missing columns) will propagate as an unhandled exception.
- **`sid` aggregation:** All unique `sid` values from the filtered detections are sorted and joined with `","` into a single string. Per-band `sid` is not tracked separately.
- **In-place mutation:** `astro_object.features` is replaced by a `pd.concat` result on every call. If called more than once on the same object, duplicate `Psi_CS` / `Psi_eta` rows will accumulate.
- **`detections` is not sorted by `mjd`:** The extractor does not sort `detections` before folding. It relies on NumPy's `argsort(folded_time)` to sort brightness by phase, so the original order of rows in `detections` does not matter.
- **`e_brightness` is never read:** Measurement uncertainties are not used in any of the statistics.

## Cross-references

- **Composites that include this extractor:**
  - `lc_classifier/lc_classifier/features/composites/ztf.py` — `ZTFFeatureExtractor`, with `unit="magnitude"`.
  - `lc_classifier/lc_classifier/features/composites/elasticc.py` — `ELAsTiCCFeatureExtractor`, with `unit` passed from composite constructor.
  - `lc_classifier/lc_classifier/features/composites/lsst.py` — `LSSTFeatureExtractor`, with `unit="magnitude"`.
- **Extractor that must run first (produces `Multiband_period`):**
  - `lc_classifier/lc_classifier/features/extractors/period_extractor.py` — `PeriodExtractor`.
- **Extractor that also reads `Multiband_period` and runs after:**
  - `lc_classifier/lc_classifier/features/extractors/harmonics_extractor.py` — `HarmonicsExtractor`.
- **Analogous turbofats implementations (not called by this extractor):**
  - `Psi_CS_v2` and `Psi_eta_v2` in `lc_classifier/lc_classifier/features/turbofats/FeatureFunctionLib.py`.
- **Test file:**
  - `lc_classifier/tests/features/test_folded_kim_feature_extractor.py`.
