# MHPSExtractor

Computes Mexican Hat Power Spectrum (MHPS) features — integrated power at two timescales and their ratio — for each photometric band of a light curve.

- **Source:** `lc_classifier/lc_classifier/features/extractors/mhps_extractor.py`
- **Version:** `1.0.0`
- **Base class:** `FeatureExtractor`
- **External libs:** `mhps` (version 0.1.1, Cython extension; paper: https://arxiv.org/abs/1207.5825)

## Purpose / Meaning

The Mexican Hat Power Spectrum measures the integrated power of a light curve at two user-defined timescales (`t1`, `t2`) by convolving the light curve with Mexican Hat (Ricker wavelet) kernels of width `t1` (low-frequency / long-timescale) and `t2` (high-frequency / short-timescale). The ratio `MHPS_low / MHPS_high` discriminates between sources whose variability is dominated by long-timescale structure (e.g. long-period variables, active galactic nuclei) versus short-timescale structure (e.g. fast transients, eclipsing binaries). The `MHPS_PN_flag` signals when Poisson noise dominates the convolved power, which indicates either a photometrically quiet source or insufficient data.

## Input

### Constructor arguments

| Name   | Type        | Default   | Meaning |
|--------|-------------|-----------|---------|
| `bands` | `List[str]` | (required) | Photometric band identifiers to process (e.g. `["g", "r"]` for ZTF, `["u","g","r","i","z","y"]` for LSST). |
| `unit`  | `str`       | (required) | Brightness unit; must be `"magnitude"` or `"diff_flux"`. Raises `ValueError` if any other value is passed. |
| `t1`   | `float`     | `100.0`   | Long-timescale kernel half-width in days (low-frequency probe). |
| `t2`   | `float`     | `10.0`    | Short-timescale kernel half-width in days (high-frequency probe). |
| `dt`   | `float`     | `3.0`     | Stored as `self.dt` but **never passed** to `mhps.statistics` or `mhps.flux_statistics`. The library uses its own internal default for the convolution time step when `dt` is omitted. This parameter is dead code in the current implementation. |

### `AstroObject` fields read

- `detections` — columns: `mjd`, `brightness`, `e_brightness`, `fid`, `unit`, `sid`.

`forced_photometry` is not read by this extractor.

### Pre-filtering applied

1. Rows where `detections["unit"] != self.unit` are dropped.
2. Rows where `detections["brightness"]` is `NaN` are dropped.
3. Per band: rows where `detections["fid"] != band` are dropped.
4. Per band: detections are sorted ascending by `mjd` (in-place on a `.copy()`).

### Valid `unit` values

- `"magnitude"`: arrays are cast to `np.double` before calling `mhps.statistics`.
- `"diff_flux"`: arrays are cast to `np.float32` before calling `mhps.flux_statistics`. The compiled extension enforces `float32` for `flux_statistics`; passing `float64` raises a `ValueError: Buffer dtype mismatch`.

## Output

Every row appended to `astro_object.features` has columns `name`, `value`, `fid`, `sid`, `version`.

`sid` is set to a comma-joined, sorted string of all unique `sid` values present in the (unit-filtered) detections — not band-specific.

### When `t1 == 100.0` and `t2 == 10.0` (default case)

| Feature `name`    | `fid` scope | Meaning |
|-------------------|-------------|---------|
| `MHPS_ratio`      | per band    | `MHPS_low / MHPS_high`: ratio of integrated power at low frequency to high frequency. Values >> 1 indicate slow variability; values << 1 indicate fast variability. |
| `MHPS_low`        | per band    | Integrated power (sigma-clipped) at the long-timescale kernel (`t1`). Units depend on the brightness unit. |
| `MHPS_high`       | per band    | Integrated power (sigma-clipped) at the short-timescale kernel (`t2`). |
| `MHPS_non_zero`   | per band    | Count of time bins with non-zero power in the convolved spectrum (proxy for number of usable epochs after internal filtering). |
| `MHPS_PN_flag`    | per band    | Integer flag: `0` = signal dominates; `1` = Poisson noise dominates (or insufficient data relative to kernel scale). |

### When `t1 != 100.0` or `t2 != 10.0` (non-default timescales)

Only three features are emitted per band; `non_zero` and `pn_flag` are silently discarded:

| Feature `name`                      | `fid` scope | Meaning |
|-------------------------------------|-------------|---------|
| `MHPS_ratio_{int(t1)}_{int(t2)}`    | per band    | Power ratio, as above, with timescale values embedded in the name (e.g. `MHPS_ratio_365_30`). |
| `MHPS_low_{int(t1)}`                | per band    | Low-frequency integrated power (e.g. `MHPS_low_365`). |
| `MHPS_high_{int(t2)}`               | per band    | High-frequency integrated power (e.g. `MHPS_high_30`). |

### Sentinel values

All five outputs from `mhps.statistics` / `mhps.flux_statistics` are `np.nan` when:
- The band has zero detections after pre-filtering (the extractor short-circuits and assigns `np.nan` to all five variables before calling the library).
- The library itself returns all-`nan`: observed when the number of detections is too small relative to the kernel half-width, or when the time baseline is shorter than the kernel scale. From empirical testing with two observations over a 1-day span against `t1=100, t2=10`, all outputs are `nan`.

## Underlying library / math

### `mhps.statistics` (magnitude mode)

- **Function:** `mhps.statistics(mag, magerr, time, t1, t2[, dt[, mag0[, epsilon]]])`
- **Algorithm:** Mexican Hat Power Spectrum — convolves the magnitude time series with two Mexican Hat (Ricker wavelet) kernels of half-widths `t1` (days) and `t2` (days) on a uniform time grid, applies sigma-clipping to the resulting power spectrum, and integrates the clipped power. The reference paper is Elorrieta et al. 2013, cited in the package metadata as https://arxiv.org/abs/1207.5825.
- **Input dtypes:** `mag`, `magerr`, `time` must be `np.double` (float64). Mismatched sizes raise `ValueError: mag, magerr and time array should be the same size`.
- **Returns:** 5-tuple `(ratio, Ik2_low_freq, Ik2_high_freq, non_zero, PN_flag)`.
  - `ratio` = `Ik2_low_freq / Ik2_high_freq` (confirmed by direct calculation).
  - `Ik2_low_freq`: sigma-clipped integrated power at scale `t1`.
  - `Ik2_high_freq`: sigma-clipped integrated power at scale `t2`.
  - `non_zero`: integer count of time bins with non-zero power (in practice equals the number of input observations when the baseline is sufficient).
  - `PN_flag`: `0` or `1`; `1` signals Poisson-noise-dominated regime.
- **Library defaults inherited:** When `dt`, `mag0`, and `epsilon` are omitted, the library applies its own internal defaults. The extractor never passes `dt` despite storing `self.dt`.

### `mhps.flux_statistics` (diff_flux mode)

- **Function:** `mhps.flux_statistics(flux, fluxerr, time, t1, t2)`
- **Algorithm:** Same MHPS algorithm adapted for flux (difference-image photometry). Internally calls `_flux_statistics32`.
- **Input dtypes:** `flux`, `fluxerr`, `time` must be `np.float32`. Passing `float64` raises `ValueError: Buffer dtype mismatch`.
- **Returns:** same 5-tuple `(ratio, Ik2_low_freq, Ik2_high_freq, non_zero, PN_flag)`.
- **Error message on size mismatch:** `flux, fluxerr and time array should be the same size`.

### `mhps.sigma_clip`

Used internally by the library (not called directly by the extractor). Operates on `float32` arrays; passing `float64` raises `ValueError`.

## Hardcoded values

- `valid_units = ["magnitude", "diff_flux"]` — constructor guard; baked in.
- `np.double` cast for `unit == "magnitude"` — baked in; required by the C extension.
- `np.float32` cast for `unit == "diff_flux"` — baked in; required by the C extension.
- Feature naming branch `if self.t1 == 100.0 and self.t2 == 10.0` — the default-timescale names (`MHPS_ratio`, `MHPS_low`, `MHPS_high`, `MHPS_non_zero`, `MHPS_PN_flag`) are emitted only for the exact floating-point values `100.0` and `10.0`. Any deviation (e.g. `t1=100.1`) silently switches to the parametric naming scheme and drops `non_zero` and `pn_flag` from the output.
- `self.dt = 3.0` — stored but never forwarded to the library; effectively dead code.

## Important considerations

- **`dt` is never used.** The constructor accepts and stores `dt` (default `3.0`) with a `# TODO: check extra params of mhps.statistics` comment, but neither `mhps.statistics` nor `mhps.flux_statistics` is called with it. The library silently applies its own internal default for the time-grid step size.
- **`non_zero` and `pn_flag` are silently dropped for non-default timescales.** When `t1 != 100.0` or `t2 != 10.0`, the library still returns these values but the extractor discards them without warning.
- **Empty band short-circuit.** If a band has no detections after filtering, the extractor sets all five outputs to `np.nan` and still appends feature rows. This means downstream consumers always receive exactly 5 (or 3) rows per band; they should check for `NaN` rather than the absence of a row.
- **`sid` aggregation.** `sid` is derived from all unit-filtered detections across all bands, not per-band. All feature rows for the object share the same `sid` string.
- **Dtype traps.** `mhps.flux_statistics` will raise `ValueError` at runtime if `brightness` or `e_brightness` contains values that cannot be safely cast to `float32` (overflow). The extractor does not guard against this.
- **Time baseline vs. kernel scale.** Results are `nan` when the observation time span is shorter than `t1`. With `t1=365`, a light curve shorter than ~1 year is likely to yield all-`nan` output for the low-frequency features.
- **No `min_length` guard.** Unlike some other extractors, `MHPSExtractor` does not enforce a minimum number of observations; the library itself returns `nan` for very short series.
- **Stateless between calls.** The extractor holds no mutable state from one `compute_features_single_object` call to the next; it is safe to reuse across objects.

## Cross-references

- **Composites that include this extractor:**
  - `ZTFFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/ztf.py`): instantiates `MHPSExtractor(bands, unit="diff_flux")` (defaults `t1=100, t2=10`) and `MHPSExtractor(bands, unit="diff_flux", t1=365.0, t2=30.0)`.
  - `LSSTFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/lsst.py`): same two instances as ZTF.
  - `ElasticcFeatureExtractor` (`lc_classifier/lc_classifier/features/composites/elasticc.py`): instantiates a single `MHPSExtractor(bands, unit)` with default timescales.
- **Other extractors reading the same fields:** any extractor that reads `detections["brightness"]`, `detections["mjd"]`, `detections["e_brightness"]`, or `detections["fid"]` — e.g. `TurboFatsExtractor`, `HarmonicsExtractor`, `FoldedKimExtractor`.
- **Consumers of emitted feature names:** no other extractor in the repo reads `MHPS_*` features from `astro_object.features`. These features feed directly into downstream classification models.
