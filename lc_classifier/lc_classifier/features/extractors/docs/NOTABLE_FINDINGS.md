# Notable Findings Across Feature Extractors

Cross-extractor summary of behaviours that are surprising, undocumented, or load-bearing in non-obvious ways. Compiled from the per-extractor reference docs in this directory. Read this before relying on any feature numerically — several values are not what their names suggest.

Each entry links to the extractor's own doc for the full context.

---

## Process-global side effects

These are mutations that affect the entire Python process, not just the extractor that owns them.

- **`tde_extractor.py` and `ulens_extractor.py`** both call `jax.config.update("jax_enable_x64", True)` at **module import time**. Importing either extractor (e.g. via `from lc_classifier.features.extractors import ...`) silently promotes all JAX computations in the process to 64-bit for its remaining lifetime. Anything else in the process that assumed 32-bit JAX will run slower and may produce different numerics. See [tde_extractor.md](tde_extractor.md), [ulens_extractor.md](ulens_extractor.md).

---

## Silently wrong units, scales, or pivots

Values whose physical interpretation is different from what their name suggests.

- **`SPMExtractor` rescales flux by `× 0.001` before fitting** (legacy µJy → mJy conversion). `SPM_A` is therefore in a rescaled mJy-equivalent unit, not the pipeline's native µJy. Any downstream consumer treating `SPM_A` as a physical amplitude in µJy is wrong by 1000×. See [spm_extractor.md](spm_extractor.md).

- **`SPMExtractor` cosmological correction pivots at `z = 0.3`, not `z = 0`**. Flux at the host redshift is normalised relative to `z = 0.3`, so `SPM_A` is *not* a redshift-corrected intrinsic luminosity proxy — it systematically depends on whether the object is above or below the pivot. See [spm_extractor.md](spm_extractor.md).

- **`MicroLensExtractor` reduced χ² uses `N − 4` despite the model having 5 free parameters** (`u0`, `tE`, `fs`, `t0`, `mag_0`). `ulens_chi` is systematically lower than the textbook reduced χ² definition. Intent is undocumented. See [ulens_extractor.md](ulens_extractor.md).

- **`PeriodExtractor` / `P4J.set_data` silently casts time arrays to `float32`**. At ZTF/LSST MJDs (~58000–70000) this gives only ~4 s of time resolution, degrading period accuracy near the `smallest_period = 0.045 d` lower bound. See [period_extractor.md](period_extractor.md).

- **`CoordinateExtractor` averages RA directly in degrees**, which is wrong near the 0°/360° wrap. The source explicitly comments that the unit-vector-average fix is known and intentionally skipped for performance — any object whose detections straddle that boundary will silently produce incorrect Cartesian coordinates. See [coordinate_extractor.md](coordinate_extractor.md).

---

## Order- or context-dependent features

Features that depend on other extractors having run first, or on the order of operations inside the extractor.

- **`turbofats_extractor.py`**: the feature-name list is load-bearing. `SF_ML_gamma` reads `shared_data["g_sf"]` deposited by `SF_ML_amplitude` — reordering the list or removing `SF_ML_amplitude` raises an exception that is then **swallowed to `NaN`** with no loud signal. See [turbofats_extractor.md](turbofats_extractor.md).

- **`FoldedKimExtractor` and `HarmonicsExtractor` both require a preceding `PeriodExtractor`** to have written `Multiband_period` into `astro_object.features`. There is no soft-fail path: a missing period raises an uncaught exception. See [folded_kim_extractor.md](folded_kim_extractor.md), [harmonics_extractor.md](harmonics_extractor.md).

- **`FoldedKimExtractor` folds over a `2 × period` window**, not one full cycle. This is a silent deviation from the turbofats `Psi_CS_v2`/`Psi_eta_v2` definitions and from Kim et al. — values are not numerically comparable to the turbofats equivalents with the same name. See [folded_kim_extractor.md](folded_kim_extractor.md).

- **`SNExtractor` derives `first_detection_mjd` across all bands combined, not per-band**. A band with an earlier first detection than other bands ends up with zero "before" epochs and 5/10 of its features become `NaN`, even when substantial pre-detection baseline data exist in that band. See [sn_extractor.md](sn_extractor.md).

- **`AllwiseColorsFeatureExtractor`**: the consecutive AllWISE colors (`W1-W2`, `W2-W3`, `W3-W4`) are gated on the optical detection count after preprocessing. They are forced to `NaN` whenever no optical detections survive the filter, even when all four AllWISE magnitudes are present in metadata — an undocumented coupling between the IR-only colors and the optical path. See [allwise_colors_feature_extractor.md](allwise_colors_feature_extractor.md).

- **`HarmonicsExtractor` reads the period via `period["value"][0]` (label-based)** rather than `.iloc[0]`. After a DataFrame concat where the integer label `0` is absent or duplicated, this silently returns the wrong value or raises a `KeyError`. See [harmonics_extractor.md](harmonics_extractor.md).

---

## Silent failure modes

Errors that should be loud but become `NaN`, default values, or success-looking output.

- **`GPDRWExtractor` never checks `sol.success`** after `scipy.optimize.minimize`. Non-converged fits emit numeric `GP_DRW_sigma` / `GP_DRW_tau` values indistinguishable from converged ones in the feature table. See [gp_drw_extractor.md](gp_drw_extractor.md).

- **`MHPSExtractor` silently drops `non_zero` and `pn_flag` outputs** when `t1 != 100.0` or `t2 != 10.0`, despite the library still computing them. Pure information loss with no warning. See [mhps_extractor.md](mhps_extractor.md).

- **`MHPSExtractor`'s `self.dt = 3.0` is stored but never passed to `mhps.statistics` / `mhps.flux_statistics`**. The library applies its own internal default for the convolution time step. The constructor argument is misleading. See [mhps_extractor.md](mhps_extractor.md).

- **`turbofats.IAR_phi` emits `1e10` (not `NaN`) when `|phi| >= 1`** internally, before the outer exception catcher converts the bad value to `NaN`. Inspecting raw turbofats output directly (outside the extractor wrapper) will show sentinel values, not NaNs. See [turbofats_extractor.md](turbofats_extractor.md).

- **`TimespanExtractor` has no guard against an empty `detections` DataFrame**. Pandas silently returns `NaN` for `min`/`max`; `Timespan` becomes `NaN` with no sentinel path. See [timespan_extractor.md](timespan_extractor.md).

---

## Hardcoded thresholds and sentinels

Magic numbers that materially shape feature values and are not exposed as constructor arguments.

- **`e_brightness < 1.0` magnitude-error cap** — hardcoded in `preprocess_detections` and reused across multiple extractors (`AllwiseColorsFeatureExtractor`, `ColorFeatureExtractor`, etc.). No override. Sparse-band detections with reasonable but loose uncertainty (e.g. 1.2 mag) are silently dropped. See [allwise_colors_feature_extractor.md](allwise_colors_feature_extractor.md), [color_feature_extractor.md](color_feature_extractor.md).

- **`ColorFeatureExtractor` flux-ratio denominator adds `+ 1` µJy** (`band_p90_list[i+1] + 1`). For faint objects with p90 fluxes in the single-digit µJy range, this hardcoded additive bias compresses color ratios toward 1. Not tunable. See [color_feature_extractor.md](color_feature_extractor.md).

- **`ColorFeatureExtractor` differential-flux path has no error quality filter at all** (unlike the magnitude path which uses the 1.0 mag cap). `_mean` / `_max` features without the `_corr` suffix can include arbitrarily low-SNR detections. See [color_feature_extractor.md](color_feature_extractor.md).

- **`PanStarrsFeatureExtractor` non-detection sentinel is hardcoded at `< -30.0` mag** (PanSTARRS/ZTF convention). Catalogues that flag non-detections with `NaN` or `99.0` will silently produce finite colors. See [panstarrs_feature_extractor.md](panstarrs_feature_extractor.md).

- **`ReferenceFeatureExtractor` clips `distnr` at 5 arcsec**, justified in a source comment as a ZTF training-set limitation. Detections beyond that radius are silently discarded. See [reference_feature_extractor.md](reference_feature_extractor.md).

- **`HarmonicsExtractor` uses two asymmetric error floors**: an unconditional `10⁻²` added to `e_brightness` for the fit weights, and a separate `error_tol` (`1e-2` for mags, `1e-3` for `diff_flux`) added only to the χ² denominator. For `unit="diff_flux"` the fit is regularised with a floor 10× larger than the χ² floor. See [harmonics_extractor.md](harmonics_extractor.md).

- **`PeriodExtractor` PPE has a `1e-2` additive floor in its entropy**, so PPE is never exactly 0 or 1, and a uniform top-100 periodogram can produce a slightly negative PPE. The feature's range is not a clean `[0, 1]` probability. See [period_extractor.md](period_extractor.md).

- **`turbofats.Meanvariance` (`std / mean`) has no guard against near-zero mean**, which is a real failure mode for `unit="diff_flux"` (used in the ELAsTiCC composite) on faint sources centred near zero flux. See [turbofats_extractor.md](turbofats_extractor.md).

- **`TDETailExtractor` calls `np.abs(brightness)` before converting to magnitude**, so negative-flux difference epochs (host-galaxy subtraction residuals, pre-peak baselines) are treated as same-magnitude positive detections. A large-absolute-value negative epoch can misplace `t_d` (the peak epoch). See [tde_extractor.md](tde_extractor.md).

---

## Dead code / misleading scaffolding

Code that looks load-bearing but isn't, or imports/state that suggests behaviour the extractor doesn't have.

- **`DummyExtractor` never sets `self.version`**. Any caller that reads `extractor.version` (production extractors all do, when stamping the `version` column in `astro_object.features`) will hit `AttributeError`. `lsst.py` imports `DummyExtractor` at module level but never instantiates it — dead import in production code. See [dummy_extractor.md](dummy_extractor.md).

- **`PanStarrsFeatureExtractor` imports `functools.lru_cache` but never uses it** — suggests a copy-paste artifact from another extractor. See [panstarrs_feature_extractor.md](panstarrs_feature_extractor.md).

- **`GPDRWExtractor` instantiates `celerite2.RealTerm(a=1.0, c=10.0)` once per band**, but those values are overwritten on the first `neg_log_like` call — constructor values have zero effect. (Note: celerite2 itself warns that `RealTerm` "will generally behave poorly" and recommends `SHOTerm` for most users — the choice here is a deliberate strict-DRW model decision.) See [gp_drw_extractor.md](gp_drw_extractor.md).

- **`SNExtractor`'s `forced_photometry = forced_photometry[forced_photometry["unit"] == self.unit]`** sits inside the band loop and re-applies the same filter on every iteration — harmless but redundant after the first pass. See [sn_extractor.md](sn_extractor.md).

---

## Unguarded type / data assumptions

Latent crashes that depend on input shape rather than physical content.

- **`ReferenceFeatureExtractor`'s `rfid`-to-`int` cast (line 62)** has no exception guard. Non-integer float values raise `ValueError` and propagate uncaught, while every other NaN/empty edge case in the extractor emits `np.nan` gracefully — an asymmetric failure mode. See [reference_feature_extractor.md](reference_feature_extractor.md).

- **`CoordinateExtractor` has no NaN guard or minimum-detection count**. A single all-NaN `ra`/`dec` column propagates `NaN` through to features with no deliberate sentinel-emission path. See [coordinate_extractor.md](coordinate_extractor.md).

- **`HarmonicsExtractor` `period["value"][0]` is a label-based lookup, not positional**. Works on a fresh DataFrame; breaks silently or loudly after a concat. See [harmonics_extractor.md](harmonics_extractor.md).

---

## Schema / structural surprises

Output shapes that diverge from per-band conventions used elsewhere.

- **`TimespanExtractor` emits its row with `fid = None` and `sid` as a comma-joined sorted *string*** (e.g. `"LSST,ZTF"`), not the usual bare identifier or list. Consumers iterating `astro_object.features` and assuming `sid` is a single survey identifier will mis-handle this row. See [timespan_extractor.md](timespan_extractor.md).

- **`TurboFatsExtractor` suffixes every feature name with `_flux` when `unit="diff_flux"`** but emits the same name otherwise — a single feature column can mean different things across runs depending on the configured unit. See [turbofats_extractor.md](turbofats_extractor.md).
