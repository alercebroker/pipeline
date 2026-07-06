# Reference-MJD extractor features — expose `*_mjd_ref` as official features

**Date:** 2026-07-06
**Status:** implemented + verified on real OID (36028941624528297) — see §5
**Related:** `feature_step/features/offline/FLOW.md`,
`feature_step/features/offline/feature_lut.py`,
`docs/superpowers/specs/2026-06-25-offline-ztf-feature-writer-design.md`.

---

## 1. Problem

Four extractors internally compute a reference MJD (the "time origin" they subtract
before fitting) but never emit it as a feature. Downstream consumers need these
reference epochs to reconstruct absolute times from the fitted model parameters
(`SPM_t0`, `ulens_t0`, `fleet_t0`, TDE decay start). We add them as official
features.

## 2. The four features

| Feature | Extractor | Source value | Band |
|---|---|---|---|
| `SPM_mjd_ref` | `spm_extractor.py` `SPMExtractor` | `mjd_first_detection = np.min(detections["mjd"])` (line 62) | **0** (whole-object) |
| `ulens_mjd_ref` | `ulens_extractor.py` `MicroLensExtractor` | `np.min(observations["mjd"])` in `get_observations` (line 57) | **0** (whole-object) |
| `TDE_mjd_ref` | `tde_extractor.py` `TDETailExtractor` | `t_d = brightest_obs.mjd` per band (line 62) | **per band** (1,2,3…) |
| `fleet_mjd_ref` | `tde_extractor.py` `FleetExtractor` | `first_mjd` per band (line 166) | **per band** (1,2,3…) |

## 3. Decisions (approved)

- **Band 0 convention.** SPM/ulens refs are one value for the whole object, not per
  band. The established whole-object convention in this codebase is `fid=None`
  (see `TimespanExtractor`, `CoordinateExtractor`), which both `fid_mapper_for_db`
  (ZTF) and `fid_mapper_for_db_lsst` map to band `0`.
  **Not** `",".join(bands)` — for ZTF that is `"g,r"` → `12` (a color-pair code),
  which would be wrong.
- **Per-band refs.** TDE/fleet refs are emitted per band with `fid=band` (the
  extractors already loop per band), landing at 1,2,3… via the existing mapper —
  same as their sibling features (`TDE_decay`, `fleet_a`, …).
- **NaN consistency (must match sibling features).** A `*_mjd_ref` is NaN exactly
  when the corresponding fit produced no result — scientifically right (a reference
  epoch is meaningless without a fit to interpret against it) and consistent with
  the extractor's other outputs:
  - `SPM_mjd_ref`: NaN when `len(observations) == 0` (the sole SPM NaN condition).
  - `ulens_mjd_ref` (whole-object): NaN unless ≥1 band fit succeeded (`any_band_fit`).
  - `TDE_mjd_ref` (per band): NaN in the `< 2` branch; `t_d` on success.
  - `fleet_mjd_ref` (per band): NaN in the `< 4` branch AND `except RuntimeError`;
    `first_mjd` only on a successful fit.
  NaN values drop at DB-prep via the existing `value.notna()` filter.
- **Version bumps** (output contract changed): SPM `1.0.1→1.0.2`,
  ulens `1.0.2→1.0.3`, TDETail `1.0.1→1.0.2`, Fleet `1.0.2→1.0.3`.
- **Offline LUT ordering.** `feature_lut.py` is re-derived in **extractor
  (natural) order** — the order features come out of `ZTFFeatureExtractor` — not
  alphabetical. The generator (`offline_generate_feature_lut.py`) is changed to
  preserve first-occurrence emission order instead of `sorted()`. Safe to renumber:
  the DB `feature_name_lut` (sid=0) is **not yet seeded**.
- **Seed SQL.** `feature_lut.py` gains a `render_seed_sql()` (mirroring
  `classifier_taxonomy_lut.render_seed_sql`) as the single source; regenerate
  `ztf_feature_luts_seed.sql` from it.

## 4. Verification (real OID, no synthetic data)

Bundled `lc_classifier.examples` are broken (AstroObject now requires an `oid`
column the fixtures lack). Test end-to-end against the live DB instead:

```
conda run --no-capture-output -n training_py310 python \
    feature_step/scripts/offline_compute_features.py --oid 36028941624528297
```

Confirm in the resulting feature frame:
1. `SPM_mjd_ref` and `ulens_mjd_ref` present, one row each, `band == 0`.
2. `TDE_mjd_ref` and `fleet_mjd_ref` present per band, `band ∈ {1,2}` for ZTF g/r.
3. Values are plausible MJDs (~5.8e4–6.0e4), matching each extractor's origin.

The same run's `ao.features["name"]` (pre-NaN-filter) yields the extractor order
used to regenerate `feature_lut.py`.

## 5. Result (verified 2026-07-06, oid 36028941624528297)

Generator produced 127 names in extractor order; new ids: `SPM_mjd_ref`=90,
`TDE_mjd_ref`=94, `fleet_mjd_ref`=100, `ulens_mjd_ref`=118 — each immediately
after its extractor's last feature. `compute_db_features` on the real OID:

| feature | band | value |
|---|---|---|
| `SPM_mjd_ref` | 0 | 58476.54 |
| `ulens_mjd_ref` | 0 | 58476.54 |
| `TDE_mjd_ref` | 1 (g) / 2 (r) | 60760.45 / 60639.56 |
| `fleet_mjd_ref` | 1/2 (raw), dropped | NaN (Fleet did not fit this OID — all `fleet_*` NaN; emission verified in raw `ao.features` with `fid=g/r`) |

`feature_lut.render_seed_sql()` regenerated `ztf_feature_luts_seed.sql` (127
names). Tests: `tests/unittest/test_offline_feature_lut.py` updated for the new
ordering invariant + `render_seed_sql`; 16 offline tests pass. DB `feature_name_lut`
(sid=0) still **not seeded** — apply `ztf_feature_luts_seed.sql` when ready.
