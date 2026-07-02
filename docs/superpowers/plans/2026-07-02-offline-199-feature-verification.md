# Offline 199-Feature Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the offline ZTF pipeline emits every one of the deployed BHRF (squidward 2.1.0) model's 199 band-suffixed feature names, so `SquidwardFeaturesClassifier.predict` cannot `KeyError` on `features[self.feature_list]`.

**Architecture:** A checked-in authority module pins the 199 expected names (with md5 provenance). A pure verification module reproduces the exact column set `predict()` selects on — `parse_output` → `None→NaN` → `RandomForestPreprocessor.preprocess_features` → `.columns` — and diffs it against the expected 199. A CLI script runs the diff over a diverse handful of real oids (primary check, no model load) and, with `--smoke`, loads the real model once to confirm the end-to-end `predict()` path does not raise and that `model.feature_list` matches the pinned constant.

**Tech Stack:** Python 3.10, pandas, numpy, pytest; the real `feature_step` parser (`features.utils.parsers.parse_output`) and `alerce_classifiers` RF preprocessor; conda env `training_py310` (needed because feature extraction imports `lc_classifier`, and `--smoke` imports `imblearn`).

---

## Why this is a hard contract (context for the implementer)

`HierarchicalRandomForestClassifier.classify_batch` (in `alerce_classifiers/alerce_classifiers/classifiers/hierarchical_random_forest.py`) does:

```python
features = self.preprocessor.preprocess_features(features)   # RandomForestPreprocessor
features_np = features[self.feature_list].values             # STRICT column selection
```

`RandomForestPreprocessor.preprocess_features` (`.../classifiers/preprocess.py`) fills NaN **values** with `-999.0` and renames columns (`-`→`_`, band endings `g`→`_1`, `r`→`_2`, `g,r`→`_12`), but **never adds missing columns**. So a single missing name in `feature_list` ⇒ `KeyError` at predict time — not a silent degrade. This makes full 199 coverage a prerequisite for *any* offline BHRF classification.

Note the namespaces already line up: `parse_output` band-suffixes and applies `-`→`_`, and `RandomForestPreprocessor._preprocess_feature_names` is idempotent on that output. The open question is purely whether the extractor emits every (name, band) row for a given object, and whether that is object-dependent — hence the diverse sample.

## File Structure

- **Create** `feature_step/features/offline/model_feature_list.py` — pins the 199 expected names + model provenance (version/md5/size). No heavy imports.
- **Create** `feature_step/features/offline/model_features.py` — pure verification logic: `predict_input_columns(ao, message)` and `diff_feature_coverage(produced, expected)`.
- **Create** `feature_step/scripts/offline_verify_model_features.py` — CLI: name-diff over sample oids (default) + `--smoke` end-to-end confirmation.
- **Create** `feature_step/tests/unittest/test_offline_model_features.py` — unit tests for the two pure functions + the constant.
- **Modify** `docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md` — close the "Pending: confirm offline features cover the model's 199" section; reframe as a hard prerequisite.
- **Modify** `feature_step/features/offline/FLOW.md` — flip the feature-coverage status line.

---

## Task 1: Pin the 199 expected feature names (authority module)

**Files:**
- Create: `feature_step/features/offline/model_feature_list.py`
- Test: `feature_step/tests/unittest/test_offline_model_features.py`

- [ ] **Step 1: Write the failing test**

Create `feature_step/tests/unittest/test_offline_model_features.py`:

```python
"""Tests for the offline 199-feature verification (model_feature_list + model_features)."""
from features.offline.model_feature_list import MODEL_FEATURE_LIST, MODEL_VERSION, MODEL_MD5


def test_model_feature_list_has_199_unique_names():
    assert len(MODEL_FEATURE_LIST) == 199
    assert len(set(MODEL_FEATURE_LIST)) == 199


def test_model_provenance_pins_deployed_artifact():
    assert MODEL_VERSION == "2.1.0"
    assert MODEL_MD5 == "95e8e9f18fde62f22025e31a88ad81fa"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_model_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.offline.model_feature_list'`

- [ ] **Step 3: Write minimal implementation**

Create `feature_step/features/offline/model_feature_list.py`:

```python
"""Authority: the exact 199 band-suffixed feature names the deployed BHRF model
consumes, pinned so the coverage check runs without the 1.72 GB model download.

Provenance: squidward 2.1.0 hierarchical_random_forest_model.pkl, sourced from the
md5-verified deployed artifact
    https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl
    size 1,720,755,396 bytes, md5 95e8e9f18fde62f22025e31a88ad81fa (Last-Modified 2 Jun 2025)
These are the POST-preprocess column names classify_batch selects via
`features[self.feature_list]` (see hierarchical_random_forest.py). The offline
`--smoke` path asserts a freshly loaded model's feature_list equals this list, so
drift is caught rather than silently trusted.
"""

MODEL_VERSION = "2.1.0"
MODEL_MD5 = "95e8e9f18fde62f22025e31a88ad81fa"
MODEL_SIZE_BYTES = 1_720_755_396

# 199 names, verbatim from loaded_data["feature_list"].
MODEL_FEATURE_LIST = [
    'Amplitude_1', 'Amplitude_2', 'AndersonDarling_1', 'AndersonDarling_2',
    'Autocor_length_1', 'Autocor_length_2', 'Beyond1Std_1', 'Beyond1Std_2',
    'Con_1', 'Con_2', 'Coordinate_x', 'Coordinate_y',
    'Coordinate_z', 'Eta_e_1', 'Eta_e_2', 'ExcessVar_1',
    'ExcessVar_2', 'GP_DRW_sigma_1', 'GP_DRW_sigma_2', 'GP_DRW_tau_1',
    'GP_DRW_tau_2', 'Gskew_1', 'Gskew_2', 'Harmonics_chi_1',
    'Harmonics_chi_2', 'Harmonics_mag_1_1', 'Harmonics_mag_1_2', 'Harmonics_mag_2_1',
    'Harmonics_mag_2_2', 'Harmonics_mag_3_1', 'Harmonics_mag_3_2', 'Harmonics_mag_4_1',
    'Harmonics_mag_4_2', 'Harmonics_mag_5_1', 'Harmonics_mag_5_2', 'Harmonics_mag_6_1',
    'Harmonics_mag_6_2', 'Harmonics_mag_7_1', 'Harmonics_mag_7_2', 'Harmonics_mse_1',
    'Harmonics_mse_2', 'Harmonics_phase_2_1', 'Harmonics_phase_2_2', 'Harmonics_phase_3_1',
    'Harmonics_phase_3_2', 'Harmonics_phase_4_1', 'Harmonics_phase_4_2', 'Harmonics_phase_5_1',
    'Harmonics_phase_5_2', 'Harmonics_phase_6_1', 'Harmonics_phase_6_2', 'Harmonics_phase_7_1',
    'Harmonics_phase_7_2', 'IAR_phi_1', 'IAR_phi_2', 'LinearTrend_1',
    'LinearTrend_2', 'MHPS_PN_flag_1', 'MHPS_PN_flag_2', 'MHPS_high_30_1',
    'MHPS_high_30_2', 'MHPS_high_1', 'MHPS_high_2', 'MHPS_low_365_1',
    'MHPS_low_365_2', 'MHPS_low_1', 'MHPS_low_2', 'MHPS_non_zero_1',
    'MHPS_non_zero_2', 'MHPS_ratio_365_30_1', 'MHPS_ratio_365_30_2', 'MHPS_ratio_1',
    'MHPS_ratio_2', 'MaxSlope_1', 'MaxSlope_2', 'Mean_1',
    'Mean_2', 'Meanvariance_1', 'Meanvariance_2', 'MedianAbsDev_1',
    'MedianAbsDev_2', 'MedianBRP_1', 'MedianBRP_2', 'Multiband_period_12',
    'PPE_12', 'PairSlopeTrend_1', 'PairSlopeTrend_2', 'PercentAmplitude_1',
    'PercentAmplitude_2', 'Period_band_1', 'Period_band_2', 'Power_rate_1_2_12',
    'Power_rate_1_3_12', 'Power_rate_1_4_12', 'Power_rate_2_12', 'Power_rate_3_12',
    'Power_rate_4_12', 'Psi_CS_1', 'Psi_CS_2', 'Psi_eta_1',
    'Psi_eta_2', 'Pvar_1', 'Pvar_2', 'Q31_1',
    'Q31_2', 'Rcs_1', 'Rcs_2', 'SF_ML_amplitude_1',
    'SF_ML_amplitude_2', 'SF_ML_gamma_1', 'SF_ML_gamma_2', 'SPM_A_1',
    'SPM_A_2', 'SPM_beta_1', 'SPM_beta_2', 'SPM_chi_1',
    'SPM_chi_2', 'SPM_gamma_1', 'SPM_gamma_2', 'SPM_t0_1',
    'SPM_t0_2', 'SPM_tau_fall_1', 'SPM_tau_fall_2', 'SPM_tau_rise_1',
    'SPM_tau_rise_2', 'Skew_1', 'Skew_2', 'SmallKurtosis_1',
    'SmallKurtosis_2', 'Std_1', 'Std_2', 'StetsonK_1',
    'StetsonK_2', 'TDE_decay_chi_1', 'TDE_decay_chi_2', 'TDE_decay_1',
    'TDE_decay_2', 'Timespan', 'W1_W2', 'W2_W3',
    'W3_W4', 'color_variation_12', 'dbrightness_first_det_band_1', 'dbrightness_first_det_band_2',
    'dbrightness_forced_phot_band_1', 'dbrightness_forced_phot_band_2', 'delta_period_1', 'delta_period_2',
    'distpsnr1', 'fleet_a_1', 'fleet_a_2', 'fleet_chi_1',
    'fleet_chi_2', 'fleet_w_1', 'fleet_w_2', 'g_W1',
    'g_W2', 'g_W3', 'g_W4', 'g_r_max_corr_12',
    'g_r_max_12', 'g_r_mean_corr_12', 'g_r_mean_12', 'last_brightness_before_band_1',
    'last_brightness_before_band_2', 'max_brightness_after_band_1', 'max_brightness_after_band_2', 'max_brightness_before_band_1',
    'max_brightness_before_band_2', 'mean_chinr_12', 'mean_distnr_12', 'mean_sharpnr_12',
    'median_brightness_after_band_1', 'median_brightness_after_band_2', 'median_brightness_before_band_1', 'median_brightness_before_band_2',
    'n_forced_phot_band_after_1', 'n_forced_phot_band_after_2', 'n_forced_phot_band_before_1', 'n_forced_phot_band_before_2',
    'positive_fraction_1', 'positive_fraction_2', 'ps_g_r', 'ps_i_z',
    'ps_r_i', 'r_W1', 'r_W2', 'r_W3',
    'r_W4', 'sgscore1', 'sigma_distnr_12', 'ulens_chi_1',
    'ulens_chi_2', 'ulens_fs_1', 'ulens_fs_2', 'ulens_tE_1',
    'ulens_tE_2', 'ulens_u0_1', 'ulens_u0_2',
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_model_features.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/model_feature_list.py feature_step/tests/unittest/test_offline_model_features.py
git commit -m "feat(feature_step): pin deployed BHRF 199 feature_list as offline authority"
```

---

## Task 2: Pure verification logic (`predict_input_columns`, `diff_feature_coverage`)

**Files:**
- Create: `feature_step/features/offline/model_features.py`
- Test: `feature_step/tests/unittest/test_offline_model_features.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `feature_step/tests/unittest/test_offline_model_features.py`:

```python
import pandas as pd
from features.offline import model_features


def test_diff_feature_coverage_reports_missing_and_extra():
    produced = ["Amplitude_1", "Amplitude_2", "surprise_1"]
    expected = ["Amplitude_1", "Amplitude_2", "Std_1"]
    diff = model_features.diff_feature_coverage(produced, expected)
    assert diff["missing"] == ["Std_1"]      # would KeyError at predict
    assert diff["extra"] == ["surprise_1"]
    assert diff["covered"] == ["Amplitude_1", "Amplitude_2"]
    assert diff["n_expected"] == 3
    assert diff["n_missing"] == 1


def test_predict_input_columns_matches_predict_path(monkeypatch):
    # parse_output is monkeypatched (mirrors test_offline_classify.py) so no real
    # AstroObject/DB is needed. The band suffix + None values exercise the exact
    # SquidwardMapper.preprocess (None->NaN) + RandomForestPreprocessor path.
    out_message = {
        "oid": 123,
        "features": {"Amplitude_1": 0.5, "W1_W2": None, "ps_g_r": 1.0},
    }
    monkeypatch.setattr(model_features, "parse_output",
                        lambda aos, msgs, candids: [out_message])

    cols = model_features.predict_input_columns(object(), {"oid": 123, "measurement_id": [1]})

    # RandomForestPreprocessor is idempotent on these already-suffixed names.
    assert set(cols) == {"Amplitude_1", "W1_W2", "ps_g_r"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_model_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.offline.model_features'`

- [ ] **Step 3: Write minimal implementation**

Create `feature_step/features/offline/model_features.py`:

```python
"""Pure logic for the 199-feature coverage check.

`predict_input_columns` reproduces the EXACT column set that
HierarchicalRandomForestClassifier.classify_batch selects on via
`features[self.feature_list]`: parse_output names + band-suffixes the features,
SquidwardMapper turns None into NaN, and RandomForestPreprocessor renames columns.
Comparing that set against MODEL_FEATURE_LIST tells us whether predict() would
KeyError (missing name) for a given object.
"""
import numpy as np
import pandas as pd

from features.utils.parsers import parse_output
from alerce_classifiers.classifiers.preprocess import RandomForestPreprocessor


def predict_input_columns(ao, message: dict) -> list[str]:
    """Post-extract AstroObject + its message -> the column names predict() selects on.

    Mirrors features.offline.classify.classify_astro_object up to (but not
    including) the model call, then applies the model's own RandomForestPreprocessor
    so the returned names are in the same namespace as the model's feature_list.
    """
    candids = {message["oid"]: message.get("measurement_id", [])}
    out_message = parse_output([ao], [message], candids)[0]
    features = out_message.get("features") or {}
    df = pd.DataFrame([features], index=[message["oid"]])
    df.replace({None: np.nan}, inplace=True)          # SquidwardMapper.preprocess
    processed = RandomForestPreprocessor().preprocess_features(df)
    return list(processed.columns)


def diff_feature_coverage(produced, expected) -> dict:
    """Diff a produced name set against the expected model feature_list.

    Returns covered/missing/extra (sorted) plus counts. `missing` is the set the
    model would KeyError on.
    """
    produced_set = set(produced)
    expected_set = set(expected)
    missing = sorted(expected_set - produced_set)
    return {
        "covered": sorted(expected_set & produced_set),
        "missing": missing,
        "extra": sorted(produced_set - expected_set),
        "n_expected": len(expected_set),
        "n_missing": len(missing),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_model_features.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/model_features.py feature_step/tests/unittest/test_offline_model_features.py
git commit -m "feat(feature_step): pure 199-feature coverage diff mirroring predict path"
```

---

## Task 3: Discover a diverse sample of real oids

The sample must span lightcurve shapes (dense / sparse / forced-photometry-heavy) so an object-dependent missing feature surfaces. These oids are execution-time data (read-only DB access; they cannot be known at plan-writing time), so this task discovers them with a bounded, oid-indexed query and pins the result into a constant in Task 4.

**Files:**
- Scratch: `<scratchpad>/discover_sample_oids.py`

- [ ] **Step 1: Write the bounded discovery script**

Create `<scratchpad>/discover_sample_oids.py` (replace `<scratchpad>` with the session scratchpad dir):

```python
"""Pick a diverse handful of ZTF oids by lightcurve shape. Bounded: TABLESAMPLE
for candidates (page sample, no full scan), then per-oid indexed counts under a
statement_timeout. Read-only."""
import sys
from pathlib import Path
PIPE = Path("/home/fandrades/desktop/pipeline")
sys.path.insert(0, str(PIPE / "feature_step"))

from features.offline import db
from sqlalchemy import text

CREDS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
engine = db._make_engine(CREDS)

with engine.connect() as conn:
    conn.execute(text("SET statement_timeout = '60s'"))
    # Candidate oids from a small page sample of the detection table (cheap).
    cand = [r[0] for r in conn.execute(text(f"""
        SELECT DISTINCT oid
        FROM {db.SCHEMA}.detection TABLESAMPLE SYSTEM (0.05)
        WHERE sid = 0
        LIMIT 300
    """)).fetchall()]
    print(f"candidates: {len(cand)}")

    rows = []
    for oid in cand:
        n_det = conn.execute(text(
            f"SELECT count(*) FROM {db.SCHEMA}.detection WHERE oid=:o AND sid=0"),
            {"o": oid}).scalar()
        n_fp = conn.execute(text(
            f"SELECT count(*) FROM {db.SCHEMA}.forced_photometry WHERE oid=:o AND sid=0"),
            {"o": oid}).scalar()
        rows.append((oid, n_det, n_fp))

    dense = sorted(rows, key=lambda r: -r[1])[:3]
    sparse = [r for r in sorted(rows, key=lambda r: r[1]) if r[1] >= 5][:3]
    forced = sorted(rows, key=lambda r: -r[2])[:3]

    picked = {}
    for label, group in (("dense", dense), ("sparse", sparse), ("forced", forced)):
        for oid, nd, nf in group:
            picked[oid] = (label, nd, nf)
    # Always include the pinned LUT oid for continuity with the rest of offline.
    picked.setdefault(36028941624528297, ("lut", None, None))

    print("\n# paste into SAMPLE_OIDS (Task 4):")
    for oid, (label, nd, nf) in picked.items():
        print(f"    {oid},  # {label} n_det={nd} n_fp={nf}")
```

- [ ] **Step 2: Run discovery**

Run:
```bash
conda run --no-capture-output -n training_py310 python <scratchpad>/discover_sample_oids.py
```
Expected: a list of ~8-10 `oid,  # label ...` lines. Copy them for Task 4.

If TABLESAMPLE returns too few forced/sparse candidates, re-run (page sample varies) or raise the `SYSTEM (0.05)` percentage slightly. Do NOT remove the `statement_timeout` or the `LIMIT`.

---

## Task 4: CLI script — name-diff over the sample (primary check)

**Files:**
- Create: `feature_step/scripts/offline_verify_model_features.py`

- [ ] **Step 1: Write the script scaffold + name-diff**

Create `feature_step/scripts/offline_verify_model_features.py`, pasting the oids from Task 3 into `SAMPLE_OIDS`:

```python
#!/usr/bin/env python
"""Verify the offline pipeline emits all 199 features the deployed BHRF model needs.

Primary check (default, no model download): for each sample oid, compute features
offline and diff the resulting predict-input column set against the pinned
MODEL_FEATURE_LIST. A missing name would KeyError inside model.predict.

Confirmation (--smoke, requires MODEL_PATH + training_py310): load the real model
once, run predict on the sample, assert no KeyError and model.feature_list matches
the pinned constant.

    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_verify_model_features.py

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_verify_model_features.py --smoke
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "libs" / "apf",
          PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import argparse

from features.offline import db
from features.offline.message import build_message
from features.offline.lc_features import compute_astro_object
from features.offline.model_features import predict_input_columns, diff_feature_coverage
from features.offline.model_feature_list import MODEL_FEATURE_LIST, MODEL_VERSION

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")

# Diverse handful discovered via scripts discovery (Task 3), pinned for reproducibility.
SAMPLE_OIDS = [
    # <<< paste Task 3 output here, e.g.: >>>
    36028941624528297,  # lut
]


def _astro_object_for(oid: int, credentials: str, min_det: int):
    dets = db.fetch_detections(credentials, [oid])
    forced = db.fetch_forced_photometry(credentials, [oid])
    ps1 = db.fetch_ps1(credentials, [oid])
    allwise = db.fetch_allwise(credentials, [oid])
    refs = db.fetch_references(credentials, [oid])
    message = build_message(oid, dets, forced, ps1)
    ao = compute_astro_object(message, refs, allwise, min_det)
    return ao, message


def run_name_diff(oids, credentials, min_det) -> int:
    """Name-diff each oid; return process exit code (0 = all covered)."""
    agg_missing = set(MODEL_FEATURE_LIST)   # intersect down to names missing for ALL
    any_missing = set()                     # union of names missing for ANY oid
    checked = 0
    print(f"expected: {len(MODEL_FEATURE_LIST)} features (BHRF {MODEL_VERSION})\n")
    for oid in oids:
        ao, message = _astro_object_for(oid, credentials, min_det)
        if ao is None:
            print(f"  oid {oid}: SKIP (too few real detections)")
            continue
        cols = predict_input_columns(ao, message)
        diff = diff_feature_coverage(cols, MODEL_FEATURE_LIST)
        checked += 1
        agg_missing &= set(diff["missing"])
        any_missing |= set(diff["missing"])
        status = "OK" if diff["n_missing"] == 0 else f"MISSING {diff['n_missing']}"
        print(f"  oid {oid}: {status}"
              + (f" -> {diff['missing']}" if diff["missing"] else "")
              + (f"  (+{len(diff['extra'])} extra)" if diff["extra"] else ""))

    print(f"\nchecked {checked}/{len(oids)} oids")
    if checked == 0:
        print("FAIL: no oids produced features")
        return 1
    if any_missing:
        print(f"FAIL: {len(any_missing)} name(s) missing for at least one oid: "
              f"{sorted(any_missing)}")
        if agg_missing:
            print(f"  of which missing for ALL oids: {sorted(agg_missing)}")
        return 1
    print("PASS: all 199 features covered for every checked oid")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oid", type=int, action="append",
                    help="Override SAMPLE_OIDS (repeatable).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--min-det", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="Also load the real model (MODEL_PATH) and run predict.")
    args = ap.parse_args()

    oids = args.oid if args.oid else SAMPLE_OIDS
    code = run_name_diff(oids, args.credentials, args.min_det)

    if args.smoke:
        code |= run_smoke(oids, args.credentials, args.min_det)   # defined in Task 5

    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a temporary stub so the file imports before Task 5**

At the top of `main`-scope (just above `def main():`), temporarily add:

```python
def run_smoke(oids, credentials, min_det) -> int:
    raise SystemExit("run_smoke not implemented yet (Task 5)")
```

(Removed/replaced in Task 5. This keeps the module importable if `--smoke` is passed early.)

- [ ] **Step 3: Run the primary check for real**

Run:
```bash
cd /home/fandrades/desktop/pipeline && conda run --no-capture-output -n training_py310 \
    python feature_step/scripts/offline_verify_model_features.py
```
Expected: a per-oid report ending in either `PASS: all 199 features covered...` or `FAIL: N name(s) missing ...` with the exact missing names. **Record this output** — it is the deliverable answer (used in Task 7).

- [ ] **Step 4: Commit**

```bash
git add feature_step/scripts/offline_verify_model_features.py
git commit -m "feat(feature_step): offline 199-feature coverage verifier (name-diff)"
```

---

## Task 5: `--smoke` end-to-end confirmation

**Files:**
- Modify: `feature_step/scripts/offline_verify_model_features.py`

- [ ] **Step 1: Replace the stub with the real smoke path**

In `feature_step/scripts/offline_verify_model_features.py`, replace the temporary `run_smoke` stub from Task 4 Step 2 with:

```python
def run_smoke(oids, credentials, min_det) -> int:
    """Load the real model once and confirm predict() runs without KeyError,
    and that the loaded feature_list matches the pinned constant."""
    from features.offline.classify import load_squidward_model, classify_astro_object

    model, name, version = load_squidward_model()
    print(f"\n[smoke] model {name} version={version}")

    # Drift guard: the pinned 199 must equal the live model's feature_list.
    loaded = list(model.model.feature_list)
    if loaded != MODEL_FEATURE_LIST:
        only_model = sorted(set(loaded) - set(MODEL_FEATURE_LIST))
        only_pin = sorted(set(MODEL_FEATURE_LIST) - set(loaded))
        print(f"[smoke] FAIL: feature_list drift "
              f"(model-only={only_model}, pin-only={only_pin})")
        return 1

    failures = 0
    for oid in oids:
        ao, message = _astro_object_for(oid, credentials, min_det)
        if ao is None:
            print(f"[smoke] oid {oid}: SKIP (too few real detections)")
            continue
        try:
            out = classify_astro_object(ao, message, model)
        except KeyError as e:
            print(f"[smoke] oid {oid}: FAIL KeyError {e}")
            failures += 1
            continue
        n = 0 if out.probabilities is None else len(out.probabilities)
        print(f"[smoke] oid {oid}: OK ({n} prob row(s))")

    if failures:
        print(f"[smoke] FAIL: {failures} oid(s) raised KeyError")
        return 1
    print("[smoke] PASS: predict ran without KeyError on all checked oids")
    return 0
```

Also delete the temporary stub definition added in Task 4 Step 2.

- [ ] **Step 2: Run the smoke path for real**

Run:
```bash
cd /home/fandrades/desktop/pipeline && \
MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
conda run --no-capture-output -n training_py310 \
    python feature_step/scripts/offline_verify_model_features.py --smoke
```
Expected: the name-diff report, then `[smoke] PASS: predict ran without KeyError ...`. First run downloads ~1.72 GB to `/tmp/SquidwardFeaturesClassifier` (a few minutes). **Record the smoke result** for Task 7.

- [ ] **Step 3: Commit**

```bash
git add feature_step/scripts/offline_verify_model_features.py
git commit -m "feat(feature_step): --smoke end-to-end predict confirmation + drift guard"
```

---

## Task 6: Handle any gap the verification finds (conditional)

Only if Task 4/5 reported `FAIL` / missing names. If both PASS, skip to Task 7.

- [ ] **Step 1: Classify each missing name**

For each name in the reported `missing` set, determine the cause:
- **Extractor never emits it** (name absent from `ao.features` for all oids) → a real coverage gap in the offline extractor config vs the deployed model; document it as a finding and stop — fixing the extractor is out of this plan's scope (record it in the spec's risks and raise with the team).
- **Emitted only for some object shapes** (present for dense, missing for sparse) → expand `SAMPLE_OIDS` is not the fix; the model will still KeyError on that shape. Document as a real gap (same as above).
- **Name-normalization mismatch** (e.g. an off-by-one in band suffixing) → a bug in this verifier; fix `predict_input_columns` and re-run. Verify against the idempotency note in the header before changing anything.

- [ ] **Step 2: Record the finding**

Write the classified missing-name list into the spec (Task 7) rather than silently patching, so the decision is visible.

---

## Task 7: Documentation — close the pending item

**Files:**
- Modify: `docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md`
- Modify: `feature_step/features/offline/FLOW.md`

- [ ] **Step 1: Reframe + close the spec's pending section**

In `docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md`, replace the body of the `## Pending: confirm offline features cover the model's 199` section with the actual result. Use this template, filling the bracketed values from Task 4/5 output:

```markdown
## Resolved: offline features cover the model's 199

Verified [DATE] by `feature_step/scripts/offline_verify_model_features.py` over
[N] diverse oids (dense / sparse / forced-heavy + the LUT oid). Result:
**[PASS — all 199 covered for every oid | FAIL — missing: <names>]**.
`--smoke` loaded the deployed model (md5 `95e8e9f18fde62f22025e31a88ad81fa`) and
confirmed `predict()` runs without `KeyError`, and the model's `feature_list`
equals the pinned `MODEL_FEATURE_LIST`.

This is a **hard prerequisite** for offline BHRF classification, not just a
probability-trust nicety: `classify_batch` does `features[self.feature_list]`
(strict column selection), so any missing name raises `KeyError` at predict time.
```

Also update the earlier `### Feature-count cross-check (199 vs 123)` paragraph: change "Before trusting offline BHRF *probabilities* ... confirm ..." to note it is a hard predict-time contract (KeyError), now verified — and reference the script.

- [ ] **Step 2: Flip the FLOW.md status line**

In `feature_step/features/offline/FLOW.md`, find the line tracking the 199-vs-123 / feature-coverage verification (search for "199") and change its status from Pending to Done, referencing `scripts/offline_verify_model_features.py`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md feature_step/features/offline/FLOW.md
git commit -m "docs(feature_step): record offline 199-feature coverage verification result"
```

---

## Self-Review notes (for the implementer)

- **Env:** every `pytest`/script run uses `conda run -n training_py310` because feature extraction imports `lc_classifier` and `--smoke` imports `imblearn`. Running in the base env will fail at import, not at logic.
- **DB safety:** all DB access is read-only and oid-indexed; the only non-indexed query is the Task 3 TABLESAMPLE page-sample, which is bounded and timeout-guarded. Never scan `alerce.probability`.
- **Do not print `credentials.json`** — it holds a live password.
- **The 199 list is the contract:** if `--smoke`'s drift guard fails, the deployed model changed; re-derive `MODEL_FEATURE_LIST` from the md5-verified pickle before touching anything else.
