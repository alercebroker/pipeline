# Offline ZTF DB-ready features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline ZTF feature output emit DB-ready rows `(oid, sid, feature_id, band, version, value)` following the production-LSST save rules (drop NaN, `fid→band`, `name→feature_id` via a LUT), while leaving the named feature frame intact for classification.

**Architecture:** Fix the production `prepare_ao_features_for_db` to map ids via a `feature_name_lut` parameter (also repairing the live ZTF enumerate bug); offline supplies that LUT from a checked-in local fixture and wraps the fixed function in a new `compute_db_features`. No DB writes.

**Tech Stack:** Python, pandas, pytest; `lc_classifier` (AstroObject/extractor), `features.utils.parsers`, `features.offline.*`.

**Spec:** `docs/superpowers/specs/2026-06-25-offline-ztf-db-ready-features-design.md`

**Test command convention** (matches existing offline tests + repo scripts):
```bash
cd feature_step && conda run --no-capture-output -n training_py310 \
    python -m pytest tests/unittest/<file>.py -v
```
The `tests/conftest.py` puts `lc_classifier`, `idmapper`, `apf` on `sys.path`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `feature_step/features/utils/parsers.py` | `prepare_ao_features_for_db` (id mapping rule) + `parse_scribe_payload` wiring | Modify |
| `feature_step/tests/unittest/test_prepare_ao_features_for_db.py` | unit-test the parser fix | Create |
| `feature_step/scripts/offline_generate_feature_lut.py` | one-off generator that prints the fixture literal from a real run | Create |
| `feature_step/features/offline/feature_lut.py` | the checked-in fixture + loaders | Create |
| `feature_step/tests/unittest/test_offline_feature_lut.py` | unit-test the fixture/loaders | Create |
| `feature_step/features/offline/lc_features.py` | new `compute_db_features` | Modify |
| `feature_step/tests/unittest/test_offline_db_features.py` | unit-test `compute_db_features` | Create |
| `feature_step/scripts/offline_compute_features.py` | emit DB-ready rows | Modify |
| `feature_step/features/offline/FLOW.md`, `README.md` | document DB-ready output + fixture | Modify |

---

## Task 1: Fix `prepare_ao_features_for_db` to map ids via the LUT

**Files:**
- Modify: `feature_step/features/utils/parsers.py:404-437` (function) and `:500` (caller)
- Test: `feature_step/tests/unittest/test_prepare_ao_features_for_db.py`

- [ ] **Step 1: Write the failing test**

Create `feature_step/tests/unittest/test_prepare_ao_features_for_db.py`:

```python
"""Unit tests for prepare_ao_features_for_db — the ZTF DB-prep rule.

Uses a lightweight stand-in object (prepare only reads `.features`), so no
real AstroObject/DB is needed.
"""
import types

import numpy as np
import pandas as pd

from features.utils.parsers import prepare_ao_features_for_db


def _ao(features_df):
    return types.SimpleNamespace(features=features_df)


def test_maps_ids_from_lut_drops_nan_and_bands():
    df = pd.DataFrame({
        "name":  ["Amplitude", "Amplitude", "Period", "MHPS_ratio"],
        "fid":   ["g", "r", "g,r", None],
        "value": [0.5, 0.6, np.nan, 1.2],
    })
    # Ids chosen so they do NOT match appearance order — proves the LUT is used,
    # not the old enumerate.
    lut = {5: "Amplitude", 3: "MHPS_ratio", 7: "Period"}

    out = prepare_ao_features_for_db(_ao(df), lut)

    # NaN value (Period) dropped before id mapping
    assert "Period" not in set(out["name"])
    assert len(out) == 3
    # band codes: g->1, r->2, None->0
    assert out.loc[(out["name"] == "Amplitude") & (out["band"] == 1)].shape[0] == 1
    assert out.loc[(out["name"] == "Amplitude") & (out["band"] == 2)].shape[0] == 1
    assert out.loc[out["name"] == "MHPS_ratio", "band"].iloc[0] == 0
    # feature_id comes from the LUT, not enumerate(0,1,...)
    assert out.loc[out["name"] == "Amplitude", "feature_id"].iloc[0] == 5
    assert out.loc[out["name"] == "MHPS_ratio", "feature_id"].iloc[0] == 3
    # output columns
    assert set(out.columns) == {"name", "value", "band", "feature_id"}


def test_unmapped_name_yields_nan_id():
    df = pd.DataFrame({"name": ["Unknown"], "fid": ["g"], "value": [1.0]})
    out = prepare_ao_features_for_db(_ao(df), {0: "Amplitude"})
    assert out["feature_id"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_prepare_ao_features_for_db.py -v`
Expected: FAIL — `prepare_ao_features_for_db() takes 1 positional argument but 2 were given` (current signature has no `feature_name_lut`).

- [ ] **Step 3: Implement the fix**

In `feature_step/features/utils/parsers.py`, replace the function body at lines 404-437. Change the signature and swap the `enumerate` block for a LUT lookup; keep the NaN drop, band mapping, inf/nan→None, back-compat name fixes, and the unmapped warning:

```python
def prepare_ao_features_for_db(astro_object: AstroObject, feature_name_lut) -> pd.DataFrame:
    ao_features = astro_object.features[["name", "fid", "value"]].copy()
    ao_features = ao_features[ao_features["value"].notna()]

    ao_features["band"] = ao_features["fid"].apply(fid_mapper_for_db)
    ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)

    # backward compatibility
    ao_features["name"] = ao_features["name"].replace(
        {
            "Power_rate_1_4": "Power_rate_1/4",
            "Power_rate_1_3": "Power_rate_1/3",
            "Power_rate_1_2": "Power_rate_1/2",
        }
    )

    # Map feature names to ids using the feature_name_lut ({feature_id: feature_name}).
    name_to_id = {name: feature_id for feature_id, name in feature_name_lut.items()}
    ao_features["feature_id"] = ao_features["name"].map(name_to_id)

    # Log warning for unmapped features
    unmapped_features = ao_features[ao_features["feature_id"].isna()]["name"].unique()
    if len(unmapped_features) > 0:
        logging.getLogger("alerce.FeatureStep").warning(
            f"Features not found in lookup table: {list(unmapped_features)}"
        )

    # Drop original columns, keep only the mapped data
    ao_features.drop(columns=["fid"], inplace=True)
    return ao_features
```

- [ ] **Step 4: Wire the LUT through the caller**

In the same file, `parse_scribe_payload` (line ~500) already receives `feature_name_lut` but never forwards it. Change the call:

```python
        # for upserting features
        ao_features = prepare_ao_features_for_db(astro_object, feature_name_lut)
```

(No `step.py` change — it already loads and threads `feature_name_lut` for ZTF at `step.py:82` and `:213`/`:217`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_prepare_ao_features_for_db.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add feature_step/features/utils/parsers.py feature_step/tests/unittest/test_prepare_ao_features_for_db.py
git commit -m "fix(feature_step): map ZTF feature ids via feature_name_lut (not enumerate)"
```

---

## Task 2: Fixture generator script

**Files:**
- Create: `feature_step/scripts/offline_generate_feature_lut.py`

This is a one-off tool (needs the DB); its output seeds Task 3. No unit test — verified by a manual run.

- [ ] **Step 1: Write the generator**

Create `feature_step/scripts/offline_generate_feature_lut.py`:

```python
#!/usr/bin/env python
"""Generate the offline ZTF feature_name_lut + feature_version_lut fixture.

Runs the real extractor on one (or more) representative oid(s), collects the
FULL set of band-less feature names (NOT NaN-filtered — we want the complete
feature schema, not one object's non-NaN subset), sorts them, assigns ids
0..N-1, and prints a ready-to-paste Python literal for
feature_step/features/offline/feature_lut.py.

    conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_generate_feature_lut.py --oid 36028941624528297
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]
for p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper"):
    sys.path.insert(0, str(p))

import argparse

from features.offline import db, lc_features
from features.offline.message import build_message

# Mirror the back-compat name fixes in prepare_ao_features_for_db.
_NAME_FIXES = {
    "Power_rate_1_4": "Power_rate_1/4",
    "Power_rate_1_3": "Power_rate_1/3",
    "Power_rate_1_2": "Power_rate_1/2",
}
DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")


def collect(credentials, oids):
    names, versions = set(), set()
    for oid in oids:
        dets = db.fetch_detections(credentials, [oid])
        forced = db.fetch_forced_photometry(credentials, [oid])
        ps1 = db.fetch_ps1(credentials, [oid])
        allwise = db.fetch_allwise(credentials, [oid])
        refs = db.fetch_references(credentials, [oid])
        message = build_message(oid, dets, forced, ps1)
        ao = lc_features.compute_astro_object(message, refs, allwise)
        if ao is None:
            print(f"# oid {oid}: too few detections, skipped", file=sys.stderr)
            continue
        feats = ao.features  # NOT NaN-filtered
        names.update(feats["name"].replace(_NAME_FIXES))
        versions.update(feats["version"].dropna().unique())
    return sorted(names), sorted(versions)


def main():
    ap = argparse.ArgumentParser(description="Generate offline ZTF feature LUT fixture.")
    ap.add_argument("--oid", type=int, action="append", required=True,
                    help="Multisurvey bigint oid (repeat to union name sets).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    args = ap.parse_args()

    names, versions = collect(args.credentials, args.oid)
    print(f"# {len(names)} feature names; versions={versions}")
    print("FEATURE_NAME_LUT = {")
    for i, n in enumerate(names):
        print(f"    {i}: {n!r},")
    print("}")
    print("FEATURE_VERSION_LUT = {")
    for i, v in enumerate(versions):
        print(f"    {i}: {v!r},")
    print("}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs (manual, needs DB)**

Run: `conda run --no-capture-output -n training_py310 python feature_step/scripts/offline_generate_feature_lut.py --oid 36028941624528297`
Expected: prints a non-empty `FEATURE_NAME_LUT = { 0: '...', ... }` block and a `FEATURE_VERSION_LUT` block with at least one version string. **Save this output** — it seeds Task 3.

- [ ] **Step 3: Commit**

```bash
git add feature_step/scripts/offline_generate_feature_lut.py
git commit -m "feat(feature_step): offline ZTF feature LUT generator script"
```

---

## Task 3: Local fixture module + loaders

**Files:**
- Create: `feature_step/features/offline/feature_lut.py`
- Test: `feature_step/tests/unittest/test_offline_feature_lut.py`

- [ ] **Step 1: Write the failing test**

Create `feature_step/tests/unittest/test_offline_feature_lut.py`:

```python
"""Unit tests for the offline ZTF feature LUT fixture + loaders.

Asserts structural invariants that hold for any valid generated fixture, so
the tests don't hard-code the feature-name list.
"""
from features.offline.feature_lut import (
    FEATURE_NAME_LUT,
    FEATURE_VERSION_LUT,
    load_feature_name_lut,
    version_name_to_id,
)


def test_name_lut_non_empty():
    assert len(FEATURE_NAME_LUT) > 0


def test_name_lut_ids_contiguous_from_zero():
    assert sorted(FEATURE_NAME_LUT) == list(range(len(FEATURE_NAME_LUT)))


def test_name_lut_sorted_by_name():
    names = [FEATURE_NAME_LUT[i] for i in sorted(FEATURE_NAME_LUT)]
    assert names == sorted(names)


def test_load_returns_independent_copy():
    lut = load_feature_name_lut()
    assert lut == FEATURE_NAME_LUT
    lut[10_000] = "x"
    assert 10_000 not in FEATURE_NAME_LUT


def test_version_round_trips():
    vid, vname = next(iter(FEATURE_VERSION_LUT.items()))
    assert version_name_to_id(vname) == vid


def test_unknown_version_warns_and_returns_negative_one():
    assert version_name_to_id("__definitely_not_a_version__") == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_feature_lut.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.offline.feature_lut'`.

- [ ] **Step 3: Create the module with loaders, then paste the generated LUT**

Create `feature_step/features/offline/feature_lut.py`:

```python
"""Offline ZTF feature LUTs (local fixture).

Stand-in for the DB feature_name_lut / feature_version_lut (sid=0), which are
empty (FLOW §3d). Generated by scripts/offline_generate_feature_lut.py and
checked in as static data. Ids are offline's own until the DB LUT is seeded;
the shape and rules match production, the id VALUES reconcile later.
"""
import logging

log = logging.getLogger(__name__)

# {feature_id: band-less feature_name}, sid=0. Ids 0..N-1 by sorted name.
# Generated: scripts/offline_generate_feature_lut.py --oid 36028941624528297
FEATURE_NAME_LUT = {
    # >>> PASTE the generator's FEATURE_NAME_LUT body here <<<
}

# {version_id: version_name}
FEATURE_VERSION_LUT = {
    # >>> PASTE the generator's FEATURE_VERSION_LUT body here <<<
}


def load_feature_name_lut() -> dict:
    """Return {feature_id: feature_name} — drop-in for database.get_feature_name_lut."""
    return dict(FEATURE_NAME_LUT)


def version_name_to_id(version_name) -> int:
    """Reverse-map a version string to its smallint id. Warn + return -1 if unknown."""
    rev = {name: vid for vid, name in FEATURE_VERSION_LUT.items()}
    if version_name not in rev:
        log.warning("Unknown feature version %r; not in FEATURE_VERSION_LUT", version_name)
        return -1
    return rev[version_name]
```

Then replace the two `>>> PASTE ... <<<` markers with the `FEATURE_NAME_LUT` / `FEATURE_VERSION_LUT` bodies printed by the Task 2 generator run (the dict entries only — keep the assignment lines already in the file). This is a data paste, not logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_feature_lut.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/feature_lut.py feature_step/tests/unittest/test_offline_feature_lut.py
git commit -m "feat(feature_step): offline ZTF feature_name/version LUT fixture"
```

---

## Task 4: `compute_db_features`

**Files:**
- Modify: `feature_step/features/offline/lc_features.py`
- Test: `feature_step/tests/unittest/test_offline_db_features.py`

- [ ] **Step 1: Write the failing test**

Create `feature_step/tests/unittest/test_offline_db_features.py`:

```python
"""Unit tests for compute_db_features — the DB-ready offline output.

compute_astro_object (heavy, real extractor) is monkeypatched to return a stub
AstroObject with a `.features` frame.
"""
import types

import numpy as np
import pandas as pd

from features.offline import lc_features


def test_compute_db_features_emits_feature_table_rows(monkeypatch):
    feats = pd.DataFrame({
        "name":    ["Amplitude", "Amplitude", "Period"],
        "fid":     ["g", "r", "g,r"],
        "value":   [0.5, 0.6, np.nan],
        "sid":     [0, 0, 0],
        "version": ["lc_v1", "lc_v1", "lc_v1"],
    })
    ao = types.SimpleNamespace(features=feats)
    monkeypatch.setattr(lc_features, "compute_astro_object", lambda *a, **k: ao)
    monkeypatch.setattr(lc_features, "version_name_to_id", lambda v: 7)

    lut = {0: "Amplitude", 1: "Period"}
    out = lc_features.compute_db_features({"oid": 123}, None, None, feature_name_lut=lut)

    assert list(out.columns) == ["oid", "sid", "feature_id", "band", "version", "value"]
    assert (out["oid"] == 123).all()
    assert (out["sid"] == 0).all()
    assert (out["version"] == 7).all()
    assert out["value"].notna().all()        # NaN Period row dropped
    assert len(out) == 2
    assert set(out["band"]) == {1, 2}         # only the surviving g + r rows
    assert set(out["feature_id"]) == {0}      # both rows are Amplitude -> id 0


def test_compute_db_features_returns_none_when_no_astro_object(monkeypatch):
    monkeypatch.setattr(lc_features, "compute_astro_object", lambda *a, **k: None)
    assert lc_features.compute_db_features({"oid": 1}, None, None, feature_name_lut={}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_db_features.py -v`
Expected: FAIL — `AttributeError: module 'features.offline.lc_features' has no attribute 'compute_db_features'`.

- [ ] **Step 3: Implement `compute_db_features`**

In `feature_step/features/offline/lc_features.py`, add imports near the top (after the existing imports):

```python
from importlib.metadata import version as _pkg_version

from features.utils.parsers import prepare_ao_features_for_db
from features.offline.feature_lut import load_feature_name_lut, version_name_to_id

SID_ZTF = 0
```

Then add the function at the end of the file:

```python
def compute_db_features(message: dict, references_db, allwise, min_detections: int = 1,
                        preprocessor=None, extractor=None, feature_name_lut=None,
                        version_name=None):
    """Per-oid path -> DB-ready feature rows, following the production save rules.

    Returns a DataFrame with exactly the `feature` table columns
    [oid, sid, feature_id, band, version, value] (NaN values dropped,
    fid->band code, name->feature_id via the LUT), or None if too few real
    detections. `compute_features`/`compute_astro_object` keep the named,
    NaN-inclusive frame for classification.

    `version` mirrors production: it is the single `feature-step` package
    version (`version("feature-step")`, e.g. "27.5.7a31"), mapped to a smallint
    via the fixture's FEATURE_VERSION_LUT — NOT the per-module ao.features
    ["version"] column. Override `version_name` for tests.
    """
    ao = compute_astro_object(message, references_db, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None

    lut = feature_name_lut if feature_name_lut is not None else load_feature_name_lut()
    rows = prepare_ao_features_for_db(ao, lut)  # [name, value, band, feature_id]

    if version_name is None:
        version_name = _pkg_version("feature-step")

    rows = rows.copy()
    rows["oid"] = int(message["oid"])
    rows["sid"] = SID_ZTF
    rows["version"] = version_name_to_id(version_name)
    rows = rows.drop(columns=["name"])
    return rows[["oid", "sid", "feature_id", "band", "version", "value"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_db_features.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/lc_features.py feature_step/tests/unittest/test_offline_db_features.py
git commit -m "feat(feature_step): compute_db_features — DB-ready offline ZTF rows"
```

---

## Task 5: Repoint `offline_compute_features.py` to the DB-ready output

**Files:**
- Modify: `feature_step/scripts/offline_compute_features.py:51-58`

- [ ] **Step 1: Update the compute + print block**

Replace lines 51-58 (the `compute_features(...)` call and its validation/print) with:

```python
    features = lc_features.compute_db_features(message, refs, allwise)
    if features is None or len(features) == 0:
        print("\nFAIL: empty DB-ready features frame")
        sys.exit(1)
    print(f"\nDB-ready features: {features.shape}; columns={list(features.columns)}")
    print(features.head(20).to_string())
    print("\nOK: DB-ready feature rows produced.")
```

- [ ] **Step 2: Verify it runs (manual, needs DB)**

Run: `conda run --no-capture-output -n training_py310 python feature_step/scripts/offline_compute_features.py --oid 36028941624528297`
Expected: prints `DB-ready features: (N, 6); columns=['oid', 'sid', 'feature_id', 'band', 'version', 'value']` with populated rows and exit 0. Spot-check that `feature_id`/`band`/`version` are integers and there are no NaN `value`s.

- [ ] **Step 3: Commit**

```bash
git add feature_step/scripts/offline_compute_features.py
git commit -m "feat(feature_step): offline_compute_features emits DB-ready rows"
```

---

## Task 6: Documentation

**Files:**
- Modify: `feature_step/features/offline/FLOW.md`, `feature_step/features/offline/README.md`

- [ ] **Step 1: Update FLOW.md**

In `FLOW.md` §5 (Compute features), add a paragraph after the `compute_astro_object`/`compute_features` bullet:

```markdown
- `compute_db_features` is the **DB-ready** output: it runs the same compute,
  then applies the production save rules via `prepare_ao_features_for_db`
  (drop NaN, `fid→band` code, `name→feature_id` via the LUT) and attaches
  `oid, sid=0, version`. Result columns match the `feature` table exactly:
  `[oid, sid, feature_id, band, version, value]`. The `feature_id`/`version`
  maps come from the **local fixture** `offline/feature_lut.py` (the DB ZTF
  `feature_name_lut`/`feature_version_lut` are still empty — §3d); fixture ids
  are offline's own until that LUT is seeded. `compute_features` keeps the
  named, NaN-inclusive frame for `classify.py` / `compare_vs_alerce`.
```

In `FLOW.md` §8 (File map), add rows:

```markdown
| `feature_lut.py` | Local ZTF feature_name/version LUT fixture + loaders (`load_feature_name_lut`, `version_name_to_id`). |
| `scripts/offline_generate_feature_lut.py` | One-off generator that prints the fixture from a real run. |
```

- [ ] **Step 2: Update README.md**

In `README.md`'s file table, add a row for `feature_lut.py` and note next to `lc_features.py` that it now also exposes `compute_db_features` (DB-ready rows: drop NaN + `fid→band` + `name→feature_id` via the fixture LUT).

- [ ] **Step 3: Commit**

```bash
git add feature_step/features/offline/FLOW.md feature_step/features/offline/README.md
git commit -m "docs(feature_step): document compute_db_features + feature LUT fixture"
```

---

## Final verification

- [ ] **Run the full offline unit suite**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_prepare_ao_features_for_db.py tests/unittest/test_offline_feature_lut.py tests/unittest/test_offline_db_features.py tests/unittest/test_offline_classify.py tests/unittest/test_offline_feature_compare.py -v`
Expected: all PASS (the existing offline tests still pass — `compute_features`/`classify` are untouched).
