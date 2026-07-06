# ZTF BHRF Classifier + Taxonomy Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seed the BHRF (Squidward 2.1.0) hierarchical LC classifier and its taxonomy into `multisurvey_ztf` — 5 `classifier` rows + 45 `taxonomy` rows — so BHRF probabilities will later have a DB home.

**Architecture:** Follow the existing feature-LUT idiom exactly: a Python fixture module (`classifier_taxonomy_lut.py`) is the single source of truth, and the idempotent `ztf_classifier_taxonomy_seed.sql` is *generated from it* by a `render_seed_sql()` function. Unit tests assert structural invariants on the fixture and that the committed SQL matches the render (drift guard). An optional standalone script cross-checks the fixture's class names against the deployed model pickle (the correctness-critical guarantee: an exact class-name match is required or `class_name → class_id` lookups return garbage). Applying the SQL to live is a manual, write-credentialed step.

**Tech Stack:** Python 3.10 (`training_py310` conda env for tests/pickle), pytest, PostgreSQL (`multisurvey_ztf` schema), `alerce_classifiers` submodule (pickle load only).

---

## Context the engineer needs (read first)

- **Spec:** `docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md` — approved design. Class names are **locked to `SESN`** (not `SNIbc`), md5-verified against the deployed S3 pickle.
- **Sibling pattern (copy its shape):** `feature_step/features/offline/feature_lut.py` (fixture) + `feature_step/features/offline/ztf_feature_luts_seed.sql` (generated SQL) + `feature_step/tests/unittest/test_offline_feature_lut.py` (invariant tests).
- **Live table schemas** (`libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py`):
  - `classifier`: `classifier_id (int)`, `classifier_name (varchar)`, `classifier_version (varchar)`, `tid (smallint)`, `created_date (server default)`. **PK = `(classifier_id)`** (`pk_classifier_classifierid`).
  - `taxonomy`: `class_id (int)`, `class_name (varchar)`, `order (int)`, `classifier_id (smallint)`, `created_date (server default)`. **PK = `(class_id, classifier_id)`** (`pk_taxonomy_classid_classifierid`).
  - `order` is a **reserved SQL word** — it must be quoted as `"order"` in the INSERT.
  - `created_date` has a server default — do **not** supply it in the INSERT (the feature-LUT seed omits it too).
- **Authority file drift note (for the back-port task):** `libs/db-plugins-multisurvey/db_plugins/db/sql/_initial_data_pipeline.py` is *already ahead* of what the spec described — it now uses `index_elements: ["class_id", "classifier_id"]` for taxonomy and carries `classifier_id 3` taxonomy rows. But its `classifier` block still only has ids **1 and 2** (stamp). The live DB has ids **1–4**. So the back-port must still reconcile the missing live ids 3–4 before adding BHRF 5–9. Back-port is **out of scope for this plan** (recorded as pending only).

### The locked class lists (from the spec, md5-verified against the deployed pickle)

`classifier_version = "2.1.0"`, `tid = 0` for all 5. IDs are next-free (live max = 4 → **5–9**), **re-verify at apply time**.

| classifier_id | classifier_name | classes in `class_id` = `order` = 0..n order |
|---|---|---|
| 5 | `lc_classifier_BHRF_forced_phot` (flat, 21) | AGN, Blazar, CEP, CV/Nova, DSCT, EA, EB/EW, LPV, Microlensing, Periodic-Other, QSO, RRLab, RRLc, RSCVn, SESN, SLSN, SNII, SNIIn, SNIa, TDE, YSO |
| 6 | `lc_classifier_BHRF_forced_phot_top` (3) | Periodic, Stochastic, Transient |
| 7 | `lc_classifier_BHRF_forced_phot_transient` (6) | SESN, SLSN, SNII, SNIIn, SNIa, TDE |
| 8 | `lc_classifier_BHRF_forced_phot_stochastic` (6) | AGN, Blazar, CV/Nova, Microlensing, QSO, YSO |
| 9 | `lc_classifier_BHRF_forced_phot_periodic` (9) | CEP, DSCT, EA, EB/EW, LPV, Periodic-Other, RRLab, RRLc, RSCVn |

Per classifier, `class_id` is per-classifier 0-indexed in the order above, and `order = class_id`.

### How to run tests

```bash
cd /home/fandrades/desktop/pipeline/feature_step
conda run -n training_py310 python -m pytest tests/unittest/<file> -q
```

---

## Task 1: Fixture module — `classifier_taxonomy_lut.py` (source of truth + SQL renderer)

**Files:**
- Create: `feature_step/features/offline/classifier_taxonomy_lut.py`
- Test: `feature_step/tests/unittest/test_offline_classifier_taxonomy_lut.py`

- [ ] **Step 1: Write the failing test**

```python
# feature_step/tests/unittest/test_offline_classifier_taxonomy_lut.py
"""Unit tests for the BHRF classifier + taxonomy seed fixture and SQL renderer."""
from features.offline.classifier_taxonomy_lut import (
    CLASSIFIER_LUT,
    TAXONOMY_LUT,
    CLASSIFIER_VERSION,
    render_seed_sql,
)

FLAT_ID = 5


def test_five_classifiers_ids_5_to_9():
    ids = [c["classifier_id"] for c in CLASSIFIER_LUT]
    assert ids == [5, 6, 7, 8, 9]


def test_classifier_names_and_version():
    by_id = {c["classifier_id"]: c for c in CLASSIFIER_LUT}
    assert by_id[5]["classifier_name"] == "lc_classifier_BHRF_forced_phot"
    assert by_id[6]["classifier_name"] == "lc_classifier_BHRF_forced_phot_top"
    assert by_id[7]["classifier_name"] == "lc_classifier_BHRF_forced_phot_transient"
    assert by_id[8]["classifier_name"] == "lc_classifier_BHRF_forced_phot_stochastic"
    assert by_id[9]["classifier_name"] == "lc_classifier_BHRF_forced_phot_periodic"
    assert all(c["classifier_version"] == "2.1.0" for c in CLASSIFIER_LUT)
    assert all(c["tid"] == 0 for c in CLASSIFIER_LUT)
    assert CLASSIFIER_VERSION == "2.1.0"


def test_taxonomy_class_counts():
    counts = {cid: len(classes) for cid, classes in TAXONOMY_LUT.items()}
    assert counts == {5: 21, 6: 3, 7: 6, 8: 6, 9: 9}
    total = sum(counts.values())
    assert total == 45


def test_transient_uses_sesn_not_snibc():
    assert "SESN" in TAXONOMY_LUT[7]
    assert "SNIbc" not in TAXONOMY_LUT[7]
    assert "SESN" in TAXONOMY_LUT[FLAT_ID]
    assert "SNIbc" not in TAXONOMY_LUT[FLAT_ID]


def test_flat_is_union_of_branches():
    branches = set(TAXONOMY_LUT[7]) | set(TAXONOMY_LUT[8]) | set(TAXONOMY_LUT[9])
    assert set(TAXONOMY_LUT[FLAT_ID]) == branches  # 6 + 6 + 9 = 21 leaves


def test_every_classifier_has_taxonomy():
    assert set(TAXONOMY_LUT) == {c["classifier_id"] for c in CLASSIFIER_LUT}


def test_render_seed_sql_is_idempotent_and_targets_composite_pk():
    sql = render_seed_sql()
    # classifier upsert targets the classifier_id PK
    assert "ON CONFLICT (classifier_id) DO NOTHING" in sql
    # taxonomy upsert targets the composite PK, NOT class_id alone
    assert 'ON CONFLICT (class_id, classifier_id) DO NOTHING' in sql
    # order is quoted (reserved word)
    assert '"order"' in sql
    # created_date is NOT supplied (server default)
    assert "created_date" not in sql
    # a spot-check row: flat classifier SESN at class_id 14
    assert "(14, 'SESN', 14, 5)" in sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n training_py310 python -m pytest tests/unittest/test_offline_classifier_taxonomy_lut.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.offline.classifier_taxonomy_lut'`

- [ ] **Step 3: Write the fixture module**

```python
# feature_step/features/offline/classifier_taxonomy_lut.py
"""BHRF (Squidward 2.1.0) classifier + taxonomy seed fixture — single source of truth.

Class names/order are md5-verified against the deployed S3 pickle
(``squidward/2.1.0/hierarchical_random_forest_model.pkl``); the transient class is
``SESN`` (not ``SNIbc``). See
``docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md``.

``ztf_classifier_taxonomy_seed.sql`` is generated from this module via
``render_seed_sql()`` — edit here, never hand-edit the SQL.

Names MUST match the model's ``classes_`` exactly: the multisurvey write path maps
``class_name -> class_id`` by exact string match, so a wrong label yields a garbage
class_id.
"""

CLASSIFIER_VERSION = "2.1.0"
TID = 0  # ZTF

# next-free classifier_id: live max is 4 (four stamp classifiers) -> 5..9.
# RE-VERIFY against live at apply time (another deploy could claim 5+ first).
CLASSIFIER_LUT = [
    {"classifier_id": 5, "classifier_name": "lc_classifier_BHRF_forced_phot",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 6, "classifier_name": "lc_classifier_BHRF_forced_phot_top",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 7, "classifier_name": "lc_classifier_BHRF_forced_phot_transient",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 8, "classifier_name": "lc_classifier_BHRF_forced_phot_stochastic",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
    {"classifier_id": 9, "classifier_name": "lc_classifier_BHRF_forced_phot_periodic",
     "classifier_version": CLASSIFIER_VERSION, "tid": TID},
]

# classifier_id -> ordered class names (class_id = order = list position).
# Ordering is the model's classes_ (branches alphabetical; flat = list_of_classes).
TAXONOMY_LUT = {
    5: [  # flat, 21 leaves
        "AGN", "Blazar", "CEP", "CV/Nova", "DSCT", "EA", "EB/EW", "LPV",
        "Microlensing", "Periodic-Other", "QSO", "RRLab", "RRLc", "RSCVn",
        "SESN", "SLSN", "SNII", "SNIIn", "SNIa", "TDE", "YSO",
    ],
    6: ["Periodic", "Stochastic", "Transient"],  # top
    7: ["SESN", "SLSN", "SNII", "SNIIn", "SNIa", "TDE"],  # transient
    8: ["AGN", "Blazar", "CV/Nova", "Microlensing", "QSO", "YSO"],  # stochastic
    9: ["CEP", "DSCT", "EA", "EB/EW", "LPV", "Periodic-Other",
        "RRLab", "RRLc", "RSCVn"],  # periodic
}

SCHEMA = "multisurvey_ztf"


def _sql_str(value: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def render_seed_sql() -> str:
    """Render the idempotent seed SQL from the fixture (single source of truth)."""
    lines = [
        "-- BHRF (Squidward 2.1.0) classifier + taxonomy seed for ZTF (tid=0).",
        "-- Generated from feature_step/features/offline/classifier_taxonomy_lut.py",
        "--   (single source of truth). Do not hand-edit; regenerate from the fixture.",
        "-- Run with:  psql \"<conn>\" -f ztf_classifier_taxonomy_seed.sql",
        "--   (needs INSERT privileges, not readonly_user).",
        "--",
        "-- RE-VERIFY next-free classifier_id against live before applying:",
        "--   SELECT classifier_id, classifier_name, classifier_version, tid",
        f"--     FROM {SCHEMA}.classifier ORDER BY classifier_id;",
        "",
        "-- 1. classifier (5 rows: flat + top + 3 branches)",
        f"INSERT INTO {SCHEMA}.classifier "
        "(classifier_id, classifier_name, classifier_version, tid) VALUES",
    ]
    crows = [
        f"({c['classifier_id']}, {_sql_str(c['classifier_name'])}, "
        f"{_sql_str(c['classifier_version'])}, {c['tid']})"
        for c in CLASSIFIER_LUT
    ]
    lines.append(",\n".join(crows))
    lines.append("ON CONFLICT (classifier_id) DO NOTHING;")
    lines.append("")
    lines.append("-- 2. taxonomy (45 rows: 21 + 3 + 6 + 6 + 9; class_id per-classifier 0-indexed)")
    lines.append(
        f"INSERT INTO {SCHEMA}.taxonomy "
        '(class_id, class_name, "order", classifier_id) VALUES'
    )
    trows = []
    for classifier_id in sorted(TAXONOMY_LUT):
        for class_id, class_name in enumerate(TAXONOMY_LUT[classifier_id]):
            trows.append(
                f"({class_id}, {_sql_str(class_name)}, {class_id}, {classifier_id})"
            )
    lines.append(",\n".join(trows))
    lines.append("ON CONFLICT (class_id, classifier_id) DO NOTHING;")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n training_py310 python -m pytest tests/unittest/test_offline_classifier_taxonomy_lut.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/classifier_taxonomy_lut.py \
        feature_step/tests/unittest/test_offline_classifier_taxonomy_lut.py
git commit -m "feat(feature_step): BHRF classifier+taxonomy seed fixture (SESN) + SQL renderer"
```

---

## Task 2: Generate the committed SQL + drift-guard test

**Files:**
- Create: `feature_step/features/offline/ztf_classifier_taxonomy_seed.sql` (generated)
- Modify: `feature_step/tests/unittest/test_offline_classifier_taxonomy_lut.py` (add drift test)

- [ ] **Step 1: Write the failing drift-guard test**

Append to `test_offline_classifier_taxonomy_lut.py`:

```python
import pathlib

_SQL_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "features" / "offline" / "ztf_classifier_taxonomy_seed.sql"
)


def test_committed_sql_matches_render():
    # The .sql on disk must be exactly what render_seed_sql() produces, so the
    # fixture stays the single source of truth (no hand-edited drift).
    assert _SQL_PATH.exists(), f"missing generated SQL: {_SQL_PATH}"
    assert _SQL_PATH.read_text() == render_seed_sql()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n training_py310 python -m pytest tests/unittest/test_offline_classifier_taxonomy_lut.py::test_committed_sql_matches_render -q`
Expected: FAIL — `AssertionError: missing generated SQL: .../ztf_classifier_taxonomy_seed.sql`

- [ ] **Step 3: Generate the SQL file from the fixture**

Run (writes the file from the single source of truth):

```bash
cd /home/fandrades/desktop/pipeline/feature_step
conda run -n training_py310 python -c "from features.offline.classifier_taxonomy_lut import render_seed_sql; open('features/offline/ztf_classifier_taxonomy_seed.sql','w').write(render_seed_sql())"
```

- [ ] **Step 4: Run test to verify it passes + eyeball the SQL**

Run: `conda run -n training_py310 python -m pytest tests/unittest/test_offline_classifier_taxonomy_lut.py -q`
Expected: PASS (9 passed)

Then visually confirm the file: `cat features/offline/ztf_classifier_taxonomy_seed.sql` — 5 classifier rows, 45 taxonomy rows, `"order"` quoted, both `ON CONFLICT` targets correct.

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/ztf_classifier_taxonomy_seed.sql \
        feature_step/tests/unittest/test_offline_classifier_taxonomy_lut.py
git commit -m "feat(feature_step): generate ztf_classifier_taxonomy_seed.sql from fixture"
```

---

## Task 3: Optional pickle cross-check script (correctness guard)

Mirrors the `--smoke` guard in `offline_verify_model_features.py`: loads the **deployed** model pickle and asserts the fixture's class names/order equal the model's `list_of_classes` (flat) and each branch's `classes_`. This is the concrete defense against the `SNIbc`/`SESN` class-name risk called out in the spec. It is **manual/optional** — it needs the ~1.72 GB pickle and `imblearn` — so it is a standalone script, not a unit test.

**Files:**
- Create: `feature_step/scripts/offline_verify_taxonomy.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Cross-check the classifier_taxonomy_lut fixture against the deployed BHRF pickle.

Loads the model at MODEL_PATH (URL or local md5-verified copy) and asserts the
fixture's class names + order match the model's list_of_classes (flat) and each
branch's classes_. Exit 0 on match, 1 on any mismatch.

    MODEL_PATH=/path/to/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        python feature_step/scripts/offline_verify_taxonomy.py

Requires an env where imblearn/alerce_classifiers import cleanly (training_py310).
"""
import os
import sys

from features.offline.classifier_taxonomy_lut import TAXONOMY_LUT

# fixture classifier_id -> model branch key in dict_of_rf
BRANCH_KEY = {6: "top", 7: "Transient", 8: "Stochastic", 9: "Periodic"}


def _load_model():
    from classify import load_squidward_model  # reuse the deployed loader
    return load_squidward_model()


def main() -> int:
    if not os.environ.get("MODEL_PATH"):
        print("MODEL_PATH not set — point it at the deployed pickle.", file=sys.stderr)
        return 2
    model = _load_model()
    hrf = model.model  # HierarchicalRandomForestClassifier

    problems = []

    flat_model = list(hrf.list_of_classes)
    if flat_model != TAXONOMY_LUT[5]:
        problems.append(
            f"flat: fixture {TAXONOMY_LUT[5]} != model list_of_classes {flat_model}"
        )

    for cid, key in BRANCH_KEY.items():
        model_classes = list(hrf.dict_of_rf[key].classes_)
        if model_classes != TAXONOMY_LUT[cid]:
            problems.append(
                f"{key} (classifier_id {cid}): fixture {TAXONOMY_LUT[cid]} "
                f"!= model classes_ {model_classes}"
            )

    if problems:
        print("MISMATCH — fixture disagrees with the deployed model:")
        for p in problems:
            print("  -", p)
        return 1
    print("OK — fixture class names + order match the deployed model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Note:** `model.model` and `hrf.dict_of_rf` / `hrf.list_of_classes` are the attribute
> names read in the spec (§"Deployed model artifact"). If the loaded object exposes
> them under different names, adjust to match — do **not** guess; print `dir(model)` /
> `dir(hrf)` first. If `load_squidward_model` needs `MAPPER_CLASS`, the script inherits
> the same env-var contract as `offline_classify.py`.

- [ ] **Step 2: Run it against the md5-verified local pickle (manual)**

Run:
```bash
cd /home/fandrades/desktop/pipeline/feature_step
MODEL_PATH=/home/fandrades/desktop/alerce_models/squidward/2.1.0/hierarchical_random_forest_model.pkl \
  PYTHONPATH=../alerce_classifiers:$PYTHONPATH \
  conda run -n training_py310 python scripts/offline_verify_taxonomy.py
```
Expected: `OK — fixture class names + order match the deployed model.` (exit 0)

If it prints MISMATCH, **stop** — fix `classifier_taxonomy_lut.py` to match the model, regenerate the SQL (Task 2 Step 3), and re-run Task 1/2 tests before proceeding.

- [ ] **Step 3: Commit**

```bash
git add feature_step/scripts/offline_verify_taxonomy.py
git commit -m "feat(feature_step): offline taxonomy cross-check vs deployed BHRF pickle"
```

---

## Task 4: Docs — FLOW.md, README, back-port note

**Files:**
- Modify: `feature_step/features/offline/FLOW.md` (§3d table, §7 Done/Pending, §8 file map)
- Modify: `feature_step/features/offline/README.md` (Modules/file map + status)

- [ ] **Step 1: Update FLOW.md §3d LUT table**

In the `## 3d.` table, change the `classifier` and `taxonomy` rows from **Pending** to **Done**, matching the actual seed. Replace the existing `classifier` and `taxonomy` rows with:

```markdown
| `classifier` | `(classifier_id)` | ids 1–4 stamp **+ ids 5–9 BHRF** (flat + top + 3 branches, `classifier_version = "2.1.0"`, `tid = 0`) | **Done** — seeded via `ztf_classifier_taxonomy_seed.sql`. Back-port to authority file pending. |
| `taxonomy` | `(class_id, classifier_id)` | flat stamp taxonomy **+ 45 BHRF rows** (21+3+6+6+9; `class_id` per-classifier 0-indexed, `order = class_id`; transient uses **`SESN`**) | **Done** — same seed file. Back-port pending. |
```

- [ ] **Step 2: Update FLOW.md §7 status lists**

In `**Done & working:**` add a bullet:

```markdown
- **BHRF `classifier` + `taxonomy` LUTs seeded** (5 classifier rows ids 5–9 +
  45 taxonomy rows) via `ztf_classifier_taxonomy_seed.sql`, generated from the
  `classifier_taxonomy_lut.py` fixture and cross-checked against the deployed
  pickle (`scripts/offline_verify_taxonomy.py`). Class names locked to **`SESN`**.
  ⚠ seeded directly, not yet back-ported to the db-plugins authority file (§3d).
```

In `**Pending / deferred:**`, replace the "Seed the BHRF `classifier` + `taxonomy` LUTs" bullet with a back-port-only bullet:

```markdown
- **Back-port the BHRF `classifier` + `taxonomy` rows to the authority file** —
  seeded directly via `ztf_classifier_taxonomy_seed.sql`; `_initial_data_pipeline.py`
  does not yet carry BHRF ids 5–9. Must first reconcile the missing live
  `classifier` ids 3–4 (this checkout's `classifier` block stops at id 2) before
  adding 5–9, or it will renumber over real ids (§3d).
```

- [ ] **Step 3: Update FLOW.md §8 file map**

Add two rows to the §8 table (after the `feature_lut.py` row and the `ztf_feature_luts_seed.sql` row respectively):

```markdown
| `classifier_taxonomy_lut.py` | BHRF classifier + taxonomy seed fixture (source of truth) + `render_seed_sql()`. §3d |
| `ztf_classifier_taxonomy_seed.sql` | Idempotent SQL seeding the BHRF `classifier` (ids 5–9) + `taxonomy` (45 rows). Generated from `classifier_taxonomy_lut.py`. §3d |
```

- [ ] **Step 4: Update README.md Modules table + Scripts table**

Add to the `## Modules` table:

```markdown
| `classifier_taxonomy_lut.py` | BHRF (Squidward 2.1.0) `classifier` (ids 5–9) + `taxonomy` (45 rows) seed fixture — single source of truth. `render_seed_sql()` generates `ztf_classifier_taxonomy_seed.sql`. Class names locked to `SESN` (md5-verified vs the deployed pickle). |
```

Add to the `## Scripts` table:

```markdown
| `offline_verify_taxonomy.py` | Cross-checks the taxonomy fixture's class names/order against the deployed BHRF pickle at `MODEL_PATH`. Exit 0 on match. |
```

- [ ] **Step 5: Verify docs render + commit**

Run: `git diff --stat feature_step/features/offline/FLOW.md feature_step/features/offline/README.md`
Expected: both files modified.

```bash
git add feature_step/features/offline/FLOW.md feature_step/features/offline/README.md
git commit -m "docs(feature_step): record BHRF classifier+taxonomy seed (Done)"
```

---

## Task 5: Apply to live (MANUAL — write credentials, not part of automated run)

> This is a **manual operator step**, done once, with write-capable DB credentials.
> Offline default credentials are read-only. Same procedure used for the feature LUTs.

- [ ] **Step 1: Re-verify next-free `classifier_id` on live**

Run (read-only is fine):
```sql
SELECT classifier_id, classifier_name, classifier_version, tid
FROM multisurvey_ztf.classifier ORDER BY classifier_id;
```
Expected: ids **1–4** present (all stamp), **none ≥ 5**. If anything already occupies 5–9, **stop** — renumber the fixture (Task 1) + regenerate SQL (Task 2) before applying.

- [ ] **Step 2: Re-confirm the deployed model's class labels (Task 3)**

Run `offline_verify_taxonomy.py` against the pickle currently on the production S3 URL (not any `/tmp` copy). Expected: `OK`. This is the last guard before writing — the write path maps `class_name → class_id` by exact match.

- [ ] **Step 3: Apply the seed**

```bash
psql "<write-capable conn to multisurvey_ztf>" \
  -f feature_step/features/offline/ztf_classifier_taxonomy_seed.sql
```
Expected: `INSERT 0 5` (classifier) then `INSERT 0 45` (taxonomy) on a fresh apply; `INSERT 0 0` on a re-run (idempotent).

- [ ] **Step 4: Verify the rows landed**

```sql
SELECT classifier_id, classifier_name, classifier_version
FROM multisurvey_ztf.classifier WHERE classifier_id BETWEEN 5 AND 9 ORDER BY classifier_id;

SELECT classifier_id, count(*)
FROM multisurvey_ztf.taxonomy WHERE classifier_id BETWEEN 5 AND 9
GROUP BY classifier_id ORDER BY classifier_id;
```
Expected: 5 classifier rows; taxonomy counts `5→21, 6→3, 7→6, 8→6, 9→9`.

---

## Self-Review (completed against the spec)

- **Spec coverage:** Deliverable §1 (SQL file) → Tasks 1–2. §1's PK/conflict targets (`classifier` → `(classifier_id)`, `taxonomy` → `(class_id, classifier_id)`) → asserted in Task 1 test + rendered in Task 2. §2 (apply direct-to-live) → Task 5. §3 docs (FLOW §3d/§7, README, back-port pending note) → Task 4. `SESN` lock → Task 1 tests + Task 3 pickle guard. Next-free-id risk → Task 5 Step 1. Wrong-artifact risk → Task 3 + Task 5 Step 2.
- **Out of scope (correctly excluded):** `probability_writer.py`, `offline_classify.py --save`, the `classifier_version` smallint convention, actually persisting probabilities, and the db-plugins back-port (recorded as pending in Task 4 only).
- **Placeholder scan:** none — every code/SQL/command step is concrete. Task 3's one soft spot (model attribute names) is flagged with an explicit "don't guess, print `dir()`" instruction because it depends on the loaded object's shape.
- **Type/name consistency:** `CLASSIFIER_LUT`, `TAXONOMY_LUT`, `CLASSIFIER_VERSION`, `render_seed_sql`, `_SQL_PATH` used identically across Tasks 1–3.
