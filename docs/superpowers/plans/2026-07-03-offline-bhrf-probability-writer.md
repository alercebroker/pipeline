# Offline BHRF Probability Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist offline BHRF (Squidward 2.1.0) probabilities into `multisurvey_ztf.probability` — one row per class per classifier across all 5 seeded classifiers (ids 5–9) — via a `probability_writer.py` that mirrors `feature_writer.py` (dry-run by default), wired into `offline_classify.py --save`.

**Architecture:** A pure `build_probability_rows(output_dto, oid, lastmjd, taxonomy_by_classifier)` turns the BHRF `OutputDTO` (flat + hierarchical frames) into DB-ready row dicts, mapping each model class *name* → integer `class_id` using a `{classifier_id: {class_name: class_id}}` map **read from the DB `taxonomy` table** (the authority for the `probability.class_id` relationship, exactly as production's `get_taxonomy_by_classifier_id` does). The map is fetched read-only by `db.fetch_taxonomy_maps(...)` and passed into the pure builder (so the builder stays unit-testable with a literal dict). `write_probabilities(...)` upserts the rows in one transaction, dry-run unless `execute=True`. `classify.py` gains a save-capable path that also returns `lastmjd`; `offline_classify.py --save` wires it all together.

**Tech Stack:** Python 3.10 (`training_py310` conda env), pandas, SQLAlchemy, pytest. PostgreSQL `multisurvey_ztf` schema.

---

## Context the engineer needs (read first)

- **Design spec:** `docs/superpowers/specs/2026-07-03-offline-bhrf-probability-writer-design.md`.
- **Pattern to mirror for the writer shape:** `feature_step/features/offline/feature_writer.py` + its tests `feature_step/tests/unittest/test_offline_feature_writer.py` (fake-engine, dry-run-doesn't-connect, native-type coercion). Read both before starting. **Difference to note:** `feature_writer` maps names→ids from the *local fixture* only because the DB feature LUT is unseeded; the `taxonomy` table **is** seeded, so this writer reads ids from the **DB** instead (the DB is authoritative for the `probability.class_id` FK).
- **Production reference (already traced):** `stamp_classifier_2025_multisurvey_step/stamp_classifier_2025_multisurvey_step/db/db.py` — `get_taxonomy_by_classifier_id` (reads `{class_name: class_id}` from the DB, `ORDER BY "order"`), `format_probability_records` (melt→rank→map class_id→version smallint), `classifier_version_str_to_small_integer`, `class_name_to_id` (`-1` on miss).
- **The seeded taxonomy (already applied to live):** `multisurvey_ztf.taxonomy` holds the 45 BHRF rows for classifier_ids 5–9. The local fixture `feature_step/features/offline/classifier_taxonomy_lut.py` (`TAXONOMY_LUT`, `CLASSIFIER_VERSION == "2.1.0"`) is the *source* the seed SQL was generated from — useful as a **test oracle**, but the runtime `class_id` source is the DB.
- **The `OutputDTO` shape (runtime-verified):** From `alerce_classifiers/squidward/{model.py,mapper.py}` + `classifiers/hierarchical_random_forest.py::classify_batch`:
  - `out.probabilities` — flat frame, columns = the model's 21 `list_of_classes`, indexed by oid.
  - `out.hierarchical["top"]` — 3-class frame (columns `Periodic, Stochastic, Transient`), indexed by oid.
  - `out.hierarchical["children"]["Transient"|"Stochastic"|"Periodic"]` — 6/6/9-class frames, indexed by oid.
  - `out.hierarchical` is a plain dict (accessed as `out.hierarchical.get("top")`, see `classify.py`).
  - An empty/can't-predict `OutputDTO` has `probabilities` empty and `hierarchical == {"top": <empty>, "children": {}}`.
- **The `probability` table** (`libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py:968`):
  - Columns: `oid (bigint)`, `sid (smallint)`, `classifier_id (smallint)`, `classifier_version (smallint)`, `class_id (smallint)`, `probability (real)`, `ranking (smallint, nullable)`, `lastmjd (double, NOT NULL)`. **No `updated_date` column** (unlike `feature`).
  - **PK / conflict target = `(oid, sid, classifier_id, class_id)`**. Hash-partitioned on `oid`.
- **`db.py` conventions** (`feature_step/features/offline/db.py`): `import sqlalchemy as sa`, `from sqlalchemy import text`; `SCHEMA` const (env `OFFLINE_DB_SCHEMA`, default `multisurvey_ztf`); `_make_engine(credentials_json)`; schema is interpolated with an f-string (trusted operator input, same convention as all readers).

### Frame → classifier_id map (used throughout)

| classifier_id | source frame |
|---|---|
| 5 | `out.probabilities` (flat 21) |
| 6 | `out.hierarchical["top"]` (3) |
| 7 | `out.hierarchical["children"]["Transient"]` (6) |
| 8 | `out.hierarchical["children"]["Stochastic"]` (6) |
| 9 | `out.hierarchical["children"]["Periodic"]` (9) |

### Locked decisions (from the spec's open points)

- **Conflict policy: `DO UPDATE`** (refresh probability/classifier_version/ranking/lastmjd) — matches `feature_writer`, supports re-runs.
- **Write all 5 classifiers.**
- **`classifier_version` smallint:** `"2.1.0" → 210` via the production rule.
- **`class_id` source: the DB `taxonomy` table** (authority), fetched read-only and passed into the pure builder. (Corrects an earlier draft that used the local fixture.)
- **`lastmjd` = max `mjd` over all epochs the classifier consumed (real detections + forced photometry).** This refines the design's "real detections" wording: `lastmjd` is `NOT NULL` and BHRF is a forced-phot classifier, so its data horizon includes forced epochs. **Do NOT subtract 2400000.5** — the offline pipeline is already in MJD (a JD→MJD subtraction here would yield a nonsense negative epoch).

### How to run tests

```bash
cd /home/fandrades/desktop/pipeline/feature_step
conda run -n training_py310 python -m pytest tests/unittest/<file> -q
```

### Git hygiene (applies to every commit in this plan)

The repo has unrelated uncommitted changes (P4J/*.c/*.html, mhps, `message.py`, other offline scripts, untracked plan/spec docs). **Stage only the files each task names.** Never `git add -A`.

---

## Task 1: `probability_writer.py` — pure row builder (taxonomy passed in) + version helper

**Files:**
- Create: `feature_step/features/offline/probability_writer.py`
- Test: `feature_step/tests/unittest/test_offline_probability_writer.py`

- [ ] **Step 1: Write the failing tests**

```python
# feature_step/tests/unittest/test_offline_probability_writer.py
"""Unit tests for probability_writer — pure row building + fake-engine write."""
import pandas as pd
import pytest

from alerce_classifiers.base.dto import OutputDTO
from features.offline import probability_writer as pw
from features.offline.classifier_taxonomy_lut import TAXONOMY_LUT

OID = 36028941624528297

# Test oracle: {classifier_id: {class_name: class_id}} derived from the fixture
# (the runtime source is the DB; here we simulate what fetch_taxonomy_maps returns).
TAX_MAPS = {
    cid: {name: idx for idx, name in enumerate(names)}
    for cid, names in TAXONOMY_LUT.items()
}


def _frame(classifier_id, probs=None):
    """Build a 1-oid frame whose columns are the model's class labels for a classifier."""
    names = TAXONOMY_LUT[classifier_id]
    if probs is None:
        probs = [1.0 / (i + 1) for i in range(len(names))]  # strictly decreasing
    return pd.DataFrame([dict(zip(names, probs))], index=[OID])


def _full_dto():
    return OutputDTO(
        _frame(5),
        {"top": _frame(6),
         "children": {"Transient": _frame(7),
                      "Stochastic": _frame(8),
                      "Periodic": _frame(9)}},
    )


def test_version_to_smallint():
    assert pw.classifier_version_to_smallint("2.1.0") == 210
    assert pw.classifier_version_to_smallint("1.0.4") == 104
    assert pw.classifier_version_to_smallint("2.1.0_rc1") == 210  # patch suffix stripped
    assert pw.classifier_version_to_smallint("weird") == 0


def test_classifier_ids_constant():
    assert pw.CLASSIFIER_IDS == [5, 6, 7, 8, 9]


def test_build_rows_fans_out_to_45_rows():
    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    assert len(rows) == 45  # 21 + 3 + 6 + 6 + 9
    by_cls = {}
    for r in rows:
        by_cls[r["classifier_id"]] = by_cls.get(r["classifier_id"], 0) + 1
    assert by_cls == {5: 21, 6: 3, 7: 6, 8: 6, 9: 9}


def test_build_rows_field_values():
    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    r = next(r for r in rows if r["classifier_id"] == 7 and r["class_id"] == 0)
    # transient class_id 0 == "SESN" (fixture/DB), highest prob (1/1) -> ranking 1
    assert r["oid"] == OID and isinstance(r["oid"], int)
    assert r["sid"] == 0
    assert r["classifier_version"] == 210
    assert r["probability"] == pytest.approx(1.0)
    assert r["ranking"] == 1
    assert r["lastmjd"] == 60000.5
    # ranking is dense-desc within the classifier: class_id 5 (TDE, prob 1/6) -> rank 6
    last = next(r for r in rows if r["classifier_id"] == 7 and r["class_id"] == 5)
    assert last["ranking"] == 6


def test_build_rows_uses_class_ids_from_the_map_not_position():
    # A taxonomy map whose class_ids are NOT the enumerate position — the builder
    # must use the map's ids verbatim (proves it reads the DB map, not list index).
    maps = {6: {"Periodic": 42, "Stochastic": 43, "Transient": 44}}
    dto = OutputDTO(_frame(6), {"top": _frame(6), "children": {}})
    # only classifier 6 present in the map; probabilities frame (id 5) must also map,
    # so include id 5 too:
    maps[5] = TAX_MAPS[5]
    rows = pw.build_probability_rows(dto, OID, 1.0, maps)
    ids_for_6 = sorted(r["class_id"] for r in rows if r["classifier_id"] == 6)
    assert ids_for_6 == [42, 43, 44]


def test_build_rows_unknown_class_raises():
    bad = OutputDTO(
        pd.DataFrame([{"AGN": 0.5, "SNIbc": 0.5}], index=[OID]),  # SNIbc not in taxonomy
        {"top": pd.DataFrame(), "children": {}},
    )
    with pytest.raises(ValueError, match="SNIbc"):
        pw.build_probability_rows(bad, OID, 1.0, {5: {"AGN": 0}})


def test_build_rows_missing_classifier_in_map_raises():
    # classifier 5 present in output but absent from the taxonomy map -> hard error
    dto = OutputDTO(_frame(5), {"top": pd.DataFrame(), "children": {}})
    with pytest.raises(ValueError, match="classifier_id=5"):
        pw.build_probability_rows(dto, OID, 1.0, {6: {}})


def test_build_rows_empty_output_returns_empty():
    empty = OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})
    assert pw.build_probability_rows(empty, OID, 1.0, TAX_MAPS) == []


def test_build_rows_requires_lastmjd():
    with pytest.raises(ValueError, match="lastmjd"):
        pw.build_probability_rows(_full_dto(), OID, None, TAX_MAPS)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_probability_writer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.offline.probability_writer'`

- [ ] **Step 3: Implement the pure part of `probability_writer.py`**

```python
"""Persist offline BHRF probabilities into <schema>.probability.

Mirrors feature_writer.py (writing lives here, not in db.py). Turns the BHRF
OutputDTO (flat + hierarchical frames) into DB-ready rows across all 5 seeded
classifiers (ids 5-9). class_name -> class_id uses a {classifier_id: {class_name:
class_id}} map READ FROM THE DB taxonomy table (the authority for the
probability.class_id FK; see db.fetch_taxonomy_maps), passed in so this stays a
pure function. Upsert is ON CONFLICT (oid, sid, classifier_id, class_id) DO UPDATE.
Dry-run unless execute=True.
"""
import logging
from typing import Optional

from sqlalchemy import text

from features.offline import db
from features.offline.classifier_taxonomy_lut import CLASSIFIER_VERSION

log = logging.getLogger(__name__)

# The 5 seeded BHRF classifiers, in id order (flat, top, transient, stochastic, periodic).
CLASSIFIER_IDS = [5, 6, 7, 8, 9]


def classifier_version_to_smallint(version: str) -> int:
    """'2.1.0' -> 210 (production rule). Strips a '_suffix' on the patch part."""
    parts = version.split(".")
    if len(parts) == 3:
        parts[-1] = parts[-1].split("_")[0]
        return int("".join(parts))
    return 0


def _iter_frames(output_dto):
    """Yield (classifier_id, frame) for the 5 BHRF classifiers, in id order."""
    hierarchical = output_dto.hierarchical or {}
    children = hierarchical.get("children", {}) or {}
    return [
        (5, output_dto.probabilities),
        (6, hierarchical.get("top")),
        (7, children.get("Transient")),
        (8, children.get("Stochastic")),
        (9, children.get("Periodic")),
    ]


def build_probability_rows(output_dto, oid: int, lastmjd: float,
                           taxonomy_by_classifier: dict, *,
                           version: str = CLASSIFIER_VERSION, sid: int = 0) -> list:
    """BHRF OutputDTO (single oid) -> DB-ready probability row dicts (all 5 classifiers).

    taxonomy_by_classifier: {classifier_id: {class_name: class_id}} from the DB
    (db.fetch_taxonomy_maps). A class name not in its classifier's map, or a
    classifier with no map at all, raises (mirrors the -1 miss that would store
    garbage). Returns [] for an empty/can't-predict OutputDTO. ranking =
    per-classifier dense rank descending.
    """
    if output_dto is None or output_dto.probabilities is None or len(output_dto.probabilities) == 0:
        return []
    if lastmjd is None:
        raise ValueError("lastmjd is required (probability.lastmjd is NOT NULL)")

    version_smallint = classifier_version_to_smallint(version)
    rows = []
    for classifier_id, frame in _iter_frames(output_dto):
        if frame is None or len(frame) == 0:
            continue
        class_id_of = taxonomy_by_classifier.get(classifier_id)
        if not class_id_of:
            raise ValueError(
                f"no taxonomy map for classifier_id={classifier_id} "
                "(fetch_taxonomy_maps returned nothing for it — is the taxonomy seeded?)"
            )
        series = frame.iloc[0]  # single oid -> Series: class_name -> probability
        ranks = series.rank(ascending=False, method="dense").astype(int)
        for class_name in series.index:
            if class_name not in class_id_of:
                raise ValueError(
                    f"class '{class_name}' not in taxonomy for classifier_id={classifier_id} "
                    "(model/taxonomy mismatch — cannot map to class_id)"
                )
            rows.append({
                "oid": int(oid),
                "sid": int(sid),
                "classifier_id": int(classifier_id),
                "classifier_version": int(version_smallint),
                "class_id": int(class_id_of[class_name]),
                "probability": float(series[class_name]),
                "ranking": int(ranks[class_name]),
                "lastmjd": float(lastmjd),
            })
    return rows
```

- [ ] **Step 4: Run the tests, confirm all pass**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_probability_writer.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/probability_writer.py feature_step/tests/unittest/test_offline_probability_writer.py
git commit -m "feat(feature_step): pure BHRF probability row builder (class_id from DB taxonomy map)"
```

---

## Task 2: `db.fetch_taxonomy_maps` — read `{classifier_id: {class_name: class_id}}` from the DB

**Files:**
- Modify: `feature_step/features/offline/db.py`
- Test: `feature_step/tests/unittest/test_offline_db_taxonomy.py`

- [ ] **Step 1: Write the failing test**

```python
# feature_step/tests/unittest/test_offline_db_taxonomy.py
"""Unit test for db.fetch_taxonomy_maps — fake engine, no real DB."""
from features.offline import db


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self._rows


class _Conn:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((str(sql), params))
        return _Result(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, rows):
        self._rows = rows
        self.conn = _Conn(rows)

    def connect(self):
        return self.conn


def test_fetch_taxonomy_maps_groups_by_classifier(monkeypatch):
    rows = [
        {"classifier_id": 6, "class_id": 0, "class_name": "Periodic"},
        {"classifier_id": 6, "class_id": 1, "class_name": "Stochastic"},
        {"classifier_id": 6, "class_id": 2, "class_name": "Transient"},
        {"classifier_id": 7, "class_id": 0, "class_name": "SESN"},
        {"classifier_id": 7, "class_id": 1, "class_name": "SLSN"},
    ]
    engine = _Engine(rows)
    monkeypatch.setattr(db, "_make_engine", lambda _c: engine)

    maps = db.fetch_taxonomy_maps("creds", [6, 7], schema="multisurvey_ztf")

    assert maps == {
        6: {"Periodic": 0, "Stochastic": 1, "Transient": 2},
        7: {"SESN": 0, "SLSN": 1},
    }
    # class_id must be a native int, and the query targets the right schema/table
    assert isinstance(maps[7]["SESN"], int)
    sql, params = engine.conn.executed[0]
    assert "multisurvey_ztf.taxonomy" in sql
    assert params["cids"] == [6, 7]


def test_fetch_taxonomy_maps_default_schema(monkeypatch):
    engine = _Engine([])
    monkeypatch.setattr(db, "_make_engine", lambda _c: engine)
    db.fetch_taxonomy_maps("creds", [5])
    assert f"{db.SCHEMA}.taxonomy" in engine.conn.executed[0][0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_db_taxonomy.py -q`
Expected: FAIL — `AttributeError: module 'features.offline.db' has no attribute 'fetch_taxonomy_maps'`

- [ ] **Step 3: Add `fetch_taxonomy_maps` to `db.py`**

Append to `feature_step/features/offline/db.py` (uses the module's existing `sa`/`text`/`SCHEMA`/`_make_engine`). Add `from typing import Optional` at the top if not already present:

```python
def fetch_taxonomy_maps(credentials_json: str, classifier_ids: list,
                        schema: Optional[str] = None) -> dict:
    """Return {classifier_id: {class_name: class_id}} from <schema>.taxonomy.

    The authoritative class_name -> class_id mapping for writing probabilities
    (mirrors production's get_taxonomy_by_classifier_id). Read-only. Ordered by
    "order" per classifier (cosmetic for the dict, matches production).
    """
    schema = schema or SCHEMA
    engine = _make_engine(credentials_json)
    # schema is trusted operator input (env / CLI), same f-string convention as the
    # other readers; classifier_ids are bound as an expanding parameter.
    sql = text(
        f'SELECT classifier_id, class_id, class_name FROM {schema}.taxonomy '
        'WHERE classifier_id IN :cids ORDER BY classifier_id, "order"'
    ).bindparams(sa.bindparam("cids", expanding=True))

    maps: dict = {}
    with engine.connect() as conn:
        for row in conn.execute(sql, {"cids": _py_oids(classifier_ids)}).mappings():
            maps.setdefault(int(row["classifier_id"]), {})[row["class_name"]] = int(row["class_id"])
    return maps
```

> `_py_oids` already exists in `db.py` (casts numpy scalars to plain Python) — reuse it so ints bind cleanly.

- [ ] **Step 4: Run the test, confirm pass**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_db_taxonomy.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/db.py feature_step/tests/unittest/test_offline_db_taxonomy.py
git commit -m "feat(feature_step): db.fetch_taxonomy_maps reads class_name->class_id from taxonomy"
```

---

## Task 3: `write_probabilities` — the upsert (fake-engine tested)

**Files:**
- Modify: `feature_step/features/offline/probability_writer.py`
- Modify: `feature_step/tests/unittest/test_offline_probability_writer.py`

- [ ] **Step 1: Add the failing tests**

Append to `test_offline_probability_writer.py`:

```python
class _RecordingConn:
    def __init__(self):
        self.calls = []

    def execute(self, sql, records):
        self.calls.append((sql, records))


class _FakeEngine:
    def __init__(self):
        self.conn = _RecordingConn()

    def begin(self):
        conn = self.conn

        class _Ctx:
            def __enter__(self_):
                return conn

            def __exit__(self_, *a):
                return False

        return _Ctx()


def test_write_dry_run_does_not_connect(monkeypatch):
    def _boom(_creds):
        raise AssertionError("dry-run must not open a connection")
    monkeypatch.setattr(pw.db, "_make_engine", _boom)

    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    result = pw.write_probabilities(rows, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 45}


def test_write_execute_upserts(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(pw.db, "_make_engine", lambda _c: fake)

    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    result = pw.write_probabilities(rows, "creds", schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 45}
    assert len(fake.conn.calls) == 1
    sql, records = fake.conn.calls[0]
    sql_str = str(sql)
    assert "multisurvey_ztf.probability" in sql_str
    assert "ON CONFLICT (oid, sid, classifier_id, class_id)" in sql_str
    assert "DO UPDATE SET" in sql_str
    assert "updated_date" not in sql_str  # probability has no updated_date column
    assert records[0]["oid"] == OID and isinstance(records[0]["oid"], int)


def test_write_default_schema_is_db_schema(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(pw.db, "_make_engine", lambda _c: fake)
    rows = pw.build_probability_rows(_full_dto(), OID, 1.0, TAX_MAPS)
    pw.write_probabilities(rows, "creds", execute=True)  # no schema=
    assert f"{pw.db.SCHEMA}.probability" in str(fake.conn.calls[0][0])


def test_write_empty_rows_execute_no_call(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(pw.db, "_make_engine", lambda _c: fake)
    result = pw.write_probabilities([], "creds", execute=True)
    assert result == {"executed": True, "written": 0}
    assert fake.conn.calls == []
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_probability_writer.py -q`
Expected: FAIL — `AttributeError: module 'features.offline.probability_writer' has no attribute 'write_probabilities'`

- [ ] **Step 3: Add `write_probabilities` to `probability_writer.py`**

Append to the module:

```python
def write_probabilities(rows: list, credentials: str, schema: Optional[str] = None,
                        execute: bool = False) -> dict:
    """Upsert DB-ready probability rows into <schema>.probability.

    Dry-run by default (execute=False): returns {"executed": False,
    "would_write": N} and opens no DB connection. With execute=True, upserts all
    rows in one transaction. schema defaults to db.SCHEMA.
    """
    schema = schema or db.SCHEMA
    n = len(rows)
    if not execute:
        return {"executed": False, "would_write": n}

    # schema is a trusted operator-supplied identifier (db.SCHEMA env / CLI), not
    # user input — same f-string convention as db.py / feature_writer.py.
    sql = text(
        f"INSERT INTO {schema}.probability "
        "(oid, sid, classifier_id, classifier_version, class_id, probability, ranking, lastmjd) "
        "VALUES (:oid, :sid, :classifier_id, :classifier_version, :class_id, "
        ":probability, :ranking, :lastmjd) "
        "ON CONFLICT (oid, sid, classifier_id, class_id) "
        "DO UPDATE SET probability = EXCLUDED.probability, "
        "classifier_version = EXCLUDED.classifier_version, "
        "ranking = EXCLUDED.ranking, lastmjd = EXCLUDED.lastmjd"
    )
    engine = db._make_engine(credentials)
    with engine.begin() as conn:
        if rows:
            conn.execute(sql, rows)
    return {"executed": True, "written": n}
```

- [ ] **Step 4: Run all writer tests, confirm pass**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_probability_writer.py -q`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/probability_writer.py feature_step/tests/unittest/test_offline_probability_writer.py
git commit -m "feat(feature_step): probability_writer upsert into <schema>.probability (dry-run default)"
```

---

## Task 4: `classify.py` — save-capable path returning `lastmjd`

**Files:**
- Modify: `feature_step/features/offline/classify.py`
- Modify: `feature_step/tests/unittest/test_offline_classify.py`

Refactor the DB fetch out of `classify_oid` into a helper (no behavior change), then add `classify_oid_for_save` that also returns `lastmjd = max mjd over detections + forced`.

- [ ] **Step 1: Add the failing test**

Append to `feature_step/tests/unittest/test_offline_classify.py`:

```python
def test_classify_oid_for_save_returns_lastmjd(monkeypatch):
    import pandas as pd
    # Fake the DB readers + message/AO build so no real DB is needed.
    monkeypatch.setattr(classify.db, "fetch_detections",
                        lambda c, oids: pd.DataFrame({"mjd": [59000.0, 59010.5]}))
    monkeypatch.setattr(classify.db, "fetch_forced_photometry",
                        lambda c, oids: pd.DataFrame({"mjd": [59020.25]}))  # forced later than dets
    monkeypatch.setattr(classify.db, "fetch_ps1", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_allwise", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_references", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify, "build_message", lambda oid, d, f, p: {"oid": oid})
    monkeypatch.setattr(classify, "compute_astro_object",
                        lambda *a, **k: object())  # non-None AO
    monkeypatch.setattr(classify, "classify_astro_object",
                        lambda ao, msg, model: OutputDTO(pd.DataFrame({"AGN": [0.9]}, index=[123]),
                                                         {"top": pd.DataFrame(), "children": {}}))

    dto, lastmjd = classify.classify_oid_for_save(123, "creds", model=object())
    assert lastmjd == 59020.25            # max over detections + forced, already MJD
    assert dto.probabilities.loc[123, "AGN"] == 0.9


def test_classify_oid_for_save_none_when_no_ao(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(classify.db, "fetch_detections", lambda c, oids: pd.DataFrame({"mjd": []}))
    monkeypatch.setattr(classify.db, "fetch_forced_photometry", lambda c, oids: pd.DataFrame({"mjd": []}))
    monkeypatch.setattr(classify.db, "fetch_ps1", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_allwise", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify.db, "fetch_references", lambda c, oids: pd.DataFrame())
    monkeypatch.setattr(classify, "build_message", lambda oid, d, f, p: {"oid": oid})
    monkeypatch.setattr(classify, "compute_astro_object", lambda *a, **k: None)  # too few dets

    dto, lastmjd = classify.classify_oid_for_save(1, "creds", model=object())
    assert dto is None and lastmjd is None
```

> **Note on imports:** the test monkeypatches `classify.db`, `classify.build_message`, and `classify.compute_astro_object`. For that to work they must be module-level names on `classify`. Currently `classify_oid` imports `db`/`build_message` *inside* the function. Step 3 moves them to module top-level (`compute_astro_object` is already top-level).

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_classify.py -q`
Expected: FAIL — `AttributeError: module 'features.offline.classify' has no attribute 'classify_oid_for_save'` (and/or the monkeypatch of `classify.db`/`classify.build_message` failing until they're module-level).

- [ ] **Step 3: Refactor `classify.py`**

At the top of `classify.py`, add the two imports that are currently function-local (keep the existing `from .lc_features import compute_astro_object`):

```python
from features.offline import db
from features.offline.message import build_message
from .lc_features import compute_astro_object
```

Replace the existing `classify_oid` function with a shared fetch helper + the two public functions:

```python
def _fetch_oid_inputs(oid: int, credentials: str):
    """DB -> (message, references, allwise, detections, forced) for one oid."""
    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    allwise = db.fetch_allwise(credentials, oids)
    refs = db.fetch_references(credentials, oids)
    message = build_message(oid, dets, forced, ps1)
    return message, refs, allwise, dets, forced


def _lc_lastmjd(dets, forced):
    """Max MJD over all epochs the classifier consumed (detections + forced).

    Already MJD (db.py reads mjd) — do NOT subtract 2400000.5. None if no epochs.
    """
    mjds = []
    if dets is not None and len(dets):
        mjds.append(float(dets["mjd"].max()))
    if forced is not None and len(forced):
        mjds.append(float(forced["mjd"].max()))
    return max(mjds) if mjds else None


def classify_oid(oid: int, credentials: str, model, min_detections: int = 1,
                 preprocessor=None, extractor=None):
    """DB -> message -> features -> probabilities for one oid.

    Returns an OutputDTO, or None if the object has too few real detections."""
    message, refs, allwise, _dets, _forced = _fetch_oid_inputs(oid, credentials)
    ao = compute_astro_object(message, refs, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None
    return classify_astro_object(ao, message, model)


def classify_oid_for_save(oid: int, credentials: str, model, min_detections: int = 1,
                          preprocessor=None, extractor=None):
    """Like classify_oid but also returns lastmjd for persistence.

    Returns (OutputDTO, lastmjd), or (None, None) if too few real detections.
    lastmjd = max MJD over detections + forced (see _lc_lastmjd)."""
    message, refs, allwise, dets, forced = _fetch_oid_inputs(oid, credentials)
    ao = compute_astro_object(message, refs, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None, None
    return classify_astro_object(ao, message, model), _lc_lastmjd(dets, forced)
```

(Delete the old function-local `from features.offline import db` / `from features.offline.message import build_message` lines that were inside the old `classify_oid`.)

- [ ] **Step 4: Run the classify tests, confirm pass**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && conda run -n training_py310 python -m pytest tests/unittest/test_offline_classify.py -q`
Expected: PASS (all prior tests + 2 new)

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/classify.py feature_step/tests/unittest/test_offline_classify.py
git commit -m "feat(feature_step): classify_oid_for_save returns lastmjd (det+forced max MJD)"
```

---

## Task 5: Wire `offline_classify.py --save`

**Files:**
- Modify: `feature_step/scripts/offline_classify.py`

Mirror the `--save / --execute / --write-credentials` flag shape of `offline_compute_features.py`. The taxonomy map is fetched **read-only** (via `--credentials`), separate from the write.

- [ ] **Step 1: Update the CLI**

At the top of `feature_step/scripts/offline_classify.py`, update the imports:

```python
from features.offline.classify import load_squidward_model, classify_oid_for_save
from features.offline import db, probability_writer
```

Replace the body of `main()`:

```python
def main():
    ap = argparse.ArgumentParser(
        description="Offline DB->features->BHRF probabilities for one ZTF oid."
    )
    ap.add_argument("--oid", type=int, required=True,
                    help="Multisurvey bigint oid (e.g. 36028941624528297).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON (read; also used to read taxonomy).")
    ap.add_argument("--min-det", type=int, default=1,
                    help="Minimum real detections required to classify.")
    ap.add_argument("--save", action="store_true",
                    help="Persist probabilities into <schema>.probability "
                         "(dry-run unless --execute).")
    ap.add_argument("--execute", action="store_true",
                    help="With --save, actually write. Requires --write-credentials.")
    ap.add_argument("--write-credentials", default=None, dest="write_credentials",
                    help="Write-capable DB credentials JSON (default credentials are read-only).")
    args = ap.parse_args()

    if args.save and args.execute and not args.write_credentials:
        ap.error("--execute requires --write-credentials (the default credentials are read-only)")

    model, name, version = load_squidward_model()
    print(f"model: {name} version={version}")
    print(f"oid: {args.oid}")

    result, lastmjd = classify_oid_for_save(args.oid, args.credentials, model,
                                            min_detections=args.min_det)
    if result is None or result.probabilities is None or len(result.probabilities) == 0:
        print("\nFAIL: no probabilities (too few detections or can't predict)")
        sys.exit(1)

    print(f"\nprobabilities:\n{result.probabilities.to_string()}")
    top = result.hierarchical.get("top")
    if top is not None and len(top):
        print(f"\ntop:\n{top.to_string()}")
    print("\nOK: probabilities produced.")

    if args.save:
        # class_id authority is the DB taxonomy — read it (read-only creds).
        taxonomy_maps = db.fetch_taxonomy_maps(args.credentials,
                                               probability_writer.CLASSIFIER_IDS)
        rows = probability_writer.build_probability_rows(result, args.oid, lastmjd,
                                                         taxonomy_maps, version=version)
        write_creds = args.write_credentials or args.credentials
        outcome = probability_writer.write_probabilities(
            rows, write_creds, execute=args.execute)
        print(f"\nsave: {outcome} (lastmjd={lastmjd})")
        if not args.execute:
            print("(dry-run — pass --execute with --write-credentials to write)")
```

- [ ] **Step 2: Verify the CLI parses and the guard fires (no model/DB needed)**

Run:
```bash
cd /home/fandrades/desktop/pipeline/feature_step
conda run -n training_py310 python scripts/offline_classify.py --help 2>&1 | grep -E "\-\-save|\-\-execute|\-\-write-credentials"
conda run -n training_py310 python scripts/offline_classify.py --oid 1 --save --execute 2>&1 | tail -2
```
Expected: the three flags appear in `--help`; the second command exits with argparse error `--execute requires --write-credentials` (fires at arg-parse time, before any model load).

- [ ] **Step 3: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/scripts/offline_classify.py
git commit -m "feat(feature_step): offline_classify.py --save persists BHRF probabilities (class_id from DB taxonomy)"
```

---

## Task 6: Docs — FLOW.md + README

**Files:**
- Modify: `feature_step/features/offline/FLOW.md`
- Modify: `feature_step/features/offline/README.md`

- [ ] **Step 1: Update FLOW.md §7**

In `**Done & working:**` add:

```markdown
- **BHRF probabilities persist to `multisurvey_ztf.probability`** via
  `probability_writer.py` (`offline_classify.py --save`): one row per class per
  classifier across all 5 seeded classifiers (ids 5–9). `class_name→class_id` read
  from the **DB `taxonomy`** (`db.fetch_taxonomy_maps`, the authority — not the
  fixture); `classifier_version` smallint (`2.1.0`→`210`); `ranking` =
  per-classifier dense rank desc; `lastmjd` = max MJD over detections+forced
  (already MJD, no JD subtraction); upsert `ON CONFLICT (oid, sid, classifier_id,
  class_id) DO UPDATE`. Dry-run by default.
```

Remove the now-done **Pending / deferred** bullet "Persist BHRF probabilities to `multisurvey_ztf.probability` — *not built.*" and its 4 sub-steps. (The scribe-consumer unknown it described is resolved: the multisurvey path writes directly; we mirror it.)

- [ ] **Step 2: Update FLOW.md §3c and §8**

§3c currently says there's nothing to write/compare in `probability`. Add a sentence: we now **write** BHRF LC probabilities there (distinct from the stamp rows already present) via `probability_writer.py`, mapping class names through the seeded `taxonomy` (§3d).

Add to the §8 file map table:

```markdown
| `probability_writer.py` | Build + upsert BHRF probability rows into `<schema>.probability` (`build_probability_rows` + `write_probabilities`; class_id via `db.fetch_taxonomy_maps`). §6 |
```

- [ ] **Step 3: Update README.md**

Add to the `## Modules` table:

```markdown
| `probability_writer.py` | `build_probability_rows(output_dto, oid, lastmjd, taxonomy_by_classifier)` → DB-ready rows for all 5 BHRF classifiers (class_id from the DB taxonomy map, version smallint, ranking, lastmjd); `write_probabilities(rows, credentials, schema, execute=False)` upserts into `<schema>.probability` (`ON CONFLICT … DO UPDATE`, dry-run unless `execute=True`). |
```

Also mention `db.fetch_taxonomy_maps` in the `db.py` row of the Modules table (append to its description): `; fetch_taxonomy_maps(creds, classifier_ids) → {classifier_id: {class_name: class_id}} from taxonomy`.

Update the `offline_classify.py` row in the `## Scripts` table:

```markdown
| `offline_classify.py --oid <bigint> [--credentials PATH] [--min-det M] [--save [--execute] [--write-credentials PATH]]` | DB -> message -> features -> BHRF probabilities for one oid. Requires `MODEL_PATH`. `--save` upserts into `<schema>.probability` (dry-run unless `--execute` + `--write-credentials`). |
```

- [ ] **Step 4: Confirm both docs modified + commit**

Run: `git diff --stat feature_step/features/offline/FLOW.md feature_step/features/offline/README.md`
Expected: both modified.

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/FLOW.md feature_step/features/offline/README.md
git commit -m "docs(feature_step): record BHRF probability writer (Done)"
```

---

## Task 7: Live smoke — dry-run against a real oid (MANUAL verification)

> Verifies the whole path end-to-end against the live DB + real model, including the
> DB taxonomy read. The dry-run is safe (read-only, no write). The actual
> `--execute` write is the operator's call and requires write credentials.

- [ ] **Step 1: Dry-run save on a known dense oid**

Run:
```bash
cd /home/fandrades/desktop/pipeline/feature_step
MODEL_PATH=/home/fandrades/desktop/alerce_models/squidward/2.1.0/hierarchical_random_forest_model.pkl \
  conda run -n training_py310 python scripts/offline_classify.py \
  --oid 36028941624528297 --save
```
Expected: prints the flat probabilities + top frame, then `save: {'executed': False, 'would_write': 45} (lastmjd=<a plausible MJD ~59000-60000>)` and the dry-run hint. Confirm `would_write == 45` and `lastmjd` is a positive MJD (NOT negative — that would mean a JD/MJD bug). If it raises "class ... not in taxonomy", the seeded taxonomy disagrees with the model output — stop and reconcile (should not happen; `offline_verify_taxonomy.py` already passed).

- [ ] **Step 2: (operator-only, optional) execute the write**

Only if you intend to persist. Requires write-capable credentials JSON:
```bash
cd /home/fandrades/desktop/pipeline/feature_step
MODEL_PATH=/home/fandrades/desktop/alerce_models/squidward/2.1.0/hierarchical_random_forest_model.pkl \
  conda run -n training_py310 python scripts/offline_classify.py \
  --oid 36028941624528297 --save --execute --write-credentials <path-to-write-creds.json>
```
Expected: `save: {'executed': True, 'written': 45}`.

Then verify in DB (a separate connection — the writer commits its own transaction):
```sql
SELECT classifier_id, count(*) FROM multisurvey_ztf.probability
WHERE oid = 36028941624528297 AND classifier_id BETWEEN 5 AND 9
GROUP BY classifier_id ORDER BY classifier_id;
```
Expected: `5→21, 6→3, 7→6, 8→6, 9→9`.

---

## Self-Review (against the design spec)

- **Spec coverage:** `probability_writer.py` build → Task 1; DB taxonomy read (authority) → Task 2; upsert → Task 3; `OutputDTO`→5-classifier fan-out → Task 1 `_iter_frames` + tests; per-row rules (rank/version-smallint/class_id/lastmjd) → Task 1; `lastmjd` = det+forced max MJD, no JD subtraction → Task 4 `_lc_lastmjd` + test; wire `--save` (fetch maps read-only, pass in) → Task 5; docs → Task 6; live check → Task 7.
- **Locked decisions honored:** DO UPDATE ✓; all 5 classifiers ✓; version 210 ✓; **class_id from DB taxonomy** ✓ (corrected from the fixture); lastmjd det+forced ✓.
- **Out of scope (excluded):** batch/backfill loop, any change to the live streaming step/scribe, and the (negative) stored-probability *compare* (`offline_compare_probabilities.py` stays a stub).
- **Placeholder scan:** none — every code/SQL/command step is concrete.
- **Type/name consistency:** `build_probability_rows(output_dto, oid, lastmjd, taxonomy_by_classifier)`, `write_probabilities`, `classifier_version_to_smallint`, `CLASSIFIER_IDS`, `_iter_frames`, `db.fetch_taxonomy_maps`, `classify_oid_for_save`, `_fetch_oid_inputs`, `_lc_lastmjd` used consistently across tasks; `OutputDTO.hierarchical` accessed as a dict everywhere; probability column list identical in Task 3 SQL and the Task 1 row dicts.
```
