# Offline ZTF feature writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the offline DB-ready feature rows (`compute_db_features` output) into `multisurvey_ztf.feature` via a per-row upsert, exposed through a `--save` flag on the existing CLI (dry-run by default).

**Architecture:** A new `feature_writer.py` holds a single `write_features(rows, credentials, schema, execute)` that sanitizes the DataFrame to native types and upserts it `ON CONFLICT (oid, sid, feature_id, band) DO UPDATE` in one transaction. Writing stays out of the read-only `db.py`. The CLI gains `--save` / `--execute` / `--write-credentials`.

**Tech Stack:** Python, pandas, SQLAlchemy (`db._make_engine`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-25-offline-ztf-feature-writer-design.md`

**Test command convention:**
```bash
cd feature_step && conda run --no-capture-output -n training_py310 \
    python -m pytest tests/unittest/<file>.py -v
```

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `feature_step/features/offline/feature_writer.py` | sanitize + upsert rows into `feature` | Create |
| `feature_step/tests/unittest/test_offline_feature_writer.py` | unit-test the writer (no real DB) | Create |
| `feature_step/scripts/offline_compute_features.py` | add `--save` / `--execute` / `--write-credentials` | Modify |
| `feature_step/features/offline/FLOW.md`, `README.md` | document the writer | Modify |

---

## Task 1: `feature_writer.py` + unit tests

**Files:**
- Create: `feature_step/features/offline/feature_writer.py`
- Test: `feature_step/tests/unittest/test_offline_feature_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `feature_step/tests/unittest/test_offline_feature_writer.py`:

```python
"""Unit tests for feature_writer.write_features — no real DB.

The engine is faked: db._make_engine is monkeypatched so execute() records the
SQL + records, and dry-run is proven to never call it.
"""
import numpy as np
import pandas as pd
import pytest

from features.offline import feature_writer


def _df(rows):
    """rows: list of (oid, sid, feature_id, band, version, value)"""
    return pd.DataFrame(
        rows, columns=["oid", "sid", "feature_id", "band", "version", "value"]
    )


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


def test_dry_run_does_not_connect(monkeypatch):
    def _boom(_creds):
        raise AssertionError("dry-run must not open a connection")
    monkeypatch.setattr(feature_writer.db, "_make_engine", _boom)

    df = _df([(36028941624528297, 0, 0, 1, 0, 0.5),
              (36028941624528297, 0, 0, 2, 0, 0.6)])
    result = feature_writer.write_features(df, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 2}


def test_execute_upserts_native_types(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(feature_writer.db, "_make_engine", lambda _c: fake)

    df = _df([(36028941624528297, 0, 5, 12, 0, 0.5),
              (36028941624528297, 0, 7, 0, 0, None)])
    result = feature_writer.write_features(df, "creds", schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 2}
    assert len(fake.conn.calls) == 1
    sql, records = fake.conn.calls[0]
    sql_str = str(sql)
    assert "multisurvey_ztf.feature" in sql_str
    assert "ON CONFLICT (oid, sid, feature_id, band)" in sql_str
    assert "DO UPDATE SET" in sql_str
    # native types, big oid preserved (not float), None value passes through
    r0 = records[0]
    assert r0["oid"] == 36028941624528297 and isinstance(r0["oid"], int)
    assert isinstance(r0["feature_id"], int) and isinstance(r0["band"], int)
    assert r0["value"] == 0.5
    assert records[1]["value"] is None


def test_nan_feature_id_rows_dropped(monkeypatch, caplog):
    df = _df([(36028941624528297, 0, 0, 1, 0, 0.5),
              (36028941624528297, 0, np.nan, 2, 0, 0.6)])
    import logging
    with caplog.at_level(logging.WARNING, logger="features.offline.feature_writer"):
        result = feature_writer.write_features(df, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 1}
    assert any("feature_id" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_feature_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.offline.feature_writer'`.

- [ ] **Step 3: Implement `feature_writer.py`**

Create `feature_step/features/offline/feature_writer.py`:

```python
"""Persist offline ZTF DB-ready feature rows into <schema>.feature.

Writing lives here, not in db.py (which is read-only). Takes the DataFrame
produced by lc_features.compute_db_features and upserts it one row at a time
via ON CONFLICT (oid, sid, feature_id, band) DO UPDATE.
"""
import logging

import pandas as pd
from sqlalchemy import text

from features.offline import db

log = logging.getLogger(__name__)


def _records(rows: pd.DataFrame) -> list:
    """Sanitize the frame into native-typed dict records for the upsert.

    - Drop rows with NaN feature_id (cannot satisfy the NOT-NULL PK); warn.
    - Warn if any version == -1 (version_name absent from feature_version_lut).
    - Cast to native Python types so the >2**53 oid bigint and the smallints are
      not coerced to float by the driver; NaN/None value -> SQL NULL.
    """
    n_before = len(rows)
    rows = rows[rows["feature_id"].notna()]
    dropped = n_before - len(rows)
    if dropped:
        log.warning("Dropping %d feature row(s) with unmapped (NaN) feature_id", dropped)

    if len(rows) and (rows["version"] == -1).any():
        log.warning("%d row(s) have version=-1 (version_name not in feature_version_lut)",
                    int((rows["version"] == -1).sum()))

    records = []
    for r in rows.to_dict("records"):
        value = r["value"]
        records.append({
            "oid": int(r["oid"]),
            "sid": int(r["sid"]),
            "feature_id": int(r["feature_id"]),
            "band": int(r["band"]),
            "version": int(r["version"]),
            "value": None if value is None or pd.isna(value) else float(value),
        })
    return records


def write_features(rows: pd.DataFrame, credentials: str, schema: str = None,
                   execute: bool = False) -> dict:
    """Upsert DB-ready feature rows into <schema>.feature.

    Dry-run by default (execute=False): returns {"executed": False,
    "would_write": N} and opens no DB connection. With execute=True, upserts all
    records in one transaction. schema defaults to db.SCHEMA.
    """
    schema = schema or db.SCHEMA
    records = _records(rows)
    n = len(records)

    if not execute:
        return {"executed": False, "would_write": n}

    sql = text(
        f"INSERT INTO {schema}.feature (oid, sid, feature_id, band, version, value) "
        "VALUES (:oid, :sid, :feature_id, :band, :version, :value) "
        "ON CONFLICT (oid, sid, feature_id, band) "
        "DO UPDATE SET value = EXCLUDED.value, version = EXCLUDED.version, updated_date = now()"
    )
    engine = db._make_engine(credentials)
    with engine.begin() as conn:
        if records:
            conn.execute(sql, records)
    return {"executed": True, "written": n}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_feature_writer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/feature_writer.py feature_step/tests/unittest/test_offline_feature_writer.py
git commit -m "feat(feature_step): offline feature_writer — upsert DB-ready rows into feature"
```

---

## Task 2: Wire `--save` / `--execute` / `--write-credentials` into the CLI

**Files:**
- Modify: `feature_step/scripts/offline_compute_features.py`

- [ ] **Step 1: Add the import**

Change line 19:

```python
from features.offline import db, lc_features
```
to:
```python
from features.offline import db, lc_features, feature_writer
```

- [ ] **Step 2: Add the three CLI args**

After the existing `--feature-version` argument block (ends at line 36), before `args = ap.parse_args()`, add:

```python
    ap.add_argument("--save", action="store_true",
                    help="Persist the DB-ready features into <schema>.feature.")
    ap.add_argument("--execute", action="store_true",
                    help="With --save, actually write (otherwise dry-run). "
                         "Requires --write-credentials.")
    ap.add_argument("--write-credentials", default=None, dest="write_credentials",
                    help="Credentials JSON with INSERT privileges; required when --execute "
                         "(the default credentials are read-only).")
```

- [ ] **Step 3: Add fail-fast validation right after `args = ap.parse_args()`**

Immediately after `args = ap.parse_args()` (currently line 37), add:

```python
    if args.execute and not args.save:
        ap.error("--execute only applies together with --save")
    if args.save and args.execute and not args.write_credentials:
        ap.error("--execute requires --write-credentials (the default credentials are read-only)")
```

- [ ] **Step 4: Add the save block at the end of `main()`**

After the final `print("\nOK: DB-ready feature rows produced.")` line, add:

```python
    if args.save:
        write_creds = args.write_credentials or credentials
        result = feature_writer.write_features(features, write_creds, execute=args.execute)
        if result["executed"]:
            print(f"\nSAVED: {result['written']} rows upserted into feature.")
        else:
            print(f"\nDRY RUN: would write {result['would_write']} rows "
                  f"(pass --execute with --write-credentials to write).")
```

- [ ] **Step 5: Verify arg validation (no DB needed)**

Run: `conda run --no-capture-output -n training_py310 python feature_step/scripts/offline_compute_features.py --oid 1 --save --execute`
Expected: exits non-zero with argparse error `--execute requires --write-credentials …` (it must fail BEFORE any DB read).

- [ ] **Step 6: Verify dry-run end-to-end (needs live read DB)**

Run: `conda run --no-capture-output -n training_py310 python feature_step/scripts/offline_compute_features.py --oid 36028941624528297 --save`
Expected: prints the DB-ready frame, then `DRY RUN: would write N rows …` with N>0 and exit 0 (no write). If DB unreachable, note it and mark DONE_WITH_CONCERNS.

- [ ] **Step 7: Commit**

```bash
git add feature_step/scripts/offline_compute_features.py
git commit -m "feat(feature_step): --save/--execute/--write-credentials on offline_compute_features"
```

---

## Task 3: Documentation

**Files:**
- Modify: `feature_step/features/offline/FLOW.md`, `feature_step/features/offline/README.md`

- [ ] **Step 1: Update FLOW.md**

In `FLOW.md` §5, after the `compute_db_features` bullet (added previously), add:

```markdown
- `feature_writer.write_features(rows, credentials, schema=db.SCHEMA, execute=False)`
  persists the DB-ready rows into `<schema>.feature` via
  `ON CONFLICT (oid, sid, feature_id, band) DO UPDATE` (refresh value/version/
  updated_date — matches production's scribe upsert). Dry-run by default; opens no
  connection unless `execute=True`. Exposed as `offline_compute_features.py --save`
  (`--execute` + `--write-credentials` to actually write; the default credentials
  are read-only).
```

In `FLOW.md` §8 (File map), add a row:

```markdown
| `feature_writer.py` | Upsert DB-ready feature rows into `<schema>.feature` (`write_features`). |
```

- [ ] **Step 2: Update README.md**

In the `README.md` file table, add a row near `lc_features.py`:

```markdown
| `feature_writer.py` | `write_features(rows, credentials, schema, execute=False)` upserts the DB-ready rows into `<schema>.feature` (`ON CONFLICT … DO UPDATE`); dry-run unless `execute=True`. |
```

- [ ] **Step 3: Commit**

```bash
git add feature_step/features/offline/FLOW.md feature_step/features/offline/README.md
git commit -m "docs(feature_step): document offline feature_writer"
```

---

## Final verification

- [ ] **Run the offline unit suite (regression check)**

Run: `cd feature_step && conda run --no-capture-output -n training_py310 python -m pytest tests/unittest/test_offline_feature_writer.py tests/unittest/test_offline_db_features.py tests/unittest/test_offline_feature_lut.py tests/unittest/test_prepare_ao_features_for_db.py tests/unittest/test_offline_classify.py tests/unittest/test_offline_feature_compare.py -q`
Expected: all PASS.
