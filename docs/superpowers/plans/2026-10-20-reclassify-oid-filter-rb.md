# rb-aware oid filter for the 2.2.0 reclassification: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A script that turns the 20.6M-oid dump list into the subset with at least N (default 8) non-forced ZTF detections surviving the rb cut, as a sorted `.npy` the reclassify runner takes with `--oid-file`.

**Architecture:** Two new readers in `features/offline/db.py` (the n_det pre-filter and the rb count, both plain SQL against the named schema) and one new script, `scripts/offline_select_oids_rb.py`, that reuses the unit/pool/resume machinery of `offline_run_batch.py` (`R`). The readers use an expanding `IN` instead of `= ANY(:oids)` so the unit tests run the real SQL on an in-memory sqlite with an attached `multisurvey` database; no DB connection in tests.

**Tech Stack:** Python 3 (`~/.pyenv/versions/feature-step/bin/python`), numpy, pandas, SQLAlchemy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-20-reclassify-oid-filter-rb-design.md` (PR #641, branch `reclass/squidward-2.2.0`).

## Global Constraints

- Every new or changed line carries a `#` comment; every public function has a docstring (spec criterion 6).
- rb rule, verbatim from the spec: a `detection` row counts when `sid = 0 AND (rb IS NULL OR rb >= 0.55)`; forced photometry never counts.
- One flag `--min-detections` (default 8) drives both the `object.n_det` pre-filter and the rb count.
- Read credentials only: default `R.DEFAULT_CREDENTIALS` (`read_credentials.json`). The script never writes to the DB.
- Tests go in a new file `feature_step/tests/unittest/test_offline_select_oids_rb.py`; committed test files are not edited by the agent.
- Done: `cd <repo> && ~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step` exits 0.
- Run pytest from `feature_step/` as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=<repo>/alerce_classifiers ~/.pyenv/versions/feature-step/bin/python -m pytest tests/unittest/test_offline_select_oids_rb.py -q`.

## Review Focus

1. numpy int64 oids handed to the driver: sqlite and psycopg2 both refuse them. Readers must cast with `db._py_oids`. Pinned in Task 1 (`test_rb_count_accepts_numpy_oids`).
2. An empty pre-filter (wrong schema, N too high): the script must exit 1 with a message, never write an empty `--out`. Pinned in Task 2 (`test_empty_prefilter_exits_1`).
3. A unit that fails after retries: no unit file, `--out` not written, exit 1 naming the unit. Pinned in Task 2 (`test_failed_unit_blocks_merge`).
4. `--max-units` smoke: unit files written, `--out` NOT written (a partial list must never look like the real one). Pinned in Task 2 (`test_max_units_does_not_write_out`).
5. Duplicates or unsorted oids in `--oid-file`: output is sorted unique. Pinned in Task 2 (`test_prefilter_sorted_unique`).

---

### Task 1: the two readers in `features/offline/db.py`

**Files:**
- Modify: `feature_step/features/offline/db.py` (append after `fetch_probability_lastmjd`, line ~427)
- Test: `feature_step/tests/unittest/test_offline_select_oids_rb.py`

**Interfaces:**
- Consumes: `db._make_engine(credentials_json)`, `db._py_oids(oids)`, `db.SID` (0), `db.SCHEMA`.
- Produces:
  - `fetch_oids_min_n_det(credentials_json: str, min_n_det: int, schema: str = SCHEMA) -> np.ndarray` — sorted int64 oids of `<schema>.object WHERE sid = :sid AND n_det >= :min_n_det`, streamed with `chunksize=1_000_000` like `R.select_oids`, sorted in numpy. Empty array when none.
  - `fetch_oids_with_good_detections(credentials_json: str, oids, min_detections: int, schema: str = SCHEMA) -> np.ndarray` — sorted int64 oids among `oids` with `count(*) >= :min_detections` over `<schema>.detection d JOIN <schema>.ztf_detection z ON d.oid = z.oid AND d.measurement_id = z.measurement_id WHERE d.oid IN :oids AND d.sid = :sid AND (z.rb IS NULL OR z.rb >= 0.55) GROUP BY d.oid HAVING count(*) >= :min_detections`. `:oids` is `sa.bindparam("oids", expanding=True)`. Empty array when none.

- [ ] **Step 1: Write the failing tests** (sqlite fixture shared by both readers)

```python
"""offline_select_oids_rb: readers run on sqlite with an attached `multisurvey` db; script on fakes."""
import numpy as np
import pytest
import sqlalchemy as sa

from features.offline import db

SCHEMA = "multisurvey"

@pytest.fixture
def engine(monkeypatch):
    """In-memory sqlite with `multisurvey.object`, `.detection`, `.ztf_detection`; db._make_engine returns it."""
    eng = sa.create_engine("sqlite://", poolclass=sa.pool.StaticPool,
                           connect_args={"check_same_thread": False})
    with eng.begin() as c:
        c.exec_driver_sql(f"ATTACH DATABASE ':memory:' AS {SCHEMA}")
        c.exec_driver_sql(f"CREATE TABLE {SCHEMA}.object (oid INTEGER, sid INTEGER, n_det INTEGER)")
        c.exec_driver_sql(f"CREATE TABLE {SCHEMA}.detection (oid INTEGER, sid INTEGER, measurement_id INTEGER)")
        c.exec_driver_sql(f"CREATE TABLE {SCHEMA}.ztf_detection (oid INTEGER, measurement_id INTEGER, rb REAL)")
    monkeypatch.setattr(db, "_make_engine", lambda _c: eng)
    return eng

def _dets(engine, oid, rbs, sid=0):
    """Insert one detection + ztf_detection row per rb value (None -> NULL)."""
    with engine.begin() as c:
        for i, rb in enumerate(rbs):
            c.exec_driver_sql(f"INSERT INTO {SCHEMA}.detection VALUES (?, ?, ?)", (oid, sid, i))
            c.exec_driver_sql(f"INSERT INTO {SCHEMA}.ztf_detection VALUES (?, ?, ?)", (oid, i, rb))

def test_min_n_det_reader_filters_sid_and_threshold_sorted(engine):
    with engine.begin() as c:
        c.exec_driver_sql(f"INSERT INTO {SCHEMA}.object VALUES (30, 0, 8), (10, 0, 9), (20, 0, 7), (40, 1, 50)")
    out = db.fetch_oids_min_n_det("creds", 8, schema=SCHEMA)
    assert out.dtype == np.int64 and out.tolist() == [10, 30]

def test_rb_count_drops_oid_with_one_rb_below_threshold(engine):   # spec criterion 3, first case
    _dets(engine, 1, [0.9] * 7 + [0.5])
    assert db.fetch_oids_with_good_detections("creds", [1], 8, schema=SCHEMA).tolist() == []

def test_rb_count_keeps_oid_with_null_rb(engine):                   # spec criterion 3, second case
    _dets(engine, 2, [0.55] * 7 + [None])
    assert db.fetch_oids_with_good_detections("creds", [2], 8, schema=SCHEMA).tolist() == [2]

def test_rb_count_ignores_other_sid_and_oids_not_asked(engine):
    _dets(engine, 3, [0.9] * 8, sid=1)       # right count, wrong survey
    _dets(engine, 4, [0.9] * 8)              # passes but not in the list
    _dets(engine, 5, [0.9] * 8)
    assert db.fetch_oids_with_good_detections("creds", [3, 5], 8, schema=SCHEMA).tolist() == [5]

def test_rb_count_accepts_numpy_oids(engine):                       # review focus 1
    _dets(engine, 6, [0.9] * 8)
    out = db.fetch_oids_with_good_detections("creds", np.array([6], dtype=np.int64), 8, schema=SCHEMA)
    assert out.tolist() == [6] and out.dtype == np.int64
```

- [ ] **Step 2: Run the tests, expect failure** — `AttributeError: module 'features.offline.db' has no attribute 'fetch_oids_min_n_det'`.

- [ ] **Step 3: Implement both readers in `db.py`** with the signatures and SQL above. `fetch_oids_min_n_det` streams like `R.select_oids` (`execution_options(stream_results=True)`, `pd.read_sql_query(..., chunksize=1_000_000)`), then `np.sort`. `fetch_oids_with_good_detections` builds `sa.text(sql).bindparams(sa.bindparam("oids", expanding=True))`, passes `{"oids": _py_oids(oids), "sid": SID, "min_detections": int(min_detections)}`, returns `np.sort(np.asarray(..., dtype=np.int64))`.

- [ ] **Step 4: Run the tests, expect 5 passed.**

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/offline/db.py feature_step/tests/unittest/test_offline_select_oids_rb.py
git commit -m "feat(feature_step): readers for the rb-aware oid filter (n_det pre-filter, good-detection count)"
```

---

### Task 2: the script `scripts/offline_select_oids_rb.py`

**Files:**
- Create: `feature_step/scripts/offline_select_oids_rb.py`
- Test: `feature_step/tests/unittest/test_offline_select_oids_rb.py` (append)

**Interfaces:**
- Consumes: Task 1 readers; from `offline_run_batch as R`: `DEFAULT_CREDENTIALS`, `default_workers()`, `default_start_method()`, `load_oids(path)`, `make_units(oids, unit_size) -> [(index, slice)]`, `_retry(fn, attempts, what)`, `run_fingerprint(oids, unit_size)`, `check_or_write_fingerprint(out_dir, fp, force)`, `_write_json(path, payload)`. Import pattern as in `offline_reclassify_batch.py` lines 50-66 (thread pinning block + `sys.path` insert).
- Produces (module-level, testable without a pool):
  - `prefilter(oids: np.ndarray, credentials: str, min_detections: int, schema: str) -> np.ndarray` — `np.intersect1d(np.unique(oids), db.fetch_oids_min_n_det(...), assume_unique=True)`; sorted unique.
  - `unit_path(out_dir: Path, index: int) -> Path` — `out_dir / "units" / f"unit_{index:05d}.npy"`.
  - `process_unit(unit: tuple) -> dict` — worker: for each `cfg["minibatch"]` slice of the unit's oids, `R._retry(lambda: db.fetch_oids_with_good_detections(cfg["credentials"], chunk, cfg["min_detections"], schema=cfg["schema"]), cfg["retries"], f"unit {index} rb count")`; concatenates, writes `unit_path` via `.tmp` + `os.replace`; returns `{"unit": index, "n_in": int, "n_kept": int, "seconds": float}`. Worker config in a module dict `_W` filled by `_init_worker(cfg)` like the sibling.
  - `merge_units(out_dir: Path, n_units: int) -> np.ndarray` — raises `FileNotFoundError` listing the missing unit indexes; else concatenation in unit order (already sorted).
  - `run_pool(units: list, cfg: dict, workers: int, start_method: str) -> list[dict]` — the only function that touches `ProcessPoolExecutor` (`initializer=_init_worker`); a unit that raises comes back as `{"unit": index, "error": repr(exc)}` instead of propagating; prints each unit as it lands.
  - `main(argv=None) -> int` — argparse per spec criterion 1 (`--oid-file` required, `--out`, `--out-dir`, `--min-detections` default 8, `--credentials` default `R.DEFAULT_CREDENTIALS`, `--schema` default `"multisurvey"`, `--workers`, `--unit-size` default 50_000, `--minibatch` default 500, `--max-units`, `--retries` default 4, `--start-method`, `--force-resume`). Flow: load + prefilter; exit 1 printing `"no oid passed the n_det >= N pre-filter"` when empty; fingerprint; units; skip units whose file exists; cap with `--max-units`; `db.dispose_engines()`; `run_pool`; after it, if `--max-units` was given or any unit has `"error"`: print the state (naming each failed `unit i`) and return 1 without writing `--out`; else `merge_units`, `np.save(out)`, `R._write_json(Path(out + ".json"), {...})` with keys `n_input, n_after_n_det, n_after_rb, min_detections, schema, unit_size, elapsed_s`, return 0.

- [ ] **Step 1: Write the failing tests** (append; the script module imported as `S` with `sys.path` pointing at `feature_step/scripts`)

```python
import importlib, json, sys
from pathlib import Path

@pytest.fixture
def S():
    """The script module (scripts/ is not a package)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("offline_select_oids_rb")

def test_prefilter_sorted_unique(S, monkeypatch):                   # review focus 5
    monkeypatch.setattr(S.db, "fetch_oids_min_n_det", lambda c, n, schema: np.array([1, 3, 5, 7], dtype=np.int64))
    out = S.prefilter(np.array([7, 3, 3, 2, 9], dtype=np.int64), "creds", 8, "multisurvey")
    assert out.tolist() == [3, 7] and out.dtype == np.int64

def test_process_unit_writes_kept_oids_in_minibatches(S, tmp_path, monkeypatch):
    calls = []
    def fake(c, oids, n, schema):
        calls.append(list(oids)); return np.array([o for o in oids if o % 2 == 0], dtype=np.int64)
    monkeypatch.setattr(S.db, "fetch_oids_with_good_detections", fake)
    S._init_worker({"credentials": "creds", "schema": "multisurvey", "min_detections": 8,
                    "minibatch": 2, "retries": 1, "out_dir": str(tmp_path)})
    m = S.process_unit((3, np.array([1, 2, 3, 4, 5], dtype=np.int64)))
    assert calls == [[1, 2], [3, 4], [5]]
    assert np.load(S.unit_path(tmp_path, 3)).tolist() == [2, 4]
    assert m["unit"] == 3 and m["n_in"] == 5 and m["n_kept"] == 2
    assert not list((tmp_path / "units").glob("*.tmp"))

def test_merge_units_concatenates_in_order_and_names_missing(S, tmp_path):
    (tmp_path / "units").mkdir()
    np.save(S.unit_path(tmp_path, 0), np.array([1, 2], dtype=np.int64))
    np.save(S.unit_path(tmp_path, 1), np.array([5], dtype=np.int64))
    assert S.merge_units(tmp_path, 2).tolist() == [1, 2, 5]
    with pytest.raises(FileNotFoundError, match="2"):
        S.merge_units(tmp_path, 3)

def _run(S, monkeypatch, tmp_path, oids, prefiltered, keep, extra=()):
    """Drive main() with both readers faked and the pool replaced by in-process calls."""
    monkeypatch.setattr(S.db, "fetch_oids_min_n_det", lambda c, n, schema: np.array(prefiltered, dtype=np.int64))
    monkeypatch.setattr(S.db, "fetch_oids_with_good_detections",
                        lambda c, o, n, schema: np.array([x for x in o if x in keep], dtype=np.int64))
    monkeypatch.setattr(S, "run_pool", lambda units, cfg, workers, start_method:
                        [S._init_worker(cfg) or S.process_unit(u) for u in units])
    src = tmp_path / "in.npy"; np.save(src, np.array(oids, dtype=np.int64))
    out = tmp_path / "out.npy"
    rc = S.main(["--oid-file", str(src), "--out", str(out), "--out-dir", str(tmp_path / "run"),
                 "--unit-size", "2", "--minibatch", "1", "--credentials", "creds", *extra])
    return rc, out

def test_main_writes_sorted_npy_and_json(S, monkeypatch, tmp_path):     # criteria 4, 5
    rc, out = _run(S, monkeypatch, tmp_path, oids=[1, 2, 3, 4, 5], prefiltered=[1, 2, 4, 5], keep={1, 4, 5})
    assert rc == 0 and np.load(out).tolist() == [1, 4, 5]
    meta = json.loads(Path(str(out) + ".json").read_text())
    assert (meta["n_input"], meta["n_after_n_det"], meta["n_after_rb"], meta["min_detections"]) == (5, 4, 3, 8)

def test_main_resumes_skipping_finished_units(S, monkeypatch, tmp_path):   # criterion 4
    _run(S, monkeypatch, tmp_path, [1, 2, 3, 4], [1, 2, 3, 4], {1, 2, 3, 4})
    seen = []
    monkeypatch.setattr(S, "run_pool", lambda units, cfg, w, s: seen.extend(u[0] for u in units) or [])
    rc, out = _run(S, monkeypatch, tmp_path, [1, 2, 3, 4], [1, 2, 3, 4], {1, 2, 3, 4})
    assert rc == 0 and seen == []            # both unit files exist: nothing to run, merge only

def test_empty_prefilter_exits_1(S, monkeypatch, tmp_path, capsys):   # review focus 2
    rc, out = _run(S, monkeypatch, tmp_path, [1, 2], prefiltered=[], keep=set())
    assert rc == 1 and not out.exists() and "pre-filter" in capsys.readouterr().out

def test_max_units_does_not_write_out(S, monkeypatch, tmp_path):     # review focus 4
    rc, out = _run(S, monkeypatch, tmp_path, [1, 2, 3, 4], [1, 2, 3, 4], {1, 2, 3, 4}, extra=("--max-units", "1"))
    assert rc == 1 and not out.exists()
    assert S.unit_path(tmp_path / "run", 0).exists() and not S.unit_path(tmp_path / "run", 1).exists()

def test_failed_unit_blocks_merge(S, monkeypatch, tmp_path, capsys):   # review focus 3
    def boom(c, o, n, schema):
        if 3 in list(o): raise RuntimeError("db down")
        return np.array(list(o), dtype=np.int64)
    monkeypatch.setattr(S.db, "fetch_oids_min_n_det", lambda c, n, schema: np.array([1, 2, 3, 4], dtype=np.int64))
    monkeypatch.setattr(S.db, "fetch_oids_with_good_detections", boom)
    def pool(units, cfg, workers, start_method):
        S._init_worker(cfg); res = []
        for u in units:
            try: res.append(S.process_unit(u))
            except Exception as e: res.append({"unit": u[0], "error": repr(e)})
        return res
    monkeypatch.setattr(S, "run_pool", pool)
    src = tmp_path / "in.npy"; np.save(src, np.array([1, 2, 3, 4], dtype=np.int64))
    out = tmp_path / "out.npy"
    rc = S.main(["--oid-file", str(src), "--out", str(out), "--out-dir", str(tmp_path / "run"),
                 "--unit-size", "2", "--minibatch", "1", "--retries", "1", "--credentials", "creds"])
    assert rc == 1 and not out.exists() and "unit 1" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests, expect failure** — `ModuleNotFoundError: No module named 'offline_select_oids_rb'`.

- [ ] **Step 3: Implement the script** with the functions and defaults in the Interfaces block. Module docstring: what it is for, the rb rule, and the cpu1 usage:

```
python scripts/offline_select_oids_rb.py \
    --oid-file $RUN/oids/squidward_v210_oids.npy \
    --out $RUN/oids/squidward_v210_oids_ndet8_rb.npy --out-dir $RUN/select_ndet8 \
    --credentials features/offline/read_credentials.json --workers 32
```

`main` prints: input count, after pre-filter, plan (units, unit size, minibatch, workers), per-unit lines as they land (`unit i: n_in -> n_kept, s`), and the final counts.

- [ ] **Step 4: Run the whole new test file, expect 13 passed.**

- [ ] **Step 5: Run the done command**

```bash
cd <repo> && ~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step
```
Expected: exit 0 (the 8 baseline failures only).

- [ ] **Step 6: Commit**

```bash
git add feature_step/scripts/offline_select_oids_rb.py feature_step/tests/unittest/test_offline_select_oids_rb.py
git commit -m "feat(feature_step): offline_select_oids_rb.py: oid list with >= N detections after the rb cut"
```
