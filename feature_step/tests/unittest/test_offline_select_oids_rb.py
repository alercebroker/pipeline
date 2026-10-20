"""offline_select_oids_rb: readers run on sqlite with an attached `multisurvey` db; script on fakes."""  # module docstring
import numpy as np  # arrays and dtype checks
import pytest  # fixtures
import sqlalchemy as sa  # in-memory engine

from features.offline import db  # module under test

SCHEMA = "multisurvey"  # schema name attached in sqlite


@pytest.fixture  # shared by every test
def engine(monkeypatch):  # sqlite engine standing in for postgres
    """In-memory sqlite with `multisurvey.object`, `.detection`, `.ztf_detection`; db._make_engine returns it."""
    eng = sa.create_engine("sqlite://", poolclass=sa.pool.StaticPool,  # one shared in-memory connection
                           connect_args={"check_same_thread": False})  # allow reuse across threads
    with eng.begin() as c:  # create the tables in one transaction
        c.exec_driver_sql(f"ATTACH DATABASE ':memory:' AS {SCHEMA}")  # fake schema
        c.exec_driver_sql(f"CREATE TABLE {SCHEMA}.object (oid INTEGER, sid INTEGER, n_det INTEGER)")  # object table
        c.exec_driver_sql(f"CREATE TABLE {SCHEMA}.detection (oid INTEGER, sid INTEGER, measurement_id INTEGER)")  # detection table
        c.exec_driver_sql(f"CREATE TABLE {SCHEMA}.ztf_detection (oid INTEGER, measurement_id INTEGER, rb REAL)")  # ztf extras
    monkeypatch.setattr(db, "_make_engine", lambda _c: eng)  # readers get the sqlite engine
    return eng  # hand it to the tests


def _dets(engine, oid, rbs, sid=0):  # insert helper
    """Insert one detection + ztf_detection row per rb value (None -> NULL)."""
    with engine.begin() as c:  # one transaction
        for i, rb in enumerate(rbs):  # one row per rb
            c.exec_driver_sql(f"INSERT INTO {SCHEMA}.detection VALUES (?, ?, ?)", (oid, sid, i))  # detection row
            c.exec_driver_sql(f"INSERT INTO {SCHEMA}.ztf_detection VALUES (?, ?, ?)", (oid, i, rb))  # matching ztf row


def test_min_n_det_reader_filters_sid_and_threshold_sorted(engine):  # n_det pre-filter
    """Pins the n_det reader: sid 0 only, n_det >= threshold, sorted int64."""
    with engine.begin() as c:  # seed objects
        c.exec_driver_sql(f"INSERT INTO {SCHEMA}.object VALUES (30, 0, 8), (10, 0, 9), (20, 0, 7), (40, 1, 50)")  # 40 is another survey
    out = db.fetch_oids_min_n_det("creds", 8, schema=SCHEMA)  # run the reader
    assert out.dtype == np.int64 and out.tolist() == [10, 30]  # sorted, sid 0, n_det >= 8


def test_rb_count_drops_oid_with_one_rb_below_threshold(engine):  # spec criterion 3, first case
    """Pins spec criterion 3: 7 good + 1 rb < 0.55 detection leaves 7, below 8, so the oid is dropped."""
    _dets(engine, 1, [0.9] * 7 + [0.5])  # 7 good, 1 bad
    assert db.fetch_oids_with_good_detections("creds", [1], 8, schema=SCHEMA).tolist() == []  # only 7 survive


def test_rb_count_keeps_oid_with_null_rb(engine):  # spec criterion 3, second case
    """Pins spec criterion 3: rb == 0.55 and rb NULL both count, so 8 survive and the oid is kept."""
    _dets(engine, 2, [0.55] * 7 + [None])  # boundary rb and a NULL
    assert db.fetch_oids_with_good_detections("creds", [2], 8, schema=SCHEMA).tolist() == [2]  # all 8 survive


def test_rb_count_ignores_other_sid_and_oids_not_asked(engine):  # sid and oid scoping
    """Pins the sid filter and the oid list: other-survey detections and unasked oids never appear."""
    _dets(engine, 3, [0.9] * 8, sid=1)  # right count, wrong survey
    _dets(engine, 4, [0.9] * 8)  # passes but not in the list
    _dets(engine, 5, [0.9] * 8)  # passes and asked
    assert db.fetch_oids_with_good_detections("creds", [3, 5], 8, schema=SCHEMA).tolist() == [5]  # only 5


def test_rb_count_accepts_numpy_oids(engine):  # review focus 1
    """Pins numpy input: an int64 array of oids is accepted and the result is int64."""
    _dets(engine, 6, [0.9] * 8)  # passes
    out = db.fetch_oids_with_good_detections("creds", np.array([6], dtype=np.int64), 8, schema=SCHEMA)  # numpy input
    assert out.tolist() == [6] and out.dtype == np.int64  # plain result, int64


# --- Task 2: the script scripts/offline_select_oids_rb.py, on faked readers ---
import importlib, json, sys  # module import by name, json sidecar, sys.path
from pathlib import Path  # paths


@pytest.fixture  # the script as a module
def S():  # scripts/ is on sys.path only here
    """The script module (scripts/ is not a package)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # feature_step/scripts
    return importlib.import_module("offline_select_oids_rb")  # cached after the first import


def test_prefilter_sorted_unique(S, monkeypatch):                   # review focus 5
    """Pins prefilter: duplicates dropped, only oids the n_det reader returned, sorted int64."""
    monkeypatch.setattr(S.db, "fetch_oids_min_n_det", lambda c, n, schema: np.array([1, 3, 5, 7], dtype=np.int64))  # fake reader
    out = S.prefilter(np.array([7, 3, 3, 2, 9], dtype=np.int64), "creds", 8, "multisurvey")  # unsorted, duplicated input
    assert out.tolist() == [3, 7] and out.dtype == np.int64  # intersection, sorted, int64


def test_process_unit_writes_kept_oids_in_minibatches(S, tmp_path, monkeypatch):  # worker
    """Pins process_unit: one reader call per minibatch, kept oids in the unit file, no .tmp left."""
    calls = []  # oids each reader call received
    def fake(c, oids, n, schema):  # keeps even oids
        """Record the call and keep the even oids."""
        calls.append(list(oids)); return np.array([o for o in oids if o % 2 == 0], dtype=np.int64)  # record, filter
    monkeypatch.setattr(S.db, "fetch_oids_with_good_detections", fake)  # fake reader
    S._init_worker({"credentials": "creds", "schema": "multisurvey", "min_detections": 8,  # worker config:
                    "minibatch": 2, "retries": 1, "out_dir": str(tmp_path)})  # worker config
    m = S.process_unit((3, np.array([1, 2, 3, 4, 5], dtype=np.int64)))  # unit 3, five oids
    assert calls == [[1, 2], [3, 4], [5]]  # minibatches of 2
    assert np.load(S.unit_path(tmp_path, 3)).tolist() == [2, 4]  # the kept oids
    assert m["unit"] == 3 and m["n_in"] == 5 and m["n_kept"] == 2  # the returned counts
    assert not list((tmp_path / "units").glob("*.tmp"))  # atomic write left nothing behind


def test_merge_units_concatenates_in_order_and_names_missing(S, tmp_path):  # merge
    """Pins merge_units: unit order concatenation, and a missing unit is named in the error."""
    (tmp_path / "units").mkdir()  # the unit directory
    np.save(S.unit_path(tmp_path, 0), np.array([1, 2], dtype=np.int64))  # unit 0
    np.save(S.unit_path(tmp_path, 1), np.array([5], dtype=np.int64))  # unit 1
    assert S.merge_units(tmp_path, 2).tolist() == [1, 2, 5]  # both units, in order
    with pytest.raises(FileNotFoundError, match="2"):  # unit 2 never ran
        S.merge_units(tmp_path, 3)  # asks for three units


def _run(S, monkeypatch, tmp_path, oids, prefiltered, keep, extra=()):  # main() driver
    """Drive main() with both readers faked and the pool replaced by in-process calls."""
    monkeypatch.setattr(S.db, "fetch_oids_min_n_det", lambda c, n, schema: np.array(prefiltered, dtype=np.int64))  # n_det reader
    monkeypatch.setattr(S.db, "fetch_oids_with_good_detections",  # rb reader:
                        lambda c, o, n, schema: np.array([x for x in o if x in keep], dtype=np.int64))  # rb reader
    monkeypatch.setattr(S, "run_pool", lambda units, cfg, workers, start_method:  # pool replaced:
                        [S._init_worker(cfg) or S.process_unit(u) for u in units])  # no processes
    src = tmp_path / "in.npy"; np.save(src, np.array(oids, dtype=np.int64))  # input oid file
    out = tmp_path / "out.npy"  # output oid file
    rc = S.main(["--oid-file", str(src), "--out", str(out), "--out-dir", str(tmp_path / "run"),  # flags:
                 "--unit-size", "2", "--minibatch", "1", "--credentials", "creds", *extra])  # run the script
    return rc, out  # exit code and output path


def test_main_writes_sorted_npy_and_json(S, monkeypatch, tmp_path):     # criteria 4, 5
    """Pins the output: sorted kept oids in --out and the counts in its .json sidecar."""
    rc, out = _run(S, monkeypatch, tmp_path, oids=[1, 2, 3, 4, 5], prefiltered=[1, 2, 4, 5], keep={1, 4, 5})  # full run
    assert rc == 0 and np.load(out).tolist() == [1, 4, 5]  # success, kept oids
    meta = json.loads(Path(str(out) + ".json").read_text())  # sidecar
    assert (meta["n_input"], meta["n_after_n_det"], meta["n_after_rb"], meta["min_detections"]) == (5, 4, 3, 8)  # counts


def test_main_resumes_skipping_finished_units(S, monkeypatch, tmp_path):   # criterion 4
    """Pins resume: a rerun with every unit file present runs nothing and only merges."""
    _run(S, monkeypatch, tmp_path, [1, 2, 3, 4], [1, 2, 3, 4], {1, 2, 3, 4})  # first pass
    seen = []  # units the second pass would run
    monkeypatch.setattr(S, "run_pool", lambda units, cfg, w, s: seen.extend(u[0] for u in units) or [])  # record only
    rc, out = _run(S, monkeypatch, tmp_path, [1, 2, 3, 4], [1, 2, 3, 4], {1, 2, 3, 4})  # second pass
    assert rc == 0 and seen == []            # both unit files exist: nothing to run, merge only


def test_empty_prefilter_exits_1(S, monkeypatch, tmp_path, capsys):   # review focus 2
    """Pins the empty pre-filter: exit 1, a message naming the pre-filter, no output file."""
    rc, out = _run(S, monkeypatch, tmp_path, [1, 2], prefiltered=[], keep=set())  # nothing passes n_det
    assert rc == 1 and not out.exists() and "pre-filter" in capsys.readouterr().out  # refused loudly


def test_max_units_does_not_write_out(S, monkeypatch, tmp_path):     # review focus 4
    """Pins --max-units: the capped units run, but --out is never written from a partial run."""
    rc, out = _run(S, monkeypatch, tmp_path, [1, 2, 3, 4], [1, 2, 3, 4], {1, 2, 3, 4}, extra=("--max-units", "1"))  # one unit
    assert rc == 1 and not out.exists()  # partial: no output
    assert S.unit_path(tmp_path / "run", 0).exists() and not S.unit_path(tmp_path / "run", 1).exists()  # only unit 0 ran


def test_failed_unit_blocks_merge(S, monkeypatch, tmp_path, capsys):   # review focus 3
    """Pins a failed unit: exit 1, the unit named in the log, no output file."""
    def boom(c, o, n, schema):  # fails on oid 3
        """Raise for the minibatch holding oid 3, keep everything else."""
        if 3 in list(o): raise RuntimeError("db down")  # unit 1 fails
        return np.array(list(o), dtype=np.int64)  # keep all
    monkeypatch.setattr(S.db, "fetch_oids_min_n_det", lambda c, n, schema: np.array([1, 2, 3, 4], dtype=np.int64))  # n_det reader
    monkeypatch.setattr(S.db, "fetch_oids_with_good_detections", boom)  # rb reader
    def pool(units, cfg, workers, start_method):  # in-process pool with run_pool's error contract
        """Run units in-process; a raising unit comes back as {"unit", "error"}."""
        S._init_worker(cfg); res = []  # configure, collect
        for u in units:  # each unit
            try: res.append(S.process_unit(u))  # success
            except Exception as e: res.append({"unit": u[0], "error": repr(e)})  # failure as data
        return res  # all results
    monkeypatch.setattr(S, "run_pool", pool)  # no processes
    src = tmp_path / "in.npy"; np.save(src, np.array([1, 2, 3, 4], dtype=np.int64))  # input oid file
    out = tmp_path / "out.npy"  # output oid file
    rc = S.main(["--oid-file", str(src), "--out", str(out), "--out-dir", str(tmp_path / "run"),  # flags:
                 "--unit-size", "2", "--minibatch", "1", "--retries", "1", "--credentials", "creds"])  # run the script
    assert rc == 1 and not out.exists() and "unit 1" in capsys.readouterr().out  # failure named, no output
