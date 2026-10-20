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
    with engine.begin() as c:  # seed objects
        c.exec_driver_sql(f"INSERT INTO {SCHEMA}.object VALUES (30, 0, 8), (10, 0, 9), (20, 0, 7), (40, 1, 50)")  # 40 is another survey
    out = db.fetch_oids_min_n_det("creds", 8, schema=SCHEMA)  # run the reader
    assert out.dtype == np.int64 and out.tolist() == [10, 30]  # sorted, sid 0, n_det >= 8


def test_rb_count_drops_oid_with_one_rb_below_threshold(engine):  # spec criterion 3, first case
    _dets(engine, 1, [0.9] * 7 + [0.5])  # 7 good, 1 bad
    assert db.fetch_oids_with_good_detections("creds", [1], 8, schema=SCHEMA).tolist() == []  # only 7 survive


def test_rb_count_keeps_oid_with_null_rb(engine):  # spec criterion 3, second case
    _dets(engine, 2, [0.55] * 7 + [None])  # boundary rb and a NULL
    assert db.fetch_oids_with_good_detections("creds", [2], 8, schema=SCHEMA).tolist() == [2]  # all 8 survive


def test_rb_count_ignores_other_sid_and_oids_not_asked(engine):  # sid and oid scoping
    _dets(engine, 3, [0.9] * 8, sid=1)  # right count, wrong survey
    _dets(engine, 4, [0.9] * 8)  # passes but not in the list
    _dets(engine, 5, [0.9] * 8)  # passes and asked
    assert db.fetch_oids_with_good_detections("creds", [3, 5], 8, schema=SCHEMA).tolist() == [5]  # only 5


def test_rb_count_accepts_numpy_oids(engine):  # review focus 1
    _dets(engine, 6, [0.9] * 8)  # passes
    out = db.fetch_oids_with_good_detections("creds", np.array([6], dtype=np.int64), 8, schema=SCHEMA)  # numpy input
    assert out.tolist() == [6] and out.dtype == np.int64  # plain result, int64
