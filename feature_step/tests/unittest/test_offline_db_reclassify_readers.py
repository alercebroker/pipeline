"""The two readers behind the reclassification: SQL shape and parameter types.

Both must be primary-key lookups (`oid = ANY(:oids)` plus sid / classifier_id)
on the named schema: multisurvey.feature and multisurvey.probability hold
billions of rows and have no other usable index. pandas' read_sql_query is
replaced so no connection is opened; what is asserted is the SQL and the bound
parameters it was handed, and the shape of what comes back.
"""
import numpy as np
import pandas as pd

from features.offline import db


class _Engine:
    """Stands in for the SQLAlchemy engine: connect() yields a no-op context manager."""
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def connect(self):
        return self._Conn()


def _capture(monkeypatch, frame):
    """Replace read_sql_query; return the list that collects (sql, params) calls."""
    calls = []

    def fake(query, conn, params=None):
        calls.append((str(query), params))
        return frame
    monkeypatch.setattr(db, "_make_engine", lambda _c: _Engine())
    monkeypatch.setattr(db.pd, "read_sql_query", fake)
    return calls


def test_fetch_feature_rows_reads_by_primary_key_on_the_named_schema(monkeypatch):
    """One index range per oid: the predicate is oid = ANY plus sid, on <schema>.feature."""
    frame = pd.DataFrame(columns=["oid", "feature_name", "band", "value", "updated_date"])
    calls = _capture(monkeypatch, frame)

    out = db.fetch_feature_rows("creds", np.array([1, 2], dtype=np.int64), schema="multisurvey")

    sql, params = calls[0]
    assert "multisurvey.feature f" in sql and "multisurvey.feature_name_lut l" in sql
    assert "f.oid = ANY(:oids)" in sql and "f.sid = :sid" in sql
    assert params == {"oids": [1, 2], "sid": 0}
    assert all(type(o) is int for o in params["oids"])   # numpy ints are not adapted by psycopg2
    assert list(out.columns) == ["oid", "feature_name", "band", "value", "updated_date"]


def test_fetch_probability_lastmjd_reads_one_classifier_by_primary_key(monkeypatch):
    """max(lastmjd) per oid over the rows of one classifier, keyed by oid, sid and classifier_id."""
    frame = pd.DataFrame({"oid": np.array([1, 2], dtype=np.int64),
                          "lastmjd": np.array([60000.5, 60001.0], dtype=np.float32)})
    calls = _capture(monkeypatch, frame)

    out = db.fetch_probability_lastmjd("creds", [1, 2, 3], classifier_id=5, schema="multisurvey")

    sql, params = calls[0]
    assert "multisurvey.probability" in sql
    assert "oid = ANY(:oids)" in sql and "sid = :sid" in sql and "classifier_id = :cid" in sql
    assert "GROUP BY oid" in sql
    assert params == {"oids": [1, 2, 3], "sid": 0, "cid": 5}
    assert out == {1: 60000.5, 2: 60001.0}               # oid 3 had no row: absent, not None
    assert all(type(k) is int and type(v) is float for k, v in out.items())
