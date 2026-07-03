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
