"""Unit tests for the DB-backed feature LUT readers — fake engine, no real DB.

These close the gap FLOW.md §3d warns about: the feature rows we write are
stamped with `feature_id` / `version` ids taken from the LOCAL fixture, while
`<schema>.feature` has no FK to the LUTs. If the DB were ever re-seeded with a
different numbering, we would silently write correctly-shaped rows that mean
something else. The writer must resolve both ids against the DB, the way
`fetch_taxonomy_maps` already does for `class_id`.
"""
import pytest

from features.offline import db


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self._rows

    def scalar(self):
        return self._rows[0] if self._rows else None


class _Conn:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((str(sql), params))
        return _Result(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, rows):
        self.conn = _Conn(rows)

    def connect(self):
        return self.conn


def test_fetch_feature_name_lut_maps_id_to_name(monkeypatch):
    rows = [
        {"feature_id": 0, "feature_name": "g-r_mean"},
        {"feature_id": 1, "feature_name": "g-r_max"},
        {"feature_id": 15, "feature_name": "sgscore1"},
    ]
    engine = _Engine(rows)
    monkeypatch.setattr(db, "_make_engine", lambda _c: engine)

    lut = db.fetch_feature_name_lut("creds", schema="multisurvey_ztf")

    assert lut == {0: "g-r_mean", 1: "g-r_max", 15: "sgscore1"}
    # Same {id: name} shape as the fixture loader, so it is a drop-in.
    assert isinstance(next(iter(lut)), int)
    sql, params = engine.conn.executed[0]
    assert "multisurvey_ztf.feature_name_lut" in sql
    assert params["sid"] == db.SID


def test_fetch_feature_name_lut_default_schema(monkeypatch):
    engine = _Engine([])
    monkeypatch.setattr(db, "_make_engine", lambda _c: engine)
    db.fetch_feature_name_lut("creds")
    assert f"{db.SCHEMA}.feature_name_lut" in engine.conn.executed[0][0]


def test_fetch_feature_version_id_returns_native_int(monkeypatch):
    engine = _Engine([0])
    monkeypatch.setattr(db, "_make_engine", lambda _c: engine)

    vid = db.fetch_feature_version_id("creds", "27.5.7a31", schema="multisurvey_ztf")

    assert vid == 0
    assert isinstance(vid, int)
    sql, params = engine.conn.executed[0]
    assert "multisurvey_ztf.feature_version_lut" in sql
    assert params["version_name"] == "27.5.7a31"
    assert params["sid"] == db.SID


def test_fetch_feature_version_id_raises_when_version_absent(monkeypatch):
    """The fixture path returns -1 and logs; that silently mislabels every row.

    There is no FK on <schema>.feature, so a -1 would be accepted by the DB.
    Refusing to return an id is the only thing that stops the write.
    """
    engine = _Engine([])
    monkeypatch.setattr(db, "_make_engine", lambda _c: engine)

    with pytest.raises(LookupError, match="27.5.7a31"):
        db.fetch_feature_version_id("creds", "27.5.7a31")
