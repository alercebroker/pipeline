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
