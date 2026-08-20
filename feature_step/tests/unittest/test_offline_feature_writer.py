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


class _FakeCursor:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeRaw:
    """Stands in for a psycopg2 connection taken from the pool."""

    def __init__(self, owner):
        self.owner = owner

    def cursor(self):
        return _FakeCursor(self.owner)

    def commit(self):
        self.owner.commits += 1

    def rollback(self):
        self.owner.rollbacks += 1

    def close(self):
        self.owner.closed += 1


class _FakeEngine:
    def __init__(self):
        self.calls = []          # one entry per execute_values call
        self.commits = self.rollbacks = self.closed = 0

    def raw_connection(self):
        return _FakeRaw(self)


def _capture(engine):
    """Replacement for execute_values that records (sql, tuples, page_size)."""
    def _execute_values(cur, sql, argslist, page_size=None, **kw):
        engine.calls.append((str(sql), list(argslist), page_size))
    return _execute_values


def test_dry_run_does_not_connect(monkeypatch):
    def _boom(_creds):
        raise AssertionError("dry-run must not open a connection")
    monkeypatch.setattr(feature_writer.db, "_make_engine", _boom)

    df = _df([(36028941624528297, 0, 0, 1, 0, 0.5),
              (36028941624528297, 0, 0, 2, 0, 0.6)])
    result = feature_writer.write_features(df, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 2}


def test_execute_sends_all_rows_in_one_batched_statement(monkeypatch):
    """One execute_values call, not one round trip per row.

    psycopg2's executemany is a Python loop that waits for the server once per
    row: measured against the real table, 19,054 feature rows took 161s that way
    (118 rows/s) versus 1.4s batched (13,789 rows/s). At ~193 rows per object and
    millions of objects, the per-row round trip is the whole cost of the write.
    """
    fake = _FakeEngine()
    monkeypatch.setattr(feature_writer.db, "_make_engine", lambda _c: fake)
    monkeypatch.setattr(feature_writer, "execute_values", _capture(fake))

    df = _df([(36028941624528297, 0, 5, 12, 0, 0.5),
              (36028941624528297, 0, 7, 0, 0, None)])
    result = feature_writer.write_features(df, "creds", schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 2}
    assert len(fake.calls) == 1
    sql, tuples, page_size = fake.calls[0]
    assert "multisurvey_ztf.feature" in sql
    assert "VALUES %s" in sql          # the placeholder execute_values expands
    assert "ON CONFLICT (oid, sid, feature_id, band)" in sql
    assert "DO UPDATE SET" in sql
    assert page_size and page_size > 1
    assert fake.commits == 1 and fake.closed == 1
    # native types, big oid preserved (not float), None value passes through
    assert tuples[0][0] == 36028941624528297 and isinstance(tuples[0][0], int)
    assert isinstance(tuples[0][2], int) and isinstance(tuples[0][3], int)
    assert tuples[0][5] == 0.5
    assert tuples[1][5] is None


def test_execute_refuses_a_batch_with_a_duplicated_key(monkeypatch):
    """Two rows with the same PK in ONE statement is a Postgres error.

    "ON CONFLICT DO UPDATE command cannot affect row a second time" — and it
    could not happen under executemany, where each row was its own statement and
    the second simply updated the first. Batching makes it possible, so a caller
    that assembled the same oid twice must hear about it as a clear error rather
    than as that message from the driver.
    """
    df = _df([(36028941624528297, 0, 5, 12, 0, 0.5),
              (36028941624528297, 0, 5, 12, 0, 0.9)])
    with pytest.raises(ValueError, match="duplicate"):
        feature_writer.write_features(df, "ignored", execute=False)


def test_nan_feature_id_rows_dropped(monkeypatch, caplog):
    df = _df([(36028941624528297, 0, 0, 1, 0, 0.5),
              (36028941624528297, 0, np.nan, 2, 0, 0.6)])
    import logging
    with caplog.at_level(logging.WARNING, logger="features.offline.feature_writer"):
        result = feature_writer.write_features(df, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 1}
    assert any("feature_id" in r.message for r in caplog.records)


def test_default_schema_is_db_schema(monkeypatch):
    """No schema arg -> writes target db.SCHEMA (the offline default), not a literal."""
    fake = _FakeEngine()
    monkeypatch.setattr(feature_writer.db, "_make_engine", lambda _c: fake)
    monkeypatch.setattr(feature_writer, "execute_values", _capture(fake))
    df = _df([(36028941624528297, 0, 0, 1, 0, 0.5)])
    feature_writer.write_features(df, "creds", execute=True)  # no schema=
    assert f"{feature_writer.db.SCHEMA}.feature" in fake.calls[0][0]


def test_version_minus_one_warns(caplog):
    import logging
    df = _df([(36028941624528297, 0, 0, 1, -1, 0.5)])
    with caplog.at_level(logging.WARNING, logger="features.offline.feature_writer"):
        result = feature_writer.write_features(df, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 1}
    assert any("version=-1" in r.getMessage() for r in caplog.records)
