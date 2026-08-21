"""The oid list must come out ascending, and Postgres must not be the one sorting.

Ascending order is load-bearing twice over: it is what makes a work unit's oids
land on adjacent index pages instead of scattering across the heap, and it is
what makes the run fingerprint (a hash of the array) reproducible, so a rerun
resumes instead of starting over.

Asking the server for it is the expensive way to get it. The only index that
serves the filter is on n_det, so the plan is a bitmap heap scan followed by an
external sort of ~26M rows; the client can sort the same array in memory in
seconds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_run_batch as R


class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execution_options(self, **kw):
        return self


class _Engine:
    def connect(self):
        return _Conn()


def _capture(monkeypatch, rows):
    """Run select_oids against a stubbed engine; return (array, sql text)."""
    seen = {}

    def fake_read_sql_query(sql, conn, params=None, chunksize=None):
        seen["sql"] = str(sql)
        return iter([pd.DataFrame({"oid": rows})])

    monkeypatch.setattr(R.db, "_make_engine", lambda c: _Engine())
    monkeypatch.setattr(R.pd, "read_sql_query", fake_read_sql_query)
    return R.select_oids("creds.json", 2), seen["sql"]


def test_the_oid_list_is_ascending_even_when_the_server_returns_it_scattered(monkeypatch):
    oids, _ = _capture(monkeypatch, [30, 10, 20])
    assert oids.tolist() == [10, 20, 30]


def test_the_server_is_not_asked_to_sort(monkeypatch):
    _, sql = _capture(monkeypatch, [1, 2, 3])
    assert "ORDER BY" not in sql.upper()
