"""Tests for features/offline/xmatch.py — the offline AllWISE crossmatch.

No network: a fake client stands in for XmatchClient.conesearch_with_metadata,
returning MatchWithMetadata-shaped dicts (as Xwave/xmatch_client would)."""
import numpy as np
import pandas as pd
import pytest

from features.offline import xmatch


def _match(oid, w1, w2, w3, w4, match_id="J1", distance=0.1, catalog="allwise"):
    return {
        "oid": str(oid),
        "match_id": match_id,
        "catalog": catalog,
        "distance": distance,
        "metadata": {
            "w1mpro": {"Float64": w1},
            "w2mpro": {"Float64": w2},
            "w3mpro": {"Float64": w3},
            "w4mpro": {"Float64": w4},
        },
    }


class _FakeClient:
    """Stands in for XmatchClient with the real `conesearch_with_metadata`
    signature (single `catalog` string). Records one entry per call and returns
    canned matches, mirroring the per-catalog request loop the step uses."""
    def __init__(self, matches):
        self._matches = matches
        self.calls = []

    def conesearch_with_metadata(self, ras, decs, oids, radius=1.5, catalog=None):
        self.calls.append({"ras": ras, "decs": decs, "oids": oids,
                           "radius": radius, "catalog": catalog})
        return self._matches

    @property
    def called_with(self):
        return self.calls[-1] if self.calls else None


def test_compute_matches_passes_string_oids_and_returns_matches():
    m = _match(100, 12.0, 11.0, 9.0, 7.0)
    client = _FakeClient([m])
    out = xmatch.compute_matches([100], [285.7], [76.5], client=client)
    assert out == [m]
    # default catalogs=(allwise,) -> exactly one per-catalog request
    assert len(client.calls) == 1
    assert client.called_with["catalog"] == "allwise"
    # oids stringified, coords forwarded as-is
    assert client.called_with["oids"] == ["100"]
    assert client.called_with["ras"] == [285.7]
    assert client.called_with["decs"] == [76.5]


def test_compute_matches_one_request_per_catalog():
    m = _match(100, 12.0, 11.0, 9.0, 7.0)
    client = _FakeClient([m])
    xmatch.compute_matches([100], [285.7], [76.5], client=client,
                           catalogs=("allwise", "gaia"))
    assert [c["catalog"] for c in client.calls] == ["allwise", "gaia"]


def test_compute_matches_none_catalogs_is_single_global_call():
    client = _FakeClient([])
    xmatch.compute_matches([100], [285.7], [76.5], client=client, catalogs=None)
    assert len(client.calls) == 1
    assert client.called_with["catalog"] is None


def test_compute_matches_empty_short_circuits():
    # No client needed when there is nothing to search.
    assert xmatch.compute_matches([], [], []) == []


def test_compute_matches_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        xmatch.compute_matches([1, 2], [10.0], [20.0], client=_FakeClient([]))


def test_matches_to_allwise_df_shape_and_values():
    df = xmatch.matches_to_allwise_df([_match(100, 12.0, 11.0, 9.0, 7.0)])
    assert list(df.columns) == ["oid", "W1", "W2", "W3", "W4"]
    row = df.iloc[0]
    assert row["oid"] == 100
    assert (row["W1"], row["W2"], row["W3"], row["W4"]) == (12.0, 11.0, 9.0, 7.0)


def test_matches_to_allwise_df_keeps_nearest_per_oid():
    far = _match(100, 1.0, 1.0, 1.0, 1.0, match_id="far", distance=1.4)
    near = _match(100, 12.0, 11.0, 9.0, 7.0, match_id="near", distance=0.2)
    df = xmatch.matches_to_allwise_df([far, near])
    assert len(df) == 1
    assert df.iloc[0]["W1"] == 12.0  # nearest (smallest distance) wins


def test_matches_to_allwise_df_ignores_non_allwise_and_empty():
    other = _match(100, 12.0, 11.0, 9.0, 7.0, catalog="gaia")
    df = xmatch.matches_to_allwise_df([other])
    assert list(df.columns) == ["oid", "W1", "W2", "W3", "W4"]
    assert len(df) == 0
    assert len(xmatch.matches_to_allwise_df([])) == 0


def test_matches_to_allwise_df_missing_metadata_is_nan():
    m = _match(100, 12.0, 11.0, 9.0, 7.0)
    del m["metadata"]["w3mpro"]
    df = xmatch.matches_to_allwise_df([m])
    assert np.isnan(df.iloc[0]["W3"])
    assert df.iloc[0]["W1"] == 12.0


def test_build_xmatch_rows_mirrors_scribe_db_columns():
    # Exact multisurvey_ztf.xmatch columns (no `catalog` col in the table).
    rows = xmatch.build_xmatch_rows([_match(100, 12.0, 11.0, 9.0, 7.0,
                                            match_id="J190248", distance=0.33)])
    assert rows == [{
        "oid": 100, "sid": xmatch.SID_ZTF, "catid": xmatch.ALLWISE_CATID,
        "oid_catalog": "J190248", "dist": 0.33,
    }]


def test_build_xmatch_rows_non_allwise_catid_is_unknown():
    # parse_xmatch maps unknown catalogs to catid -999.
    rows = xmatch.build_xmatch_rows([_match(100, 1.0, 1.0, 1.0, 1.0, catalog="gaia")])
    assert rows[0]["catid"] == -999


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeRaw:
    """Stands in for the psycopg2 connection taken from the pool."""

    def __init__(self, owner):
        self.owner = owner

    def cursor(self):
        return _FakeCursor()

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


def _patch_engine(monkeypatch):
    from features.offline import db
    fake = _FakeEngine()
    monkeypatch.setattr(db, "_make_engine", lambda _c: fake)
    monkeypatch.setattr(xmatch, "execute_values", _capture(fake))
    return fake


def test_persist_matches_dry_run_counts_and_does_not_connect(monkeypatch):
    def _boom(_creds):
        raise AssertionError("dry-run must not open a connection")
    from features.offline import db
    monkeypatch.setattr(db, "_make_engine", _boom)

    result = xmatch.persist_matches([_match(100, 12.0, 11.0, 9.0, 7.0)], execute=False)
    assert result == {"executed": False, "would_write": 1}


def test_persist_matches_sends_every_row_in_one_batched_statement(monkeypatch):
    """One execute_values call for the whole unit, not a round trip per match.

    A work unit produces ~4.3k link rows; the per-row executemany this replaced
    cost ~8.5 ms each, which would have made persisting the crossmatch cost more
    than the 19k feature rows and the 225k probability rows combined.
    """
    fake = _patch_engine(monkeypatch)
    matches = [_match(36028941624528297 + i, 12.0, 11.0, 9.0, 7.0,
                      match_id=f"J{i}", distance=0.3) for i in range(2500)]

    result = xmatch.persist_matches(matches, write_credentials="creds",
                                    schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 2500}
    assert len(fake.calls) == 1
    sql, tuples, page_size = fake.calls[0]
    assert "multisurvey_ztf.xmatch" in sql
    assert "VALUES %s" in sql
    assert "ON CONFLICT (oid, sid, catid)" in sql
    assert "DO UPDATE SET" in sql
    assert page_size and page_size > 1
    assert len(tuples) == 2500
    assert fake.commits == 1 and fake.closed == 1


def test_persist_matches_execute_upserts_native_types(monkeypatch):
    fake = _patch_engine(monkeypatch)

    result = xmatch.persist_matches(
        [_match(36028941624528297, 12.0, 11.0, 9.0, 7.0, match_id="J190248", distance=0.33)],
        write_credentials="creds", schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 1}
    oid, sid, catid, oid_catalog, dist = fake.calls[0][1][0]
    # native types, big oid preserved (not float)
    assert oid == 36028941624528297 and isinstance(oid, int)
    assert isinstance(sid, int) and isinstance(catid, int)
    assert oid_catalog == "J190248" and dist == pytest.approx(0.33)


def test_persist_matches_keeps_the_nearest_of_two_counterparts(monkeypatch, caplog):
    """Xwave can return two counterparts for one oid inside the same cone.

    Both collapse to the same (oid, sid, catid), and Postgres refuses to touch a
    row twice in one ON CONFLICT statement. Raising would wedge the unit for
    good -- it fails, the rerun recomputes the same matches and fails again --
    so collapse to the nearest, which is what the old per-row path ended up
    storing anyway, and say so in the log.
    """
    import logging
    fake = _patch_engine(monkeypatch)
    dup = [_match(100, 1.0, 1.0, 1.0, 1.0, match_id="far", distance=0.9),
           _match(100, 1.0, 1.0, 1.0, 1.0, match_id="near", distance=0.2),
           _match(101, 1.0, 1.0, 1.0, 1.0, match_id="other", distance=0.5)]

    with caplog.at_level(logging.WARNING, logger="features.offline.xmatch"):
        result = xmatch.persist_matches(dup, write_credentials="creds", execute=True)

    assert result == {"executed": True, "written": 2}
    by_oid = {t[0]: t for t in fake.calls[0][1]}
    assert by_oid[100][3] == "near" and by_oid[100][4] == pytest.approx(0.2)
    assert by_oid[101][3] == "other"
    assert any("counterpart" in r.getMessage() for r in caplog.records)


def test_persist_matches_dedupe_applies_to_the_dry_run_too(monkeypatch):
    """would_write has to be the number that will actually land, or the dry run
    is not a preview of anything."""
    dup = [_match(100, 1.0, 1.0, 1.0, 1.0, match_id="a", distance=0.9),
           _match(100, 1.0, 1.0, 1.0, 1.0, match_id="b", distance=0.2)]
    assert xmatch.persist_matches(dup, execute=False) == {"executed": False,
                                                          "would_write": 1}


def test_persist_matches_drops_null_dist_or_catalog(monkeypatch, caplog):
    import logging
    fake = _patch_engine(monkeypatch)

    good = _match(100, 1.0, 1.0, 1.0, 1.0, match_id="ok", distance=0.2)
    no_dist = _match(101, 1.0, 1.0, 1.0, 1.0, match_id="x", distance=None)
    no_cat = _match(102, 1.0, 1.0, 1.0, 1.0, match_id=None, distance=0.3)
    with caplog.at_level(logging.WARNING, logger="features.offline.xmatch"):
        result = xmatch.persist_matches([good, no_dist, no_cat],
                                        write_credentials="creds", execute=True)
    assert result == {"executed": True, "written": 1}
    assert [t[0] for t in fake.calls[0][1]] == [100]
    assert any("NULL dist/oid_catalog" in r.getMessage() for r in caplog.records)


def test_persist_matches_empty_execute_makes_no_call(monkeypatch):
    fake = _patch_engine(monkeypatch)
    result = xmatch.persist_matches([], write_credentials="creds", execute=True)
    assert result == {"executed": True, "written": 0}
    assert fake.calls == []


def test_persist_matches_default_schema_is_db_schema(monkeypatch):
    fake = _patch_engine(monkeypatch)
    xmatch.persist_matches([_match(100, 1.0, 1.0, 1.0, 1.0)],
                           write_credentials="creds", execute=True)  # no schema=
    from features.offline import db as _db
    assert f"{_db.SCHEMA}.xmatch" in fake.calls[0][0]


def test_resolve_url_prefers_arg_then_env(monkeypatch):
    monkeypatch.setenv("XMATCH_URL", "http://from-env:8081")
    assert xmatch._resolve_url("http://explicit:8081") == "http://explicit:8081"
    assert xmatch._resolve_url(None) == "http://from-env:8081"
    monkeypatch.delenv("XMATCH_URL", raising=False)
    with pytest.raises(ValueError, match="XMATCH_URL"):
        xmatch._resolve_url(None)


def test_allwise_for_oid_computes_live_when_a_url_is_given(monkeypatch):
    """With a URL, the AllWISE colors come from Xwave — never from the DB.

    <schema>.allwise is empty for ZTF (the catalog rows are bulk-loaded by a
    separate process, and that load never ran for multisurvey_ztf), so a DB read
    silently yields no WISE and the features come out NaN. That is the exact
    shape of the production bug in WISE_NULL_CLASSIFICATION_IMPACT.md, so the
    live path must not fall back to it.
    """
    def _boom(*a, **k):
        raise AssertionError("must not read the DB when a crossmatch URL is given")
    monkeypatch.setattr(xmatch, "_db_allwise", _boom)
    monkeypatch.setattr(xmatch, "compute_matches",
                        lambda *a, **k: [_match(7, 13.7, 13.1, 9.7, 7.2)])

    allwise, matches = xmatch.allwise_for_oid(
        7, 222.98, 4.92, "creds", xmatch_url="http://127.0.0.1:8081")

    assert list(allwise.columns) == ["oid", "W1", "W2", "W3", "W4"]
    assert allwise.iloc[0]["W1"] == 13.7
    assert len(matches) == 1          # raw matches returned so they can be persisted


def test_allwise_for_oid_reads_the_db_when_no_url_is_given(monkeypatch):
    frame = pd.DataFrame({"oid": [7], "W1": [1.0], "W2": [2.0], "W3": [3.0], "W4": [4.0]})
    monkeypatch.setattr(xmatch, "_db_allwise", lambda creds, oids: frame)
    monkeypatch.setattr(xmatch, "compute_matches",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not hit Xwave without a URL")))

    allwise, matches = xmatch.allwise_for_oid(7, 222.98, 4.92, "creds", xmatch_url=None)

    assert allwise.equals(frame)
    assert matches == []              # nothing was computed, so nothing to persist
