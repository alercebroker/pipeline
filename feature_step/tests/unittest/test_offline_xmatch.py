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


def test_persist_matches_dry_run_counts_and_does_not_connect(monkeypatch):
    def _boom(_creds):
        raise AssertionError("dry-run must not open a connection")
    from features.offline import db
    monkeypatch.setattr(db, "_make_engine", _boom)

    result = xmatch.persist_matches([_match(100, 12.0, 11.0, 9.0, 7.0)], execute=False)
    assert result == {"executed": False, "would_write": 1}


def test_persist_matches_execute_upserts_native_types(monkeypatch):
    fake = _FakeEngine()
    from features.offline import db
    monkeypatch.setattr(db, "_make_engine", lambda _c: fake)

    result = xmatch.persist_matches(
        [_match(36028941624528297, 12.0, 11.0, 9.0, 7.0, match_id="J190248", distance=0.33)],
        write_credentials="creds", schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 1}
    assert len(fake.conn.calls) == 1
    sql, records = fake.conn.calls[0]
    sql_str = str(sql)
    assert "multisurvey_ztf.xmatch" in sql_str
    assert "ON CONFLICT (oid, sid, catid)" in sql_str
    assert "DO UPDATE SET" in sql_str
    r0 = records[0]
    # native types, big oid preserved (not float)
    assert r0["oid"] == 36028941624528297 and isinstance(r0["oid"], int)
    assert isinstance(r0["sid"], int) and isinstance(r0["catid"], int)
    assert r0["oid_catalog"] == "J190248" and r0["dist"] == pytest.approx(0.33)


def test_persist_matches_drops_null_dist_or_catalog(monkeypatch, caplog):
    import logging
    fake = _FakeEngine()
    from features.offline import db
    monkeypatch.setattr(db, "_make_engine", lambda _c: fake)

    good = _match(100, 1.0, 1.0, 1.0, 1.0, match_id="ok", distance=0.2)
    no_dist = _match(101, 1.0, 1.0, 1.0, 1.0, match_id="x", distance=None)
    no_cat = _match(102, 1.0, 1.0, 1.0, 1.0, match_id=None, distance=0.3)
    with caplog.at_level(logging.WARNING, logger="features.offline.xmatch"):
        result = xmatch.persist_matches([good, no_dist, no_cat],
                                        write_credentials="creds", execute=True)
    assert result == {"executed": True, "written": 1}
    records = fake.conn.calls[0][1]
    assert [r["oid"] for r in records] == [100]
    assert any("NULL dist/oid_catalog" in r.getMessage() for r in caplog.records)


def test_persist_matches_default_schema_is_db_schema(monkeypatch):
    fake = _FakeEngine()
    from features.offline import db
    monkeypatch.setattr(db, "_make_engine", lambda _c: fake)
    xmatch.persist_matches([_match(100, 1.0, 1.0, 1.0, 1.0)],
                           write_credentials="creds", execute=True)  # no schema=
    from features.offline import db as _db
    assert f"{_db.SCHEMA}.xmatch" in str(fake.conn.calls[0][0])


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
