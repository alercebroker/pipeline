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
    """Records the conesearch args and returns canned matches."""
    def __init__(self, matches):
        self._matches = matches
        self.called_with = None

    def conesearch_with_metadata(self, ras, decs, oids, radius=1.5, catalogs=None):
        self.called_with = {"ras": ras, "decs": decs, "oids": oids,
                            "radius": radius, "catalogs": catalogs}
        return self._matches


def test_compute_matches_passes_string_oids_and_returns_matches():
    m = _match(100, 12.0, 11.0, 9.0, 7.0)
    client = _FakeClient([m])
    out = xmatch.compute_matches([100], [285.7], [76.5], client=client)
    assert out == [m]
    # oids stringified, coords forwarded as-is
    assert client.called_with["oids"] == ["100"]
    assert client.called_with["ras"] == [285.7]
    assert client.called_with["decs"] == [76.5]


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


def test_persist_matches_dry_run_returns_rows():
    rows = xmatch.persist_matches([_match(100, 12.0, 11.0, 9.0, 7.0)], execute=False)
    assert len(rows) == 1 and rows[0]["oid"] == 100


def test_persist_matches_execute_is_placeholder():
    with pytest.raises(NotImplementedError, match="placeholder"):
        xmatch.persist_matches([_match(100, 12.0, 11.0, 9.0, 7.0)], execute=True)


def test_resolve_url_prefers_arg_then_env(monkeypatch):
    monkeypatch.setenv("XMATCH_URL", "http://from-env:8081")
    assert xmatch._resolve_url("http://explicit:8081") == "http://explicit:8081"
    assert xmatch._resolve_url(None) == "http://from-env:8081"
    monkeypatch.delenv("XMATCH_URL", raising=False)
    with pytest.raises(ValueError, match="XMATCH_URL"):
        xmatch._resolve_url(None)
