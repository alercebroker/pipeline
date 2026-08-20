"""Unit tests for compute_db_features — the DB-ready offline output.

compute_astro_object (heavy, real extractor) is monkeypatched to return a stub
AstroObject with a `.features` frame; version_name_to_id is monkeypatched so
the test doesn't depend on the fixture's exact version string.
"""
import types

import numpy as np
import pandas as pd

from features.offline import lc_features


def test_compute_db_features_emits_feature_table_rows(monkeypatch):
    feats = pd.DataFrame({
        "name":    ["Amplitude", "Amplitude", "Period"],
        "fid":     ["g", "r", "g,r"],
        "value":   [0.5, 0.6, np.nan],
        "sid":     [0, 0, 0],
        "version": ["mod_a", "mod_b", "mod_a"],  # per-module; must NOT be used
    })
    ao = types.SimpleNamespace(features=feats)
    monkeypatch.setattr(lc_features, "compute_astro_object", lambda *a, **k: ao)
    monkeypatch.setattr(lc_features, "version_name_to_id", lambda v: 7)

    lut = {0: "Amplitude", 1: "Period"}
    out = lc_features.compute_db_features(
        {"oid": 123}, None, None, feature_name_lut=lut, version_name="anything"
    )

    assert list(out.columns) == ["oid", "sid", "feature_id", "band", "version", "value"]
    assert (out["oid"] == 123).all()
    assert (out["sid"] == 0).all()
    assert (out["version"] == 7).all()
    assert out["value"].notna().all()        # NaN Period row dropped
    assert len(out) == 2
    assert set(out["band"]) == {1, 2}         # surviving g + r rows
    assert set(out["feature_id"]) == {0}      # both rows are Amplitude -> id 0


def test_compute_db_features_returns_none_when_no_astro_object(monkeypatch):
    monkeypatch.setattr(lc_features, "compute_astro_object", lambda *a, **k: None)
    assert lc_features.compute_db_features(
        {"oid": 1}, None, None, feature_name_lut={}, version_name="x"
    ) is None


def test_compute_db_features_falls_back_when_package_not_installed(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    feats = pd.DataFrame({
        "name":  ["Amplitude"],
        "fid":   ["g"],
        "value": [0.5],
    })
    ao = types.SimpleNamespace(features=feats)
    monkeypatch.setattr(lc_features, "compute_astro_object", lambda *a, **k: ao)

    # Simulate feature-step NOT installed: _pkg_version raises.
    def _raise(_name):
        raise PackageNotFoundError(_name)
    monkeypatch.setattr(lc_features, "_pkg_version", _raise)

    # Spy on the fallback: default_version_name returns a known string, and
    # version_name_to_id records what it was asked to map.
    monkeypatch.setattr(lc_features, "default_version_name", lambda: "FALLBACK_VER")
    seen = {}
    def _v2id(v):
        seen["v"] = v
        return 3
    monkeypatch.setattr(lc_features, "version_name_to_id", _v2id)

    out = lc_features.compute_db_features({"oid": 1}, None, None, feature_name_lut={0: "Amplitude"})
    assert seen["v"] == "FALLBACK_VER"   # used the fallback, not a crash
    assert (out["version"] == 3).all()


def test_compute_db_features_uses_injected_version_id_without_the_fixture(monkeypatch):
    """A caller that resolved the id against the DB must bypass the fixture.

    <schema>.feature has no FK to feature_version_lut, so a fixture that drifted
    from the DB would stamp every row with a version that resolves to something
    else — and nothing would reject it. When version_id is supplied, the fixture
    reverse-map must not be consulted at all.
    """
    feats = pd.DataFrame({"name": ["Amplitude"], "fid": ["g"], "value": [0.5]})
    ao = types.SimpleNamespace(features=feats)
    monkeypatch.setattr(lc_features, "compute_astro_object", lambda *a, **k: ao)

    def _boom(_v):
        raise AssertionError("fixture version_name_to_id must not be consulted")
    monkeypatch.setattr(lc_features, "version_name_to_id", _boom)

    out = lc_features.compute_db_features(
        {"oid": 1}, None, None, feature_name_lut={0: "Amplitude"}, version_id=4
    )

    assert (out["version"] == 4).all()
