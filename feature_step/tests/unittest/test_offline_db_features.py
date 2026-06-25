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
