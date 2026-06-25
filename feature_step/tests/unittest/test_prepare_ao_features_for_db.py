"""Unit tests for prepare_ao_features_for_db — the ZTF DB-prep rule.

Uses a lightweight stand-in object (prepare only reads `.features`), so no
real AstroObject/DB is needed.
"""
import logging
import types

import numpy as np
import pandas as pd

from features.utils.parsers import prepare_ao_features_for_db


def _ao(features_df):
    return types.SimpleNamespace(features=features_df)


def test_maps_ids_from_lut_drops_nan_and_bands():
    df = pd.DataFrame({
        "name":  ["Amplitude", "Amplitude", "Period", "MHPS_ratio"],
        "fid":   ["g", "r", "g,r", None],
        "value": [0.5, 0.6, np.nan, 1.2],
    })
    # Ids chosen so they do NOT match appearance order — proves the LUT is used,
    # not the old enumerate.
    lut = {5: "Amplitude", 3: "MHPS_ratio", 7: "Period"}

    out = prepare_ao_features_for_db(_ao(df), lut)

    # NaN value (Period) dropped before id mapping
    assert "Period" not in set(out["name"])
    assert len(out) == 3
    # band codes: g->1, r->2, None->0
    assert out.loc[(out["name"] == "Amplitude") & (out["band"] == 1)].shape[0] == 1
    assert out.loc[(out["name"] == "Amplitude") & (out["band"] == 2)].shape[0] == 1
    assert out.loc[out["name"] == "MHPS_ratio", "band"].iloc[0] == 0
    # feature_id comes from the LUT, not enumerate(0,1,...)
    assert out.loc[out["name"] == "Amplitude", "feature_id"].iloc[0] == 5
    assert out.loc[out["name"] == "MHPS_ratio", "feature_id"].iloc[0] == 3
    # output columns
    assert set(out.columns) == {"name", "value", "band", "feature_id"}


def test_unmapped_name_yields_nan_id(caplog):
    df = pd.DataFrame({"name": ["Unknown"], "fid": ["g"], "value": [1.0]})
    with caplog.at_level(logging.WARNING, logger="alerce.FeatureStep"):
        out = prepare_ao_features_for_db(_ao(df), {0: "Amplitude"})
    assert out["feature_id"].isna().all()
    # the warning is the only runtime signal that id-mapping silently failed
    assert any("Unknown" in r.message for r in caplog.records)
