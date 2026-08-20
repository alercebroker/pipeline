"""Unit tests for the offline ZTF feature LUT fixture + loaders.

Asserts structural invariants that hold for any valid generated fixture, so
the tests don't hard-code the feature-name list.
"""
from features.offline.feature_lut import (
    FEATURE_NAME_LUT,
    FEATURE_VERSION_LUT,
    default_version_name,
    load_feature_name_lut,
    render_seed_sql,
    version_name_to_id,
)


def test_name_lut_non_empty():
    assert len(FEATURE_NAME_LUT) > 0


def test_name_lut_ids_contiguous_from_zero():
    assert sorted(FEATURE_NAME_LUT) == list(range(len(FEATURE_NAME_LUT)))


def test_name_lut_in_extractor_order_not_alphabetical():
    # Ids follow extractor (natural) emission order, NOT alphabetical. Guard
    # against a regenerator accidentally re-sorting the fixture by name.
    names = [FEATURE_NAME_LUT[i] for i in sorted(FEATURE_NAME_LUT)]
    assert names != sorted(names)


def test_mjd_ref_features_follow_their_extractor_siblings():
    # Each reference-epoch feature sits immediately after its extractor's last
    # feature — the invariant the extractor-order LUT exists to preserve.
    pos = {name: i for i, name in FEATURE_NAME_LUT.items()}
    for sibling, mjd_ref in [
        ("SPM_chi", "SPM_mjd_ref"),
        ("TDE_mag0", "TDE_mjd_ref"),
        ("fleet_t0", "fleet_mjd_ref"),
        ("ulens_mag0", "ulens_mjd_ref"),
    ]:
        assert pos[mjd_ref] == pos[sibling] + 1


def test_load_returns_independent_copy():
    lut = load_feature_name_lut()
    assert lut == FEATURE_NAME_LUT
    lut[10_000] = "x"
    assert 10_000 not in FEATURE_NAME_LUT


def test_version_round_trips():
    vid, vname = next(iter(FEATURE_VERSION_LUT.items()))
    assert version_name_to_id(vname) == vid


def test_unknown_version_warns_and_returns_negative_one():
    assert version_name_to_id("__definitely_not_a_version__") == -1


def test_default_version_name_is_latest_entry():
    # default is the version_name of the highest version_id in the LUT
    expected = FEATURE_VERSION_LUT[max(FEATURE_VERSION_LUT)]
    assert default_version_name() == expected


def test_default_version_name_round_trips_to_an_id():
    assert version_name_to_id(default_version_name()) == max(FEATURE_VERSION_LUT)


def test_render_seed_sql_has_a_row_per_feature_name():
    sql = render_seed_sql()
    for feature_id, name in FEATURE_NAME_LUT.items():
        assert f"({feature_id}, '{name}', 0, 0)" in sql
    assert "INSERT INTO multisurvey_ztf.feature_name_lut" in sql
    assert "ON CONFLICT (feature_id, sid) DO NOTHING;" in sql


def test_render_seed_sql_seeds_every_version():
    sql = render_seed_sql()
    for version_id, version_name in FEATURE_VERSION_LUT.items():
        assert f"({version_id}, '{version_name}', 0, 0)" in sql
    assert "ON CONFLICT (version_id, sid) DO NOTHING;" in sql


def test_version_ids_follow_the_production_numbering_convention():
    """version_id must start at 1, never 0 — that is what production assigns.

    `features.database.get_or_create_version_id` inserts
    COALESCE(MAX(version_id), 0) + 1, so on an empty table the first version is
    1. The column has no default or sequence in either schema, so the number is
    whatever the inserter supplies: a seed that hand-writes 0 (as ours did)
    makes the same version_name resolve to a different id per schema, and every
    cross-schema comparison by version_id silently compares the wrong versions.
    """
    assert min(FEATURE_VERSION_LUT) >= 1
