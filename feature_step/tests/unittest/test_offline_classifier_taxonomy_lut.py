"""Unit tests for the BHRF classifier + taxonomy seed fixture and SQL renderer."""
from features.offline.classifier_taxonomy_lut import (
    CLASSIFIER_LUT,
    TAXONOMY_LUT,
    CLASSIFIER_VERSION,
    render_seed_sql,
)

FLAT_ID = 5


def test_five_classifiers_ids_5_to_9():
    ids = [c["classifier_id"] for c in CLASSIFIER_LUT]
    assert ids == [5, 6, 7, 8, 9]


def test_classifier_names_and_version():
    by_id = {c["classifier_id"]: c for c in CLASSIFIER_LUT}
    assert by_id[5]["classifier_name"] == "lc_classifier_BHRF_forced_phot"
    assert by_id[6]["classifier_name"] == "lc_classifier_BHRF_forced_phot_top"
    assert by_id[7]["classifier_name"] == "lc_classifier_BHRF_forced_phot_transient"
    assert by_id[8]["classifier_name"] == "lc_classifier_BHRF_forced_phot_stochastic"
    assert by_id[9]["classifier_name"] == "lc_classifier_BHRF_forced_phot_periodic"
    assert all(c["classifier_version"] == "2.1.0" for c in CLASSIFIER_LUT)
    assert all(c["tid"] == 0 for c in CLASSIFIER_LUT)
    assert CLASSIFIER_VERSION == "2.1.0"


def test_taxonomy_class_counts():
    counts = {cid: len(classes) for cid, classes in TAXONOMY_LUT.items()}
    assert counts == {5: 21, 6: 3, 7: 6, 8: 6, 9: 9}
    total = sum(counts.values())
    assert total == 45


def test_transient_uses_sesn_not_snibc():
    assert "SESN" in TAXONOMY_LUT[7]
    assert "SNIbc" not in TAXONOMY_LUT[7]
    assert "SESN" in TAXONOMY_LUT[FLAT_ID]
    assert "SNIbc" not in TAXONOMY_LUT[FLAT_ID]


def test_flat_is_union_of_branches():
    branches = set(TAXONOMY_LUT[7]) | set(TAXONOMY_LUT[8]) | set(TAXONOMY_LUT[9])
    assert set(TAXONOMY_LUT[FLAT_ID]) == branches  # 6 + 6 + 9 = 21 leaves


def test_every_classifier_has_taxonomy():
    assert set(TAXONOMY_LUT) == {c["classifier_id"] for c in CLASSIFIER_LUT}


def test_render_seed_sql_is_idempotent_and_targets_composite_pk():
    sql = render_seed_sql()
    # classifier upsert targets the classifier_id PK
    assert "ON CONFLICT (classifier_id) DO NOTHING" in sql
    # taxonomy upsert targets the composite PK, NOT class_id alone
    assert 'ON CONFLICT (class_id, classifier_id) DO NOTHING' in sql
    # order is quoted (reserved word)
    assert '"order"' in sql
    # created_date is NOT supplied (server default)
    assert "created_date" not in sql
    # a spot-check row: flat classifier SESN at class_id 14
    assert "(14, 'SESN', 14, 5)" in sql
