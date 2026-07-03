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


def test_render_seed_sql_row_counts():
    sql = render_seed_sql()
    lines = sql.splitlines()
    cls_start = next(
        i for i, l in enumerate(lines)
        if l.endswith("(classifier_id, classifier_name, classifier_version, tid) VALUES")
    )
    cls_end = next(
        i for i in range(cls_start + 1, len(lines))
        if lines[i].startswith("ON CONFLICT (classifier_id)")
    )
    assert len(lines[cls_start + 1:cls_end]) == 5

    tax_start = next(
        i for i, l in enumerate(lines)
        if l.endswith('(class_id, class_name, "order", classifier_id) VALUES')
    )
    tax_end = next(
        i for i in range(tax_start + 1, len(lines))
        if lines[i].startswith("ON CONFLICT (class_id, classifier_id)")
    )
    assert len(lines[tax_start + 1:tax_end]) == 45


def test_taxonomy_order_equals_class_id():
    import re
    sql = render_seed_sql()
    # taxonomy tuples are (int, 'str', int, int); classifier tuples are (int, 'str', 'str', int)
    # so this pattern (3rd group all-digits) matches only taxonomy rows.
    matches = re.findall(r"\((\d+), '[^']*', (\d+), \d+\)", sql)
    assert matches, "no taxonomy rows matched"
    for class_id, order in matches:
        assert order == class_id


import pathlib

_SQL_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "features" / "offline" / "ztf_classifier_taxonomy_seed.sql"
)


def test_committed_sql_matches_render():
    # The .sql on disk must be exactly what render_seed_sql() produces, so the
    # fixture stays the single source of truth (no hand-edited drift).
    assert _SQL_PATH.exists(), f"missing generated SQL: {_SQL_PATH}"
    assert _SQL_PATH.read_text() == render_seed_sql()
