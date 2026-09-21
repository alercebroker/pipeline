"""scripts/create_private_tables.py creates the three private tables from the
db_plugins models, and nothing else: no create_all over the whole schema.

No test here opens a database; the SQL the script would run is checked as
text, and the create step is checked against a recording engine.
"""
from unittest import mock

from db_plugins.db.sql.models_pipeline import ClassifierPrivate, ProbabilityPrivate, TaxonomyPrivate

from scripts.create_private_tables import PRIVATE_MODELS, create_tables, statements

SCHEMA = "multisurvey_ztf"


def test_only_the_three_private_models_are_created():
    assert set(PRIVATE_MODELS) == {ClassifierPrivate, TaxonomyPrivate, ProbabilityPrivate}


def test_create_tables_creates_exactly_the_private_tables_and_their_partitions():
    engine = mock.MagicMock()
    conn = mock.MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    with mock.patch("scripts.create_private_tables.Base.metadata.create_all") as create_all:
        create_tables(engine, SCHEMA)

    tables = create_all.call_args.kwargs["tables"]
    assert {t.name for t in tables} == {"classifier_private", "taxonomy_private", "probability_private"}
    partition_ddl = [str(c.args[0]) for c in conn.execute.call_args_list]
    assert len(partition_ddl) == ProbabilityPrivate.__n_partitions__
    assert all(f"{SCHEMA}.probability_private_part_" in ddl for ddl in partition_ddl)
    assert not any("probability_part_" in ddl for ddl in partition_ddl)


def test_drop_statements_only_for_the_tables_asked_to_recreate():
    sql = statements(SCHEMA, recreate=["probability_private"], reader=None, writer=None)

    assert sql == ["DROP TABLE IF EXISTS multisurvey_ztf.probability_private CASCADE"]


def test_recreate_refuses_a_public_table():
    try:
        statements(SCHEMA, recreate=["probability"], reader=None, writer=None)
    except ValueError as e:
        assert "probability" in str(e)
    else:
        raise AssertionError("public table accepted")


def test_grant_statements_read_for_the_reader_and_write_only_on_probability_for_the_writer():
    sql = statements(SCHEMA, recreate=[], reader="readonly_user", writer="rubin_stamp_ms")

    assert sql == [
        "GRANT SELECT ON multisurvey_ztf.classifier_private TO readonly_user",
        "GRANT SELECT ON multisurvey_ztf.taxonomy_private TO readonly_user",
        "GRANT SELECT ON multisurvey_ztf.probability_private TO readonly_user",
        "GRANT SELECT ON multisurvey_ztf.classifier_private TO rubin_stamp_ms",
        "GRANT SELECT ON multisurvey_ztf.taxonomy_private TO rubin_stamp_ms",
        "GRANT SELECT, INSERT, UPDATE ON multisurvey_ztf.probability_private TO rubin_stamp_ms",
    ]
