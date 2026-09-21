"""Private copies of classifier, taxonomy and probability.

A classifier that is not public yet (the TNS candidate hunter) writes to
*_private tables in the same schema: same columns and physical shape as the
public ones, their own index and constraint names, no grants to the API roles.
"""
import pytest

from db_plugins.db.sql.models_pipeline import (
    Base,
    Classifier,
    ClassifierPrivate,
    Probability,
    ProbabilityPrivate,
    Taxonomy,
    TaxonomyPrivate,
)


def columns(model):
    return {
        c.name: (type(c.type).__name__, c.nullable, c.primary_key)
        for c in model.__table__.columns
    }


def index_shapes(model):
    prefix = model.__tablename__ + "."
    return {
        (
            tuple(c.name for c in i.columns),
            i.kwargs.get("postgresql_using"),
            str(i.dialect_options["postgresql"].get("where")).replace(prefix, ""),
        )
        for i in model.__table__.indexes
    }


def names(model):
    return {i.name for i in model.__table__.indexes} | {model.__table__.primary_key.name}


@pytest.mark.parametrize(
    "public, private, name",
    [
        (Classifier, ClassifierPrivate, "classifier_private"),
        (Taxonomy, TaxonomyPrivate, "taxonomy_private"),
        (Probability, ProbabilityPrivate, "probability_private"),
    ],
)
def test_private_model_mirrors_the_public_columns(public, private, name):
    assert private.__tablename__ == name
    assert columns(private) == columns(public)


def test_private_probability_is_partitioned_like_probability():
    assert ProbabilityPrivate.__n_partitions__ == Probability.__n_partitions__
    assert ProbabilityPrivate.__table__.kwargs == Probability.__table__.kwargs
    assert ProbabilityPrivate.__partition_on__(3) == Probability.__partition_on__(3)


def test_private_probability_has_the_same_indexes_under_its_own_names():
    assert index_shapes(ProbabilityPrivate) == index_shapes(Probability)
    assert names(ProbabilityPrivate).isdisjoint(names(Probability))
    assert all("private" in n for n in names(ProbabilityPrivate))


@pytest.mark.parametrize("private", [ClassifierPrivate, TaxonomyPrivate])
def test_private_lookup_tables_have_their_own_primary_key_names(private):
    assert "private" in private.__table__.primary_key.name


def test_private_tables_are_part_of_the_schema_metadata():
    assert {"classifier_private", "taxonomy_private", "probability_private"} <= set(Base.metadata.tables)
