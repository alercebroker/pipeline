"""Which models the step reads the taxonomy from and writes probabilities to.

The public models are Taxonomy and Probability. A classifier that is not
public yet (hunter) writes to the private copies, so DB_CONFIG can name the
model classes: TAXONOMY_CLASS and PROBABILITY_CLASS. Rubin leaves both unset.

No test here opens a database: RecordingConnection stands in for
PSQLConnection and only records the SQL it is handed.
"""
from contextlib import contextmanager
from unittest import mock

import pytest
from db_plugins.db.sql.models_pipeline import Probability, ProbabilityPrivate, Taxonomy, TaxonomyPrivate
from sqlalchemy.dialects import postgresql

from rubin_stamp_classifier_step.db.db import get_taxonomy_by_classifier_id, store_probability
from rubin_stamp_classifier_step.step import StampClassifierStep
from tests.unit.stub_model import CLASSES

TAXONOMY = {name: idx for idx, name in enumerate(CLASSES)}
PRIVATE = {
    "TAXONOMY_CLASS": "db_plugins.db.sql.models_pipeline.TaxonomyPrivate",
    "PROBABILITY_CLASS": "db_plugins.db.sql.models_pipeline.ProbabilityPrivate",
}


class RecordingConnection:
    """Fake PSQLConnection: records what is executed, answers canned rows."""

    def __init__(self, rows=()):
        self.rows = rows
        self.executed = []

    @contextmanager
    def session(self):
        yield self

    def execute(self, statement, params=None):
        self.executed.append((statement, params))
        result = mock.Mock()
        result.mappings.return_value.all.return_value = list(self.rows)
        return result

    def commit(self):
        pass


def executed_sql(connection):
    statement, _ = connection.executed[0]
    return " ".join(str(statement.compile(dialect=postgresql.dialect())).split())


# --- db helpers ------------------------------------------------------------

def test_taxonomy_is_read_from_the_public_table_by_default():
    connection = RecordingConnection(rows=[{"class_id": 0, "class_name": "candidate"}])

    mapping = get_taxonomy_by_classifier_id(10, connection)

    assert "FROM taxonomy WHERE" in executed_sql(connection)
    assert mapping == {"candidate": 0}


def test_taxonomy_is_read_from_the_given_model():
    connection = RecordingConnection(rows=[{"class_id": 1, "class_name": "candidate"}])

    mapping = get_taxonomy_by_classifier_id(10, connection, model=TaxonomyPrivate)

    assert "FROM taxonomy_private WHERE" in executed_sql(connection)
    assert mapping == {"candidate": 1}


def prediction():
    return {"oid": 123, "sid": 1, "probabilities": {"candidate": 0.9, "not_candidate": 0.1}, "midpointMjdTai": 60000.5}


def test_probabilities_are_written_to_the_public_table_by_default():
    connection = RecordingConnection()

    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()])

    assert executed_sql(connection).startswith("INSERT INTO probability (")


def test_probabilities_are_written_to_the_given_model():
    connection = RecordingConnection()

    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()],
                      model=ProbabilityPrivate)

    sql = executed_sql(connection)
    assert sql.startswith("INSERT INTO probability_private (")
    assert "ON CONFLICT DO NOTHING" in sql
    _, rows = connection.executed[0]
    assert [(r["class_id"], r["ranking"]) for r in rows] == [(1, 1), (0, 2)]


# --- step wiring -------------------------------------------------------------

def build_step(db_config):
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
        "DB_CONFIG": db_config,
        "MODEL_CONFIG": {"CLASS": "tests.unit.stub_model.StubModel", "CLS_ID": 10},
    }
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id",
                    return_value=TAXONOMY) as read_taxonomy:
        step = StampClassifierStep(config=config)
    return step, read_taxonomy


@pytest.mark.parametrize("db_config, model", [({}, Taxonomy), (PRIVATE, TaxonomyPrivate)])
def test_step_reads_the_taxonomy_from_the_configured_model(db_config, model):
    _, read_taxonomy = build_step(db_config)

    assert read_taxonomy.call_args.kwargs["model"] is model


def prediction_messages():
    return [{"oid": 1, "sid": 1, "probabilities": {c: 0.2 for c in CLASSES}, "midpointMjdTai": 1.0}]


@pytest.mark.parametrize("db_config, model", [({}, Probability), (PRIVATE, ProbabilityPrivate)])
def test_step_writes_probabilities_to_the_configured_model(db_config, model):
    step, _ = build_step(db_config)

    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:
        step.post_execute(prediction_messages())

    assert store.call_args.kwargs["model"] is model
