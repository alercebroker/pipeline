"""Which tables the step reads the taxonomy from and writes probabilities to.

The public tables are `taxonomy` and `probability`. A classifier that is not
public yet (hunter) keeps private copies in the same schema, so DB_CONFIG can
name them: TAXONOMY_TABLE and PROBABILITY_TABLE. Rubin leaves both unset.

No test here opens a database: RecordingConnection stands in for
PSQLConnection and only records the SQL it is handed.
"""
from contextlib import contextmanager
from unittest import mock

import pytest
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql

from rubin_stamp_classifier_step.db.db import get_taxonomy_by_classifier_id, probability_table, store_probability
from rubin_stamp_classifier_step.step import StampClassifierStep
from tests.unit.stub_model import CLASSES

TAXONOMY = {name: idx for idx, name in enumerate(CLASSES)}


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


def test_taxonomy_is_read_from_the_configured_table():
    connection = RecordingConnection(rows=[{"class_id": 1, "class_name": "candidate"}])

    mapping = get_taxonomy_by_classifier_id(10, connection, table="taxonomy_hunter")

    assert "FROM taxonomy_hunter WHERE" in executed_sql(connection)
    assert mapping == {"candidate": 1}


def prediction():
    return {"oid": 123, "sid": 1, "probabilities": {"candidate": 0.9, "not_candidate": 0.1}, "midpointMjdTai": 60000.5}


def test_probabilities_are_written_to_the_public_table_by_default():
    connection = RecordingConnection()

    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()])

    assert executed_sql(connection).startswith("INSERT INTO probability (")


def test_probabilities_are_written_to_the_configured_table():
    connection = RecordingConnection()

    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()],
                      table=probability_table("probability_hunter"))

    sql = executed_sql(connection)
    assert sql.startswith("INSERT INTO probability_hunter (")
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


@pytest.mark.parametrize("db_config, table", [({}, "taxonomy"), ({"TAXONOMY_TABLE": "taxonomy_hunter"}, "taxonomy_hunter")])
def test_step_reads_the_taxonomy_from_the_configured_table(db_config, table):
    _, read_taxonomy = build_step(db_config)

    assert read_taxonomy.call_args.kwargs["table"] == table


@pytest.mark.parametrize("db_config, table", [({}, "probability"), ({"PROBABILITY_TABLE": "probability_hunter"}, "probability_hunter")])
def test_step_writes_probabilities_to_the_configured_table(db_config, table):
    step, _ = build_step(db_config)

    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:
        step.post_execute(prediction_messages())

    assert store.call_args.kwargs["table"] is step.probability_table
    assert isinstance(step.probability_table, Table)
    assert step.probability_table.name == table


def prediction_messages():
    return [{"oid": 1, "sid": 1, "probabilities": {c: 0.2 for c in CLASSES}, "midpointMjdTai": 1.0}]


def test_step_builds_the_private_table_once_not_per_insert():
    step, _ = build_step({"PROBABILITY_TABLE": "probability_hunter"})

    with mock.patch("rubin_stamp_classifier_step.db.db.probability_table") as factory:
        step.post_execute(prediction_messages())
        step.post_execute(prediction_messages())

    factory.assert_not_called()
