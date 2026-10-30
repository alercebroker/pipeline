"""The step reads the taxonomy from and writes probabilities to the public tables only.

No test here opens a database: RecordingConnection stands in for
PSQLConnection and only records the SQL it is handed.
"""  # module docstring: what the file checks
import inspect  # used to inspect the signatures of the db helpers
from contextlib import contextmanager  # builds the fake session context manager
from unittest import mock  # patches the DB connection and canned results

from sqlalchemy.dialects import postgresql  # compiles statements to SQL text

from rubin_stamp_classifier_step.db.db import get_taxonomy_by_classifier_id, store_probability  # helpers under test
from rubin_stamp_classifier_step.step import StampClassifierStep  # step under test
from tests.unit.stub_model import CLASSES  # class names of the stub model

TAXONOMY = {name: idx for idx, name in enumerate(CLASSES)}  # taxonomy the patched DB read returns
LEFTOVER = {"TAXONOMY_CLASS": "nowhere.Bogus", "PROBABILITY_CLASS": "nowhere.Bogus"}  # old keys that must be ignored


class RecordingConnection:  # fake PSQLConnection
    """Fake PSQLConnection: records what is executed, answers canned rows."""

    def __init__(self, rows=(), written=()):  # rows: what a SELECT returns; written: what RETURNING oid gives
        self.rows = rows  # canned rows
        self.written = written  # canned RETURNING oids
        self.executed = []  # (statement, params) pairs seen

    @contextmanager  # lets the fake be used in a with block
    def session(self):  # mimics PSQLConnection.session
        """Yield the connection itself as the session."""
        yield self  # the fake is its own session

    def execute(self, statement, params=None):  # mimics Session.execute
        """Record the statement and return a result with the canned rows."""
        self.executed.append((statement, params))  # keep the call for the asserts
        result = mock.Mock()  # stand-in for the SQLAlchemy result
        result.mappings.return_value.all.return_value = list(self.rows)  # canned rows
        result.scalars.return_value.all.return_value = list(self.written)  # oids a RETURNING would give back
        return result  # hand it back

    def commit(self):  # mimics Session.commit
        """Do nothing: there is no real transaction."""


def executed_sql(connection):  # compiles the first recorded statement
    """Return the first recorded statement compiled to one-line PostgreSQL text."""
    statement, _ = connection.executed[0]  # first statement executed
    return " ".join(str(statement.compile(dialect=postgresql.dialect())).split())  # collapse whitespace


def prediction():  # one prediction message
    """Return one prediction message with two class probabilities."""
    return {"oid": 123, "sid": 1, "probabilities": {"candidate": 0.9, "not_candidate": 0.1}, "midpointMjdTai": 60000.5}  # sample


def prediction_messages():  # messages for post_execute
    """Return one prediction message per object, with the stub classes."""
    return [{"oid": 1, "sid": 1, "probabilities": {c: 0.2 for c in CLASSES}, "midpointMjdTai": 1.0}]  # sample


def build_step(db_config):  # builds the step with the DB patched out
    """Build a StampClassifierStep with DB_CONFIG=db_config and no real database.

    Returns the step and the mock that replaced get_taxonomy_by_classifier_id.
    """
    config = {  # minimal step config
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},  # no real consumer
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},  # no real producer
        "DB_CONFIG": db_config,  # the part under test
        "MODEL_CONFIG": {"CLASS": "tests.unit.stub_model.StubModel", "CLS_ID": 10},  # stub model
    }
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id",
                    return_value=TAXONOMY) as read_taxonomy:  # patch DB access
        step = StampClassifierStep(config=config)  # construct inside the patches
    return step, read_taxonomy  # step and the read mock


def test_taxonomy_is_read_from_the_public_table():  # taxonomy read
    """The taxonomy query targets the public taxonomy table."""
    connection = RecordingConnection(rows=[{"class_id": 0, "class_name": "candidate"}])  # canned row
    mapping = get_taxonomy_by_classifier_id(10, connection)  # read the taxonomy
    assert "FROM taxonomy WHERE" in executed_sql(connection)  # public table in the SQL
    assert mapping == {"candidate": 0}  # mapping built from the row


def test_probabilities_are_written_to_the_public_table():  # probability write
    """The probability insert targets the public probability table, ranked by value."""
    connection = RecordingConnection()  # records the insert
    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()])  # write
    sql = executed_sql(connection)  # compiled insert
    assert sql.startswith("INSERT INTO probability (")  # public table
    assert "ON CONFLICT ON CONSTRAINT pk_probability_oid_classifierid_classid DO UPDATE" in sql  # multisurvey upsert
    assert "WHERE probability.lastmjd >= excluded.lastmjd" in sql  # an older, or the same, alert overwrites
    assert sql.endswith("RETURNING probability.oid")  # the written oids come back
    _, rows = connection.executed[0]  # rows handed to the insert
    assert [(r["class_id"], r["ranking"]) for r in rows] == [(1, 1), (0, 2)]  # ranked by probability


def test_store_probability_returns_the_written_oids():  # RETURNING contract
    """store_probability returns the set of oids the upsert reported back, deduplicated across classes."""
    connection = RecordingConnection(written=[123, 123])  # one oid per written (oid, class) row
    written = store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()])  # write
    assert written == {123}  # collapsed to oids


def test_store_probability_keeps_the_oldest_alert_per_oid():  # batch dedup
    """Two alerts of one object in a batch write one row set, the older alert's."""
    connection = RecordingConnection()  # records the insert
    older, newer = prediction(), prediction()  # same oid
    older["midpointMjdTai"], newer["midpointMjdTai"] = 60000.1, 60000.9  # older first
    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [newer, older])  # newer first on purpose
    _, rows = connection.executed[0]  # rows handed to the insert
    assert {r["lastmjd"] for r in rows} == {60000.1}  # only the older alert's rows
    assert len(rows) == 2  # one per class, not four


def test_step_ignores_leftover_class_keys():  # old config keys
    """Old TAXONOMY_CLASS / PROBABILITY_CLASS keys are not read and no model is passed."""
    step, read_taxonomy = build_step(LEFTOVER)  # must not raise: the keys are not read
    assert "model" not in read_taxonomy.call_args.kwargs  # taxonomy read takes no model
    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:  # capture the write
        step.post_execute(prediction_messages())  # run the write path
    assert "model" not in store.call_args.kwargs  # probability write takes no model


def test_db_helpers_take_no_model_argument():  # signatures
    """Neither db helper has a model parameter."""
    assert "model" not in inspect.signature(get_taxonomy_by_classifier_id).parameters  # taxonomy helper
    assert "model" not in inspect.signature(store_probability).parameters  # probability helper
