"""feature_version_lut rows are chosen by (package version, FEATURE_VERSION, sid, tid).

Covers the model column, get_or_create_version_id against a fake session that
records the SQL text and parameters, and the FEATURE_VERSION handling of
FeatureStep.__init__. No database and no docker.
"""
import unittest  # test case base class
from contextlib import contextmanager  # build the session context manager of the fake db
from importlib.metadata import version  # package version the step registers
from unittest import mock  # patch the lookups the step calls

from db_plugins.db.sql.models_pipeline import FeatureVersionLut  # model under test
from sqlalchemy import VARCHAR  # type the new column must have

import features.step as step_module  # module whose lookups are patched
from features.database import get_or_create_version_id  # function under test
from features.step import FeatureStep  # step under test

from .test_surveys_golden import CONSUMER_CONFIG, PRODUCER_CONFIG, SCRIBE_PRODUCER_CONFIG  # mocked Kafka configs

SELECT_SQL = "SELECT version_id FROM multisurvey.feature_version_lut WHERE version_name = :version_name AND feature_version = :feature_version AND sid = :sid AND tid = :tid"  # expected select
MAX_SQL = "SELECT COALESCE(MAX(version_id), 0) FROM multisurvey.feature_version_lut WHERE sid = :sid AND tid = :tid"  # expected max query
INSERT_SQL = "INSERT INTO multisurvey.feature_version_lut (version_id, version_name, feature_version, sid, tid) VALUES (:version_id, :version_name, :feature_version, :sid, :tid)"  # expected insert
V = ("27.8.0", "0.1.0", 1, 1)  # version_name, feature_version, sid, tid


class FakeResult:
    """Scripted query result: fetchone returns the row, or None when nothing was found."""

    def __init__(self, value, found=True):
        """Keep the value and whether the query finds a row."""
        self.value = value  # first column of the row
        self.found = found  # False makes fetchone return None

    def fetchone(self):
        """Return a one-column row, or None when the scripted query finds nothing."""
        return (self.value,) if self.found else None  # sqlalchemy rows index like tuples


def row(value, found=True):
    """Return a scripted result holding `value` in its first column."""
    return FakeResult(value, found)  # for FakeSession.execute


class FakeSession:
    """Session that records (sql text, params) and returns scripted results in order."""

    def __init__(self, results):
        """Keep the scripted results; calls and commits start empty."""
        self.results = list(results)  # popped one per execute
        self.calls = []  # (str(query), params) of every execute
        self.commits = 0  # number of commit() calls

    def execute(self, query, params=None):
        """Record the call and return the next scripted result, raising it when it is an exception."""
        self.calls.append((str(query), params))  # what the code sent
        result = self.results.pop(0)  # next scripted answer
        if isinstance(result, Exception):  # a scripted failure
            raise result  # as a driver error would
        return result  # a FakeResult or None

    def commit(self):
        """Count the commit."""
        self.commits += 1  # for the assertions


class FakeDb:
    """Connection whose session() yields one FakeSession; calls and commits are read through it."""

    def __init__(self, results):
        """Build the single session with the scripted results."""
        self.fake_session = FakeSession(results)  # shared by every session() call

    @contextmanager  # makes session() a context manager
    def session(self):
        """Yield the fake session like PSQLConnection.session does."""
        yield self.fake_session  # no real transaction

    @property  # read as an attribute, not called
    def calls(self):
        """Calls recorded by the session."""
        return self.fake_session.calls  # (sql, params) pairs

    @property  # read as an attribute, not called
    def commits(self):
        """Commits recorded by the session."""
        return self.fake_session.commits  # count


class ModelTestCase(unittest.TestCase):
    """FeatureVersionLut declares the feature_version column (criterion 1)."""

    def test_model_has_feature_version_column(self):
        """VARCHAR, not null, server default 1.0.0, primary key unchanged."""
        col = FeatureVersionLut.__table__.c.feature_version  # column under test
        self.assertIsInstance(col.type, VARCHAR)  # text column
        self.assertFalse(col.nullable)  # not null
        self.assertEqual(col.server_default.arg, "1.0.0")  # backfills old rows
        pk = [c.name for c in FeatureVersionLut.__table__.primary_key]  # key columns
        self.assertEqual(pk, ["version_id", "sid"])  # unchanged


class LookupTestCase(unittest.TestCase):
    """get_or_create_version_id selects, inserts and raises as the spec says (criteria 2, 3, 4, 7)."""

    def test_existing_combination_returns_its_id(self):
        """A present row is returned with one select and no insert."""
        db = FakeDb([row(7)])  # select finds id 7
        self.assertEqual(get_or_create_version_id(db, "multisurvey", *V), 7)  # its id
        self.assertEqual(len(db.calls), 1)  # one select only
        self.assertEqual(db.commits, 0)  # nothing inserted

    def test_select_filters_by_the_four_values(self):
        """The select compares version_name, feature_version, sid and tid."""
        db = FakeDb([row(7)])  # select finds id 7
        get_or_create_version_id(db, "multisurvey", *V)  # call under test
        sql, params = db.calls[0]  # the select
        self.assertEqual(" ".join(sql.split()), SELECT_SQL)  # full statement, whitespace collapsed: four filters joined by AND
        self.assertEqual(params, {"version_name": "27.8.0", "feature_version": "0.1.0", "sid": 1, "tid": 1})  # and bound

    def test_new_combination_inserts_next_id(self):
        """A miss inserts the four values with MAX(version_id) + 1 taken over sid and tid."""
        db = FakeDb([row(None, found=False), row(4), None])  # select misses, max is 4, insert
        self.assertEqual(get_or_create_version_id(db, "multisurvey", *V), 5)  # next id
        sql, params = db.calls[2]  # the insert
        self.assertEqual(" ".join(sql.split()), INSERT_SQL)  # full insert, values bound, none literal
        self.assertEqual(" ".join(db.calls[0][0].split()), SELECT_SQL)  # the miss came from the full select
        self.assertEqual(" ".join(db.calls[1][0].split()), MAX_SQL)  # max over sid and tid only
        self.assertEqual(params, {"version_id": 5, "version_name": "27.8.0", "feature_version": "0.1.0", "sid": 1, "tid": 1})  # four values and id
        self.assertEqual(db.calls[1][1], {"sid": 1, "tid": 1})  # max is taken over sid and tid only
        self.assertEqual(db.commits, 1)  # committed once

    def test_select_error_is_raised(self):
        """An error in the select propagates."""
        db = FakeDb([RuntimeError("column feature_version does not exist")])  # select fails
        with self.assertRaises(RuntimeError):  # no fallback to 1
            get_or_create_version_id(db, "multisurvey", *V)  # call under test

    def test_insert_error_is_raised(self):
        """An error in the insert propagates."""
        db = FakeDb([row(None, found=False), row(4), RuntimeError("duplicate key")])  # insert fails
        with self.assertRaises(RuntimeError):  # no fallback to 1
            get_or_create_version_id(db, "multisurvey", *V)  # call under test

    def test_no_connection_returns_none(self):
        """Without a connection the function returns None."""
        self.assertIsNone(get_or_create_version_id(None, "multisurvey", *V))  # unchanged behaviour


def config_with(**extra):
    """Return the mocked step config with SURVEY lsst and `extra` keys added."""
    return {  # three mocked Kafka configs of test_surveys_golden
        "PRODUCER_CONFIG": PRODUCER_CONFIG,  # mocked producer
        "CONSUMER_CONFIG": CONSUMER_CONFIG,  # mocked consumer
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,  # mocked scribe producer
        "SURVEY": "lsst",  # sid 1, tid 1
        **extra,  # FEATURE_VERSION or nothing
    }  # end of the config


class StepTestCase(unittest.TestCase):
    """FeatureStep passes FEATURE_VERSION to the lookup and rejects a missing one (criteria 5, 6)."""

    def build(self, **extra):
        """Build a step with both lookups patched; return (step, version_id mock, db_sql)."""
        with mock.patch.object(step_module, "get_or_create_version_id") as version_id:  # no version lookup in the db
            with mock.patch.object(step_module, "get_feature_name_lut"):  # no LUT read from the db
                db_sql = mock.MagicMock()  # connection handed to the lookups
                step = FeatureStep(config=config_with(**extra), db_sql=db_sql)  # step under test
        return step, version_id, db_sql  # for the assertions

    def assert_rejected(self, **extra):
        """Assert construction raises ValueError naming FEATURE_VERSION before any lookup or DB call."""
        db_sql = mock.MagicMock()  # connection that must stay untouched
        with mock.patch.object(step_module, "get_or_create_version_id") as version_id:  # must stay unused
            with self.assertRaises(ValueError) as raised:  # the check under test
                FeatureStep(config=config_with(**extra), db_sql=db_sql)  # bad config
        self.assertIn("FEATURE_VERSION", str(raised.exception))  # names the key
        version_id.assert_not_called()  # before any lookup
        self.assertEqual(db_sql.mock_calls, [])  # before any DB call

    def test_step_passes_config_version(self):
        """The lookup gets the package version, the config version, sid and tid."""
        step, version_id, db_sql = self.build(FEATURE_VERSION="0.1.0")  # valid config
        version_id.assert_called_once_with(db_sql, "multisurvey", version("feature-step"), "0.1.0", 1, 1, step.logger)  # five values plus logger
        self.assertEqual(step.feature_version, "0.1.0")  # kept on the step

    def test_missing_version_raises(self):
        """No FEATURE_VERSION key raises."""
        self.assert_rejected()  # key absent

    def test_empty_version_raises(self):
        """An empty FEATURE_VERSION raises."""
        self.assert_rejected(FEATURE_VERSION="")  # empty string

    def test_blank_version_raises(self):
        """A whitespace-only FEATURE_VERSION raises."""
        self.assert_rejected(FEATURE_VERSION="   ")  # blanks only

    def test_version_is_stripped(self):
        """Surrounding blanks are removed before the lookup."""
        _, version_id, _ = self.build(FEATURE_VERSION=" 0.1.0 ")  # padded value
        self.assertEqual(version_id.call_args[0][3], "0.1.0")  # fourth argument is stripped

    def test_non_string_version_raises(self):
        """A float, int, bool or list raises instead of being converted (yaml reads an unquoted 1.10 as 1.1)."""
        for value in (1.1, 0, False, ["0.1.0"]):  # values an unquoted yaml scalar or a list can give
            with self.subTest(value=value):  # report each value on its own
                self.assert_rejected(FEATURE_VERSION=value)  # ValueError naming the key, no calls
