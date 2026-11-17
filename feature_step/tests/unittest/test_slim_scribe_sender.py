"""Scribe commands are produced in order with one flush per list (slim refactor, task 1).

Covers features.scribe_sender.send_commands and the use FeatureStep.produce_to_scribe
makes of it. Mock producer only: no Kafka, no database.
"""
import json  # encode the expected values
import unittest  # test case base class
from unittest import mock  # fake producer and survey
from unittest.mock import call  # expected producer calls

from features.step import FeatureStep  # step under test


def command(oid):
    """Return a scribe command whose payload carries `oid`."""
    return {"step": "s", "payload": {"oid": oid}}  # minimal command shape the sender reads


def expected_produce(cmd):
    """Return the produce call expected for `cmd` on topic "t"."""
    oid = cmd["payload"]["oid"]  # key source
    return call.produce(topic="t", value=json.dumps(cmd).encode("utf-8"), key=str(oid).encode("utf-8"))  # call as the producer sees it


class SendCommandsTestCase(unittest.TestCase):
    """send_commands produces every command in order and flushes once at the end."""

    def test_commands_are_produced_in_order_and_flushed_once(self):
        """Three commands give three produce calls in list order, then one flush."""
        from features.scribe_sender import send_commands  # imported here so a missing module fails this test only

        producer = mock.Mock()  # records every call
        cmds = [command(3), command(1), command(2)]  # deliberately unsorted oids
        send_commands(producer, "t", cmds)  # call under test
        self.assertEqual(producer.mock_calls, [expected_produce(c) for c in cmds] + [call.flush()])  # order kept, one flush last

    def test_empty_list_does_nothing(self):
        """An empty list neither produces nor flushes."""
        from features.scribe_sender import send_commands  # imported here so a missing module fails this test only

        producer = mock.Mock()  # records every call
        send_commands(producer, "t", [])  # call under test
        self.assertEqual(producer.mock_calls, [])  # untouched


class StepProduceTestCase(unittest.TestCase):
    """FeatureStep.produce_to_scribe sends the object updates, then the feature upserts."""

    def build(self, commands):
        """Return (step, producer) with survey.scribe_commands returning `commands`."""
        step = FeatureStep.__new__(FeatureStep)  # no __init__: no Kafka, no database
        step.survey = mock.Mock()  # survey that builds the commands
        step.survey.scribe_commands.return_value = commands  # scripted commands
        step.extractor_version = 1  # unused by the mock, read by the step
        step.extractor_group = "g"  # unused by the mock, read by the step
        step.feature_name_lut = {}  # unused by the mock, read by the step
        step.scribe_topic_name = "t"  # topic the commands go to
        step.scribe_producer = mock.Mock()  # holder of the kafka producer
        return step, step.scribe_producer.producer  # producer whose calls are asserted

    def test_step_sends_objects_then_features(self):
        """Object updates, one flush, then the feature upserts, one flush."""
        a, b, c = command(1), command(2), command(3)  # one object update, two upserts
        step, producer = self.build({"update_object": [a], "upserting_features": [b, c]})  # scripted survey
        step.produce_to_scribe([])  # call under test
        self.assertEqual(  # exact sequence on the producer
            producer.mock_calls,
            [expected_produce(a), call.flush(), expected_produce(b), expected_produce(c), call.flush()],
        )

    def test_step_without_object_updates_flushes_once(self):
        """No update_object key: only the upserts are produced, with one flush."""
        b = command(2)  # single upsert
        step, producer = self.build({"upserting_features": [b]})  # scripted survey
        step.produce_to_scribe([])  # call under test
        self.assertEqual(producer.mock_calls, [expected_produce(b), call.flush()])  # no flush for the missing list
