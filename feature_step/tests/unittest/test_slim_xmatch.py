"""Xmatch query, attach and scribe commands live outside the step (slim refactor, task 2).

Covers features.xmatch.XmatchEnricher and the use FeatureStep.pre_execute makes of it.
Mock client and producer only: no network, no Kafka, no database.
"""
import unittest  # test case base class
from unittest import mock  # fake client, survey and producer

from features.step import FeatureStep  # step under test


def msg(oid, sid=0):
    """Return a minimal message with the fields the xmatch reads."""
    return {"oid": oid, "sid": sid, "meanra": oid * 0.5, "meandec": oid * -0.5}  # coordinates derived from the oid


def match(oid, catalog="allwise", **extra):
    """Return a service match for `oid` (an oid string, as the service returns it)."""
    return {"oid": str(oid), "catalog": catalog, "distance": 1.5, "match_id": f"{catalog}-{oid}", **extra}  # service shape


class QueryTestCase(unittest.TestCase):
    """XmatchEnricher.query asks the service once per catalog."""

    def test_query_asks_once_per_catalog_and_skips_sid_2(self):
        """Sids 0, 2, 0 and two catalogs: two calls with the two non-2 messages, results concatenated."""
        from features.xmatch import XmatchEnricher  # imported here so a missing module fails this test only

        client = mock.Mock()  # fake service client
        client.conesearch_with_metadata.side_effect = [[match(1)], [match(1, "gaia")]]  # one answer per catalog
        enricher = XmatchEnricher(client, ["allwise", "gaia"], "ztf")  # object under test
        results = enricher.query([msg(1), msg(2, sid=2), msg(3)])  # call under test
        expected = [  # one call per catalog, same arguments
            mock.call(ras=[0.5, 1.5], decs=[-0.5, -1.5], oids=["1", "3"], catalog=c)  # sid 2 left out
            for c in ("allwise", "gaia")  # catalog order of the config
        ]  # end of the expected calls
        self.assertEqual(client.conesearch_with_metadata.mock_calls, expected)  # exactly these requests
        self.assertEqual(results, [match(1), match(1, "gaia")])  # concatenation in catalog order


class AttachTestCase(unittest.TestCase):
    """XmatchEnricher.attach sets msg["xmatches"] from the allwise matches."""

    def build(self):
        """Return an enricher whose client is never called."""
        from features.xmatch import XmatchEnricher  # imported here so a missing module fails this test only

        return XmatchEnricher(mock.Mock(), ["allwise", "gaia"], "ztf")  # attach does not use the client

    def test_attach_sets_allwise_match_without_scalar_metadata(self):
        """The allwise match loses its scalar metadata on a copy; gaia is not attached; no match gives None."""
        allwise = match(1, metadata={"id": "x", "w1mpro": {"a": 1}})  # scalar and map-valued entries
        gaia = match(2, "gaia")  # other catalog
        messages = [msg(1), msg(2), msg(3)]  # matched, gaia only, unmatched
        self.build().attach(messages, [allwise, gaia])  # call under test
        self.assertEqual(messages[0]["xmatches"], {**allwise, "metadata": {"w1mpro": {"a": 1}}})  # scalars dropped
        self.assertEqual(allwise["metadata"], {"id": "x", "w1mpro": {"a": 1}})  # original untouched
        self.assertIsNone(messages[1]["xmatches"])  # gaia is not attached
        self.assertIsNone(messages[2]["xmatches"])  # no match at all

    def test_attach_with_no_results_sets_none_everywhere(self):
        """An empty result list gives xmatches None on every message."""
        messages = [msg(1), msg(2)]  # two messages
        self.build().attach(messages, [])  # call under test
        self.assertEqual([m["xmatches"] for m in messages], [None, None])  # explicit None


class CommandsTestCase(unittest.TestCase):
    """XmatchEnricher.commands builds one scribe command per match."""

    def build(self):
        """Return an enricher named "ztf" whose client is never called."""
        from features.xmatch import XmatchEnricher  # imported here so a missing module fails this test only

        return XmatchEnricher(mock.Mock(), ["allwise", "gaia"], "ztf")  # commands does not use the client

    def test_commands_shape(self):
        """One command per match, in order, with the payload the scribe reads."""
        results = [match(1), match(2, "gaia")]  # full list, all catalogs
        cmds = self.build().commands(results, [msg(1, sid=0), msg(2, sid=3)])  # call under test
        self.assertEqual(cmds, [  # exact commands
            {"step": "xmatch", "survey": "ztf", "payload": {"oid": 1, "sid": 0, "catalog": "allwise", "dist": 1.5, "oid_catalog": "allwise-1"}},  # first match
            {"step": "xmatch", "survey": "ztf", "payload": {"oid": 2, "sid": 3, "catalog": "gaia", "dist": 1.5, "oid_catalog": "gaia-2"}},  # second match
        ])  # end of the expected commands

    def test_commands_defaults(self):
        """No catalog gives allwise, an oid outside the batch gives sid 1, repeated oids take the last sid."""
        no_catalog = {"oid": "1", "distance": 0.5, "match_id": "m"}  # match without catalog
        cmds = self.build().commands([no_catalog, match(9), match(2)], [msg(1, sid=0), msg(2, sid=0), msg(2, sid=3)])  # call under test
        self.assertEqual(cmds[0]["payload"]["catalog"], "allwise")  # default catalog
        self.assertEqual(cmds[1]["payload"]["sid"], 1)  # oid 9 is not in the batch
        self.assertEqual(cmds[2]["payload"]["sid"], 3)  # last message of oid 2 wins

    def test_commands_empty_results_is_empty_list(self):
        """No results give an empty list."""
        self.assertEqual(self.build().commands([], [msg(1)]), [])  # nothing to send


class StepXmatchTestCase(unittest.TestCase):
    """FeatureStep.pre_execute drives the enricher."""

    def build(self, results):
        """Return a step with the xmatch on, a client answering `results` for every catalog, and a mock scribe producer."""
        from features.xmatch import XmatchEnricher  # imported here so a missing module fails this test only

        step = FeatureStep.__new__(FeatureStep)  # no __init__: no Kafka, no database
        step.use_xmatch = True  # xmatch on
        step.survey_name = "ztf"  # name written in the commands
        step.xmatch_catalogs = ["allwise"]  # one request
        step.scribe_topic_name = "t"  # scribe topic
        step.scribe_producer = mock.MagicMock()  # records producer calls
        step.survey = mock.Mock()  # survey whose epochs are not needed
        step.survey.epochs_from_message.return_value = [{}]  # one epoch per message
        step.min_detections_features = 1  # keep every message
        step.xmatch_client = mock.Mock()  # client the step holds
        step.xmatch_client.conesearch_with_metadata.return_value = results  # scripted answer
        step.xmatch = XmatchEnricher(step.xmatch_client, step.xmatch_catalogs, "ztf")  # as __init__ builds it
        step.xmatch_client = mock.Mock()  # replaced after construction, as committed tests do
        step.xmatch_client.conesearch_with_metadata.return_value = results  # scripted answer on the new client
        return step  # ready for pre_execute

    def test_step_uses_the_client_it_currently_holds(self):
        """A client set on the step after construction is the one queried, and the match is attached and sent."""
        step = self.build([match(1)])  # one allwise match
        out = step.pre_execute([msg(1)])  # call under test
        step.xmatch_client.conesearch_with_metadata.assert_called_once()  # the replacement client was queried
        self.assertEqual(out[0]["xmatches"]["catalog"], "allwise")  # attached
        self.assertEqual(step.scribe_producer.producer.produce.call_count, 1)  # one command sent
        step.scribe_producer.producer.flush.assert_called_once_with()  # one flush

    def test_no_results_leaves_the_scribe_producer_untouched(self):
        """Without matches the producer attribute of the scribe producer is never read."""
        step = self.build([])  # service answers nothing
        step.scribe_producer = mock.Mock(spec=[])  # has no `producer` attribute: reading it raises AttributeError
        out = step.pre_execute([msg(1)])  # call under test, must not raise
        self.assertIsNone(out[0]["xmatches"])  # no match attached
