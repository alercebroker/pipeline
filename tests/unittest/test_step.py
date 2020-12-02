import unittest
import os
import pandas as pd
from unittest import mock
from apf.consumers import AVROFileConsumer
from apf.producers import GenericProducer

FILE_PATH = os.path.dirname(__file__)
from xmatch_step import XmatchStep, XmatchClient, SQLConnection

EXPECTED_XMATCH = pd.read_csv(
    os.path.join(FILE_PATH, "../examples/expected_xmatch.csv")
)


def mock_response(
    df, input_type, catalog_alias, columns, selection, output_type, radius
):
    df.rename(columns={"oid": "oid_in", "ra": "ra_in", "dec": "dec_in"}, inplace=True)
    return EXPECTED_XMATCH


class StepTest(unittest.TestCase):
    def setUp(self):
        DB_CONFIG = {"SQL": {}}
        CONSUMER_CONFIG = {
            "DIRECTORY_PATH": os.path.join(FILE_PATH, "../examples/avro_test"),
            "NUM_MESSAGES": 5,
        }
        PRODUCER_CONFIG = {"CLASS": "apf.producers.GenericProducer"}
        XMATCH_CONFIG = {
            "CATALOG": {
                "name": "allwise",
                "columns": [
                    "AllWISE",
                    "RAJ2000",
                    "DEJ2000",
                    "W1mag",
                    "W2mag",
                    "W3mag",
                    "W4mag",
                    "e_W1mag",
                    "e_W2mag",
                    "e_W3mag",
                    "e_W4mag",
                    "Jmag",
                    "e_Jmag",
                    "Hmag",
                    "e_Hmag",
                    "Kmag",
                    "e_Kmag",
                ],
            }
        }
        mock_db_connection = mock.create_autospec(SQLConnection)
        mock_xmatch_client = mock.create_autospec(XmatchClient)
        mock_producer = mock.create_autospec(GenericProducer)
        self.step = XmatchStep(
            consumer=AVROFileConsumer(CONSUMER_CONFIG),
            config={
                "DB_CONFIG": DB_CONFIG,
                "PRODUCER_CONFIG": PRODUCER_CONFIG,
                "METRICS_CONFIG": {},
                "STEP_METADATA": {
                    "STEP_VERSION": "xmatch",
                    "STEP_ID": "xmatch",
                    "STEP_NAME": "xmatch",
                    "STEP_COMMENTS": "xmatch",
                },
                "XMATCH_CONFIG": XMATCH_CONFIG,
            },
            db_connection=mock_db_connection,
            xmatch_client=mock_xmatch_client,
            producer=mock_producer,
            test_mode=True,
        )

    def tearDown(self):
        del self.step

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.step.driver.query().get_or_create.assert_called_once()

    def test_save_xmatch(self):
        df_object = pd.DataFrame({"oid": ["ZTF"]})
        self.step.save_xmatch(EXPECTED_XMATCH, df_object)
        calls = self.step.driver.query().bulk_insert.mock_calls
        self.assertEqual(len(calls), 3)

    @mock.patch("xmatch_step.XmatchStep.save_xmatch")
    def test_execute(self, save_xmatch_mock):
        self.step.xmatch_client.execute = mock_response
        self.step.start()
        save_xmatch_mock.assert_called()
        self.step.producer.produce.assert_called()

    def test_convert_null_to_none(self):
        message = {"candidate": {"drb": "null"}}
        expected = {"candidate": {"drb": None}}
        self.step.convert_null_to_none(["drb"], message["candidate"])
        self.assertEqual(message, expected)
