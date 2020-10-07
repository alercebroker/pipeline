import unittest
import os
import pandas as pd
from unittest import mock
from apf.consumers import AVROFileConsumer
from apf.producers import GenericProducer

FILE_PATH = os.path.dirname(__file__)
from xmatch_step import XmatchStep, XmatchClient, Step, Object, Allwise, Xmatch

EXPECTED_XMATCH = pd.read_csv(
    os.path.join(FILE_PATH, "../examples/expected_xmatch.csv")
)


def mock_response(
    df, input_type, catalog_alias, columns, selection, output_type, radius
):
    df.rename(columns={"oid": "oid_in", "ra": "ra_in", "dec": "dec_in"}, inplace=True)
    oids = df.oid_in.values
    result = EXPECTED_XMATCH.loc[EXPECTED_XMATCH.oid_in.isin(oids)]
    return result


class PSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        DB_CONFIG = {
            "SQL": {
                "ENGINE": "postgresql",
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "PORT": 5432,
                "DB_NAME": "postgres",
            }
        }
        CONSUMER_CONFIG = {
            "DIRECTORY_PATH": os.path.join(FILE_PATH, "../examples/avro_test"),
            "NUM_MESSAGES": 10,
        }
        PRODUCER_CONFIG = {}
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
        mock_xmatch_client = mock.create_autospec(XmatchClient)
        mock_xmatch_client.execute = mock_response
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
            xmatch_client=mock_xmatch_client,
            producer=mock_producer,
            test_mode=True,
        )

    @classmethod
    def tearDownClass(self):
        self.step.driver.drop_db()
        self.step.driver.session.close()

    def setUp(self):
        self.step.driver.create_db()

    def tearDown(self):
        self.step.driver.session.close()
        self.step.driver.drop_db()


    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.assertEqual(len(self.step.driver.query(Step).all()), 1)

    def test_execute(self):
        self.step.start()
        objs = self.step.driver.session.query(Object).all()
        self.assertEqual(len(objs), len(EXPECTED_XMATCH.oid_in.unique()))
        allwise = self.step.driver.session.query(Allwise).all()
        self.assertEqual(len(allwise), len(EXPECTED_XMATCH.oid_in.unique()))
        xmatch = self.step.driver.session.query(Xmatch).all()
        self.assertEqual(len(xmatch), len(EXPECTED_XMATCH.oid_in.unique()))
