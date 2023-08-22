import unittest
import pandas as pd
from unittest import mock
from earlyclassifier.step import (
    EarlyClassifier,
    requests,
    SQLConnection,
    Probability,
    StampClassifier,
    KafkaProducer,
    FULL_ASTEROID_PROBABILITY,
)


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


class EarlyClassifierTest(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {"SQL": {}},
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
                "CLASSIFIER_VERSION": "test",
                "CLASSIFIER_NAME": "stamp_test",
            },
            "API_URL": "",
            "N_RETRY": 5,
        }
        self.mock_database_connection = mock.create_autospec(SQLConnection)
        self.mock_session = mock.create_autospec(requests.Session)
        self.mock_stamp_classifier = mock.create_autospec(StampClassifier)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.step = EarlyClassifier(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            request_session=self.mock_session,
            stamp_classifier=self.mock_stamp_classifier,
            producer=self.mock_producer,
            test_mode=True,
        )

    @mock.patch("earlyclassifier.step.EarlyClassifier.insert_db")
    def test_execute(self, insert_mock: unittest.mock.Mock):
        message = {
            "objectId": "ZTF1",
            "candidate": {
                "candid": 1,
                "ndethist": 0,
                "ncovhist": 0,
                "jdstarthist": 2400000.5,
                "jdendhist": 2400000.5,
                "jd": 2400000.5,
                "ra": 0,
                "dec": 0,
                "ssdistnr": -999.0,
                "sgscore1": 0.0,
                "distpsnr1": 1,
                "isdiffpos": 1,
            },
            "cutoutTemplate": {"stampData": b""},
            "cutoutScience": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
        }
        self.step.model.execute.return_value = pd.DataFrame(
            {"SN": [1, 2], "asteroid": [3, 4]}
        )
        self.step.execute(message)
        insert_mock.assert_called_with(
            {"SN": 1, "asteroid": 3},
            message["objectId"],
            {
                "ndethist": 0,
                "ncovhist": 0,
                "mjdstarthist": 0.0,
                "mjdendhist": 0.0,
                "firstmjd": 0.0,
                "lastmjd": 0.0,
                "ndet": 1,
                "deltajd": 0,
                "meanra": 0.0,
                "meandec": 0.0,
                "step_id_corr": "0.0.0",
                "corrected": False,
                "stellar": False,
            },
        )
        self.mock_producer.produce.assert_called_with(
            {
                "objectId": "ZTF1",
                "candid": 1,
                "probabilities": {"SN": 1, "asteroid": 3},
            },
            key="ZTF1",
        )

    def test_insert_db_doesnt_exist(self):
        probabilities = {
            "VS": 5,
            "asteroid": 4,
            "bogus": 3,
            "SN": 2,
            "AGN": 1,
        }
        probabilities_with_ranking = {
            "AGN": {"probability": 1, "ranking": 5},
            "SN": {"probability": 2, "ranking": 4},
            "bogus": {"probability": 3, "ranking": 3},
            "asteroid": {"probability": 4, "ranking": 2},
            "VS": {"probability": 5, "ranking": 1},
        }
        oid = "test"
        object_data = {}
        mock_probability = mock.create_autospec(Probability)
        self.mock_database_connection.query().get_or_create.return_value = (
            mock_probability,
            True,
        )
        self.step.insert_db(probabilities, oid, object_data)
        calls = [
            mock.call(filter_by={"oid": oid, **{}}),
        ]
        for prob in probabilities:
            calls.append(
                mock.call(
                    filter_by={
                        "oid": oid,
                        "class_name": prob,
                        "classifier_name": self.step_config["STEP_METADATA"][
                            "CLASSIFIER_NAME"
                        ],
                        "classifier_version": self.step_config["STEP_METADATA"][
                            "CLASSIFIER_VERSION"
                        ],
                    },
                    probability=probabilities_with_ranking[prob]["probability"],
                    ranking=probabilities_with_ranking[prob]["ranking"],
                )
            )
        self.mock_database_connection.query().get_or_create.assert_has_calls(
            calls, any_order=False
        )

    def test_insert_db_already_exist(self):
        probabilities = {
            "AGN": 1,
            "SN": 2,
            "bogus": 3,
            "asteroid": 4,
            "VS": 5,
        }
        probabilities_with_ranking = {
            "AGN": {"probability": 1, "ranking": 5},
            "SN": {"probability": 2, "ranking": 4},
            "bogus": {"probability": 3, "ranking": 3},
            "asteroid": {"probability": 4, "ranking": 2},
            "VS": {"probability": 5, "ranking": 1},
        }
        oid = "test"
        object_data = {}
        mock_probability = mock.create_autospec(Probability)
        self.mock_database_connection.query().get_or_create.return_value = (
            mock_probability,
            False,
        )
        self.step.insert_db(probabilities, oid, object_data)
        calls = [
            mock.call(filter_by={"oid": oid, **{}}),
        ]
        for prob in probabilities:
            calls.append(
                mock.call(
                    filter_by={
                        "oid": oid,
                        "class_name": prob,
                        "classifier_name": self.step_config["STEP_METADATA"][
                            "CLASSIFIER_NAME"
                        ],
                        "classifier_version": self.step_config["STEP_METADATA"][
                            "CLASSIFIER_VERSION"
                        ],
                    },
                    probability=probabilities_with_ranking[prob]["probability"],
                    ranking=probabilities_with_ranking[prob]["ranking"],
                )
            )
        test_pass = False
        for call in calls:
            if (
                call
                not in self.mock_database_connection.query().get_or_create.mock_calls
            ):
                test_pass = True
        self.assertTrue(test_pass)

    @mock.patch("earlyclassifier.step.EarlyClassifier.insert_db")
    @mock.patch("earlyclassifier.step.EarlyClassifier.produce")
    def test_asteroid_inference(
        self, producer_mock: unittest.mock.Mock, insert_db: unittest.mock.Mock
    ):
        message = {
            "objectId": "ZTFtest",
            "candidate": {
                "candid": 1,
                "ndethist": 0,
                "ncovhist": 0,
                "jdstarthist": 2400000.5,
                "jdendhist": 2400000.5,
                "jd": 2400000.5,
                "ra": 0,
                "dec": 0,
                "ssdistnr": 1,
            },
            "cutoutTemplate": {"stampData": b""},
            "cutoutScience": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
        }
        self.step.execute(message)
        insert_db.assert_called_with(
            FULL_ASTEROID_PROBABILITY,
            message["objectId"],
            {
                "ndethist": 0,
                "ncovhist": 0,
                "mjdstarthist": 0.0,
                "mjdendhist": 0.0,
                "firstmjd": 0.0,
                "lastmjd": 0.0,
                "ndet": 1,
                "deltajd": 0,
                "meanra": 0.0,
                "meandec": 0.0,
                "step_id_corr": "0.0.0",
                "corrected": False,
                "stellar": False,
            },
        )

    def test_sn_must_be_saved(self):
        message = {
            "objectId": "test",
            "candidate": {"isdiffpos": 1, "sgscore1": 0.6, "distpsnr1": 1},
        }
        # If object has the necessary conditions must be saved
        probabilities = {"SN": 1, "asteroid": 0, "bogus": 0}
        response = self.step.sn_must_be_saved(message, probabilities)
        self.assertTrue(response)

        # If object has not the necessary conditions must not be saved
        message["candidate"]["isdiffpos"] = 0
        response = self.step.sn_must_be_saved(message, probabilities)
        self.assertTrue(not response)

        # If object has not the necessary conditions must not be saved
        message["candidate"]["isdiffpos"] = 1
        message["candidate"]["distpsnr1"] = 0.4
        message["candidate"]["sgscore1"] = 0.6
        response = self.step.sn_must_be_saved(message, probabilities)
        self.assertTrue(not response)

        # If object has the necessary conditions must be saved
        message["candidate"]["distpsnr1"] = 0.1
        message["candidate"]["sgscore1"] = 0.5
        response = self.step.sn_must_be_saved(message, probabilities)
        self.assertTrue(response)

        # If object is not a SN must be saved
        probabilities = {"SN": 0, "asteroid": 1, "bogus": 0}
        response = self.step.sn_must_be_saved(message, probabilities)
        self.assertTrue(response)

    def test_produce(self):
        self.step.produce("ZTF1", 1, {})
        self.mock_producer.produce.assert_called()


class EarlyClassifierWithoutProducerTest(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {"SQL": {}},
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
                "CLASSIFIER_VERSION": "test",
                "CLASSIFIER_NAME": "stamp_test",
            },
            "API_URL": "",
            "N_RETRY": 5,
        }
        self.mock_database_connection = mock.create_autospec(SQLConnection)
        self.mock_session = mock.create_autospec(requests.Session)
        self.mock_stamp_classifier = mock.create_autospec(StampClassifier)
        self.step = EarlyClassifier(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            request_session=self.mock_session,
            stamp_classifier=self.mock_stamp_classifier,
            test_mode=True,
        )

    @mock.patch("earlyclassifier.step.EarlyClassifier.insert_db")
    @mock.patch("earlyclassifier.step.EarlyClassifier.produce")
    def test_execute(
        self, producer_mock: unittest.mock.Mock, insert_mock: unittest.mock.Mock
    ):
        message = {
            "objectId": "ZTF1",
            "candidate": {
                "candid": 1,
                "ndethist": 0,
                "ncovhist": 0,
                "jdstarthist": 2400000.5,
                "jdendhist": 2400000.5,
                "jd": 2400000.5,
                "ra": 0,
                "dec": 0,
                "ssdistnr": -999.0,
                "sgscore1": 0.0,
                "distpsnr1": 1,
                "isdiffpos": 1,
            },
            "cutoutTemplate": {"stampData": b""},
            "cutoutScience": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
        }
        self.step.model.execute.return_value = pd.DataFrame(
            {"SN": [1, 2], "asteroid": [3, 4]}
        )
        self.step.execute(message)
        producer_mock.assert_not_called()


