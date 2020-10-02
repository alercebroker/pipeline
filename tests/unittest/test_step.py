import unittest
from unittest import mock
from earlyclassifier.step import (
    EarlyClassifier,
    datetime,
    requests,
    SQLConnection,
    Probability,
)
import earlyclassifier


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
        self.mock_session = mock.create_autospec(earlyclassifier.step.requests.Session)
        self.step = EarlyClassifier(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            request_session=self.mock_session,
            test_mode=True,
        )

    @mock.patch("earlyclassifier.step.EarlyClassifier.insert_db")
    def test_execute(self, insert_mock):
        message = {
            "objectId": "ZTF1",
            "candidate": {
                "ndethist": 0,
                "ncovhist": 0,
                "jdstarthist": 2400000.5,
                "jdendhist": 2400000.5,
                "jd": 2400000.5,
                "ra": 0,
                "dec": 0,
            },
            "cutoutTemplate": {"stampData": b""},
            "cutoutScience": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
        }
        self.step.requests_session.post.return_value = MockResponse(
            {"status": "SUCCESS", "probabilities": {}}, 200
        )
        self.step.execute(message)
        insert_mock.assert_called_with(
            self.step.requests_session.post.return_value.json_data["probabilities"],
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

    def test_insert_db_doesnt_exist(self):
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
