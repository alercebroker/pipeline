import unittest
import pytest
import pandas as pd
from unittest import mock

from apf.producers import KafkaProducer
from ingestion_step.utils.multi_driver.connection import MultiDriverConnection
from ingestion_step.step import IngestionStep

from data.messages import (
    generate_message_atlas,
    generate_message_ztf,
)


DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    },
    "MONGO": {
        "HOST": "localhost",
        "USER": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
    },
}


class StepTestCase(unittest.TestCase):
    def setUp(self) -> None:
        step_config = {
            "DB_CONFIG": DB_CONFIG,
            "STEP_METADATA": {
                "STEP_ID": "ingestion",
                "STEP_NAME": "ingestion",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "test version",
            },
        }
        mock_database_connection = mock.create_autospec(MultiDriverConnection)
        mock_database_connection.connect.return_value = None
        mock_producer = mock.create_autospec(KafkaProducer)
        self.step = IngestionStep(
            config=step_config,
            db_connection=mock_database_connection,
            producer=mock_producer,
        )

    def tearDown(self) -> None:
        del self.step

    # We need define function for each method in the class?
    def test_get_objects(self):
        oids = ["ZTF1", "ZTF2"]
        self.step.get_objects(oids)
        self.step.driver.query("Object", engine="mongo").find_all.assert_called_with(
            filter_by={"aid": {"$in": oids}}, paginate=False
        )

    def test_get_detections(self):
        oids = [12345, 45678]
        self.step.get_detections(oids, engine="mongo")
        self.step.driver.query("Detection", engine="mongo").find_all.assert_called_with(
            filter_by={"aid": {"$in": oids}}, paginate=False
        )

    def test_get_non_detections(self):
        oids = [12345, 45678]
        self.step.get_non_detections(oids)
        self.step.driver.query(
            "NonDetection", engine="mongo"
        ).find_all.assert_called_with(
            filter_by={"aid": {"$in": oids}},
            paginate=False,
        )

    def test_insert_objects_without_updates(self):
        objects = {
            "aid": [12345],
            "oid": ["ZTF1"],
            "firstmjd": [53000],
            "lastmjd": [54000],
            "ndet": [2],
            "meanra": [20.0],
            "meandec": [30.0],
            "sigmara": [0.1],
            "sigmadec": [0.2],
            "extra_fields": [{}],
            "new": [True],
        }
        df_objects = pd.DataFrame(objects)
        self.step.insert_objects(df_objects, engine="mongo")
        self.step.driver.query("Object", engine="mongo").bulk_insert.assert_called()
        self.step.driver.query("Object", engine="mongo").bulk_update.assert_not_called()
        insert_call = self.step.driver.query(
            "Object", engine="mongo"
        ).bulk_insert.mock_calls[0]
        name, args, kwargs = insert_call
        self.assertIsInstance(args, tuple)
        self.assertIsInstance(args[0][0], dict)

    def test_insert_objects_without_inserts(self):
        objects = {
            "aid": [12345],
            "oid": ["ZTF1"],
            "firstmjd": [53000],
            "lastmjd": [54000],
            "ndet": [2],
            "meanra": [20.0],
            "meandec": [30.0],
            "sigmara": [0.1],
            "sigmadec": [0.2],
            "new": [False],
        }
        df_objects = pd.DataFrame(objects)
        self.step.insert_objects(df_objects)
        self.step.driver.query().bulk_insert.assert_not_called()
        self.step.driver.query().bulk_update.assert_called()

    def test_insert_detections(self):
        detection = {
            "tid": ["test"],
            "aid": ["test"],
            "candid": ["test"],
            "mjd": ["test"],
            "fid": ["test"],
            "ra": ["test"],
            "dec": ["test"],
            "rb": ["test"],
            "mag": ["test"],
            "e_mag": ["test"],
            "rfid": ["test"],
            "e_ra": ["test"],
            "e_dec": ["test"],
            "isdiffpos": ["test"],
            "magpsf_corr": ["test"],
            "sigmapsf_corr": ["test"],
            "sigmapsf_corr_ext": ["test"],
            "corrected": ["test"],
            "dubious": ["test"],
            "parent_candid": ["test"],
            "has_stamp": ["test"],
            "step_id_corr": ["test"],
            "rbversion": ["test"],
        }
        df_detection = pd.DataFrame(detection)
        self.step.insert_detections(df_detection)
        self.step.driver.query().bulk_insert.assert_called()

    def test_insert_non_detections(self):
        non_detection = {
            "tid": ["test"],
            "aid": ["test"],
            "mjd": ["test"],
            "fid": ["test"],
            "diffmaglim": ["test"],
        }
        df_non_detection = pd.DataFrame(non_detection)
        self.step.insert_non_detections(df_non_detection)
        self.step.driver.query().bulk_insert.assert_called()

    def test_compute_meanra_correct(self):
        df = pd.DataFrame({"ra": [200, 100, 100], "e_ra": [0.1, 0.1, 0.1]})
        mean_ra, _ = self.step.compute_meanra(df["ra"], df["e_ra"])
        self.assertGreaterEqual(mean_ra, 0.0)
        self.assertLessEqual(mean_ra, 360.0)

    def test_compute_meanra_incorrect(self):
        df = pd.DataFrame({"ra": [-200, -100, -100], "e_ra": [0.1, 0.1, 0.1]})
        with pytest.raises(ValueError):
            mean_ra, _ = self.step.compute_meanra(df["ra"], df["e_ra"])

    def test_compute_meandec_correct(self):
        df = pd.DataFrame({"dec": [90, 90, 90], "e_dec": [0.1, 0.1, 0.1]})
        mean_dec, _ = self.step.compute_meandec(df["dec"], df["e_dec"])
        self.assertGreaterEqual(mean_dec, -90.0)
        self.assertLessEqual(mean_dec, 90.0)

    def test_compute_meandec_incorrect(self):
        df = pd.DataFrame({"dec": [200, 100, 100], "e_dec": [0.1, 0.1, 0.1]})
        print(self.step)
        with pytest.raises(ValueError):
            mean_dec, _ = self.step.compute_meandec(df["dec"], df["e_dec"])

    def test_execute_with_ZTF_stream(self):
        ZTF_messages = generate_message_ztf(10)
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 12

    def test_execute_with_ZTF_stream_non_detections(self):
        ZTF_messages = generate_message_ztf(10)
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 12

    def test_execute_with_ATLAS_stream(self):
        ATLAS_messages = generate_message_atlas(10)
        self.step.execute(ATLAS_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3

    def test_produce(self):
        alerts = pd.DataFrame([{"aid": "a", "candid": 1}])
        objects = pd.DataFrame(
            [{"aid": "a", "meanra": 1, "meandec": 1, "ndet": 1, "lastmjd": 1}]
        )
        light_curves = {
            "detections": pd.DataFrame([{"aid": "a", "candid": 1, "new": True}]),
            "non_detections": pd.DataFrame(
                [{"aid": "a", "candid": None, "new": False}]
            ),
        }
        self.step.produce(alerts, objects, light_curves)
        self.assertEqual(len(self.step.producer.produce.mock_calls), 1)
