import unittest
import pytest
import pandas as pd
from unittest import mock

from apf.producers import KafkaProducer
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo.connection import MongoConnection
from ingestion_step.step import IngestionStep

from data.messages import (
    generate_message_atlas,
    generate_message_ztf,
    generate_message,
)


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {},
            "PRODUCER_CONFIG": {"fake": "fake"},
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
            },
        }
        self.mock_database_connection = mock.create_autospec(MongoConnection)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.step = IngestionStep(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            producer=self.mock_producer,
        )

    def tearDown(self):
        del self.step

    # We need define function for each method in the class?
    def test_get_objects(self):
        oids = ["ZTF1", "ZTF2"]
        self.step.get_objects(oids)
        self.step.driver.query().find_all.assert_called_with(
            model=Object, filter_by={"aid": {"$in": oids}}, paginate=False
        )

    def test_get_detections(self):
        oids = [12345, 45678]
        self.step.get_detections(oids)
        self.step.driver.query().find_all.assert_called_with(
            model=Detection, filter_by={"aid": {"$in": oids}}, paginate=False
        )

    def test_get_non_detections(self):
        oids = [12345, 45678]
        self.step.get_non_detections(oids)
        self.step.driver.query().find_all.assert_called_with(
            model=NonDetection,
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
        self.step.insert_objects(df_objects)
        self.step.driver.query().bulk_insert.assert_called()
        self.step.driver.query().bulk_update.assert_not_called()

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
        with pytest.raises(ValueError):
            mean_dec, _ = self.step.compute_meandec(df["dec"], df["e_dec"])

    def test_execute_with_ZTF_stream(self):
        ZTF_messages = generate_message_ztf(10, 0)
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3

    def test_execute_with_ZTF_stream_and_existing_objects(self):
        ZTF_messages = generate_message_ztf(1, 0)

        def side_effect(*args, **kwargs):
            if kwargs["model"] == Object:
                return [
                    Object(
                        aid=ZTF_messages[0]["aid"],
                        oid=["test"],
                        lastmjd=1,
                        firstmjd=1,
                        ndet=1,
                        meanra=1,
                        meandec=1,
                    )
                ]
            if kwargs["model"] == Detection:
                return [
                    Detection(
                        tid="ZTF",
                        aid=ZTF_messages[0]["aid"],
                        candid=1,
                        mjd=1,
                        fid=1,
                        ra=1,
                        dec=1,
                        rb=1,
                        mag=20,
                        e_mag=0.1,
                        rfid=1,
                        e_ra=0.1,
                        e_dec=0.1,
                        isdiffpos=0.1,
                        corrected=False,
                        parent_candid=None,
                        has_stamp=False,
                        step_id_corr="aaa",
                        rbversion="ooo",
                        oid="test",
                    )
                ]
            return []

        self.step.driver.query().find_all.side_effect = side_effect
        self.step.execute(ZTF_messages)
        self.step.driver.query().bulk_update.assert_called()
        for call in self.step.driver.query().bulk_update.mock_calls:
            name, args, kwargs = call
            assert args[0] == args[1]
            assert args[0][0]["oid"] == ["test", "ZTFoid0"]

    def test_execute_with_existing_objects(self):
        num_messages = 10
        messages = generate_message(num_messages, num_prv_candidates=0)

        def side_effect(*args, **kwargs):
            if kwargs["model"] == Object:
                return [
                    Object(
                        aid=messages[0]["aid"],
                        oid=["test"],
                        lastmjd=1,
                        firstmjd=1,
                        ndet=1,
                        meanra=1,
                        meandec=1,
                    )
                ]
            if kwargs["model"] == Detection:
                return [
                    Detection(
                        tid="ZTF",
                        aid=messages[0]["aid"],
                        candid=1,
                        mjd=1,
                        fid=1,
                        ra=1,
                        dec=1,
                        rb=1,
                        mag=20,
                        e_mag=0.1,
                        rfid=1,
                        e_ra=0.1,
                        e_dec=0.1,
                        isdiffpos=0.1,
                        corrected=False,
                        parent_candid=None,
                        has_stamp=False,
                        step_id_corr="aaa",
                        rbversion="ooo",
                        oid="test",
                    )
                ]
            return []

        self.step.driver.query().find_all.side_effect = side_effect
        self.step.execute(messages)
        self.step.driver.query().bulk_update.assert_called()
        assert len(self.step.driver.query().bulk_update.mock_calls) == 1
        name, args, kwargs = self.step.driver.query().bulk_update.mock_calls[0]
        assert args[0] == args[1]
        assert len(args[0][0]["oid"]) == 2
        insert_calls = self.step.driver.query().bulk_insert.mock_calls
        assert len(insert_calls) == 3
        insert_object_call = insert_calls[0]
        name, args, kwargs = insert_object_call
        assert len(args[0]) == num_messages - 1

    def test_execute_with_ZTF_stream_non_detections(self):
        ZTF_messages = generate_message_ztf(10, 10)
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3

    def test_execute_with_ATLAS_stream(self):
        ATLAS_messages = generate_message_atlas(10)
        self.step.execute(ATLAS_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3
