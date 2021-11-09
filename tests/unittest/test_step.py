import unittest
import pytest
import pandas as pd
import os
from unittest import mock
from fastavro import reader

from apf.producers import KafkaProducer
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo.connection import MongoConnection
from generic_save_step.step import GenericSaveStep


# Auxiliary function to read AVRO
def get_content(file_path):
    with open(file_path, "rb") as f:
        content = reader(f).next()
    return content


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
        self.step = GenericSaveStep(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            producer=self.mock_producer
        )
    
    def tearDown(self):
        del self.step
    
    # We need define function for each method in the class?
    def test_get_objects(self):
        oids = ["ZTF1", "ZTF2"]
        self.step.get_objects(oids)
        self.step.driver.query().find_all.assert_called_with(model=Object,
                                                             filter_by={"aid": {"$in": oids}},
                                                             paginate=False)

    def test_get_detections(self):
        oids = [12345, 45678]
        self.step.get_detections(oids)
        self.step.driver.query().find_all.assert_called_with(model=Detection,
                                                             filter_by={"aid": {"$in": oids}},
                                                             paginate=False)
    
    def test_get_non_detections(self):
        oids = [12345, 45678]
        self.step.get_non_detections(oids)
        self.step.driver.query().find_all.assert_called_with(model=NonDetection,
                                                             filter_by={"aid": {"$in": oids}},
                                                             paginate=False)
    
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
            "new": [True]
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
            "new": [False]
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
            "diffmaglim": ["test"]
        }
        df_non_detection = pd.DataFrame(non_detection)
        self.step.insert_non_detections(df_non_detection)
        self.step.driver.query().bulk_insert.assert_called()

    def test_compute_meanra_correct(self):
        df = pd.DataFrame({"ra": [200, 100, 100],
                           "e_ra": [0.1, 0.1, 0.1]})
        mean_ra, _ = self.step.compute_meanra(df["ra"], df["e_ra"])
        self.assertGreaterEqual(mean_ra, 0.0)
        self.assertLessEqual(mean_ra, 360.0)
    
    def test_compute_meanra_incorrect(self):
        df = pd.DataFrame({"ra": [-200, -100, -100],
                           "e_ra": [0.1, 0.1, 0.1]})
        with pytest.raises(ValueError):
            mean_ra, _ = self.step.compute_meanra(df["ra"], df["e_ra"])

    def test_compute_meandec_correct(self):
        df = pd.DataFrame({"dec": [90, 90, 90],
                           "e_dec": [0.1, 0.1, 0.1]})
        mean_dec, _ = self.step.compute_meandec(df["dec"], df["e_dec"])
        self.assertGreaterEqual(mean_dec, -90.0)
        self.assertLessEqual(mean_dec, 90.0)
    
    def test_compute_meandec_incorrect(self):
        df = pd.DataFrame({"dec": [200, 100, 100],
                           "e_dec": [0.1, 0.1, 0.1]})
        with pytest.raises(ValueError):
            mean_dec, _ = self.step.compute_meandec(df["dec"], df["e_dec"])

    def test_execute_with_ZTF_stream(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        files = os.listdir(os.path.join(dir_path, "data/ZTF"))
        ZTF_messages = [get_content(os.path.join(dir_path, "data/ZTF", x)) for x in files]
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3
    
    def test_execute_with_ZTF_stream_non_detections(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        files = os.listdir(os.path.join(dir_path, "data/ZTF"))
        ZTF_messages = [get_content(os.path.join(dir_path, "data/ZTF", x)) for x in files]
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3

    def test_execute_with_ATLAS_stream(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        files = os.listdir(os.path.join(dir_path, "data/ATLAS"))
        ATLAS_messages = [get_content(os.path.join(dir_path, "data/ATLAS", x)) for x in files]
        self.step.execute(ATLAS_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3
