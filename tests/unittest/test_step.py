import unittest
from unittest import mock
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo.connection import MongoConnection
from apf.producers import KafkaProducer
from generic_save_step.step import GenericSaveStep
import pandas as pd
import pickle

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
        #ADD MONGO CONNECTION
        self.mock_database_connection = mock.create_autospec(MongoConnection)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        # Inside the step some objects are instantiated, 
        # we need to set mocks or patch?
        # ALeRCEParser()
        # Processor with ZTFPrvCandidatesStrategy
        # Corrector with ZTFCorrectionStrategy
        self.step = GenericSaveStep(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            producer=self.mock_producer,
            test_mode=True,
        )
    
    def tearDown(self):
        del self.step
    
    # We need define function for each method in the class?
    def test_get_objects(self):
        
        oids=["ZTF1", "ZTF2"]
        self.step.get_objects(oids)
        self.step.driver.query().find_all.assert_called_with(model=Object, filter_by={"_id": {"$in": oids}}, paginate=False)

    def test_get_detections(self):
        oids = [12345, 45678]
        self.step.get_detections(oids)
        self.step.driver.query().find_all.assert_called_with(model=Detection, filter_by={"aid": {"$in": oids}}, paginate=False)
    
    def test_get_non_detections(self):
        oids = [12345, 45678]
        self.step.get_non_detections(oids)
        self.step.driver.query().find_all.assert_called_with(model=NonDetection, filter_by={"aid": {"$in": oids}}, paginate=False)
    
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
            "extra_fields":[{}], 
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
            "tid": ["test"], # Telescope id (this gives the spatial coordinates of the observatory, e.g. ZTF, ATLAS-HKO, ATLAS-MLO)
            "aid": ["test"],
            "candid": ["test"],
            "mjd": ["test"],
            "fid": ["test"],
            "ra": ["test"],
            "dec": ["test"],
            "rb": ["test"],
            "mag": ["test"],
            "sigmag": ["test"],
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
            "tid": ["test"], # Telescope id (this gives the spatial coordinates of the observatory, e.g. ZTF, ATLAS-HKO, ATLAS-MLO)
            "aid": ["test"],
            "mjd": ["test"],
            "fid": ["test"],
            "diffmaglim": ["test"]
        }
        df_non_detection = pd.DataFrame(non_detection)
        self.step.insert_non_detections(df_non_detection)
        self.step.driver.query().bulk_insert.assert_called()
    
    def test_execute_with_ZTF_stream(self):
        ZTF_messages = [{
            "objectId": "ZTF1",
            "publisher": "ZTF",
            "prv_candidates": None,
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
                "rb": 1,
                "pid": "pid",
                "fid": 1,
                "rfid": 100,
                "magpsf": 20,
                "sigmapsf": 1,
                "rbversion": 1,
                "distnr": 1,
                "magnr": 1,
                "sigmagnr": 1,
            },
            "cutoutTemplate": {"stampData": b""},
            "cutoutScience": {"stampData": b""},
            "cutoutDifference": {"stampData": b""},
        }]
        self.step.execute(ZTF_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3
    
    def test_execute_with_ATLAS_stream(self):
        with open("tests/unittest/data/ATLAS_stream.pkl", "rb") as f:
            ATLAS_messages = pickle.load(f)
        
        self.step.execute(ATLAS_messages)
        # Verify 3 inserts calls: objects, detections, non_detections
        assert len(self.step.driver.query().bulk_insert.mock_calls) == 3