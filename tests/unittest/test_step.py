import unittest
from unittest import mock
from db_plugins.db.models import Object, Detection, NonDetection
from db_plugins.db.mongo.connection import MongoConnection
from apf.producers import KafkaProducer
from generic_save_step.step import GenericSaveStep
import pandas as pd

ATLAS_MESSAGE = {}
ZTF_MESSAGE = {}

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
    
    
    # We need define function for each method in the class?
    def test_get_objects(self):
        
        oids=["ZTF1", "ZTF2"]
        self.step.get_objects(oids)
        self.step.driver.query(Object).find_all.assert_called_with(collection=Object, filter_by={"_id": {"$in": oids}}, paginate=False)

    def test_get_detections(self):
        oids = [12345, 45678]
        self.step.get_detections(oids)
        self.step.driver.query(Detection).find_all.assert_called_with(collection=Detection, filter_by={"aid": {"$in": oids}}, paginate=False)
    
    def test_get_non_detections(self):
        oids = [12345, 45678]
        self.step.get_non_detections(oids)
        self.step.driver.query(NonDetection).find_all.assert_called_with(collection=NonDetection, filter_by={"aid": {"$in": oids}}, paginate=False)
    
    def test_insert_objects_without_updates(self):
        pass
