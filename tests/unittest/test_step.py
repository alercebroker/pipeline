import unittest
from unittest import mock
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
from apf.producers import KafkaProducer
from generic_save_step.step import GenericSaveStep

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
        self.mock_database_connection = mock.create_autospec(new_DBConnection)
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
    def test_get_objects():
        pass