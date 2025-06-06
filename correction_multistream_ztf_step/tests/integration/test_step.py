
import unittest
import logging
import pytest
from apf.core.settings import config_from_yaml_file
from db_plugins.db.sql._connection import PsqlDatabase
from correction_multistream_ztf_step.step import CorrectionMultistreamZTFStep

import json 
import pandas as pd
from tests.test_utils.test_step_utils import OUTPUT, psql_config

with open("tests/integration/data/data_input_step.json", "r") as file:
    data_consumer = json.load(file)

with open("tests/integration/data/expected_output.json", "r") as file:
    expected_output = json.load(file)

@pytest.mark.usefixtures("psql_service")
class TestStep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all test methods."""
        cls.settings = config_from_yaml_file("tests/test_utils/config_w_scribe.yaml")
        cls.logger = cls._set_logger(cls.settings)
        cls.db_sql = PsqlDatabase(psql_config)
        cls.step_params = {"config": cls.settings, "db_sql": cls.db_sql}

    def setUp(self):
        self.settings = config_from_yaml_file("tests/test_utils/config_w_scribe.yaml")
        self.db_sql = PsqlDatabase(psql_config)
        self.step = self._step_creator()

    @classmethod
    def _set_logger(cls, settings):
        """Set up the logger with the appropriate level and format."""
        level = logging.INFO
        if settings.get("LOGGING_DEBUG"):
            level = logging.DEBUG

        logger = logging.getLogger("alerce")
        logger.setLevel(level)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)7s %(name)36s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        logger.addHandler(handler)
        return logger


    def _step_creator(self):
        """Create an instance of the CorrectionMultistreamZTFStep."""
        step = CorrectionMultistreamZTFStep(**self.step_params)

        if self.settings["FEATURE_FLAGS"]["SKIP_MJD_FILTER"]:
            self.logger.info(
                "This step won't filter detections by MJD. \
                Keep this in mind when using for ELAsTiCC"
            )
        return step


    @staticmethod
    def validate_oids(data: list[dict]):

        oids = [msg[0]['oid'] for msg in data]        
        
        return set(oids) == set(OUTPUT['oids'])

    @staticmethod
    def validate_mids(data: list[dict]):

        mids = [msg[0]['measurement_id'][0] for msg in data]

        return set(mids) == set(OUTPUT['mids'])
    
    @staticmethod
    def validate_dets(data: list[dict]):

        dets = [len(msg[0]['detections']) for msg in data]

        return set(dets) == set(OUTPUT['dets'])

    @staticmethod
    def validate_ndets(data: list[dict]):

        ndets = [len(msg[0]['non_detections']) for msg in data]

        return set(ndets) == set(OUTPUT['ndets'])
    
    def test_step_execution(self):
        self.step.consumer.messages = data_consumer
        original_consume = self.step.consumer.consume
        self.step.consumer.consume = lambda: data_consumer
        try:
            # Pre-consume and configuration
            self.step._pre_consume()

            # Processing all messages in the consumer
            processed_messages = []
            for message in self.step.consumer.consume():
                preprocessed_msg = self.step._pre_execute(message)
                if len(preprocessed_msg) == 0:
                    self.logger.info("Message of len zero after pre_execute")
                    continue

                try:
                    result = self.step.execute(preprocessed_msg)

                    self.step.producer.produce(result)
                    result = self.step._post_execute(result)
                    processed_messages.extend(result)
                    result = self.step._pre_produce(result)

                except Exception as error:
                    self.logger.error(f"Error during execution: {error}")
                    raise
                    
            assert pd.DataFrame(self.step.producer.pre_produce_message).equals(pd.DataFrame(expected_output))

            assert self.validate_oids(self.step.producer.pre_produce_message)
            assert self.validate_mids(self.step.producer.pre_produce_message)
            assert self.validate_dets(self.step.producer.pre_produce_message)
            assert self.validate_ndets(self.step.producer.pre_produce_message)
            

        finally:
            # Restore the original method
            self.step.consumer.consume = original_consume

       