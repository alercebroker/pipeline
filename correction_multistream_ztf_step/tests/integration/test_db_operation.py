import unittest
import logging
import pytest
from apf.core.settings import config_from_yaml_file

import json

from db_plugins.db.sql._connection import PsqlDatabase
from correction_multistream_ztf_step.step import CorrectionMultistreamZTFStep
from db_plugins.db.sql.models import (
    Detection,
    ZtfDetection,
    ZtfForcedPhotometry,
    ForcedPhotometry,
    NonDetection,
    Object,
)
from tests.integration.data.ztf_messages import (
    detections,
    ztf_detections,
    non_detection as non_detections,
    ztf_forced_photometry,
    forced_photometry,
    objects,
)
from sqlalchemy import insert

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}

with open("tests/integration/data/data_input_prv_candidates_staging.json", "r") as file:
    data_consumer = json.load(file)


@pytest.mark.usefixtures("psql_service")
class TestCorrectionMultistreamZTF(unittest.TestCase):
    """Test class for Correction Multistream ZTF Step."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all test methods."""
        cls.settings = config_from_yaml_file("tests/test_utils/config.yaml")
        cls.logger = cls._set_logger(cls.settings)
        cls.db_sql = PsqlDatabase(psql_config)
        cls.step_params = {"config": cls.settings, "db_sql": cls.db_sql}

    def setUp(self):
        # crear db
        self.db_sql = PsqlDatabase(psql_config)
        self.db_sql.create_db()
        self.insert_test_data()

        self.step = self._step_creator()

    def insert_test_data(self):
        """Insert test data using the InsertData class pattern."""
        with self.db_sql.session() as session:
            # Insert objects first
            for obj_data in objects:
                obj = Object(**obj_data)
                session.add(obj)
            session.commit()  # commit objects before inserting other data.
            print("Object listo!")

            # Insertar detecciones
            for detection_data in detections:
                detection = Detection(**detection_data)
                session.add(detection)
            print("Detection listo!")

            # Insertar ztf_detecciones
            for ztf_detection_data in ztf_detections:
                ztf_detection = ZtfDetection(**ztf_detection_data)
                session.add(ztf_detection)
            print("ZtfDetection listo!")

            # Insertar no-detecciones
            for non_detection_data in non_detections:
                non_detection = NonDetection(**non_detection_data)
                session.add(non_detection)
            print("NonDetection listo!")

            # Insertar fotometría forzada ZTF
            if isinstance(ztf_forced_photometry, dict):
                forced_phot = ZtfForcedPhotometry(**ztf_forced_photometry)
                session.add(forced_phot)
            else:
                for fp_data in ztf_forced_photometry:
                    forced_phot = ZtfForcedPhotometry(**fp_data)
                    session.add(forced_phot)
            print("ZTFFP listo!")

            # Insertar fotometría forzada
            if isinstance(forced_photometry, dict):
                forced_phot = ForcedPhotometry(**forced_photometry)
                session.add(forced_phot)
            else:
                for fp_data in forced_photometry:
                    forced_phot = ForcedPhotometry(**fp_data)
                    session.add(forced_phot)
            print("FP listo!")

            session.commit()

    def tearDown(self):
        """Clean up after each test."""
        # Limpiar la base de datos
        self.db_sql.drop_db()

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
    def validate_alert_fields_message(data):
        """Validate that all required top-level fields are present in the messages."""
        for message in data:
            required_top_fields = [
                "oid",
                "measurement_id",
                "meanra",
                "meandec",
                "detections",
                "non_detections",
            ]
            missing_top_fields = [field for field in required_top_fields if field not in message]
            if missing_top_fields:
                return False

            unexpected_fields = [field for field in message if field not in required_top_fields]
            if unexpected_fields:
                return False

        return True

    @staticmethod
    def validate_non_detection_fields_message(data):
        """Validate that all required non-detection fields are present in the messages."""
        for message in data:
            if "non_detections" in message:
                if not isinstance(message["non_detections"], list):
                    return False

                if message["non_detections"]:
                    required_non_det_fields = ["oid", "sid", "tid", "band", "mjd", "diffmaglim"]
                    for non_det in message["non_detections"]:
                        missing_non_det_fields = [
                            field for field in required_non_det_fields if field not in non_det
                        ]
                        if missing_non_det_fields:
                            return False

                        unexpected_non_det_fields = [
                            field for field in non_det if field not in required_non_det_fields
                        ]
                        if unexpected_non_det_fields:
                            print("aqui3")
                            return False

        return True

    @staticmethod
    def validate_detection_fields_message(data):
        """Validate that all required detection fields are present in the messages."""
        for message in data:
            if "detections" in message:
                if not isinstance(message["detections"], list):
                    return False

                if message["detections"]:
                    required_det_fields = [
                        "oid",
                        "sid",
                        "tid",
                        "pid",
                        "band",
                        "measurement_id",
                        "mjd",
                        "ra",
                        "e_ra",
                        "dec",
                        "e_dec",
                        "mag",
                        "e_mag",
                        "mag_corr",
                        "e_mag_corr",
                        "e_mag_corr_ext",
                        "isdiffpos",
                        "parent_candid",
                        "has_stamp",
                        "corrected",
                        "dubious",
                        "stellar",
                        "forced",
                        "new",
                        "extra_fields",
                    ]
                    for det in message["detections"]:
                        missing_det_fields = [
                            field for field in required_det_fields if field not in det
                        ]
                        if missing_det_fields:
                            return False

                        unexpected_det_fields = [
                            field for field in det if field not in required_det_fields
                        ]
                        if unexpected_det_fields:
                            return False

        return True

    @staticmethod
    def message_validation(data):
        """Combine all validation methods to validate the entire message structure."""
        print(
            TestCorrectionMultistreamZTF.validate_alert_fields_message(data),
            TestCorrectionMultistreamZTF.validate_non_detection_fields_message(data),
            TestCorrectionMultistreamZTF.validate_detection_fields_message(data),
        )
        return (
            TestCorrectionMultistreamZTF.validate_alert_fields_message(data)  # True
            and TestCorrectionMultistreamZTF.validate_non_detection_fields_message(data)  # False
            and TestCorrectionMultistreamZTF.validate_detection_fields_message(data)  # False
        )

    @staticmethod
    def output_expected_count(data, oid, expected_dets, expected_non_dets):
        """Check if a message with the given OID has the expected number of detections and non-detections."""
        matching_dicts = [item for item in data if item.get("oid") == oid]
        if not matching_dicts:
            return False

        matching_dict = matching_dicts[0]
        len_detections = len(matching_dict["detections"])
        len_non_detections = len(matching_dict["non_detections"])
        return len_detections == expected_dets and len_non_detections == expected_non_dets

    def test_correction_step_execution(self):
        """Test the full execution of the correction step."""

        # Configure test consumer to return our messages
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
                    result = self.step._post_execute(result)
                    result = self.step._pre_produce(result)
                    self.step.producer.produce(result)
                    processed_messages.extend(result)
                except Exception as error:
                    self.logger.error(f"Error during execution: {error}")
                    raise

            # If the coe get here without errors, the basic test passed
            print("processed_messages", len(processed_messages))
            self.assertTrue(len(processed_messages) > 0, "No se procesaron mensajes")

            # Additional verification if the messages are properly processed
            if (
                hasattr(self.step.producer, "pre_produce_message")
                and self.step.producer.pre_produce_message
            ):
                messages_produce = self.step.producer.pre_produce_message[0]

                # Verifyin the message structure
                self.assertTrue(
                    self.message_validation(messages_produce), "Message validation failed"
                )

                # If there is enough data, we verify if there is specifically data
                if len(messages_produce) >= 2:
                    self.assertTrue(
                        self.output_expected_count(
                            messages_produce, "661678953", expected_dets=3, expected_non_dets=3
                        ),
                        "Failed count validation for OID 661678953",
                    )
                    self.assertTrue(
                        self.output_expected_count(
                            messages_produce, "879453281", expected_dets=6, expected_non_dets=6
                        ),
                        "Failed count validation for OID 879453281",
                    )

        finally:
            # Restore the original method
            self.step.consumer.consume = original_consume
