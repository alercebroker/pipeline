import unittest
from unittest import mock
import os
from correction import Correction, SQLConnection, KafkaProducer, pd
from apf.consumers import AVROFileConsumer

FILE_PATH = os.path.dirname(__file__)


class MockSession:
    def commit(self):
        pass


class StepTest(unittest.TestCase):
    def setUp(self):
        """
        Set settings and initialize step with mocks.
        """
        config = {
            "DB_CONFIG": {"SQL": {}},
            "PRODUCER_CONFIG": {},
            "STEP_METADATA": {
                "STEP_ID": "correction",
                "STEP_NAME": "correction",
                "STEP_VERSION": "test",
                "STEP_COMMENTS": "unittests",
            },
        }
        CONSUMER_CONFIG = {
            "DIRECTORY_PATH": os.path.join(FILE_PATH, "../examples/avro_test"),
            "NUM_MESSAGES": 10,
        }
        mock_db_connection = mock.create_autospec(SQLConnection)
        mock_db_connection.session = mock.create_autospec(MockSession)
        mock_producer = mock.create_autospec(KafkaProducer)
        self.step = Correction(
            consumer=AVROFileConsumer(CONSUMER_CONFIG),
            config=config,
            db_connection=mock_db_connection,
            producer=mock_producer,
            test_mode=True,
        )

    def tearDown(self):
        """
        Delete step instance to start clean.
        """
        del self.step

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.step.driver.query().get_or_create.assert_called_once()
        self.step.driver.session.commit.assert_called_once()

    def test_remove_stamps(self):
        alert = {"cutoutDifference": {}, "cutoutScience": {}, "cutoutTemplate": {}}
        self.step.remove_stamps(alert)
        self.assertEqual(alert, {})

    @mock.patch.object(Correction, "preprocess_ps1")
    @mock.patch.object(Correction, "preprocess_ss")
    @mock.patch.object(Correction, "preprocess_reference")
    @mock.patch.object(Correction, "preprocess_gaia")
    def test_preprocess_metadata(self, mock_gaia, mock_ref, mock_ss, mock_ps1):
        mock_ps1.return_value = "ps1"
        mock_ss.return_value = "ss"
        mock_ref.return_value = "ref"
        mock_gaia.return_value = "gaia"
        metadata = {
            "ps1_ztf": {},
            "ss_ztf": {},
            "reference": {},
            "gaia": {},
        }
        expected = {
            "ps1_ztf": "ps1",
            "ss_ztf": "ss",
            "reference": "ref",
            "gaia": "gaia",
        }
        detections = pd.DataFrame()
        self.step.preprocess_metadata(metadata, detections)
        self.assertEqual(metadata, expected)

    @mock.patch.object(Correction, "get_last_alert")
    def test_preprocess_objects(self):
        objects = pd.DataFrame()
        light_curves = pd.DataFrame()
        detections = pd.DataFrame()
        new_stats = pd.DataFrame()
        self.step.preprocess_objects(objects, light_curves, detections, new_stats)

    def test_preprocess_detections(self):
        pass

    def test_preprocess_dataquality(self):
        pass

    def test_do_correction(self):
        pass

    def test_do_dubious(self):
        pass

    def test_do_magstats(self):
        pass

    def test_do_dmdt(self):
        pass

    def test_get_objects(self):
        pass

    def test_get_detections(self):
        pass

    def test_get_non_detections(self):
        pass

    def test_get_lightcurves(self):
        pass

    def test_get_ps1(self):
        pass

    def test_get_ss(self):
        pass

    def test_get_reference(self):
        pass

    def test_get_gaia(self):
        pass

    def test_get_metadata(self):
        pass

    def test_get_magstats(self):
        pass

    def test_get_prv_candidates(self):
        pass

    def test_cast_non_detection(self):
        pass

    def test_preprocess_lightcurves(self):
        pass

    def test_preprocess_ps1(self):
        pass

    def test_preprocess_ss(self):
        pass

    def test_preprocess_reference(self):
        pass

    def test_preprocess_gaia(self):
        pass

    def test_get_colors(self):
        pass

    def test_get_last_alert(self):
        pass

    def test_get_object_data(self):
        pass

    def test_get_dataquality(self):
        pass

    def test_get_first_corrected(self):
        pass

    def test_insert_detections(self):
        pass

    def test_insert_non_detections(self):
        pass

    def test_insert_dataquality(self):
        pass

    def test_insert_objects(self):
        pass

    def test_produce(self):
        pass

    def test_execute(self):
        self.step.start()
