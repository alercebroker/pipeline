import unittest
from unittest import mock
from features.step import FeaturesComputer
from features.step import CustomHierarchicalExtractor, DetectionsPreprocessorZTF
from features.step import SQLConnection
from features.step import KafkaProducer
from features.step import pd
from features.step import np
from features.step import Feature, FeatureVersion
from db_plugins.db.sql import SQLQuery


class MockSession:
    def commit(self):
        pass

    def add(self, model):
        pass


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {},
            "PRODUCER_CONFIG": {"fake": "fake"},
            "FEATURE_VERSION": "v1",
            "STEP_VERSION": "1.0-test",
            "STEP_VERSION_PREPROCESS": "1.0-test",
        }
        self.mock_database_connection = mock.create_autospec(SQLConnection)
        self.mock_database_connection.session = mock.create_autospec(MockSession)
        self.mock_custom_hierarchical_extractor = mock.create_autospec(FeaturesComputer)
        self.mock_detections_preprocessor_ztf = mock.create_autospec(
            DetectionsPreprocessorZTF
        )
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.step = FeaturesComputer(
            config=self.step_config,
            preprocessor=self.mock_detections_preprocessor_ztf,
            features_computer=self.mock_custom_hierarchical_extractor,
            db_connection=self.mock_database_connection,
            producer=self.mock_producer,
        )

    @mock.patch.object(FeaturesComputer, "create_detections_dataframe")
    def test_preprocess_detections(self, mock_create_dataframe):
        detections = [{"oid": "oidtest", "candid": 123, "mjd": 1.0}]
        detections_preprocessed = self.step.preprocess_detections(detections)
        mock_create_dataframe.assert_called_with(detections)
        self.mock_detections_preprocessor_ztf.preprocess.assert_called()

    def test_preprocess_non_detections(self):
        non_detections = []
        non_detections = self.step.preprocess_non_detections(non_detections)
        self.assertIsInstance(non_detections, pd.DataFrame)

    def test_preprocess_xmatches(self):
        xmatches = self.step.preprocess_xmatches("test")
        self.assertEqual(xmatches, "test")

    def test_preprocess_metadata(self):
        metadata = self.step.preprocess_metadata("test")
        self.assertEqual(metadata, "test")

    def test_create_detections_dataframe(self):
        detections = {"oid": "oidtest", "candid": 123, "mjd": 1.0, "alert.sgscore1": 1}
        df = self.step.create_detections_dataframe(detections)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse("alert.sgscore1" in df.columns)
        self.assertTrue("sgscore1" in df.columns)
        self.assertEqual("oid", df.index.name)

    def test_compute_features(self):
        detections = pd.DataFrame()
        non_detections = pd.DataFrame()
        self.mock_custom_hierarchical_extractor.compute_features.return_value = (
            pd.DataFrame()
        )
        features = self.step.compute_features(detections, non_detections)
        self.mock_custom_hierarchical_extractor.compute_features.assert_called_with(
            detections, non_detections
        )
        self.assertIsInstance(features, pd.DataFrame)

    def test_convert_nan(self):
        d = {"di": {}, "x": 1, "y": np.nan}
        d = self.step.convert_nan(d)
        expected = {"di": {}, "x": 1, "y": None}
        self.assertEqual(d, expected)

    def test_insert_db_doesnt_exist(self):
        oid = "ZTF1"
        features = {"testfeature_1": 1}
        version = self.step_config["FEATURE_VERSION"]
        feature_id = self.step_config["STEP_VERSION"]
        preprocess_id = self.step_config["STEP_VERSION_PREPROCESS"]
        mock_feature_version = mock.create_autospec(FeatureVersion)
        mock_feature_version.version = version
        mock_feature_version.value = 1
        self.mock_database_connection.query().get_or_create.return_value = (
            mock_feature_version,
            False,
        )
        self.step.insert_db(oid, features)
        self.mock_database_connection.query().get_or_create.assert_has_calls(
            [
                mock.call(
                    {
                        "version": version,
                        "step_id_feature": feature_id,
                        "step_id_preprocess": preprocess_id,
                    }
                ),
                mock.call(
                    {
                        "oid": oid,
                        "name": "testfeature_1",
                        "value": 1,
                        "fid": 1,
                        "version": version,
                    }
                ),
            ]
        )
        self.mock_database_connection.query().update.assert_called_once()
        self.mock_database_connection.session.commit.assert_called_once()

    def test_insert_db_exist(self):
        oid = "ZTF1"
        features = {"testfeature_1": 1}
        version = self.step_config["FEATURE_VERSION"]
        feature_id = self.step_config["STEP_VERSION"]
        preprocess_id = self.step_config["STEP_VERSION_PREPROCESS"]
        mock_feature_version = mock.create_autospec(FeatureVersion)
        mock_feature_version.version = version
        self.mock_database_connection.query().get_or_create.return_value = (
            mock_feature_version,
            True,
        )
        self.step.insert_db(oid, features)
        self.mock_database_connection.query().get_or_create.assert_has_calls(
            [
                mock.call(
                    {
                        "version": version,
                        "step_id_feature": feature_id,
                        "step_id_preprocess": preprocess_id,
                    }
                ),
                mock.call(
                    {
                        "oid": oid,
                        "name": "testfeature_1",
                        "value": 1,
                        "fid": 1,
                        "version": version,
                    }
                ),
            ]
        )
        self.mock_database_connection.query().update.assert_not_called()
        self.mock_database_connection.session.commit.assert_called_once()

    def test_get_fid(self):
        feature = "W1"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, 0)
        feature = "g-r_max"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, 12)
        feature = "somefeature_2"
        fid = self.step.get_fid(feature)
        self.assertEqual(fid, 2)

    @mock.patch.object(FeaturesComputer, "compute_features")
    def test_execute_less_than_6(self, mock_compute):
        message = {
            "object": {"oid": "ZTF1"},
            "detections": [{"candid": 123, "oid": "ZTF1", "mjd": 456}],
            "non_detections": [],
            "xmatches": "",
            "metadata": {},
        }
        self.step.execute(message)
        mock_compute.assert_not_called()

    @mock.patch.object(FeaturesComputer, "convert_nan")
    @mock.patch.object(FeaturesComputer, "compute_features")
    def test_execute_no_features(self, mock_compute, mock_convert_nan):
        message = {
            "object": {"oid": "ZTF1"},
            "detections": [{"candid": 123, "oid": "ZTF1", "mjd": 456}],
            "non_detections": [],
            "xmatches": "",
            "metadata": {},
        }
        mock_compute.return_value = pd.DataFrame()
        self.step.execute(message)
        mock_convert_nan.assert_not_called()

    @mock.patch.object(FeaturesComputer, "compute_features")
    def test_execute_with_producer(self, mock_compute):
        message = {
            "object": {"oid": "ZTF1"},
            "detections": [{"candid": 123, "oid": "ZTF1", "mjd": 456, "fid": 1}] * 10,
            "non_detections": [],
            "xmatches": "",
            "metadata": {},
        }
        df = pd.DataFrame({"oid": ["ZTF1"] * 10, "feat_1": [1] * 10})
        df.set_index("oid", inplace=True)
        mock_compute.return_value = df
        mock_feature_version = mock.create_autospec(FeatureVersion)
        mock_feature_version.version = self.step_config["FEATURE_VERSION"]
        self.mock_database_connection.query().get_or_create.return_value = (
            mock_feature_version,
            True,
        )

        self.step.execute(message)
        self.mock_producer.produce.assert_called_once()
