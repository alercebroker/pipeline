import unittest
from unittest import mock
import os
from correction import (
    Correction,
    SQLConnection,
    KafkaProducer,
    pd,
    DATAQUALITY_KEYS,
    np,
)
from apf.consumers import AVROFileConsumer
from objects import objects as input_objects
from objects import processed_objects as output_objects
from detections import detections as input_detections
from detections import detections_not_db as input_detections_not_db
from non_detections import non_detections as input_non_detections
from stats import new_stats as input_new_stats
from stats import stats_from_db as input_stats_from_db
from metadata import ps1, ss, reference, gaia
from dataquality import dataquality as input_dataquality

FILE_PATH = os.path.dirname(__file__)


class MockSession:
    def commit(self):
        pass


class StepTest(unittest.TestCase):
    def setUp(self):
        """
        Set settings and initialize step with mocks.
        """
        self.alert_candidate = pd.read_csv(FILE_PATH + "/alert_candidate.csv")
        self.alert = pd.read_csv(
            FILE_PATH + "/alert.csv", header=None, index_col=0, squeeze=True
        )
        self.alert.prv_candidates = self.alert.prv_candidates.replace("'", '"')
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
        del self.alert_candidate
        del self.alert

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
    @mock.patch("correction.apply_object_stats_df")
    def test_preprocess_objects(self, mock_apply_object_stats_df, mock_get_last_alert):
        objects = pd.DataFrame(input_objects)
        detections = pd.DataFrame(input_detections)
        non_detections = pd.DataFrame(input_non_detections)
        detections_not_db = pd.DataFrame(input_detections_not_db)
        new_stats = pd.DataFrame(input_new_stats)
        light_curves = {"detections": detections, "non_detections": non_detections}
        mock_get_last_alert.return_value = detections_not_db[
            ["oid", "ndethist", "ncovhist", "jdstarthist", "jdendhist"]
        ].iloc[0]
        result = self.step.preprocess_objects(
            objects, light_curves, detections_not_db, new_stats
        )
        expected = pd.DataFrame(output_objects)
        pd.testing.assert_frame_equal(result, expected)

    def test_preprocess_detections(self):
        expected_columns = ["mjd", "has_stamp", "step_id_corr"]
        result = self.step.preprocess_detections(self.alert_candidate)
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_preprocess_dataquality(self):
        input = pd.DataFrame(input_detections_not_db)
        res = self.step.preprocess_dataquality(input)
        np.testing.assert_array_equal(list(res.columns).sort(), DATAQUALITY_KEYS.sort())

    @mock.patch("correction.step.apply_correction_df")
    def test_do_correction(self, mock_apply_correction_df):
        detections = pd.DataFrame(input_detections_not_db)
        self.step.do_correction(detections)
        mock_apply_correction_df.assert_called_once()

    @mock.patch("correction.step.is_dubious")
    @mock.patch.object(Correction, "get_first_corrected")
    def test_do_dubious(self, mock_get_first_corrected, mock_is_dubious):
        df = pd.DataFrame(input_detections)
        dubious = pd.Series(np.array([True] * len(df)))
        dubious.name = "dubious"
        mock_is_dubious.return_value = dubious
        res = self.step.do_dubious(df)
        mock_get_first_corrected.assert_called()
        mock_is_dubious.assert_called_once()
        pd.testing.assert_series_equal(res.dubious, dubious)

    def test_do_magstats(self):
        magstats = pd.DataFrame(input_stats_from_db)
        metadata = {
            "ps1_ztf": pd.DataFrame(ps1),
            "ss_ztf": pd.DataFrame(ss),
            "reference": pd.DataFrame(reference),
            "gaia": pd.DataFrame(gaia),
        }
        detections = pd.DataFrame(input_detections)
        non_detections = pd.DataFrame(input_non_detections)
        light_curves = {"detections": detections, "non_detections": non_detections}
        res = self.step.do_magstats(light_curves, metadata, magstats)
        self.assertIsInstance(res, pd.DataFrame)

    @mock.patch("correction.step.do_dmdt_df")
    def test_do_dmdt(self, mock_do_dmdt_df):
        non_dets = pd.DataFrame(input_non_detections)
        light_curves = {"non_detections": non_dets}
        magstats = pd.DataFrame(input_stats_from_db)
        res = self.step.do_dmdt(light_curves, magstats)
        self.assertGreaterEqual(len(res), 0)

    @mock.patch("correction.step.do_dmdt_df")
    def test_do_dmdt_empty(self, mock_do_dmdt_df):
        non_dets = pd.DataFrame(columns=["oid", "fid"])
        light_curves = {"non_detections": non_dets}
        magstats = pd.DataFrame(input_stats_from_db)
        res = self.step.do_dmdt(light_curves, magstats)
        self.assertEqual(len(res), 0)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_objects(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_objects(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_detections(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_detections(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_non_detections(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_non_detections(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch.object(Correction, "get_detections")
    @mock.patch.object(Correction, "get_non_detections")
    def test_get_lightcurves(self, mock_get_non_detections, mock_get_detections):
        mock_get_detections.return_value = "dets"
        mock_get_non_detections.return_value = "non_dets"
        res = self.step.get_lightcurves(["oid"])
        self.assertEqual(res, {"detections": "dets", "non_detections": "non_dets"})

    @mock.patch("correction.step.pd.read_sql")
    def test_get_ps1(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_ps1(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_ss(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_ss(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_reference(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_reference(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_gaia(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_gaia(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    @mock.patch.object(Correction, "get_ps1")
    @mock.patch.object(Correction, "get_ss")
    @mock.patch.object(Correction, "get_reference")
    @mock.patch.object(Correction, "get_gaia")
    def test_get_metadata(self, get_gaia, get_reference, get_ss, get_ps1):
        get_gaia.return_value = "gaia"
        get_reference.return_value = "ref"
        get_ss.return_value = "ss"
        get_ps1.return_value = "ps1"
        expected = {
            "ps1_ztf": "ps1",
            "ss_ztf": "ss",
            "reference": "ref",
            "gaia": "gaia",
        }
        res = self.step.get_metadata(["oid"])
        self.assertEqual(res, expected)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_magstats(self, mock_read_sql):
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_magstats(["oid"])
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    def test_get_prv_candidates_detection(self):
        alert_cpy = self.alert.copy(deep=True)
        alert_cpy.prv_candidates = eval(alert_cpy.prv_candidates)
        x, y = self.step.get_prv_candidates(alert_cpy)
        self.assertIsInstance(x, list)
        self.assertIsInstance(y, list)
        self.assertEqual(len(x) + len(y), 73)

    @mock.patch.object(Correction, "cast_non_detection")
    def test_get_prv_candidates_non_detection(self, mock_cast_non_detection):
        alert_cpy = self.alert.copy(deep=True)
        alert_cpy.prv_candidates = eval(alert_cpy.prv_candidates)
        for prv_cand in alert_cpy.prv_candidates:
            prv_cand["candid"] = None
        self.step.get_prv_candidates(alert_cpy)
        mock_cast_non_detection.assert_called()
        self.assertEqual(mock_cast_non_detection.call_count, 73)

    def test_get_prv_candidates_empty(self):
        alert_cpy = self.alert.copy(deep=True)
        alert_cpy.prv_candidates = None
        x, y = self.step.get_prv_candidates(alert_cpy)
        pd.testing.assert_frame_equal(x, pd.DataFrame())
        pd.testing.assert_frame_equal(y, pd.DataFrame())

    def test_cast_non_detection(self):
        oid = "oid"
        candidate = {"mjd": 123, "diffmaglim": 456, "fid": 1}
        res = self.step.cast_non_detection(oid, candidate)
        candidate["oid"] = oid
        self.assertEqual(res, candidate)

    # @unittest.skip
    @mock.patch.object(Correction, "get_lightcurves")
    @mock.patch.object(Correction, "get_prv_candidates")
    @mock.patch.object(Correction, "do_correction")
    def test_preprocess_lightcurves_no_prv_detections(
        self, mock_do_correction, mock_get_prv_candidates, mock_get_lightcurves
    ):
        self.alert = pd.read_csv(FILE_PATH + "/alerts.csv")
        detections = pd.DataFrame(input_detections_not_db)
        light_curves = {
            "detections": pd.DataFrame(input_detections),
            "non_detections": pd.DataFrame(input_non_detections),
        }
        self.alert = self.alert.drop(columns=["prv_candidates"])
        mock_get_lightcurves.return_value = light_curves
        res = self.step.preprocess_lightcurves(detections, self.alert)
        mock_get_prv_candidates.assert_not_called()
        self.assertIsInstance(res, dict)
        self.assertEqual(list(res.keys()), ["detections", "non_detections"])

    @mock.patch.object(Correction, "get_lightcurves")
    @mock.patch.object(Correction, "get_prv_candidates")
    @mock.patch.object(Correction, "do_correction")
    def test_preprocess_lightcurves_with_prv_detections_already_on_db(
        self, mock_do_correction, mock_get_prv_candidates, mock_get_lightcurves
    ):
        self.alert = pd.read_csv(FILE_PATH + "/alerts.csv")
        detections = pd.DataFrame(input_detections_not_db)
        non_detections = pd.DataFrame(input_non_detections)
        light_curves = {
            "detections": pd.DataFrame(input_detections),
            "non_detections": pd.DataFrame(input_non_detections),
        }
        mock_get_lightcurves.return_value = light_curves
        mock_get_prv_candidates.return_value = (
            detections.to_dict("records"),
            non_detections.to_dict("records"),
        )
        res = self.step.preprocess_lightcurves(detections, self.alert)
        mock_do_correction.assert_not_called()

    @mock.patch.object(Correction, "get_lightcurves")
    @mock.patch.object(Correction, "get_prv_candidates")
    @mock.patch.object(Correction, "do_correction")
    def test_preprocess_lightcurves_with_prv_detections_not_on_db(
        self, mock_do_correction, mock_get_prv_candidates, mock_get_lightcurves
    ):
        self.alert = pd.read_csv(FILE_PATH + "/alerts.csv")
        detections = pd.DataFrame(input_detections_not_db)
        new_detections = pd.DataFrame(input_detections_not_db)
        non_detections = pd.DataFrame(input_non_detections)
        light_curves = {
            "detections": pd.DataFrame(input_detections),
            "non_detections": pd.DataFrame(input_non_detections),
        }
        mock_get_lightcurves.return_value = light_curves
        self.alert.prv_candidates = [eval(self.alert.prv_candidates.loc[0])]
        new_detections.at[0, "candid"] = 123
        mock_get_prv_candidates.return_value = (
            new_detections.to_dict("records"),
            non_detections.to_dict("records"),
        )
        mock_do_correction.return_value = detections
        res = self.step.preprocess_lightcurves(detections, self.alert)
        mock_do_correction.assert_called()
        self.assertIsInstance(res, dict)
        self.assertEqual(list(res.keys()), ["detections", "non_detections"])

    @mock.patch.object(Correction, "get_lightcurves")
    @mock.patch.object(Correction, "get_prv_candidates")
    @mock.patch.object(Correction, "do_correction")
    def test_preprocess_lightcurves_prv_non_detections_already_on_db(
        self, mock_do_correction, mock_get_prv_candidates, mock_get_lightcurves
    ):
        self.alert = pd.read_csv(FILE_PATH + "/alerts.csv")
        detections = pd.DataFrame(input_detections_not_db)
        non_detections = pd.DataFrame(input_non_detections)
        light_curves = {
            "detections": pd.DataFrame(input_detections),
            "non_detections": pd.DataFrame(input_non_detections),
        }
        mock_get_lightcurves.return_value = light_curves
        mock_get_prv_candidates.return_value = (
            detections.to_dict("records"),
            non_detections.to_dict("records"),
        )
        res = self.step.preprocess_lightcurves(detections, self.alert)
        mock_get_prv_candidates.assert_called()
        self.assertIsInstance(res, dict)
        self.assertEqual(list(res.keys()), ["detections", "non_detections"])

    @mock.patch.object(Correction, "get_lightcurves")
    @mock.patch.object(Correction, "get_prv_candidates")
    @mock.patch.object(Correction, "do_correction")
    def test_preprocess_lightcurves_prv_non_detections_not_on_db(
        self, mock_do_correction, mock_get_prv_candidates, mock_get_lightcurves
    ):
        self.alert = pd.read_csv(FILE_PATH + "/alerts.csv")
        detections = pd.DataFrame(input_detections_not_db)
        non_detections = pd.DataFrame(input_non_detections)
        new_non_detections = pd.DataFrame(input_non_detections)
        light_curves = {
            "detections": pd.DataFrame(input_detections),
            "non_detections": pd.DataFrame(input_non_detections),
        }
        mock_get_lightcurves.return_value = light_curves
        new_non_detections.at[0, "fid"] = 5
        new_non_detections.at[0, "round_mjd"] = 123
        new_non_detections.at[0, "oid"] = "oid"
        mock_get_prv_candidates.return_value = (
            detections.to_dict("records"),
            new_non_detections.to_dict("records"),
        )
        res = self.step.preprocess_lightcurves(detections, self.alert)
        mock_get_prv_candidates.assert_called()
        self.assertIsInstance(res, dict)
        self.assertEqual(list(res.keys()), ["detections", "non_detections"])

    def test_preprocess_ps1(self):
        metadata = pd.DataFrame(ps1)
        detections = pd.DataFrame(input_detections_not_db)
        res = self.step.preprocess_ps1(metadata, detections)
        self.assertIsInstance(res, pd.DataFrame)

    def test_preprocess_ss(self):
        metadata = pd.DataFrame(ss)
        detections = pd.DataFrame(input_detections_not_db)
        res = self.step.preprocess_ss(metadata, detections)
        self.assertIsInstance(res, pd.DataFrame)

    def test_preprocess_reference(self):
        metadata = pd.DataFrame(reference)
        detections = pd.DataFrame(input_detections_not_db)
        res = self.step.preprocess_reference(metadata, detections)
        self.assertIsInstance(res, pd.DataFrame)

    def test_preprocess_gaia(self):
        metadata = pd.DataFrame(gaia)
        detections = pd.DataFrame(input_detections_not_db)
        res = self.step.preprocess_gaia(metadata, detections)
        self.assertIsInstance(res, pd.DataFrame)

    def test_get_last_alert(self):
        alerts = pd.DataFrame(input_detections_not_db)
        expected = alerts.loc[
            :, ["oid", "ndethist", "ncovhist", "jdstarthist", "jdendhist"]
        ].iloc[0]
        result = self.step.get_last_alert(alerts)
        pd.testing.assert_series_equal(result, expected)

    @mock.patch("correction.step.pd.read_sql")
    def test_get_dataquality(self, mock_read_sql):
        candids = []
        self.step.driver.engine = mock.Mock()
        mock_query = mock.Mock()
        mock_query.statement = "ok"
        self.step.driver.query.return_value.filter.return_value = mock_query
        self.step.get_dataquality(candids)
        self.step.driver.query().filter.assert_called_once()
        mock_read_sql.assert_called_once_with("ok", self.step.driver.engine)

    def test_get_first_corrected(self):
        df = pd.DataFrame({"candid": [1, 2], "corrected": ["ok", "not ok"]})
        res = self.step.get_first_corrected(df)
        self.assertEqual(res, "ok")

    def test_insert_detections(self):
        detections = pd.DataFrame(input_detections)
        self.step.insert_detections(detections)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_non_detections(self):
        non_detections = pd.DataFrame(input_non_detections)
        self.step.insert_non_detections(non_detections)
        self.step.driver.query().bulk_insert.assert_called_once()

    @mock.patch.object(Correction, "get_dataquality")
    def test_insert_dataquality(self, mock_get_dataquality):
        dq = pd.DataFrame(input_dataquality)
        mock_get_dataquality.return_value = dq
        self.step.insert_dataquality(dq)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_objects_new(self):
        objs = pd.DataFrame(output_objects)
        objs.loc[:, ["new"]] = True
        self.step.insert_objects(objs)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_objects_update(self):
        objs = pd.DataFrame(output_objects)
        objs.loc[:, ["new"]] = False
        self.step.driver.engine = mock.Mock()
        self.step.insert_objects(objs)
        self.step.driver.query().bulk_insert.assert_not_called()
        self.step.driver.engine.execute.assert_called_once()

    def test_insert_ss(self):
        ss[0]["new"] = True
        metadata = pd.DataFrame(ss)
        self.step.insert_ss(metadata)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_reference(self):
        reference[0]["new"] = True
        metadata = pd.DataFrame(reference)
        self.step.insert_ss(metadata)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_gaia_new(self):
        metadata = pd.DataFrame(gaia)
        metadata.loc[:, ["new"]] = True
        self.step.insert_ss(metadata)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_gaia_update(self):
        metadata = pd.DataFrame(gaia)
        metadata.loc[:, ["update1"]] = True
        self.step.driver.engine = mock.Mock()
        self.step.insert_gaia(metadata)
        self.step.driver.query().bulk_insert.assert_not_called()
        self.step.driver.engine.execute.assert_called_once()

    def test_insert_gaia_no_update(self):
        metadata = pd.DataFrame(gaia)
        self.step.driver.engine = mock.Mock()
        self.step.insert_gaia(metadata)
        self.step.driver.query().bulk_insert.assert_not_called()
        self.step.driver.engine.execute.assert_not_called()

    def test_insert_ps1_new(self):
        metadata = pd.DataFrame(ps1)
        metadata.loc[:, ["new"]] = True
        self.step.insert_ps1(metadata)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_ps1_update(self):
        metadata = pd.DataFrame(ps1)
        metadata.loc[:, ["update1"]] = True
        metadata.loc[:, ["update2"]] = True
        metadata.loc[:, ["update3"]] = True
        self.step.driver.engine = mock.Mock()
        self.step.insert_gaia(metadata)
        self.step.driver.query().bulk_insert.assert_not_called()
        self.step.driver.engine.execute.assert_called_once()

    def test_insert_ps1_no_update(self):
        metadata = pd.DataFrame(gaia)
        self.step.driver.engine = mock.Mock()
        self.step.insert_gaia(metadata)
        self.step.driver.query().bulk_insert.assert_not_called()
        self.step.driver.engine.execute.assert_not_called()

    @mock.patch.object(Correction, "insert_ss")
    @mock.patch.object(Correction, "insert_reference")
    @mock.patch.object(Correction, "insert_gaia")
    @mock.patch.object(Correction, "insert_ps1")
    def test_insert_metadata(self, mock_ps1, mock_gaia, mock_ref, mock_ss):
        metadata = {
            "ss_ztf": "",
            "reference": "",
            "gaia": "",
            "ps1_ztf": "",
        }
        self.step.insert_metadata(metadata)
        mock_ps1.assert_called_once()
        mock_gaia.assert_called_once()
        mock_ref.assert_called_once()
        mock_ss.assert_called_once()

    def test_insert_magstats_new(self):
        magstats = pd.DataFrame([input_new_stats[0]])
        magstats.loc[:, ["new"]] = True
        self.step.insert_magstats(magstats)
        self.step.driver.query().bulk_insert.assert_called_once()

    def test_insert_magstats_update(self):
        magstats = pd.DataFrame([input_new_stats[0]])
        self.step.driver.engine = mock.Mock()
        self.step.insert_magstats(magstats)
        self.step.driver.engine.execute.assert_called_once()

    def test_produce(self):
        self.alert = pd.read_csv(FILE_PATH + "/alerts.csv")
        self.alert["xmatches"] = [pd.DataFrame()]
        light_curves = {
            "detections": pd.DataFrame(input_detections),
            "non_detections": pd.DataFrame(input_non_detections),
        }
        metadata = {
            "ps1_ztf": pd.DataFrame(ps1),
            "ss_ztf": pd.DataFrame(ss),
            "reference": pd.DataFrame(reference),
            "gaia": pd.DataFrame(gaia),
        }
        self.step.produce(self.alert, light_curves, metadata)
        self.step.producer.produce.assert_called_once()

    def test_flags(self):
        detections = pd.DataFrame(input_detections)
        refs = pd.DataFrame(reference)
        obj_flags, magstat_flags = self.step.do_flags(detections, refs)
        self.assertEqual(list(obj_flags.columns), ["diffpos", "reference_change"])
        self.assertEqual(magstat_flags.name, "saturation_rate")

    # def test_execute(self):
    # self.step.driver.engine = mock.Mock()
    # self.step.start()
