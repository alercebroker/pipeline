import unittest
import datetime
from db_plugins.db.sql import SQLQuery
from unittest import mock
from late_classification.step import (
    LateClassifier,
    SQLConnection,
    KafkaProducer,
    pd,
    np,
    Probability,
    Taxonomy,
    HierarchicalRandomForest,
)

CORRECT_MESSAGE = {
    "oid": "ZTF18absgqth",
    "candid": "987687467236",
    "features": {
        "Amplitude_1": 0.32924321693078085,
        "Amplitude_2": 0.2887461326078053,
        "AndersonDarling_1": 0.9999999908478816,
        "AndersonDarling_2": 0.9999919027640196,
        "Autocor_length_1": 1.0,
        "Autocor_length_2": 1.0,
        "Beyond1Std_1": 0.3225806451612903,
        "Beyond1Std_2": 0.25,
        "Con_1": 0.0,
        "Con_2": 0.0,
        "Eta_e_1": 2.4062351274293783,
        "Eta_e_2": 3.0260537479524237,
        "ExcessVar_1": 0.00026635467562440517,
        "ExcessVar_2": 0.0002004830658447554,
        "GP_DRW_sigma_1": 0.052668477522575724,
        "GP_DRW_sigma_2": 0.034923534266397625,
        "GP_DRW_tau_1": 0.18727198958676827,
        "GP_DRW_tau_2": 0.07279946176499806,
        "Gskew_1": 0.4451775411749885,
        "Gskew_2": 0.36558160417583707,
        "Harmonics_mag_1_1": 0.2388458991170795,
        "Harmonics_mag_1_2": 0.3520269940691187,
        "Harmonics_mag_2_1": 0.11788899690990833,
        "Harmonics_mag_2_2": 0.2729752885726638,
        "Harmonics_mag_3_1": 0.07177880793656359,
        "Harmonics_mag_3_2": 0.15547607815927714,
        "Harmonics_mag_4_1": 0.0604746829568916,
        "Harmonics_mag_4_2": 0.18509812070967824,
        "Harmonics_mag_5_1": 0.08030396885426971,
        "Harmonics_mag_5_2": 0.09493966457582263,
        "Harmonics_mag_6_1": 0.0193574249760217,
        "Harmonics_mag_6_2": 0.10395047742205044,
        "Harmonics_mag_7_1": 0.015013680983202044,
        "Harmonics_mag_7_2": 0.07990341189753583,
        "Harmonics_mse_1": 0.0012812932233470758,
        "Harmonics_mse_2": 0.0007968846853428921,
        "Harmonics_phase_2_1": 0.08977580806492114,
        "Harmonics_phase_2_2": 4.799379089214803,
        "Harmonics_phase_3_1": 5.898223691839025,
        "Harmonics_phase_3_2": 2.2770331053269928,
        "Harmonics_phase_4_1": 4.584130356555323,
        "Harmonics_phase_4_2": 6.2185520140990445,
        "Harmonics_phase_5_1": 3.9873045722925733,
        "Harmonics_phase_5_2": 4.264309243298461,
        "Harmonics_phase_6_1": 4.4501641677642,
        "Harmonics_phase_6_2": 1.2005746832550557,
        "Harmonics_phase_7_1": 1.3875755800617142,
        "Harmonics_phase_7_2": 5.330388690766871,
        "IAR_phi_1": 6.432183077305056e-6,
        "IAR_phi_2": 6.021135259639006e-13,
        "LinearTrend_1": 0.00012226741858221328,
        "LinearTrend_2": 0.0006092037306233967,
        "MHPS_high_1": 9.471399198439084e-5,
        "MHPS_high_2": 5.099509471731534e-5,
        "MHPS_low_1": 0.0005241389849902446,
        "MHPS_low_2": 0.00015968785903494644,
        "MHPS_ratio_1": 5.533912931012606,
        "MHPS_ratio_2": 3.1314356786697872,
        "MaxSlope_1": 0.1971761802126926,
        "MaxSlope_2": 0.3028004484933754,
        "Meanvariance_1": 0.016337470525860497,
        "Meanvariance_2": 0.014178066244463541,
        "MedianAbsDev_1": 0.08039927999203478,
        "MedianAbsDev_2": 0.058831280818933074,
        "MedianBRP_1": 0.3548387096774194,
        "MedianBRP_2": 0.5,
        "Multiband_period": 0.13331908484257887,
        "PairSlopeTrend_1": 0.16666666666666666,
        "PairSlopeTrend_2": 0.03333333333333333,
        "PercentAmplitude_1": 0.04011843385547042,
        "PercentAmplitude_2": 0.035932341174300934,
        "Period_fit": 0.07701353755504614,
        "Power_rate_1/2": 0.03609991446137428,
        "Power_rate_1/3": 0.007769198156893253,
        "Power_rate_1/4": 0.007769198156893253,
        "Power_rate_2": 0.005786219146102667,
        "Power_rate_3": 0.013031342998147009,
        "Power_rate_4": 0.01568520814180374,
        "Psi_CS_1": 0.3891123990859871,
        "Psi_CS_2": 0.2862597426223811,
        "Psi_eta_1": 0.41710647543525176,
        "Psi_eta_2": 0.6498566105447072,
        "Pvar_1": 1.0,
        "Pvar_2": 1.0,
        "Q31_1": 0.4097436438158564,
        "Q31_2": 0.2945798961902675,
        "Rcs_1": 0.17268764659202648,
        "Rcs_2": 0.31992834003333354,
        "SF_ML_amplitude_1": 0.34389173162269665,
        "SF_ML_amplitude_2": 0.3264617286683304,
        "SF_ML_gamma_1": 0.017523248903947394,
        "SF_ML_gamma_2": 0.11175310675929083,
        "SPM_A_1": 5.7383604956940415,
        "SPM_A_2": 10.2368931253294,
        "SPM_beta_1": 62.47753589190978,
        "SPM_beta_2": 90.33728614508469,
        "SPM_chi_1": 506.31687602877986,
        "SPM_chi_2": 700.9838770453348,
        "SPM_gamma_1": 0.6387500201289004,
        "SPM_gamma_2": 0.5566182550254151,
        "SPM_t0_1": 3.9362452744709437,
        "SPM_t0_2": -9.860364412576383,
        "SPM_tau_fall_1": 99.43073734289386,
        "SPM_tau_fall_2": 93.96436061620359,
        "SPM_tau_rise_1": 31.811591277683412,
        "SPM_tau_rise_2": 99.99999999198751,
        "Skew_1": 0.5714406860410475,
        "Skew_2": 0.6775419847132063,
        "SmallKurtosis_1": -1.1286844981658832,
        "SmallKurtosis_2": -0.7695963461241306,
        "Std_1": 0.23005787218841425,
        "Std_2": 0.18759658935686968,
        "StetsonK_1": 0.8596496606105944,
        "StetsonK_2": 0.8943936730805809,
        "delta_mag_fid_1": 0.6736381495372168,
        "delta_mag_fid_2": 0.5774922652156107,
        "dmag_first_det_fid_1": 6.4846327145106,
        "dmag_first_det_fid_2": np.nan,
        "dmag_non_det_fid_1": 6.965020569435853,
        "dmag_non_det_fid_2": np.nan,
        "g-r_max": 0.5449085235595703,
        "g-r_max_corr": 0.8127238425218248,
        "g-r_mean": 0.6979525673773992,
        "g-r_mean_corr": 0.8501438654952036,
        "gal_b": -58.297109913080384,
        "gal_l": 68.5428266293821,
        "last_diffmaglim_before_fid_1": 20.645299911499023,
        "last_diffmaglim_before_fid_2": np.nan,
        "max_diffmaglim_after_fid_1": 20.846200942993164,
        "max_diffmaglim_after_fid_2": 20.491199493408203,
        "max_diffmaglim_before_fid_1": 20.859699249267575,
        "max_diffmaglim_before_fid_2": np.nan,
        "median_diffmaglim_after_fid_1": 20.22769927978516,
        "median_diffmaglim_after_fid_2": 19.902999877929688,
        "median_diffmaglim_before_fid_1": 20.794700622558594,
        "median_diffmaglim_before_fid_2": np.nan,
        "n_non_det_after_fid_1": 22.0,
        "n_non_det_after_fid_2": 27.0,
        "n_non_det_before_fid_1": 3.0,
        "n_non_det_before_fid_2": 0.0,
        "positive_fraction_1": 0.5806451612903226,
        "positive_fraction_2": 0.6,
        "rb": 0.7133333000000001,
        "sgscore1": 1.0,
        "W1-W2": -0.04600000000000115,
        "W2-W3": 0.13700000000000045,
        "r-W3": 2.301465146392946,
        "r-W2": 2.1644651463929456,
        "g-W3": 3.1516090118881515,
        "g-W2": 3.014609011888151,
        "delta_period_1": 1.8052619264463665e-6,
        "delta_period_2": 2.5425884869756388e-8,
    },
}

PREDICTION = {
    "hierarchical": {
        "top": {"Periodic": 0.998, "Stochastic": 0.002, "Transient": 0.0},
        "children": {
            "Stochastic": {
                "AGN": 0.004,
                "Blazar": 0.014,
                "CV/Nova": 0.712,
                "QSO": 0.0,
                "YSO": 0.27,
            },
            "Periodic": {
                "CEP": 0.17,
                "DSCT": 0.056,
                "E": 0.698,
                "LPV": 0.0,
                "Periodic-Other": 0.06,
                "RRL": 0.016,
            },
            "Transient": {"SLSN": 0.33, "SNII": 0.3, "SNIa": 0.182, "SNIbc": 0.188},
        },
    },
    "probabilities": {
        "AGN": 8e-06,
        "Blazar": 2.8e-05,
        "CV/Nova": 0.001424,
        "QSO": 0.0,
        "YSO": 0.00054,
        "CEP": 0.16966,
        "DSCT": 0.055888,
        "E": 0.696604,
        "LPV": 0.0,
        "Periodic-Other": 0.059879999999999996,
        "RRL": 0.015968,
        "SLSN": 0.0,
        "SNII": 0.0,
        "SNIa": 0.0,
        "SNIbc": 0.0,
    },
    "class": "E",
}

class MockSession:
    def commit(self):
        pass

    def add(self, model):
        pass

    def query(self):
        return mock.create_autospec(SQLQuery)()

class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "DB_CONFIG": {"SQL": {}},
            "PRODUCER_CONFIG": {"fake": "fake"},
            "STEP_METADATA": {
                "STEP_ID": "",
                "STEP_NAME": "",
                "STEP_VERSION": "",
                "STEP_COMMENTS": "",
                "CLASSIFIER_VERSION": "test",
                "CLASSIFIER_NAME": "hrf_test",
            },
        }
        self.mock_database_connection = mock.create_autospec(SQLConnection)
        self.mock_database_connection.session = mock.create_autospec(MockSession)
        self.mock_producer = mock.create_autospec(KafkaProducer)
        self.mock_model = mock.create_autospec(HierarchicalRandomForest)
        self.mock_model.feature_list = CORRECT_MESSAGE["features"].keys()
        self.step = LateClassifier(
            config=self.step_config,
            db_connection=self.mock_database_connection,
            producer=self.mock_producer,
            model=self.mock_model,
            test_mode=True,
        )

    def tearDown(self):
        del self.step

    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.step.driver.query().get_or_create.assert_called_once()

    def test_format_features(self):
        self.assertEqual(dict, type(CORRECT_MESSAGE["features"]))
        formatted = self.step._format_features(CORRECT_MESSAGE["features"])
        self.assertEqual(pd.DataFrame, type(formatted))

    def test_get_ranking(self):
        known_ranking = [9, 8, 6, 10, 7, 2, 4, 1, 10, 3, 5, 10, 10, 10, 10]
        ranking = self.step.get_ranking(PREDICTION["probabilities"])
        self.assertEqual(list(ranking), known_ranking)

    def test_get_probability(self):
        self.mock_database_connection.session.query().get_or_create.return_value = (
            "instance",
            "created",
        )
        instance, created = self.step.get_probability(
            "oid", "class", "classifier", "version", "proba", "ranking"
        )
        self.mock_database_connection.session.query().get_or_create.assert_called_with(
            Probability,
            filter_by={
                "oid": "oid",
                "class_name": "class",
                "classifier_name": "classifier",
                "classifier_version": "version",
            },
            probability="proba",
            ranking="ranking",
        )
        self.assertEqual(instance, "instance")
        self.assertEqual(created, "created")

    def test_set_taxonomy(self):
        self.mock_database_connection.session.query().get_or_create.return_value = (
            "instance",
            "created",
        )
        instance, created = self.step.set_taxonomy([], "classifier", "version")
        self.mock_database_connection.session.query().get_or_create.assert_called_with(
            Taxonomy,
            filter_by={
                "classifier_name": "classifier",
                "classifier_version": "version",
            },
            classes=[],
        )
        self.assertEqual(instance, "instance")
        self.assertEqual(created, "created")

    def test_insert_dict(self):
        pass

    def test_insert_db(self):
        pass

    @mock.patch.object(LateClassifier, "insert_db")
    def test_execute_missing_features(self, mock_insert):
        self.step.features_required = set("not_empty")
        self.step.execute(CORRECT_MESSAGE)
        self.step.model.predict_in_pipeline.assert_not_called()
        mock_insert.assert_not_called()
        self.step.producer.produce.assert_not_called()

    @mock.patch.object(LateClassifier, "insert_db")
    def test_execute_no_missing_features(self, mock_insert):
        self.step.model.predict_in_pipeline.return_value = PREDICTION
        self.step.execute(CORRECT_MESSAGE)
        self.step.model.predict_in_pipeline.assert_called()
        mock_insert.assert_called_with(PREDICTION, CORRECT_MESSAGE["oid"])
        new_message = {
            "oid": CORRECT_MESSAGE["oid"],
            "candid": CORRECT_MESSAGE["candid"],
            "features": CORRECT_MESSAGE["features"],
            "late_classification": PREDICTION,
        }
        self.step.producer.produce.assert_called_with(new_message, key=new_message["oid"])
