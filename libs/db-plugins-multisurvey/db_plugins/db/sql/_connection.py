import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from .models import Band, Base, CatalogIdLut, Classifier, FeatureNameLut, Taxonomy


def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


logger = logging.getLogger(__name__)


class PsqlDatabase:
    def __init__(self, db_config: dict, engine=None) -> None:
        db_url = get_db_url(db_config)
        schema = db_config.get("SCHEMA", None)
        self.schema = schema
        if schema:
            self._engine = engine or create_engine(
                db_url,
                echo=False,
                connect_args={"options": "-csearch_path={}".format(schema)},
            )
        else:
            self._engine = engine or create_engine(db_url, echo=False)

        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    def create_db(self):
        Base.metadata.create_all(self._engine)

        with self._engine.connect() as conn:
            for mapper in Base.registry.mappers:
                table_class = mapper.class_
                if table_class.__n_partitions__ is not None:
                    table_class.__create_partitions__(conn, self.schema)

        self.insert_initial_data()

    def insert_initial_data(self):
        model_tables = Base.metadata.tables

        with self._engine.connect() as conn:
            # ---------- classifier ----------
            if "classifier" in model_tables:
                stmt = (
                    insert(Classifier)
                    .values([
                        {
                            "classifier_id": 1,
                            "classifier_name": "stamp_classifier_rubin",
                            "classifier_version": "2.0.1",
                            "tid": 1,
                        },
                        {
                            "classifier_id": 2,
                            "classifier_name": "stamp_classifier_2025_beta",
                            "classifier_version": "2.1.1",
                            "tid": 0,
                        }
                    ])
                    .on_conflict_do_nothing(index_elements=["classifier_id"])
                )
                conn.execute(stmt)

            # ---------- feature_name_lut ----------
            if "feature_name_lut" in model_tables:
                feature_data = [
                    {"feature_id": 0, "feature_name": "Amplitude"},
                    {"feature_id": 1, "feature_name": "AndersonDarling"},
                    {"feature_id": 2, "feature_name": "Autocor_length"},
                    {"feature_id": 3, "feature_name": "Beyond1Std"},
                    {"feature_id": 4, "feature_name": "Con"},
                    {"feature_id": 5, "feature_name": "Coordinate_x"},
                    {"feature_id": 6, "feature_name": "Coordinate_y"},
                    {"feature_id": 7, "feature_name": "Coordinate_z"},
                    {"feature_id": 8, "feature_name": "Eta_e"},
                    {"feature_id": 9, "feature_name": "ExcessVar"},
                    {"feature_id": 10, "feature_name": "GP_DRW_sigma"},
                    {"feature_id": 11, "feature_name": "GP_DRW_tau"},
                    {"feature_id": 12, "feature_name": "Gskew"},
                    {"feature_id": 13, "feature_name": "Harmonics_chi"},
                    {"feature_id": 14, "feature_name": "Harmonics_mag_1"},
                    {"feature_id": 15, "feature_name": "Harmonics_mag_2"},
                    {"feature_id": 16, "feature_name": "Harmonics_mag_3"},
                    {"feature_id": 17, "feature_name": "Harmonics_mag_4"},
                    {"feature_id": 18, "feature_name": "Harmonics_mag_5"},
                    {"feature_id": 19, "feature_name": "Harmonics_mag_6"},
                    {"feature_id": 20, "feature_name": "Harmonics_mag_7"},
                    {"feature_id": 21, "feature_name": "Harmonics_mse"},
                    {"feature_id": 22, "feature_name": "Harmonics_phase_2"},
                    {"feature_id": 23, "feature_name": "Harmonics_phase_3"},
                    {"feature_id": 24, "feature_name": "Harmonics_phase_4"},
                    {"feature_id": 25, "feature_name": "Harmonics_phase_5"},
                    {"feature_id": 26, "feature_name": "Harmonics_phase_6"},
                    {"feature_id": 27, "feature_name": "Harmonics_phase_7"},
                    {"feature_id": 28, "feature_name": "IAR_phi"},
                    {"feature_id": 29, "feature_name": "LinearTrend"},
                    {"feature_id": 30, "feature_name": "MHPS_PN_flag"},
                    {"feature_id": 31, "feature_name": "MHPS_high"},
                    {"feature_id": 32, "feature_name": "MHPS_high_30"},
                    {"feature_id": 33, "feature_name": "MHPS_low"},
                    {"feature_id": 34, "feature_name": "MHPS_low_365"},
                    {"feature_id": 35, "feature_name": "MHPS_non_zero"},
                    {"feature_id": 36, "feature_name": "MHPS_ratio"},
                    {"feature_id": 37, "feature_name": "MHPS_ratio_365_30"},
                    {"feature_id": 38, "feature_name": "MaxSlope"},
                    {"feature_id": 39, "feature_name": "Mean"},
                    {"feature_id": 40, "feature_name": "Meanvariance"},
                    {"feature_id": 41, "feature_name": "MedianAbsDev"},
                    {"feature_id": 42, "feature_name": "MedianBRP"},
                    {"feature_id": 43, "feature_name": "Multiband_period"},
                    {"feature_id": 44, "feature_name": "PPE"},
                    {"feature_id": 45, "feature_name": "PairSlopeTrend"},
                    {"feature_id": 46, "feature_name": "PercentAmplitude"},
                    {"feature_id": 47, "feature_name": "Period_band"},
                    {"feature_id": 48, "feature_name": "Power_rate_1/2"},
                    {"feature_id": 49, "feature_name": "Power_rate_1/3"},
                    {"feature_id": 50, "feature_name": "Power_rate_1/4"},
                    {"feature_id": 51, "feature_name": "Power_rate_2"},
                    {"feature_id": 52, "feature_name": "Power_rate_3"},
                    {"feature_id": 53, "feature_name": "Power_rate_4"},
                    {"feature_id": 54, "feature_name": "Psi_CS"},
                    {"feature_id": 55, "feature_name": "Psi_eta"},
                    {"feature_id": 56, "feature_name": "Pvar"},
                    {"feature_id": 57, "feature_name": "Q31"},
                    {"feature_id": 58, "feature_name": "Rcs"},
                    {"feature_id": 59, "feature_name": "SF_ML_amplitude"},
                    {"feature_id": 60, "feature_name": "SF_ML_gamma"},
                    {"feature_id": 61, "feature_name": "SPM_A"},
                    {"feature_id": 62, "feature_name": "SPM_beta"},
                    {"feature_id": 63, "feature_name": "SPM_chi"},
                    {"feature_id": 64, "feature_name": "SPM_gamma"},
                    {"feature_id": 65, "feature_name": "SPM_t0"},
                    {"feature_id": 66, "feature_name": "SPM_tau_fall"},
                    {"feature_id": 67, "feature_name": "SPM_tau_rise"},
                    {"feature_id": 68, "feature_name": "Skew"},
                    {"feature_id": 69, "feature_name": "SmallKurtosis"},
                    {"feature_id": 70, "feature_name": "Std"},
                    {"feature_id": 71, "feature_name": "StetsonK"},
                    {"feature_id": 72, "feature_name": "TDE_decay"},
                    {"feature_id": 73, "feature_name": "TDE_decay_chi"},
                    {"feature_id": 74, "feature_name": "TDE_mag0"},
                    {"feature_id": 75, "feature_name": "Timespan"},
                    {"feature_id": 76, "feature_name": "color_variation"},
                    {"feature_id": 77, "feature_name": "dbrightness_first_det_band"},
                    {"feature_id": 78, "feature_name": "dbrightness_forced_phot_band"},
                    {"feature_id": 79, "feature_name": "delta_period"},
                    {"feature_id": 80, "feature_name": "fleet_a"},
                    {"feature_id": 81, "feature_name": "fleet_chi"},
                    {"feature_id": 82, "feature_name": "fleet_m0"},
                    {"feature_id": 83, "feature_name": "fleet_t0"},
                    {"feature_id": 84, "feature_name": "fleet_w"},
                    {"feature_id": 85, "feature_name": "g-r_max"},
                    {"feature_id": 86, "feature_name": "g-r_max_corr"},
                    {"feature_id": 87, "feature_name": "g-r_mean"},
                    {"feature_id": 88, "feature_name": "g-r_mean_corr"},
                    {"feature_id": 89, "feature_name": "i-z_max"},
                    {"feature_id": 90, "feature_name": "i-z_max_corr"},
                    {"feature_id": 91, "feature_name": "i-z_mean"},
                    {"feature_id": 92, "feature_name": "i-z_mean_corr"},
                    {"feature_id": 93, "feature_name": "last_brightness_before_band"},
                    {"feature_id": 94, "feature_name": "max_brightness_after_band"},
                    {"feature_id": 95, "feature_name": "max_brightness_before_band"},
                    {"feature_id": 96, "feature_name": "median_brightness_after_band"},
                    {"feature_id": 97, "feature_name": "median_brightness_before_band"},
                    {"feature_id": 98, "feature_name": "n_forced_phot_band_after"},
                    {"feature_id": 99, "feature_name": "n_forced_phot_band_before"},
                    {"feature_id": 100, "feature_name": "positive_fraction"},
                    {"feature_id": 101, "feature_name": "r-i_max"},
                    {"feature_id": 102, "feature_name": "r-i_max_corr"},
                    {"feature_id": 103, "feature_name": "r-i_mean"},
                    {"feature_id": 104, "feature_name": "r-i_mean_corr"},
                    {"feature_id": 105, "feature_name": "u-g_max"},
                    {"feature_id": 106, "feature_name": "u-g_max_corr"},
                    {"feature_id": 107, "feature_name": "u-g_mean"},
                    {"feature_id": 108, "feature_name": "u-g_mean_corr"},
                    {"feature_id": 109, "feature_name": "ulens_chi"},
                    {"feature_id": 110, "feature_name": "ulens_fs"},
                    {"feature_id": 111, "feature_name": "ulens_mag0"},
                    {"feature_id": 112, "feature_name": "ulens_t0"},
                    {"feature_id": 113, "feature_name": "ulens_tE"},
                    {"feature_id": 114, "feature_name": "ulens_u0"},
                    {"feature_id": 115, "feature_name": "z-y_max"},
                    {"feature_id": 116, "feature_name": "z-y_max_corr"},
                    {"feature_id": 117, "feature_name": "z-y_mean"},
                    {"feature_id": 118, "feature_name": "z-y_mean_corr"},
                ]

                stmt = (
                    insert(FeatureNameLut)
                    .values(feature_data)
                    .on_conflict_do_nothing(index_elements=["feature_id"])
                )
                conn.execute(stmt)

            # ---------- sid_lut ----------
            if "sid_lut" in model_tables:
                sid_lut_data = [
                    {"sid": 0, "tid": 0, "survey_name": "ZTF"},
                    {"sid": 1, "tid": 1, "survey_name": "LSST DIA Object"},
                    {"sid": 2, "tid": 1, "survey_name": "LSST SS Object"},
                ]
                stmt = (
                    insert(model_tables["sid_lut"])
                    .values(sid_lut_data)
                    .on_conflict_do_nothing(index_elements=["sid"])
                )
                conn.execute(stmt)

            # ---------- taxonomy ----------
            if "taxonomy" in model_tables:
                taxonomy_data = [
                    {
                        "class_id": 0,
                        "class_name": "SN",
                        "order": 0,
                        "classifier_id": 1,
                    },
                    {
                        "class_id": 1,
                        "class_name": "AGN",
                        "order": 1,
                        "classifier_id": 1,
                    },
                    {
                        "class_id": 2,
                        "class_name": "VS",
                        "order": 2,
                        "classifier_id": 1,
                    },
                    {
                        "class_id": 3,
                        "class_name": "asteroid",
                        "order": 3,
                        "classifier_id": 1,
                    },
                    {
                        "class_id": 4,
                        "class_name": "bogus",
                        "order": 4,
                        "classifier_id": 1,
                    },
                    {
                        "class_id": 5,
                        "class_name": "SN",
                        "order": 5,
                        "classifier_id": 2,
                    },
                    {
                        "class_id": 6,
                        "class_name": "AGN",
                        "order": 6,
                        "classifier_id": 2,
                    },
                    {
                        "class_id": 7,
                        "class_name": "VS",
                        "order": 7,
                        "classifier_id": 2,
                    },
                    {
                        "class_id": 8,
                        "class_name": "asteroid",
                        "order": 8,
                        "classifier_id": 2,
                    },
                    {
                        "class_id": 9,
                        "class_name": "bogus",
                        "order": 9,
                        "classifier_id": 2,
                    },
                    {
                        "class_id": 10,
                        "class_name": "satellite",
                        "order": 10,
                        "classifier_id": 2,
                    },
                ]
                stmt = (
                    insert(Taxonomy)
                    .values(taxonomy_data)
                    .on_conflict_do_nothing(index_elements=["class_id"])
                )
                conn.execute(stmt)

            # ---------- catalog_id_lut ----------
            if "catalog_id_lut" in model_tables:
                stmt = (
                    insert(CatalogIdLut)
                    .values({"catid": 0, "catalog_name": "AllWISE"})
                    .on_conflict_do_nothing(index_elements=["catid"])
                )
                conn.execute(stmt)

            # ---------- band ----------
            if "band" in model_tables:
                band_data = [
                    {"sid": 1, "tid": 1, "band": 1, "band_name": "g", "order": 1},
                    {"sid": 1, "tid": 1, "band": 2, "band_name": "r", "order": 2},
                    {"sid": 1, "tid": 1, "band": 3, "band_name": "i", "order": 3},
                    {"sid": 1, "tid": 1, "band": 4, "band_name": "z", "order": 4},
                    {"sid": 1, "tid": 1, "band": 5, "band_name": "y", "order": 5},
                    {"sid": 1, "tid": 1, "band": 6, "band_name": "u", "order": 0},
                    {"sid": 2, "tid": 1, "band": 1, "band_name": "g", "order": 1},
                    {"sid": 2, "tid": 1, "band": 2, "band_name": "r", "order": 2},
                    {"sid": 2, "tid": 1, "band": 3, "band_name": "i", "order": 3},
                    {"sid": 2, "tid": 1, "band": 4, "band_name": "z", "order": 4},
                    {"sid": 2, "tid": 1, "band": 5, "band_name": "y", "order": 5},
                    {"sid": 2, "tid": 1, "band": 6, "band_name": "u", "order": 0},
                    {"sid": 0, "tid": 0, "band": 1, "band_name": "g", "order": 0},
                    {"sid": 0, "tid": 0, "band": 2, "band_name": "r", "order": 1},
                    {"sid": 0, "tid": 0, "band": 3, "band_name": "i", "order": 2},
                ]
                stmt = (
                    insert(Band)
                    .values(band_data)
                    .on_conflict_do_nothing(index_elements=["sid", "tid", "band"])
                )
                conn.execute(stmt)

            conn.commit()

    def drop_db(self):
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self):
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            logger.exception("Session rollback because of exception")
            logger.exception(e)
            session.rollback()
            raise Exception(e)
        finally:
            session.close()
