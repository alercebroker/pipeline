import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


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
        tables = Base.metadata.tables

        # Create all tables EXCEPT feature
        tables_to_create = [
            table for table in Base.metadata.tables.values() if table.name != "feature"
        ]
        Base.metadata.create_all(self._engine, tables=tables_to_create)
       
       # If feature exists in models -> create with SQL partitions
        if "feature" in tables:
            self._create_feature_table()
            self.create_feature_partitions()
        
        # Insert static data
        self.insert_initial_data()

    def _create_feature_table(self):
        with self._engine.connect() as conn:
            schema_prefix = f"{self.schema}." if self.schema else ""
            conn.execute(text(f"DROP TABLE IF EXISTS {schema_prefix}feature CASCADE"))

            conn.execute(
                text(f"""
                CREATE TABLE {schema_prefix}feature (
                    oid bigint NOT NULL,
                    sid smallint NOT NULL,
                    feature_id smallint NOT NULL,
                    band smallint NOT NULL,
                    version smallint NOT NULL,
                    value double precision,
                    updated_date date
                ) PARTITION BY HASH (oid)
            """)
            )

            conn.execute(
                text(f"CREATE INDEX ON {schema_prefix}feature USING btree (oid)")
            )

            conn.commit()

    def create_feature_partitions(self):
        """Create the 10 partitions for feature table"""
        with self._engine.connect() as conn:
            schema_prefix = f"{self.schema}." if self.schema else ""

            for i in range(10):
                conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {schema_prefix}feature_part_{i} 
                    PARTITION OF {schema_prefix}feature 
                    FOR VALUES WITH (MODULUS 10, REMAINDER {i})
                """)
                )

            conn.commit()

    def insert_initial_data(self):
        schema = f"{self.schema}." if self.schema else ""
        model_tables = Base.metadata.tables

        with self._engine.connect() as conn:

            # ---------- classifier ----------
            if "classifier" in model_tables:
                conn.execute(text(f"""
                    INSERT INTO {schema}classifier
                    (classifier_id, classifier_name, classifier_version, tid, created_date)
                    VALUES (1, 'stamp_classifier_rubin', '2.0.1', 1, now())
                    ON CONFLICT DO NOTHING
                """))

            # ---------- feature_name_lut ----------
            if "feature_name_lut" in model_tables:
                    conn.execute(text(f"""
                    INSERT INTO {schema}feature_name_lut (feature_name)
                    VALUES
                        ('Amplitude'),
                        ('AndersonDarling'),
                        ('Autocor_length'),
                        ('Beyond1Std'),
                        ('Con'),
                        ('Coordinate_x'),
                        ('Coordinate_y'),
                        ('Coordinate_z'),
                        ('Eta_e'),
                        ('ExcessVar'),
                        ('GP_DRW_sigma'),
                        ('GP_DRW_tau'),
                        ('Gskew'),
                        ('Harmonics_chi'),
                        ('Harmonics_mag_1'),
                        ('Harmonics_mag_2'),
                        ('Harmonics_mag_3'),
                        ('Harmonics_mag_4'),
                        ('Harmonics_mag_5'),
                        ('Harmonics_mag_6'),
                        ('Harmonics_mag_7'),
                        ('Harmonics_mse'),
                        ('Harmonics_phase_2'),
                        ('Harmonics_phase_3'),
                        ('Harmonics_phase_4'),
                        ('Harmonics_phase_5'),
                        ('Harmonics_phase_6'),
                        ('Harmonics_phase_7'),
                        ('IAR_phi'),
                        ('LinearTrend'),
                        ('MHPS_PN_flag'),
                        ('MHPS_high'),
                        ('MHPS_high_30'),
                        ('MHPS_low'),
                        ('MHPS_low_365'),
                        ('MHPS_non_zero'),
                        ('MHPS_ratio'),
                        ('MHPS_ratio_365_30'),
                        ('MaxSlope'),
                        ('Mean'),
                        ('Meanvariance'),
                        ('MedianAbsDev'),
                        ('MedianBRP'),
                        ('Multiband_period'),
                        ('PPE'),
                        ('PairSlopeTrend'),
                        ('PercentAmplitude'),
                        ('Period_band'),
                        ('Power_rate_1/2'),
                        ('Power_rate_1/3'),
                        ('Power_rate_1/4'),
                        ('Power_rate_2'),
                        ('Power_rate_3'),
                        ('Power_rate_4'),
                        ('Psi_CS'),
                        ('Psi_eta'),
                        ('Pvar'),
                        ('Q31'),
                        ('Rcs'),
                        ('SF_ML_amplitude'),
                        ('SF_ML_gamma'),
                        ('SPM_A'),
                        ('SPM_beta'),
                        ('SPM_chi'),
                        ('SPM_gamma'),
                        ('SPM_t0'),
                        ('SPM_tau_fall'),
                        ('SPM_tau_rise'),
                        ('Skew'),
                        ('SmallKurtosis'),
                        ('Std'),
                        ('StetsonK'),
                        ('TDE_decay'),
                        ('TDE_decay_chi'),
                        ('TDE_mag0'),
                        ('Timespan'),
                        ('color_variation'),
                        ('dbrightness_first_det_band'),
                        ('dbrightness_forced_phot_band'),
                        ('delta_period'),
                        ('fleet_a'),
                        ('fleet_chi'),
                        ('fleet_m0'),
                        ('fleet_t0'),
                        ('fleet_w'),
                        ('g-r_max'),
                        ('g-r_max_corr'),
                        ('g-r_mean'),
                        ('g-r_mean_corr'),
                        ('i-z_max'),
                        ('i-z_max_corr'),
                        ('i-z_mean'),
                        ('i-z_mean_corr'),
                        ('last_brightness_before_band'),
                        ('max_brightness_after_band'),
                        ('max_brightness_before_band'),
                        ('median_brightness_after_band'),
                        ('median_brightness_before_band'),
                        ('n_forced_phot_band_after'),
                        ('n_forced_phot_band_before'),
                        ('positive_fraction'),
                        ('r-i_max'),
                        ('r-i_max_corr'),
                        ('r-i_mean'),
                        ('r-i_mean_corr'),
                        ('u-g_max'),
                        ('u-g_max_corr'),
                        ('u-g_mean'),
                        ('u-g_mean_corr'),
                        ('ulens_chi'),
                        ('ulens_fs'),
                        ('ulens_mag0'),
                        ('ulens_t0'),
                        ('ulens_tE'),
                        ('ulens_u0'),
                        ('z-y_max'),
                        ('z-y_max_corr'),
                        ('z-y_mean'),
                        ('z-y_mean_corr')
                        ON CONFLICT DO NOTHING
                    """))

            # ---------- sid_lut ----------
            if "sid_lut" in model_tables:
                conn.execute(text(f"""
                    INSERT INTO {schema}sid_lut (sid, tid, survey_name)
                    VALUES
                        (0, 0, 'ZTF'),
                        (1, 1, 'LSST DIA Object'),
                        (2, 1, 'LSST SS Object')
                    ON CONFLICT DO NOTHING
                """))

            # ---------- taxonomy ----------
            if "taxonomy" in model_tables:
                conn.execute(text(f"""
                    INSERT INTO {schema}taxonomy
                    (class_id, class_name, "order", classifier_id, created_date)
                    VALUES
                        (0, 'SN', 0, 1, now()),
                        (1, 'AGN', 1, 1, now()),
                        (2, 'VS', 2, 1, now()),
                        (3, 'asteroid', 3, 1, now()),
                        (4, 'bogus', 4, 1, now())
                    ON CONFLICT DO NOTHING
                """))

            # ---------- catalog_id_lut ----------
            if "catalog_id_lut" in model_tables:
                conn.execute(text(f"""
                    INSERT INTO {schema}catalog_id_lut
                    (catid, catalog_name, created_date)
                    VALUES (0, 'AllWISE', now())
                    ON CONFLICT DO NOTHING
                """))

            # ---------- band ----------
            if "band" in model_tables:
                conn.execute(text(f"""
                    INSERT INTO {schema}band (sid, tid, band, band_name, "order") VALUES
                    (1,1,1,'g',1),(1,1,2,'r',2),(1,1,3,'i',3),(1,1,4,'z',4),(1,1,5,'y',5),(1,1,6,'u',0),
                    (2,1,1,'g',1),(2,1,2,'r',2),(2,1,3,'i',3),(2,1,4,'z',4),(2,1,5,'y',5),(2,1,6,'u',0),
                    (0,0,1,'g',0),(0,0,2,'r',1),(0,0,3,'i',2)
                    ON CONFLICT DO NOTHING
                """))

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
