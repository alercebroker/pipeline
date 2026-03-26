import json
import os
import unittest

import pytest
from apf.producers.kafka import KafkaProducer
from db_plugins.db.sql._connection import PsqlDatabase
from sqlalchemy import text

from sql_scribe.step import SqlScribe

DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    }
}

PSQL_CONFIG_PGBOUNCER = {
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5433,
    "DB_NAME": "postgres",
    "POOLCLASS": "NullPool",
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": ["test_topic_sql"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_3",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 3,
    "TIMEOUT": 1,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic_sql",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
    "SCHEMA_PATH": os.path.join(os.path.dirname(__file__), "producer_schema.avsc"),
}


def _make_ztf_detection(measurement_id: int) -> dict:
    """Return a complete ZTF detection dict suitable for ZTFCorrectionCommand tests."""
    return {
        "measurement_id": measurement_id,
        "new": True,
        "sid": 1,
        "mjd": 59000.0,
        "ra": 45.0,
        "dec": 45.0,
        "band": 1,
        "pid": measurement_id,
        "diffmaglim": 20.5,
        "isdiffpos": 1,
        "nid": 1,
        "mag": 18.5,
        "e_mag": 0.05,
        "magap": 18.6,
        "sigmagap": 0.06,
        "distnr": 0.1,
        "rb": 0.95,
        "rbversion": "t17_f5_c3",
        "drb": 0.99,
        "drbversion": "d6_m7",
        "magapbig": 18.7,
        "sigmagapbig": 0.07,
        "rfid": 801225900,
        "magpsf_corr": 18.4,
        "sigmapsf_corr": 0.04,
        "sigmapsf_corr_ext": 0.03,
        "corrected": True,
        "dubious": False,
        "parent_candid": None,
        "has_stamp": True,
        # ZtfObject update fields
        "ndethist": 5,
        "ncovhist": 10,
        "jdstarthist": 2458000.5,
        "jdendhist": 2459000.5,
        # PS1 crossmatch
        "objectidps1": "123456789",
        "sgmag1": 18.5, "srmag1": 18.5, "simag1": 18.5, "szmag1": 18.5,
        "sgscore1": 0.7, "distpsnr1": 0.5,
        "objectidps2": "234567890",
        "sgmag2": 19.0, "srmag2": 19.0, "simag2": 19.0, "szmag2": 19.0,
        "sgscore2": 0.5, "distpsnr2": 1.5,
        "objectidps3": "345678901",
        "sgmag3": 19.5, "srmag3": 19.5, "simag3": 19.5, "szmag3": 19.5,
        "sgscore3": 0.3, "distpsnr3": 2.5,
        "nmtchps": 3,
        # SS
        "ssdistnr": 0.5,
        "ssmagnr": 15.0,
        "ssnamenr": "null",
        # Gaia
        "neargaia": 0.3,
        "neargaiabright": 0.5,
        "maggaia": 17.0,
        "maggaiabright": 15.0,
        # Data quality
        "xpos": 500.0, "ypos": 500.0, "chipsf": 1.5, "sky": 100.0,
        "fwhm": 2.5, "classtar": 0.9, "mindtoedge": 100.0,
        "seeratio": 1.0, "aimage": 1.0, "bimage": 1.0,
        "aimagerat": 1.0, "bimagerat": 1.0, "nneg": 0, "nbad": 0,
        "sumrat": 1.0, "scorr": 10.0, "dsnrms": 1.0, "ssnrms": 1.0,
        "magzpsci": 26.0, "magzpsciunc": 0.01, "magzpscirms": 0.01,
        "nmatches": 50, "clrcoeff": 0.1, "clrcounc": 0.01,
        "zpclrcov": 0.001, "zpmed": 26.0, "clrmed": 0.1, "clrrms": 0.01,
        "exptime": 30.0,
        # Reference
        "rcid": 25, "field": 534, "magnr": 17.0, "sigmagnr": 0.05,
        "chinr": 1.0, "sharpnr": 0.1, "ranr": 45.0, "decnr": 45.0,
        "jdstartref": 2458000.5, "jdendref": 2459000.5, "nframesref": 15,
    }


@pytest.mark.usefixtures("pgbouncer_service")
@pytest.mark.usefixtures("kafka_service")
class PsqlIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = PsqlDatabase(DB_CONFIG["PSQL"])
        cls.db.create_db()
        step_config = {
            "PSQL_CONFIG": PSQL_CONFIG_PGBOUNCER,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
        }
        cls.step = SqlScribe(config=step_config)
        cls.producer = KafkaProducer(config=PRODUCER_CONFIG)

    @classmethod
    @classmethod
    def tearDownClass(cls):
        cls.db.drop_db()

    # ------------------------------------------------------------------ helpers

    def _insert_object(self, oid: int, sid: int = 1, tid: int = 0):
        """Insert a minimal object row needed for UPDATE-based commands."""
        with self.db.session() as session:
            session.execute(
                text(
                    "INSERT INTO object(oid, tid, sid, meanra, meandec, firstmjd, "
                    "lastmjd, deltamjd, n_det, n_forced, n_non_det) "
                    "VALUES (:oid, :tid, :sid, 45.0, 45.0, 59000.0, 59001.0, 1.0, 1, 0, 0) "
                    "ON CONFLICT DO NOTHING"
                ),
                {"oid": oid, "tid": tid, "sid": sid},
            )
            session.commit()

    def _insert_ztf_object(self, oid: int):
        """Insert a minimal ztf_object row so ZTFCorrectionCommand can UPDATE it."""
        with self.db.session() as session:
            session.execute(
                text(
                    "INSERT INTO ztf_object(oid, corrected, ndethist, ncovhist) "
                    "VALUES (:oid, false, 1, 1) ON CONFLICT DO NOTHING"
                ),
                {"oid": oid},
            )
            session.commit()

    def _produce_command(self, survey: str, step: str, payload: dict):
        self.producer.produce(
            {"payload": json.dumps({"survey": survey, "step": step, "payload": payload})}
        )

    # --------------------------------------------------------- ZTFCorrectionCommand

    def test_ztf_correction_inserts_detection_and_ztf_detection(self):
        oid = 100001
        measurement_id = 1000010001
        self._insert_object(oid, sid=1)
        self._insert_ztf_object(oid)

        det = _make_ztf_detection(measurement_id)
        self._produce_command(
            "ztf",
            "correction",
            {
                "oid": oid,
                "measurement_id": [measurement_id],
                "detections": [det],
                "previous_detections": [],
                "forced_photometries": [],
            },
        )
        self.step.start()

        with self.db.session() as session:
            detections = session.execute(
                text("SELECT measurement_id FROM detection WHERE oid = :oid"),
                {"oid": oid},
            ).fetchall()
            ztf_detections = session.execute(
                text("SELECT measurement_id FROM ztf_detection WHERE oid = :oid"),
                {"oid": oid},
            ).fetchall()

        assert len(detections) == 1
        assert detections[0][0] == measurement_id
        assert len(ztf_detections) == 1
        assert ztf_detections[0][0] == measurement_id

    def test_ztf_correction_updates_ztf_object_fields(self):
        oid = 100002
        measurement_id = 1000020001
        self._insert_object(oid, sid=1)
        self._insert_ztf_object(oid)

        det = _make_ztf_detection(measurement_id)
        self._produce_command(
            "ztf",
            "correction",
            {
                "oid": oid,
                "measurement_id": [measurement_id],
                "detections": [det],
                "previous_detections": [],
                "forced_photometries": [],
            },
        )
        self.step.start()

        with self.db.session() as session:
            row = session.execute(
                text("SELECT ndethist, ncovhist FROM ztf_object WHERE oid = :oid"),
                {"oid": oid},
            ).fetchone()

        assert row is not None
        # _make_ztf_detection sets ndethist=5, ncovhist=10
        assert row[0] == 5
        assert row[1] == 10

    # --------------------------------------------------------- ZTFMagstatCommand

    def test_ztf_magstat_updates_object_and_inserts_magstat(self):
        oid = 100003
        self._insert_object(oid, sid=1)
        self._insert_ztf_object(oid)

        self._produce_command(
            "ztf",
            "magstat",
            {
                "oid": oid,
                "sid": 1,
                "meanra": 60.0,
                "meandec": 60.0,
                "sigmara": 0.1,
                "sigmadec": 0.1,
                "firstmjd": 59000.0,
                "lastmjd": 59010.0,
                "deltajd": 10.0,
                "n_det": 5,
                "n_fphot": 2,
                "n_ndet": 1,
                "corrected": True,
                "stellar": False,
                "reference_change": False,
                "diffpos": True,
                "magstats": {
                    "0": [
                        {
                            "band": 1,
                            "stellar": False,
                            "corrected": True,
                            "ndubious": 0,
                            "dmdt_first": None,
                            "dm_first": None,
                            "sigmadm_first": None,
                            "dt_first": None,
                            "magmean": 18.5,
                            "magmedian": 18.5,
                            "magmax": 19.0,
                            "magmin": 18.0,
                            "magsigma": 0.1,
                            "maglast": 18.5,
                            "magfirst": 18.5,
                            "magmean_corr": 18.5,
                            "magmedian_corr": 18.5,
                            "magmax_corr": 19.0,
                            "magmin_corr": 18.0,
                            "magsigma_corr": 0.1,
                            "maglast_corr": 18.5,
                            "magfirst_corr": 18.5,
                            "saturation_rate": 0.0,
                            "ndet": 5,
                            "firstmjd": 59000.0,
                            "lastmjd": 59010.0,
                        }
                    ]
                },
            },
        )
        self.step.start()

        with self.db.session() as session:
            obj = session.execute(
                text(
                    "SELECT meanra, n_det FROM object WHERE oid = :oid AND sid = :sid"
                ),
                {"oid": oid, "sid": 1},
            ).fetchone()
            magstat = session.execute(
                text(
                    "SELECT band FROM magstat WHERE oid = :oid AND sid = :sid"
                ),
                {"oid": oid, "sid": 1},
            ).fetchall()

        assert obj is not None
        assert abs(obj[0] - 60.0) < 1e-5
        assert obj[1] == 5
        assert len(magstat) == 1
        assert magstat[0][0] == 1

    # --------------------------------------------------------- LSSTMagstatCommand

    def test_lsst_magstat_updates_object(self):
        oid = 100004
        self._insert_object(oid, sid=2)

        self._produce_command(
            "lsst",
            "magstat",
            {
                "oid": oid,
                "sid": 2,
                "meanra": 70.0,
                "meandec": 70.0,
                "sigmara": 0.2,
                "sigmadec": 0.2,
                "firstmjd": 59000.0,
                "lastmjd": 59020.0,
                "deltajd": 20.0,
                "n_det": 10,
                "n_fphot": 3,
                "n_ndet": 2,
                "corrected": False,
                "stellar": None,
            },
        )
        self.step.start()

        with self.db.session() as session:
            row = session.execute(
                text(
                    "SELECT meanra, n_det FROM object WHERE oid = :oid AND sid = :sid"
                ),
                {"oid": oid, "sid": 2},
            ).fetchone()

        assert row is not None
        assert abs(row[0] - 70.0) < 1e-5
        assert row[1] == 10

    # --------------------------------------------------------- LSSTFeatureCommand

    def test_lsst_feature_upserts_features(self):
        oid = 100005
        self._insert_object(oid, sid=2)

        self._produce_command(
            "lsst",
            "features",
            {
                "oid": oid,
                "sid": 2,
                "features_version": "1",
                "features": [
                    {"feature_id": 1, "band": 1, "value": 3.14},
                    {"feature_id": 2, "band": 2, "value": 2.71},
                ],
            },
        )
        self.step.start()

        with self.db.session() as session:
            rows = session.execute(
                text(
                    "SELECT feature_id, value FROM feature "
                    "WHERE oid = :oid AND sid = :sid ORDER BY feature_id"
                ),
                {"oid": oid, "sid": 2},
            ).fetchall()

        assert len(rows) == 2
        assert rows[0][0] == 1
        assert abs(rows[0][1] - 3.14) < 1e-5
        assert rows[1][0] == 2
        assert abs(rows[1][1] - 2.71) < 1e-5

    def test_lsst_feature_deduplicates_duplicate_keys_within_payload(self):
        oid = 100006
        self._insert_object(oid, sid=2)

        # Two entries for (feature_id=1, band=1): only the last should survive
        self._produce_command(
            "lsst",
            "features",
            {
                "oid": oid,
                "sid": 2,
                "features_version": "1",
                "features": [
                    {"feature_id": 1, "band": 1, "value": 1.0},
                    {"feature_id": 1, "band": 1, "value": 9.99},
                ],
            },
        )
        self.step.start()

        with self.db.session() as session:
            rows = session.execute(
                text(
                    "SELECT value FROM feature "
                    "WHERE oid = :oid AND sid = :sid AND feature_id = 1 AND band = 1"
                ),
                {"oid": oid, "sid": 2},
            ).fetchall()

        assert len(rows) == 1
        assert abs(rows[0][0] - 9.99) < 1e-5

    # ------------------------------------------------------------ XmatchCommand

    def test_xmatch_inserts_ztf_row(self):
        oid = 100007

        self._produce_command(
            "ztf",
            "xmatch",
            {
                "oid": oid,
                "sid": 1,
                "catalog": "allwise",
                "oid_catalog": "J123456.78+123456.7",
                "dist": 0.5,
            },
        )
        self.step.start()

        with self.db.session() as session:
            row = session.execute(
                text(
                    "SELECT oid_catalog, dist FROM xmatch "
                    "WHERE oid = :oid AND sid = 1 AND catid = 0"
                ),
                {"oid": oid},
            ).fetchone()

        assert row is not None
        assert row[0] == "J123456.78+123456.7"
        assert abs(row[1] - 0.5) < 1e-5

    def test_xmatch_skips_sid_2(self):
        oid = 100008

        self._produce_command(
            "lsst",
            "xmatch",
            {
                "oid": oid,
                "sid": 2,
                "catalog": "allwise",
                "oid_catalog": "SKIP_ME",
                "dist": 1.0,
            },
        )
        self.step.start()

        with self.db.session() as session:
            count = session.execute(
                text("SELECT COUNT(*) FROM xmatch WHERE oid = :oid"),
                {"oid": oid},
            ).scalar()

        assert count == 0

    def test_xmatch_upserts_on_second_message(self):
        oid = 100009

        for dist in (0.5, 0.1):
            self._produce_command(
                "ztf",
                "xmatch",
                {
                    "oid": oid,
                    "sid": 1,
                    "catalog": "allwise",
                    "oid_catalog": "J000000.00+000000.0",
                    "dist": dist,
                },
            )
            self.step.start()

        with self.db.session() as session:
            rows = session.execute(
                text(
                    "SELECT COUNT(*), MIN(dist) FROM xmatch "
                    "WHERE oid = :oid AND sid = 1 AND catid = 0"
                ),
                {"oid": oid},
            ).fetchone()

        # ON CONFLICT DO UPDATE keeps only one row with the latest dist
        assert rows[0] == 1
        assert abs(rows[1] - 0.1) < 1e-5

    # --------------------------------------------------- error / edge cases

    def test_invalid_message_is_skipped_without_error(self):
        # Old-format command (missing survey/step keys) must be silently dropped.
        old_format = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {"oid": 999999, "ndet": 1},
            }
        )
        self.producer.produce({"payload": old_format})
        self.step.start()  # must not raise

        with self.db.session() as session:
            count = session.execute(
                text("SELECT COUNT(*) FROM detection WHERE oid = 999999"),
            ).scalar()

        assert count == 0
