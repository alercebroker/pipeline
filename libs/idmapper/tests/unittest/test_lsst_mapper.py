import unittest
from idmapper.mapper import catalog_oid_to_masterid, decode_masterid, SURVEY_IDS
from idmapper.lsst import encode_lsst_to_masterid_without_survey_without_db
import pytest
import psycopg2


@pytest.mark.usefixtures("psql_service")
class TestLSSTMapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 25432,
            "DB_NAME": "postgres",
        }
        cls.connection = psycopg2.connect(
            dbname=config["DB_NAME"],
            user=config["USER"],
            password=config["PASSWORD"],
            host=config["HOST"],
            port=config["PORT"],
        )

    def setUp(self):
        self.cursor = self.connection.cursor()
        self.cursor.execute(open("idmapper/lsst_init.sql", "r").read())

    def tearDown(self):
        # Delete all the tables from the database
        self.cursor.execute("DROP TABLE IF EXISTS lsst_idmapper;")

    def test_encode_without_db(self):
        lsst_oid = 123456789
        lsst_oid_bits_without_survey = "111010110111100110100010101"

        masterid_without_survey = encode_lsst_to_masterid_without_survey_without_db(
            lsst_oid
        )

        # Check that the LSST object ID is the same as the master ID
        self.assertEqual(
            masterid_without_survey,
            lsst_oid,
            "Master ID should have the correct LSST object ID bits",
        )

        masterid_without_survey_bits = "{0:b}".format(masterid_without_survey)
        self.assertEqual(masterid_without_survey_bits, lsst_oid_bits_without_survey)

    def test_encode_with_db(self):
        lsst_oid = 123456789

        masterid = catalog_oid_to_masterid(
            "LSST", lsst_oid, validate=False, db_cursor=self.cursor
        )
        masterid_without_survey = masterid & ((1 << (63 - 8)) - 1)
        survey_id = masterid >> (63 - 8)
        self.assertEqual(1, masterid_without_survey)
        self.assertEqual(
            survey_id,
            SURVEY_IDS["LSST"],
            "survey_id should be LSST",
        )

    def test_decoder(self):
        lsst_oid = 123456789
        masterid = catalog_oid_to_masterid(
            "LSST", lsst_oid, validate=False, db_cursor=self.cursor
        )
        survey, oid = decode_masterid(masterid, db_cursor=self.cursor)

        # Check that the survey is "LSST"
        self.assertEqual(survey, "LSST", "Survey should be LSST")

        # Check that the oid is the same as the original oid
        self.assertEqual(oid, lsst_oid, "OID should be the same as the original OID")
