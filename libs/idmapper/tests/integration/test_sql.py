import psycopg2
import pytest
import unittest
from idmapper.lsst import encode_lsst_to_masterid_without_survey_with_db


@pytest.mark.usefixtures("psql_service")
class SQLConnectionTest(unittest.TestCase):
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

    def test_insertion(self):
        self.cursor.execute(
            """
            INSERT INTO lsst_idmapper (lsst_diaObjectId)
            VALUES (123456789);"""
        )
        self.cursor.execute("SELECT * FROM lsst_idmapper;")
        result = self.cursor.fetchall()
        self.assertEqual(123456789, result[0][1])

    def test_encode_lsst_to_masterid_without_survey_with_db(self):
        lsst_oid = 123456789
        masterid_without_survey = encode_lsst_to_masterid_without_survey_with_db(
            lsst_oid, self.cursor
        )
        self.assertEqual(1, masterid_without_survey)
        masterid_without_survey = encode_lsst_to_masterid_without_survey_with_db(
            lsst_oid, self.cursor
        )
        self.assertEqual(1, masterid_without_survey)

        self.cursor.execute("SELECT count(*) FROM lsst_idmapper;")
        result = self.cursor.fetchone()
        self.assertEqual(1, result[0])

        lsst_oid = 987654321
        masterid_without_survey = encode_lsst_to_masterid_without_survey_with_db(
            lsst_oid, self.cursor
        )
        self.assertEqual(2, masterid_without_survey)

        self.cursor.execute("SELECT count(*) FROM lsst_idmapper;")
        result = self.cursor.fetchone()
        self.assertEqual(2, result[0])
