import unittest
import pandas as pd
from unittest import mock
from sorting_hat_step.utils import wizard
from .data.batch import generate_batch_ra_dec
from sorting_hat_step.database import MongoConnection
from pymongo.database import Database


class SortingHatTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_db = mock.create_autospec(MongoConnection)
        self.mock_db.database = mock.create_autospec(Database)

    def tearDown(self):
        del self.mock_db

    def test_id_generator(self):
        aid_1 = wizard.id_generator(0, 0)
        self.assertEqual(aid_1, 1000000001000000000)

        aid_2 = wizard.id_generator(359, 0)
        self.assertEqual(aid_2, 1235600001000000000)

        aid_3 = wizard.id_generator(312.12312311, 80.99999991)
        self.assertEqual(aid_3, 1204829541805959900)

        aid_4 = wizard.id_generator(-1, -1)
        self.assertIsInstance(aid_4, int)

        aid_5 = wizard.id_generator(359 + 360, -1)
        self.assertEqual(aid_4, aid_5)

    @mock.patch("sorting_hat_step.utils.wizard.oid_query")
    def test_find_existing_id(self, mock_query):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "aid": None},
                {"oid": "B", "aid": None},
                {"oid": "C", "aid": None},
            ]
        )
        mock_query.side_effect = ["aid1", "aid1", None]
        response = wizard.find_existing_id(self.mock_db, alerts)
        assert (response["aid"].values == ["aid1", "aid1", None]).all()

    @mock.patch("sorting_hat_step.utils.wizard.conesearch_query")
    def test_find_id_by_conesearch(self, mock_query):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "B", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "C", "aid": None, "ra": 123, "dec": 456},
            ],
        )
        mock_query.side_effect = ["aid2", "aid2"]
        response = wizard.find_id_by_conesearch(self.mock_db, alerts)
        assert (response["aid"].values == ["aid1", "aid1", "aid2"]).all()

    @mock.patch("sorting_hat_step.utils.wizard.conesearch_query")
    def test_find_id_by_conesearch_without_found_aid(self, mock_query):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "B", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "C", "aid": None, "ra": 123, "dec": 456},
            ],
        )
        mock_query.side_effect = [None]
        response = wizard.find_id_by_conesearch(self.mock_db, alerts)
        assert (response["aid"].values == ["aid1", "aid1", None]).all()

    def test_generate_new_id(self):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "B", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "C", "aid": None, "ra": 123, "dec": 456},
            ],
        )
        response = wizard.generate_new_id(alerts)
        assert response["aid"].notna().all()
        assert str(response["aid"][2]).startswith("AL")

    def test_encode(self):
        # known alerce_id: 'b' -> 1
        aid_long = 1
        aid_str = wizard.encode(aid_long)
        self.assertIsInstance(aid_str, str)
        self.assertEqual(aid_str, "b")
        # a real alerce_id
        aid_long = 1000000000000000000  # aid example of 19 digits
        aid_str = wizard.encode(aid_long)
        self.assertIsInstance(aid_str, str)
        self.assertEqual(aid_str, "kmluxinkecojo")

    def test_decode(self):
        # known alerce_id: 'b' -> 1
        aid_str = "b"
        aid_int = wizard.decode(aid_str)
        self.assertIsInstance(aid_int, int)
        self.assertEqual(aid_int, 1)
        # a real alerce_id
        aid_str = "kmluxinkecojo"
        aid_int = wizard.decode(aid_str)
        self.assertIsInstance(aid_int, int)
        self.assertEqual(aid_int, 1000000000000000000)

    @mock.patch("sorting_hat_step.utils.wizard.update_query")
    def test_write_object(self, insert_mock):
        alerts = pd.DataFrame(
            [
                {
                    "oid": 10,
                    "aid": 0,
                    "sid": "ZTF",
                    "extra_fields": {},
                    "ra": 0,
                    "dec": 0,
                    "mjd": 0,
                },
                {
                    "oid": 20,
                    "aid": 1,
                    "sid": "ZTF",
                    "extra_fields": {},
                    "ra": 0,
                    "dec": 0,
                    "mjd": 0,
                },
                {
                    "oid": 30,
                    "aid": 0,
                    "sid": "ZTF",
                    "extra_fields": {},
                    "ra": 0,
                    "dec": 0,
                    "mjd": 0,
                },
                {
                    "oid": 40,
                    "aid": 2,
                    "sid": "ZTF",
                    "extra_fields": {},
                    "ra": 0,
                    "dec": 0,
                    "mjd": 0,
                },
            ]
        )

        wizard.insert_empty_objects(self.mock_db, alerts)

        updated_records = [
            {
                "_id": 10,
                "aid": 0,
                "sid": "ZTF",
                "extra_fields": {},
                "ra": 0,
                "dec": 0,
                "mjd": 0,
            },
            {
                "_id": 20,
                "aid": 1,
                "sid": "ZTF",
                "extra_fields": {},
                "ra": 0,
                "dec": 0,
                "mjd": 0,
            },
            {
                "_id": 30,
                "aid": 0,
                "sid": "ZTF",
                "extra_fields": {},
                "ra": 0,
                "dec": 0,
                "mjd": 0,
            },
            {
                "_id": 40,
                "aid": 2,
                "sid": "ZTF",
                "extra_fields": {},
                "ra": 0,
                "dec": 0,
                "mjd": 0,
            },
        ]
        insert_mock.assert_called_with(self.mock_db, updated_records)
