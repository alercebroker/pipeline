import unittest
import pandas as pd
from unittest import mock

from db_plugins.db.mongo.connection import MongoConnection
from sorting_hat_step.utils import sorting_hat as wizard
from data.batch import generate_batch_ra_dec


class SortingHatTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_database_connection = mock.create_autospec(MongoConnection)

    def tearDown(self):
        del self.mock_database_connection

    def test_wgs_scale(self):
        val90 = wizard.wgs_scale(90.0)
        self.assertAlmostEqual(val90, 111693.97955912731)

        val0 = wizard.wgs_scale(0.0)
        self.assertAlmostEqual(val0, 110574.27582159475)

        valm90 = wizard.wgs_scale(0.0)
        self.assertAlmostEqual(valm90, 110574.27582159475)

    def test_id_generator(self):
        aid_1 = wizard.id_generator(0, 0)
        self.assertEqual(aid_1, 1000000001000000000)

        aid_2 = wizard.id_generator(359, 0)
        self.assertEqual(aid_2, 1235600001000000000)

        aid_3 = wizard.id_generator(312.12312311, 80.99999991)
        self.assertEqual(aid_3, 1204829541805959900)

    def test_internal_cross_match_no_closest_objects(self):
        # Test with 500 unique objects (no closest objects) random.seed set in data.batch.py (1313)
        example_batch = generate_batch_ra_dec(500)
        batch = wizard.internal_cross_match(example_batch)
        self.assertListEqual(
            batch.index.tolist(), batch["tmp_id"].to_list()
        )  # The index must be equal to tmp_id
        self.assertEqual(len(batch["oid"].unique()), 500)

    def test_internal_cross_match_closest_objects(self):
        # Test with 10 objects closest
        example_batch = generate_batch_ra_dec(1, nearest=9)
        batch = wizard.internal_cross_match(example_batch)
        # Check 100 unique tmp_id
        self.assertEqual(len(batch["tmp_id"].unique()), 1)

    def test_internal_cross_match_same_oid(self):
        # Test with 10 objects where 5 are close
        example_batch = generate_batch_ra_dec(10)
        example_batch.loc[:, "oid"] = "same_object"
        batch = wizard.internal_cross_match(example_batch)
        self.assertEqual(len(batch["tmp_id"].unique()), 1)

    def test_internal_cross_match_two_objects_with_oid(self):
        batch_1 = generate_batch_ra_dec(1, nearest=9)
        batch_1.loc[:, "oid"] = "same_object_1"
        batch_2 = generate_batch_ra_dec(10)
        batch_2.loc[:, "oid"] = "same_object_2"
        batch = pd.concat([batch_1, batch_2], ignore_index=True)
        batch_xmatched = wizard.internal_cross_match(batch)
        self.assertEqual(len(batch_xmatched["tmp_id"].unique()), 2)
        self.assertEqual(len(batch_xmatched["tmp_id"]), 20)

    def test_internal_cross_match_repeating_oid_and_closest(self):
        batch = generate_batch_ra_dec(1, nearest=9)
        batch.loc[:, "oid"] = "same_object_1"
        batch_xmatched = wizard.internal_cross_match(batch)
        self.assertEqual(len(batch_xmatched["tmp_id"].unique()), 1)
        self.assertEqual(len(batch_xmatched["tmp_id"]), 10)

    def test_find_existing_id(self):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "tmp_id": "X"},
                {"oid": "B", "tmp_id": "X"},
                {"oid": "C", "tmp_id": "Y"},
            ]
        )
        id_getter = mock.Mock()
        id_getter.side_effect = ["aid1", None]
        response = wizard.find_existing_id(alerts, id_getter)
        assert (response["aid"].values == ["aid1", "aid1", None]).all()

    def test_find_id_by_conesearch(self):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "tmp_id": "X", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "B", "tmp_id": "X", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "C", "tmp_id": "Y", "aid": None, "ra": 123, "dec": 456},
            ],
        )
        id_getter = mock.Mock()
        id_getter.side_effect = [
            [{"oid": "obj1", "aid": "aid2"}, {"oid": "obj2", "aid": "aid2"}]
        ]
        response = wizard.find_id_by_conesearch(alerts, id_getter)
        assert (response["aid"].values == ["aid1", "aid1", "aid2"]).all()

    def test_find_id_by_conesearch_without_found_aid(self):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "tmp_id": "X", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "B", "tmp_id": "X", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "C", "tmp_id": "Y", "aid": None, "ra": 123, "dec": 456},
            ],
        )
        id_getter = mock.Mock()
        id_getter.side_effect = [[]]
        response = wizard.find_id_by_conesearch(alerts, id_getter)
        assert (response["aid"].values == ["aid1", "aid1", None]).all()

    def test_generate_new_id(self):
        alerts = pd.DataFrame(
            [
                {"oid": "A", "tmp_id": "X", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "B", "tmp_id": "X", "aid": "aid1", "ra": 123, "dec": 456},
                {"oid": "C", "tmp_id": "Y", "aid": None, "ra": 123, "dec": 456},
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
