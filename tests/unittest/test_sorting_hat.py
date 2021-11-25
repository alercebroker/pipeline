import unittest
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
from db_plugins.db.mongo.connection import MongoConnection
from db_plugins.db.mongo.models import Object
from sorting_hat_step.utils.sorting_hat import SortingHat
from data.batch import generate_batch_ra_dec, generate_parsed_batch


class SortingHatTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_database_connection = mock.create_autospec(MongoConnection)
        self.sh = SortingHat(self.mock_database_connection)

    def tearDown(self):
        del self.mock_database_connection
        del self.sh

    def test_wgs_scale(self):
        val90 = self.sh.wgs_scale(90.)
        self.assertAlmostEqual(val90, 111693.97955912731)

        val0 = self.sh.wgs_scale(0.)
        self.assertAlmostEqual(val0, 110574.27582159475)

        valm90 = self.sh.wgs_scale(0.)
        self.assertAlmostEqual(valm90, 110574.27582159475)

    def test_id_generator(self):
        aid_1 = self.sh.id_generator(0, 0)
        self.assertEqual(aid_1, 1000000001000000000)

        aid_2 = self.sh.id_generator(359, 0)
        self.assertEqual(aid_2, 1235600001000000000)

        aid_3 = self.sh.id_generator(312.12312311, 80.99999991)
        self.assertEqual(aid_3, 1204829541805959900)

    def test_internal_cross_match(self):
        # Test with 500 unique objects (no closest objects) random.seed set in data.batch.py (1313)
        example_batch = generate_batch_ra_dec(500)
        batch = self.sh.internal_cross_match(example_batch)
        self.assertListEqual(batch.index.tolist(), batch["tmp_id"].to_list())  # The index must be equal to tmp_id

        # Test with 110 objects where 10 are close
        example_batch = generate_batch_ra_dec(100, nearest=10)
        batch = self.sh.internal_cross_match(example_batch)
        # Check 100 unique tmp_id
        self.assertEqual(len(batch["tmp_id"].unique()), 100)

    def test_oid_query(self):
        # Mock a response with elements in database
        self.mock_database_connection.query(model=Object).find.return_value = [{"aid": 1}]
        aid = self.sh.oid_query(["x", "y", "z"])
        self.assertEqual(aid, 1)
        self.mock_database_connection.query(model=Object).find.assert_called()
        # Mock a response with no elements
        self.mock_database_connection.query(model=Object).find.return_value = []
        aid = self.sh.oid_query(["x", "y", "z"])
        self.assertEqual(aid, None)
        self.mock_database_connection.query(model=Object).find.assert_called()
        # Mock a response without aid field
        self.mock_database_connection.query(model=Object).find.return_value = [{"field1": 1, "field2": 2}]
        with self.assertRaises(KeyError) as context:
            self.sh.oid_query(["x", "y", "z"])
            self.mock_database_connection.query(model=Object).find.assert_called()
        self.assertIsInstance(context.exception, Exception)

    def test_cone_search(self):
        self.mock_database_connection.query(model=Object).find.return_value = [{"aid": 1}]
        aids = self.sh.cone_search(0, 0)
        self.mock_database_connection.query(model=Object).find.assert_called()
        self.assertListEqual(aids, [{"aid": 1}])

    @mock.patch("sorting_hat_step.utils.sorting_hat.SortingHat.oid_query")
    @mock.patch("sorting_hat_step.utils.sorting_hat.SortingHat.cone_search")
    def test_one_to_name_miss(self, mock_cone_search: MagicMock, mock_oid_query: MagicMock):
        batch = generate_parsed_batch(1, nearest=10)
        # Test one batch that is the same object: (miss) test when object doesn't exist in database, so create alerce_id
        mock_cone_search.return_value = []
        mock_oid_query.return_value = None
        aid_series = self.sh._to_name(batch)
        mock_oid_query.assert_called()
        mock_cone_search.assert_called()
        self.assertIsInstance(aid_series, pd.Series)

    @mock.patch("sorting_hat_step.utils.sorting_hat.SortingHat.oid_query")
    @mock.patch("sorting_hat_step.utils.sorting_hat.SortingHat.cone_search")
    def test_one_to_name_hit_oid(self, mock_cone_search: MagicMock, mock_oid_query: MagicMock):
        batch = generate_parsed_batch(1, nearest=10)
        # Test one batch that is the same object: (hit) test when exists oid
        mock_cone_search.return_value = []
        mock_oid_query.return_value = 123
        aid_series = self.sh._to_name(batch)
        mock_oid_query.assert_called()
        mock_cone_search.assert_not_called()
        self.assertIsInstance(aid_series, pd.Series)
        self.assertEqual(aid_series["aid"], 123)

    @mock.patch("sorting_hat_step.utils.sorting_hat.SortingHat.oid_query")
    @mock.patch("sorting_hat_step.utils.sorting_hat.SortingHat.cone_search")
    def test_one_to_name_hit_cone_search(self, mock_cone_search: MagicMock, mock_oid_query: MagicMock):
        batch = generate_parsed_batch(1, nearest=10)
        # Test one batch that is the same object: (hit) test when exists objects nearest
        mock_cone_search.return_value = [{"aid": 123}]
        mock_oid_query.return_value = None
        aid_series = self.sh._to_name(batch)
        mock_oid_query.assert_called()
        mock_cone_search.assert_called()
        self.assertIsInstance(aid_series, pd.Series)
        self.assertEqual(aid_series["aid"], 123)

    def test_to_name_batch(self):
        # Test one batch that is the same object
        batch = generate_parsed_batch(100, nearest=10)
        response = self.sh.to_name(batch)
        self.assertIsInstance(response, pd.DataFrame)
        self.assertIn("aid", response.columns)
        self.assertEqual(len(batch), len(response))
        pass
