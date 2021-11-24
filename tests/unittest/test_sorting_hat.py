import unittest
from unittest import mock

from db_plugins.db.mongo.connection import MongoConnection
from sorting_hat_step.utils.sorting_hat import SortingHat
from data.batch import generate_batch_ra_dec


class SortingHatTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_database_connection = mock.create_autospec(MongoConnection)
        self.sh = SortingHat(self.mock_database_connection)

    def tearDown(self):
        del self.mock_database_connection

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
        self.assertEqual(len(batch["tmp_id"].unique()), 100)

