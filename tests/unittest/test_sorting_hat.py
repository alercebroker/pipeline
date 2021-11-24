import unittest
from unittest import mock

from db_plugins.db.mongo.connection import MongoConnection
from sorting_hat_step.utils.sorting_hat import SortingHat


class SortingHatTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_database_connection = mock.create_autospec(MongoConnection)
        self.sh = SortingHat(self.mock_database_connection)

    def tearDown(self):
        del self.mock_database_connection

    def test_wgs_scale(self):
        pass

