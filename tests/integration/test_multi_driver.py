import pytest
import unittest

from ingestion_step.utils.multi_driver import MultiDriverConnection


@pytest.mark.usefixtures("mongo_service")
@pytest.mark.usefixtures("kafka_service")
@pytest.mark.usefixtures("psql_service")
class MultiDriverTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = MultiDriverConnection()
        pass

    def setUp(self):
        self.step.driver.create_db()

    @classmethod
    def tearDownClass(cls):
        cls.step.driver.drop_db()

