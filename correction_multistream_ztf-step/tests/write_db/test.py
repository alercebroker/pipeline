import unittest

import pytest
from _connection import PsqlDatabase
from write_functions import InsertData

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "db_esta_si_que_si",
}


@pytest.mark.usefixtures("psql_service")
class BaseDbTests(unittest.TestCase):
    def __init__(self):
        # crear db
        self.connection = PsqlDatabase(psql_config)
        self.connection.create_db()

        # Insert data
        insert_data_step = InsertData(self.connection)
        insert_data_step.insert_data()
        
        # Clear DB
        self.tearDown()

    def tearDown(self):
        # limpiar la db
        self.connection.drop_db()

base_db_tests = BaseDbTests()