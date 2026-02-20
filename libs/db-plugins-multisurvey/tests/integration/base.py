import unittest

import pytest

from db_plugins.db.sql._connection import PsqlDatabase


@pytest.mark.usefixtures("psql_service", "psql_db")
class BaseConnectionTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_psql_db(self, psql_db: PsqlDatabase):
        self.psql_db = psql_db

    @classmethod
    def setUpClass(cls):
        cls.session_options = {
            "autocommit": False,
            "autoflush": True,
        }


class BaseDbTest(BaseConnectionTest):
    def setUp(self):
        """Preparar la base de datos antes de cada prueba"""
        self.psql_db.create_db()

    def tearDown(self):
        """Limpiar la base de datos después de cada prueba"""
        self.psql_db.drop_db()
