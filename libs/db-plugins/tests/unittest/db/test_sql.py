from db_plugins.db.sql._connection import PsqlDatabase
import unittest
from unittest import mock
from sqlalchemy.orm import Session


class SQLConnectionTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "USER": "nada",
            "PASSWORD": "nada",
            "HOST": "nada",
            "PORT": "nada",
            "DB_NAME": "nada",
        }
        mock_engine = mock.Mock()
        self.db = PsqlDatabase(self.config, engine=mock_engine)

    def tearDown(self):
        del self.db

    def test_session(self):
        with self.db.session() as session:
            self.assertIsInstance(session, Session)

    @mock.patch("db_plugins.db.sql._connection.Base")
    def test_create_db(self, base):
        self.db.create_db()
        base.metadata.create_all.assert_called_once()

    @mock.patch("db_plugins.db.sql._connection.Base")
    def test_drop_db(self, base):
        self.db.drop_db()
        base.metadata.drop_all.assert_called_once()
