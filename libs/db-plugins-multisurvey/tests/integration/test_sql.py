from db_plugins.db.sql.models import Object
from db_plugins.db.sql._connection import (
    PsqlDatabase,
)

from sqlalchemy import select
from sqlalchemy.inspection import inspect
import pytest
import unittest


@pytest.mark.usefixtures("psql_service")
class SQLConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config = {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5432,
            "DB_NAME": "postgres",
        }
        self.session_options = {
            "autocommit": False,
            "autoflush": True,
        }
        self.db = PsqlDatabase(config)

    def tearDown(self):
        self.db.drop_db()

    def test_create_session(self):
        with self.db.session() as session:
            self.assertIsNotNone(session)

    def test_create_db(self):
        self.db.create_db()
        engine = self.db._engine
        inspector = inspect(engine)
        self.assertGreater(len(inspector.get_table_names()), 0)

    def test_drop_db(self):
        self.db.drop_db()
        engine = self.db._engine
        inspector = inspect(engine)
        self.assertEqual(len(inspector.get_table_names()), 0)

    def test_query(self):
        self.db.create_db()
        with self.db.session() as session:
            query = select(Object)
            obj = session.execute(query)
            assert len([o for o in obj.scalars()]) == 0
