import unittest

import pytest
from sqlalchemy import select
from sqlalchemy.inspection import inspect

from db_plugins.db.sql._connection import (
    PsqlDatabase,
)
from db_plugins.db.sql.models import (
    Band,
    Base,
    CatalogIdLut,
    Classifier,
    FeatureNameLut,
    Object,
    SidLut,
    Taxonomy,
)


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
        self.assertTrue(
            set(inspector.get_table_names()) >= set(Base.metadata.tables.keys())
        )

    def test_drop_db(self):
        self.db.drop_db()
        engine = self.db._engine
        inspector = inspect(engine)
        self.assertEqual(len(inspector.get_table_names()), 0)

    def test_query(self):
        self.db.create_db()
        with self.db.session() as session:
            stmt = select(Object)
            obj = session.execute(stmt)
            assert len([o for o in obj.scalars()]) == 0

    def test_initial_data(self):
        self.db.create_db()
        with self.db.session() as session:
            checks = [
                {"table": Classifier, "expected_len": 1},
                {"table": FeatureNameLut, "expected_len": 119},
                {"table": SidLut, "expected_len": 3},
                {"table": Taxonomy, "expected_len": 5},
                {"table": CatalogIdLut, "expected_len": 1},
                {"table": Band, "expected_len": 15},
            ]
            for check in checks:
                count = session.query(check["table"]).count()
                assert count == check["expected_len"]
