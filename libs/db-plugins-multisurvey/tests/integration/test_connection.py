from base import BaseConnectionTest
from sqlalchemy import (
    inspect,
)

from db_plugins.db.sql.models import (
    Band,
    Base,
    CatalogIdLut,
    Classifier,
    FeatureNameLut,
    SidLut,
    Taxonomy,
)


class ConnectionTest(BaseConnectionTest):
    """Pruebas para verificar la conexión y funcionalidades básicas de la base de datos"""

    def test_create_session(self):
        """Verificar que se puede crear una sesión de base de datos"""
        with self.psql_db.session() as session:
            self.assertIsNotNone(session)

    def test_create_db(self):
        """Verificar que se pueden crear las tablas en la base de datos"""
        self.psql_db.create_db()
        engine = self.psql_db._engine
        inspector = inspect(engine)
        self.assertGreater(len(inspector.get_table_names()), 0)

        self.assertTrue(
            set(inspector.get_table_names()) >= set(Base.metadata.tables.keys())
        )

    def test_initial_data(self):
        self.psql_db.create_db()
        with self.psql_db.session() as session:
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

    def test_drop_db(self):
        """Verificar que se pueden eliminar todas las tablas de la base de datos"""
        self.psql_db.create_db()

        self.psql_db.drop_db()

        engine = self.psql_db._engine
        inspector = inspect(engine)
        self.assertEqual(len(inspector.get_table_names()), 0)
