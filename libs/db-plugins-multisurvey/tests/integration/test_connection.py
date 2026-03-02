from base import BaseConnectionTest
from sqlalchemy import (
    inspect,
)

from db_plugins.db.sql._initial_data import INITIAL_DATA
from db_plugins.db.sql.models import Base


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

        model_tables = Base.metadata.tables
        with self.psql_db.session() as session:
            for table_name, init_data in INITIAL_DATA.items():
                if table_name not in model_tables:
                    continue
                expected_len = len(init_data["data"])
                count = session.query(model_tables[table_name]).count()
                assert count == expected_len, (
                    f"Expected length does not match for '{table_name}', expected '{expected_len}' but got '{count}'"
                )

    def test_drop_db(self):
        """Verificar que se pueden eliminar todas las tablas de la base de datos"""
        self.psql_db.create_db()

        self.psql_db.drop_db()

        engine = self.psql_db._engine
        inspector = inspect(engine)
        self.assertEqual(len(inspector.get_table_names()), 0)
