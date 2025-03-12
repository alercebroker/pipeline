import unittest
import pytest
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, Integer, String, select, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

# Base para modelos SQLAlchemy
Base = declarative_base()

# Definición de modelos
class Object(Base):
    __tablename__ = 'objects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    description = Column(String(200))

# Clase para manejar la conexión a la base de datos
class PsqlDatabase:
    def __init__(self, config):
        self.host = config.get("HOST", "localhost")
        self.user = config.get("USER", "postgres")
        self.password = config.get("PASSWORD", "postgres")
        self.port = config.get("PORT", 5432)
        self.db_name = config.get("DB_NAME", "postgres")
        
        self._engine = self._create_engine()
        self._session_factory = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=True,
                bind=self._engine
            )
        )
    
    def _create_engine(self):
        conn_str = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
        return create_engine(conn_str)
    
    @contextmanager
    def session(self):
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_db(self):
        """Crear todas las tablas definidas en los modelos"""
        Base.metadata.create_all(self._engine)
    
    def drop_db(self):
        """Eliminar todas las tablas de la base de datos"""
        Base.metadata.drop_all(self._engine)

# Clase de prueba
@pytest.mark.usefixtures("psql_service")
class SQLConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5432,
            "DB_NAME": "postgres",
        }
        cls.session_options = {
            "autocommit": False,
            "autoflush": True,
        }
        cls.db = PsqlDatabase(config)

    def tearDown(self):
        self.db.drop_db()

    def test_create_session(self):
        """Verificar que se puede crear una sesión de base de datos"""
        with self.db.session() as session:
            self.assertIsNotNone(session)

    def test_create_db(self):
        """Verificar que se pueden crear las tablas en la base de datos"""
        self.db.create_db()
        engine = self.db._engine
        inspector = inspect(engine)
        # Verificar que hay al menos una tabla creada
        self.assertGreater(len(inspector.get_table_names()), 0)
        
        # Verificar que la tabla 'objects' existe
        self.assertIn('objects', inspector.get_table_names())

    def test_drop_db(self):
        """Verificar que se pueden eliminar todas las tablas de la base de datos"""
        # Primero crear las tablas
        self.db.create_db()
        
        # Luego eliminarlas
        self.db.drop_db()
        
        # Verificar que no quedan tablas
        engine = self.db._engine
        inspector = inspect(engine)
        self.assertEqual(len(inspector.get_table_names()), 0)

    def test_query(self):
        """Verificar que se pueden realizar consultas a la base de datos"""
        self.db.create_db()
        with self.db.session() as session:
            query = select(Object)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            assert len([o for o in obj.scalars()]) == 0
    
    def test_create_and_query_objects(self):
        """Verificar que se pueden crear y consultar objetos en la base de datos"""
        self.db.create_db()
        
        # Crear un nuevo objeto
        test_object = Object(name="Test Object", description="This is a test object")
        
        with self.db.session() as session:
            session.add(test_object)
        
        # Verificar que el objeto se ha guardado
        with self.db.session() as session:
            query = select(Object)
            objects = list(session.execute(query).scalars())
            
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0].name, "Test Object")
            self.assertEqual(objects[0].description, "This is a test object")

if __name__ == "__main__":
    unittest.main()