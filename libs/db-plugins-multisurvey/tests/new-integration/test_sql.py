import unittest
import pytest
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, Integer, String, select, text
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

from db_plugins.db.sql.models_new import Base, Object, ZtfObject, Detection

# Clase para manejar la conexión a la base de datos
class PsqlDatabase:
    def __init__(self, config):
        self.host = config.get("HOST", "localhost")
        self.user = config.get("USER", "postgres")
        self.password = config.get("PASSWORD", "postgres")
        self.port = config.get("PORT", 5435)
        self.db_name = config.get("DB_NAME", "postgres")
        
        self._engine = self._create_engine()
        self._session_factory = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=True,
                bind=self._engine
            )
        )
    
    def _create_engine(self): ### Extra arg para dar el esquema
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

# Test de conexión
@pytest.mark.usefixtures("psql_service")
class SQLConnectionTest(unittest.TestCase):
    """Pruebas para verificar la conexión y funcionalidades básicas de la base de datos"""
    
    @classmethod
    def setUpClass(cls):
        config = {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5435,
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
        self.assertIn('object', inspector.get_table_names())

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

@pytest.mark.usefixtures("psql_service")
class ObjectModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo Object"""
    
    @classmethod
    def setUpClass(cls):
        config = {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5435,
            "DB_NAME": "postgres",
        }
        cls.db = PsqlDatabase(config)

    def setUp(self):
        """Preparar la base de datos antes de cada prueba"""
        self.db.create_db()
        
    def tearDown(self):
        """Limpiar la base de datos después de cada prueba"""
        self.db.drop_db()
    
    def test_query_empty_table(self):
        """Verificar que se pueden realizar consultas a la tabla Object vacía"""
        with self.db.session() as session:
            query = select(Object)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj())), 0)
    
    def test_create_and_query_object(self):
        """Verify that objects can be created and queried in the database"""
        # Create a new object
        test_object = Object(
            oid=12345678,
            tid=1,
            sid=1,
            meanra=291.26922,
            meandec=72.38752,
            sigmara=0.000326,
            sigmade=0.000064,
            firstmjd=60058.47743,
            lastmjd=60207.21820,
            deltamjd=148.74077,
            n_det=42,
            n_forced=10,
            n_non_det=5,
            corrected=True,
            stellar=False
        )
        
        with self.db.session() as session:
            session.add(test_object)
        
        # Verify the object was saved
        with self.db.session() as session:
            query = select(Object)
            objects = list(session.execute(query).scalars())
            
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0].oid, 12345678)
    
    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Crear varios objetos
        obj_data = {
                "oid": 12345680,
                "tid": 1,
                "sid": 1,
                "meanra": 293.26922,
                "meandec": 74.38752,
                "sigmara": 0.000326,
                "sigmade": 0.000064,
                "firstmjd": 60058.47743,
                "lastmjd": 60207.21820,
                "deltamjd": 148.74077,
                "n_det": 44,
                "n_forced": 12,
                "n_non_det": 7,
                "corrected": False,
                "stellar": True
            }

        obj = Object(**obj_data)

        with self.db.session() as session:
                session.add(obj)
        
        # Verificar que se han guardado todos los objetos
        with self.db.session() as session:
            query = select(Object)
            saved_objects = list(session.execute(query).scalars())
            
            self.assertEqual(len(saved_objects), 1)
    
    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Crear varios objetos
        obj_data = {
                "oid": 12345680,
                "tid": 1,
                "sid": 1,
                "meanra": 293.26922,
                "meandec": 74.38752,
                "sigmara": 0.000326,
                "sigmade": 0.000064,
                "firstmjd": 60058.47743,
                "lastmjd": 60207.21820,
                "deltamjd": 148.74077,
                "n_det": 44,
                "n_forced": 12,
                "n_non_det": 7,
                "corrected": False,
                "stellar": True
            }

        obj = Object(**obj_data)
        
        with self.db.session() as session:
            session.add(obj)


@pytest.mark.usefixtures("psql_service")
class ZtfObjectModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo Object"""
    
    @classmethod
    def setUpClass(cls):
        config = {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5435,
            "DB_NAME": "postgres",
        }
        cls.db = PsqlDatabase(config)

    def setUp(self):
        """Preparar la base de datos antes de cada prueba"""
        self.db.create_db()
        
    def tearDown(self):
        """Limpiar la base de datos después de cada prueba"""
        self.db.drop_db()
    
    def test_query_empty_table(self):
        """Verificar que se pueden realizar consultas a la tabla ZtfObject vacía"""
        with self.db.session() as session:
            query = select(ZtfObject)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)
    
    def test_create_and_query_object(self):
        """Verificar que se pueden crear y consultar objetos en la base de datos"""
        # Crear un nuevo objeto
        test_object = ZtfObject(
            oid=12345680,
            g_r_max=0.043,
            g_r_max_corr=1.02,
            g_r_mean=3.02,
            g_r_mean_corr=0.001
        )
        
        with self.db.session() as session:
            session.add(test_object)
        
        # Verificar que el objeto se ha guardado
        with self.db.session() as session:
            query = select(ZtfObject)
            objects = list(session.execute(query))
            
            self.assertEqual(len(objects), 1)
    
    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Crear varios objetos con datos diferentes
        objects_data = [
            {
                "oid": 12345680,
                "g_r_max": 0.43,
                "g_r_max_corr": 1.2,
                "g_r_mean": 3.2,
                "g_r_mean_corr": 0.01
            },
            {
                "oid": 12345681,
                "g_r_max": 0.53,
                "g_r_max_corr": 1.4,
                "g_r_mean": 2.8,
                "g_r_mean_corr": 0.02
            },
            {
                "oid": 12345682,
                "g_r_max": 0.63,
                "g_r_max_corr": 1.6,
                "g_r_mean": 2.5,
                "g_r_mean_corr": 0.03
            }
        ]
        
        # Crear instancias de ZtfObject para cada conjunto de datos
        objects = [ZtfObject(**data) for data in objects_data]
        
        # Agregar todos los objetos en una sola sesión
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
        
        # Verificar que se han guardado todos los objetos
        with self.db.session() as session:
            query = select(ZtfObject)
            saved_objects = list(session.execute(query).scalars())
            
            # Verificar que el número de objetos guardados coincide con los creados
            self.assertEqual(len(saved_objects), len(objects_data))
    
    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Crear varios objetos
        obj_data = {"oid": 12345680,
            "g_r_max": 0.43,
            "g_r_max_corr": 1.2,
            "g_r_mean": 3.2,
            "g_r_mean_corr": 0.01
            }
        
        obj = ZtfObject(**obj_data)
        
        with self.db.session() as session:
            session.add(obj)

# class DetectionModelTest(unittest.TestCase):
#     """  Pruebas específicas para el modelo Detection"""
    
#     @classmethod
#     def setUpClas(cls):
#         config = {
#             "HOST": "localhost",
#             "USER": "postgres",
#             "PASSWORD": "postgres",
#             "PORT": 5435,
#             "DB_NAME": "postgres",
#         }
#         cls.session_options = {
#             "autocommit": False,
#             "autoflush": True,
#         }
#         cls.db = PsqlDatabase(config)
    
#     def setUp(self):
#         """Preparar la base de datos antes de cada prueba"""
#         self.db.create_db()

#     def tearDown(self):
#         """Limpiar la base de datos después de cada prueba"""
#         self.db.drop_db()

#     def test_query_empty_tables(self):
#         """Verificar que se pueden realizar consultas a la tabla Detection vacia"""
#         with self.db.session() as session:
#             query = select(Detection)
#             obj = session.execute(query)
#             # Verificar que no hay objectos (db vacia)
#             self.assertEqual(len(list(obj())), 0)

#     def test_create_and_query_objects(self):
#         """Verify that objects can be created and queried in the database"""
#         # Create a new object
#         test_object = Detection(
#             oid= 12345678901234567,  
#             measurement_id= 98765432109876543,  
#             mjd= 59000.123456,  
#             ra= 150.2345678,  
#             dec= -20.9876543,  
#             band= 2 
#         )

#         with self.db.session() as session:
#             session.add(test_object)

#         # Verificar que el objecto se ha guardado
#         with self.db.session() as session:
#             query = select(Detection)
#             objects = list(session.execute(query))

#             self.assertEqual(len(objects), 1)
#     def test_create_multiple_objects(self):




if __name__ == "__main__":
    unittest.main()