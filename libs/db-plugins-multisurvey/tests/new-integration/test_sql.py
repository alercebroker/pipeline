import unittest
import pytest
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, Integer, String, select, text
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

from db_plugins.db.sql.models_new import Base, Object, ZtfObject, Detection, ZtfDetection

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
            self.assertEqual(len(list(obj)), 0)
    
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
        # Crear varios objetos con datos diferentes

        objects_data = [
            {
                "oid": 12345678,
                "tid": 1,
                "sid": 1,
                "meanra": 291.26922,
                "meandec": 72.38752,
                "sigmara": 0.000326,
                "sigmade": 0.000064,
                "firstmjd": 60058.47743,
                "lastmjd": 60207.21820,
                "deltamjd": 148.74077,
                "n_det": 42,
                "n_forced": 10,
                "n_non_det": 5,
                "corrected": True,
                "stellar": False
            },
            {
                "oid": 12345679,
                "tid": 2,
                "sid": 1,
                "meanra": 290.54321,
                "meandec": 71.98765,
                "sigmara": 0.000412,
                "sigmade": 0.000089,
                "firstmjd": 60060.32145,
                "lastmjd": 60210.54321,
                "deltamjd": 150.22176,
                "n_det": 38,
                "n_forced": 12,
                "n_non_det": 7,
                "corrected": True,
                "stellar": True
            },
            {
                "oid": 12345680,
                "tid": 1,
                "sid": 2,
                "meanra": 292.87654,
                "meandec": 73.12345,
                "sigmara": 0.000287,
                "sigmade": 0.000053,
                "firstmjd": 60055.76543,
                "lastmjd": 60205.98765,
                "deltamjd": 150.22222,
                "n_det": 45,
                "n_forced": 8,
                "n_non_det": 3,
                "corrected": False,
                "stellar": False
            }
        ]
        
        # Crear instancias de Object para cada conjunto de datos
        objects = [Object(**data) for data in objects_data]
        
        # Agregar todos los objetos en una sola sesión
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
        
        # Verificar que se han guardado todos los objetos
        with self.db.session() as session:
            query = select(Object)
            saved_objects = list(session.execute(query).scalars())
            
            # Verificar que el número de objetos guardados coincide con los creados
            self.assertEqual(len(saved_objects), len(objects_data))
    
    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Crear varios objetos
        objects_data = [
            {
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
            },
            {
                "oid": 12345681,
                "tid": 2,
                "sid": 1,
                "meanra": 290.54321,
                "meandec": 71.98765,
                "sigmara": 0.000412,
                "sigmade": 0.000089,
                "firstmjd": 60060.32145,
                "lastmjd": 60210.54321,
                "deltamjd": 150.22176,
                "n_det": 38,
                "n_forced": 15,
                "n_non_det": 5,
                "corrected": True,
                "stellar": False
            },
            {
                "oid": 12345682,
                "tid": 1,
                "sid": 2,
                "meanra": 292.87654,
                "meandec": 73.12345,
                "sigmara": 0.000287,
                "sigmade": 0.000053,
                "firstmjd": 60055.76543,
                "lastmjd": 60205.98765,
                "deltamjd": 150.22222,
                "n_det": 45,
                "n_forced": 8,
                "n_non_det": 3,
                "corrected": True,
                "stellar": True
            }
        ]

        # Crear y agregar múltiples objetos a la base de datos
        objects = [Object(**data) for data in objects_data]
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
        
        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por stellar=True
            query1 = select(Object).where(Object.stellar == True)
            stellar_objects = list(session.execute(query1).scalars())
            self.assertEqual(len(stellar_objects), 2)  # Debería encontrar 2 objetos
            
            # Filtrar por tid=1 y stellar=True
            query2 = select(Object).where(Object.tid == 1, Object.stellar == True)
            tid1_stellar_objects = list(session.execute(query2).scalars())
            self.assertEqual(len(tid1_stellar_objects), 2)  # Debería encontrar 2 objetos
            
            # Filtrar por n_det > 40
            query3 = select(Object).where(Object.n_det > 40)
            high_det_objects = list(session.execute(query3).scalars())
            self.assertEqual(len(high_det_objects), 2)  # Debería encontrar 2 objetos
            
            # Filtrar por rango de coordenadas
            query4 = select(Object).where(
                Object.meanra.between(291.0, 295.0),
                Object.meandec.between(73.0, 75.0)
            )
            coord_range_objects = list(session.execute(query4).scalars())
            self.assertEqual(len(coord_range_objects), 2)  # Debería encontrar 2 objetos
            
            # Verificar que los objetos filtrados son los esperados (usando OID)
            self.assertEqual(
                sorted([obj.oid for obj in stellar_objects]), 
                sorted([12345680, 12345682])
            )


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
                "g_r_max": 0.55,
                "g_r_max_corr": 1.4,
                "g_r_mean": 2.8,
                "g_r_mean_corr": 0.02
            },
            {
                "oid": 12345682,
                "g_r_max": 0.65,
                "g_r_max_corr": 1.6,
                "g_r_mean": 3.4,
                "g_r_mean_corr": 0.03
            },
            {
                "oid": 12345683,
                "g_r_max": 0.35,
                "g_r_max_corr": 1.1,
                "g_r_mean": 2.5,
                "g_r_mean_corr": 0.015
            }
        ]
        
        # Crear y agregar múltiples objetos a la base de datos
        objects = [ZtfObject(**data) for data in objects_data]
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
        
        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por g_r_max > 0.5
            query1 = select(ZtfObject).where(ZtfObject.g_r_max > 0.5)
            high_gr_max_objects = list(session.execute(query1).scalars())
            self.assertEqual(len(high_gr_max_objects), 2)  # Debería encontrar 2 objetos
            
            # Filtrar por g_r_mean > 3.0
            query2 = select(ZtfObject).where(ZtfObject.g_r_mean > 3.0)
            high_gr_mean_objects = list(session.execute(query2).scalars())
            self.assertEqual(len(high_gr_mean_objects), 2)  # Debería encontrar 2 objetos
            
            # Filtrar por rango de g_r_max_corr
            query3 = select(ZtfObject).where(ZtfObject.g_r_max_corr.between(1.1, 1.5))
            mid_corr_objects = list(session.execute(query3).scalars())
            self.assertEqual(len(mid_corr_objects), 3)  # Debería encontrar 3 objetos
            
            # Filtrar por múltiples condiciones
            query4 = select(ZtfObject).where(
                ZtfObject.g_r_max > 0.4,
                ZtfObject.g_r_mean < 3.0
            )
            combined_filter_objects = list(session.execute(query4).scalars())
            self.assertEqual(len(combined_filter_objects), 1)  # Debería encontrar 1 objeto
            
            # Verificar que los objetos filtrados son los esperados (usando oid)
            self.assertEqual(
                sorted([obj.oid for obj in high_gr_max_objects]), 
                sorted([12345681, 12345682])
            )
            
            self.assertEqual(
                sorted([obj.oid for obj in high_gr_mean_objects]), 
                sorted([12345680, 12345682])
            )

@pytest.mark.usefixtures("psql_service")
class DetectionModelTest(unittest.TestCase):
    """  Pruebas específicas para el modelo Detection"""
    
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

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla Detection vacia"""
        with self.db.session() as session:
            query = select(Detection)
            obj = session.execute(query)
            # Verificar que no hay objectos (db vacia)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_objects(self):
        
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(
            oid=12345678901234567,
            tid=1,
            sid=1,
            meanra=293.26922,
            meandec=74.38752,
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
        
        # Guardar el objeto padre
        with self.db.session() as session:
            session.add(parent_object)
        
        """Verify that objects can be created and queried in the database"""
        # Create a new object
        test_object = Detection(
            oid= 12345678901234567,  
            measurement_id= 98765432109876543,  
            mjd= 59000.123456,  
            ra= 150.2345678,  
            dec= -20.9876543,  
            band= 2 
        )

        with self.db.session() as session:
            session.add(test_object)

        # Verificar que el objecto se ha guardado
        with self.db.session() as session:
            query = select(Detection)
            objects = list(session.execute(query))

            self.assertEqual(len(objects), 1)

    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(
            oid=12345680,
            tid=1,
            sid=1,
            meanra=293.26922,
            meandec=74.38752,
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
        
        # Guardar el objeto padre
        with self.db.session() as session:
            session.add(parent_object)
        
        # Crear varios objetos con datos diferentes
        objects_data = [
            {
                "oid": 12345680,
                "measurement_id": 1001,
                "mjd": 60080.5432,
                "ra": 293.26945,
                "dec": 74.38762,
                "band": 1
            },
            {
                "oid": 12345680,
                "measurement_id": 1002,
                "mjd": 60085.6543,
                "ra": 293.26930,
                "dec": 74.38758,
                "band": 2
            },
            {
                "oid": 12345680,
                "measurement_id": 1003,
                "mjd": 60090.7654,
                "ra": 293.26918,
                "dec": 74.38750,
                "band": 1
            }
        ]

        # Crear instancias de Detection para cada conjunto de datos
        objects = [Detection(**data) for data in objects_data]

        # Agregar todos los objectos en una sola sesión
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
        
        # Verificar que se han guardado todos los objectos
        with self.db.session() as session:
            query = select(Detection)
            saved_objects = list(session.execute(query).scalars())

            # Ferificar que el numero de objectos guardado coincide con los creados
            self.assertEqual(len(saved_objects), len(objects_data))

    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(
            oid=12345680,
            tid=1,
            sid=1,
            meanra=293.26922,
            meandec=74.38752,
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
        
        # Guardar el objeto padre
        with self.db.session() as session:
            session.add(parent_object)
        
        # Crear varios objetos
        objects_data = [
            {
                "oid": 12345680,
                "measurement_id": 1001,
                "mjd": 60080.5432,
                "ra": 293.26945,
                "dec": 74.38762,
                "band": 1
            },
            {
                "oid": 12345680,
                "measurement_id": 1002,
                "mjd": 60085.6543,
                "ra": 293.26930,
                "dec": 74.38758,
                "band": 2
            },
            {
                "oid": 12345680,
                "measurement_id": 1003,
                "mjd": 60090.7654,
                "ra": 293.26918,
                "dec": 74.38750,
                "band": 1
            }
        ]

        # Crear y agregar multiples objectos a la DB
        objects = [Detection(**data) for data in objects_data]
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
        
        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por band = 1
            query1 = select(Detection).where(Detection.band == 1)
            band1_detections = list(session.execute(query1).scalars())
            self.assertEqual(len(band1_detections), 2)  # Debería encontrar 2 detecciones
            
            # Filtrar por mjd > 60085.0
            query2 = select(Detection).where(Detection.mjd > 60085.0)
            recent_detections = list(session.execute(query2).scalars())
            self.assertEqual(len(recent_detections), 2)  # Debería encontrar 2 detecciones
            
            # Filtrar por rango de coordenadas
            query3 = select(Detection).where(
                Detection.ra.between(293.269, 293.270),
                Detection.dec.between(74.387, 74.388)
            )
            coord_detections = list(session.execute(query3).scalars())
            self.assertEqual(len(coord_detections), 3)  # Debería encontrar todas las detecciones
            
            # Filtro combinado
            query4 = select(Detection).where(
                Detection.band == 1,
                Detection.mjd > 60085.0
            )
            combined_detections = list(session.execute(query4).scalars())
            self.assertEqual(len(combined_detections), 1)  # Debería encontrar 1 detección
            
            # Verificar que los objetos filtrados son los esperados
            self.assertEqual(
                sorted([det.measurement_id for det in band1_detections]), 
                sorted([1001, 1003])
            )
            
            self.assertEqual(
                sorted([det.measurement_id for det in recent_detections]), 
                sorted([1002, 1003])
            )
            
            self.assertEqual(
                combined_detections[0].measurement_id,
                1003
            )

@pytest.mark.usefixtures("psql_service")
class ZtfDetectionModelTest(unittest.TestCase):
    """ Pruebas especificas para el modelo ZtfDetection """
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

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla Detection vacia"""
        with self.db.session() as session:
            query = select(ZtfDetection)
            obj = session.execute(query)
            # Verificar que no hay objectos (db vacia)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_detection(self):
        """Verificar que se pueden crear y consultar detecciones en la base de datos"""
        # Primero, crear el objeto al que hará referencia la detección
        test_object = Object(
            oid=12345680,
            ndethist=5,
            ncovhist=5,
            mjdstarthist=58000.0,
            mjdendhist=59000.0,
            corrected=False,
            stellar=False,
            ndet=5,
            firstmjd=58000.0,
            lastmjd=59000.0,
            deltajd=1000.0,
            meanra=150.0,
            meandec=20.0,
            sigmara=0.1,
            sigmadec=0.1,
            class_name="variable",
            class_score=0.9
        )
        
        with self.db.session() as session:
            session.add(test_object)
            session.commit()
        
        # Crear una nueva detección
        test_detection = ZtfDetection(
            oid=12345680,
            measurement_id=987654321,
            pid=123456,
            diffmaglim=19.5,
            isdiffpos=True,
            nid=1,
            magpsf=18.2,
            sigmapsf=0.1,
            magap=18.3,
            sigmagap=0.15,
            distnr=0.5,
            rb=0.8,
            rbversion="t2",
            drb=0.9,
            drbversion="t2",
            magapbig=18.0,
            sigmagapbig=0.2,
            rband=2,
            magpsf_corr=0,
            sigmapsf_corr=0,
            sigmapsf_corr_ext=0,
            corrected=False,
            dubious=False,
            parent_candid=12345,
            has_stamp=True,
            step_id_corr=0
        )
        
        with self.db.session() as session:
            session.add(test_detection)
        
        # Verificar que la detección se ha guardado
        with self.db.session() as session:
            query = select(ZtfDetection)
            detections = list(session.execute(query))
            
            self.assertEqual(len(detections), 1)
    
    def test_create_and_query_detection(self):
        """Verificar que se pueden crear y consultar detecciones en la base de datos"""
        # Primero, crear el objeto al que hará referencia la detección
        test_object = Object(
            oid=12345680,
            tid=1,
            sid=1,
            meanra=150.0,
            meandec=20.0,
            sigmara=0.1,
            sigmade=0.1,
            firstmjd=58000.0,
            lastmjd=59000.0,
            deltamjd=1000.0,
            n_det=5,
            n_forced=2,
            n_non_det=10,
            corrected=False,
            stellar=False
        )
        
        with self.db.session() as session:
            session.add(test_object)
            session.commit()
        
        # Crear una nueva detección
        test_detection = ZtfDetection(
            oid=12345680,
            measurement_id=987654321,
            pid=123456,
            diffmaglim=19.5,
            isdiffpos=True,
            nid=1,
            magpsf=18.2,
            sigmapsf=0.1,
            magap=18.3,
            sigmagap=0.15,
            distnr=0.5,
            rb=0.8,
            rbversion="t2",
            drb=0.9,
            drbversion="t2",
            magapbig=18.0,
            sigmagapbig=0.2,
            rband=2,
            magpsf_corr=0,
            sigmapsf_corr=0,
            sigmapsf_corr_ext=0,
            corrected=False,
            dubious=False,
            parent_candid=12345,
            has_stamp=True,
            step_id_corr=0
        )
        
        with self.db.session() as session:
            session.add(test_detection)
        
        # Verificar que la detección se ha guardado
        with self.db.session() as session:
            query = select(ZtfDetection)
            detections = list(session.execute(query))
            
            self.assertEqual(len(detections), 1)
    
    def test_create_multiple_detections(self):
        """Verificar que se pueden crear y consultar múltiples detecciones"""
        # Primero, crear los objetos a los que harán referencia las detecciones
        objects_data = [
            {
                "oid": 12345680,
                "tid": 1,
                "sid": 1,
                "meanra": 150.0,
                "meandec": 20.0,
                "sigmara": 0.1,
                "sigmade": 0.1,
                "firstmjd": 58000.0,
                "lastmjd": 59000.0,
                "deltamjd": 1000.0,
                "n_det": 5,
                "n_forced": 2,
                "n_non_det": 10,
                "corrected": False,
                "stellar": False
            },
            {
                "oid": 12345681,
                "tid": 1,
                "sid": 2,
                "meanra": 151.0,
                "meandec": 21.0,
                "sigmara": 0.1,
                "sigmade": 0.1,
                "firstmjd": 58100.0,
                "lastmjd": 59100.0,
                "deltamjd": 1000.0,
                "n_det": 6,
                "n_forced": 3,
                "n_non_det": 8,
                "corrected": False,
                "stellar": True
            }
        ]
        
        # Crear y agregar los objetos
        objects = [Object(**data) for data in objects_data]
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
            session.commit()
        
        # Crear varias detecciones con datos diferentes
        detections_data = [
            {
                "oid": 12345680,
                "measurement_id": 987654321,
                "pid": 123456,
                "diffmaglim": 19.5,
                "isdiffpos": True,
                "nid": 1,
                "magpsf": 18.2,
                "sigmapsf": 0.1,
                "magap": 18.3,
                "sigmagap": 0.15,
                "distnr": 0.5,
                "rb": 0.8,
                "rbversion": "t2",
                "drb": 0.9,
                "drbversion": "t2",
                "magapbig": 18.0,
                "sigmagapbig": 0.2,
                "rband": 2,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": False,
                "dubious": False,
                "parent_candid": 12345,
                "has_stamp": True,
                "step_id_corr": 0
            },
            {
                "oid": 12345680,
                "measurement_id": 987654322,
                "pid": 123457,
                "diffmaglim": 19.6,
                "isdiffpos": True,
                "nid": 2,
                "magpsf": 18.3,
                "sigmapsf": 0.12,
                "magap": 18.4,
                "sigmagap": 0.16,
                "distnr": 0.6,
                "rb": 0.85,
                "rbversion": "t2",
                "drb": 0.91,
                "drbversion": "t2",
                "magapbig": 18.1,
                "sigmagapbig": 0.21,
                "rband": 2,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": False,
                "dubious": False,
                "parent_candid": 12346,
                "has_stamp": True,
                "step_id_corr": 0
            },
            {
                "oid": 12345681,
                "measurement_id": 987654323,
                "pid": 123458,
                "diffmaglim": 19.7,
                "isdiffpos": False,
                "nid": 3,
                "magpsf": 18.4,
                "sigmapsf": 0.13,
                "magap": 18.5,
                "sigmagap": 0.17,
                "distnr": 0.7,
                "rb": 0.86,
                "rbversion": "t2",
                "drb": 0.92,
                "drbversion": "t2",
                "magapbig": 18.2,
                "sigmagapbig": 0.22,
                "rband": 1,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": False,
                "dubious": True,
                "parent_candid": 12347,
                "has_stamp": False,
                "step_id_corr": 0
            }
        ]
        
        # Crear instancias de ZtfDetection para cada conjunto de datos
        detections = [ZtfDetection(**data) for data in detections_data]
        
        # Agregar todas las detecciones en una sola sesión
        with self.db.session() as session:
            for det in detections:
                session.add(det)
        
        # Verificar que se han guardado todas las detecciones
        with self.db.session() as session:
            query = select(ZtfDetection)
            saved_detections = list(session.execute(query).scalars())
            
            # Verificar que el número de detecciones guardadas coincide con las creadas
            self.assertEqual(len(saved_detections), len(detections_data))
    
    def test_filter_detections(self):
        """Verificar que se pueden filtrar detecciones con criterios específicos"""
        # Primero, crear los objetos a los que harán referencia las detecciones
        objects_data = [
            {
                "oid": 12345680,
                "tid": 1,
                "sid": 1,
                "meanra": 150.0,
                "meandec": 20.0,
                "sigmara": 0.1,
                "sigmade": 0.1,
                "firstmjd": 58000.0,
                "lastmjd": 59000.0,
                "deltamjd": 1000.0,
                "n_det": 5,
                "n_forced": 2,
                "n_non_det": 10,
                "corrected": False,
                "stellar": False
            },
            {
                "oid": 12345681,
                "tid": 1,
                "sid": 2,
                "meanra": 151.0,
                "meandec": 21.0,
                "sigmara": 0.1,
                "sigmade": 0.1,
                "firstmjd": 58100.0,
                "lastmjd": 59100.0,
                "deltamjd": 1000.0,
                "n_det": 6,
                "n_forced": 3,
                "n_non_det": 8,
                "corrected": False,
                "stellar": True
            },
            {
                "oid": 12345682,
                "tid": 2,
                "sid": 1,
                "meanra": 152.0,
                "meandec": 22.0,
                "sigmara": 0.1,
                "sigmade": 0.1,
                "firstmjd": 58200.0,
                "lastmjd": 59200.0,
                "deltamjd": 1000.0,
                "n_det": 7,
                "n_forced": 4,
                "n_non_det": 12,
                "corrected": True,
                "stellar": False
            }
        ]
        
        # Crear y agregar los objetos
        objects = [Object(**data) for data in objects_data]
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)
            session.commit()
            
        # Crear varias detecciones
        detections_data = [
            {
                "oid": 12345680,
                "measurement_id": 987654321,
                "pid": 123456,
                "diffmaglim": 19.5,
                "isdiffpos": True,
                "nid": 1,
                "magpsf": 18.2,
                "sigmapsf": 0.1,
                "magap": 18.3,
                "sigmagap": 0.15,
                "distnr": 0.5,
                "rb": 0.8,
                "rbversion": "t2",
                "drb": 0.9,
                "drbversion": "t2",
                "magapbig": 18.0,
                "sigmagapbig": 0.2,
                "rband": 2,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": False,
                "dubious": False,
                "parent_candid": 12345,
                "has_stamp": True,
                "step_id_corr": 0
            },
            {
                "oid": 12345680,
                "measurement_id": 987654322,
                "pid": 123457,
                "diffmaglim": 19.6,
                "isdiffpos": True,
                "nid": 2,
                "magpsf": 17.8,
                "sigmapsf": 0.12,
                "magap": 17.9,
                "sigmagap": 0.16,
                "distnr": 0.6,
                "rb": 0.85,
                "rbversion": "t2",
                "drb": 0.91,
                "drbversion": "t2",
                "magapbig": 17.7,
                "sigmagapbig": 0.21,
                "rband": 2,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": False,
                "dubious": False,
                "parent_candid": 12346,
                "has_stamp": True,
                "step_id_corr": 0
            },
            {
                "oid": 12345681,
                "measurement_id": 987654323,
                "pid": 123458,
                "diffmaglim": 19.7,
                "isdiffpos": False,
                "nid": 3,
                "magpsf": 18.5,
                "sigmapsf": 0.13,
                "magap": 18.6,
                "sigmagap": 0.17,
                "distnr": 0.7,
                "rb": 0.86,
                "rbversion": "t2",
                "drb": 0.92,
                "drbversion": "t2",
                "magapbig": 18.3,
                "sigmagapbig": 0.22,
                "rband": 1,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": False,
                "dubious": True,
                "parent_candid": 12347,
                "has_stamp": False,
                "step_id_corr": 0
            },
            {
                "oid": 12345682,
                "measurement_id": 987654324,
                "pid": 123459,
                "diffmaglim": 19.8,
                "isdiffpos": False,
                "nid": 4,
                "magpsf": 19.0,
                "sigmapsf": 0.15,
                "magap": 19.1,
                "sigmagap": 0.19,
                "distnr": 0.8,
                "rb": 0.75,
                "rbversion": "t2",
                "drb": 0.88,
                "drbversion": "t2",
                "magapbig": 18.9,
                "sigmagapbig": 0.25,
                "rband": 1,
                "magpsf_corr": 0,
                "sigmapsf_corr": 0,
                "sigmapsf_corr_ext": 0,
                "corrected": True,
                "dubious": False,
                "parent_candid": 12348,
                "has_stamp": True,
                "step_id_corr": 1
            }
        ]
        
        # Crear y agregar múltiples detecciones a la base de datos
        detections = [ZtfDetection(**data) for data in detections_data]
        with self.db.session() as session:
            for det in detections:
                session.add(det)
        
        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por magpsf > 18.0
            query1 = select(ZtfDetection).where(ZtfDetection.magpsf > 18.0)
            bright_detections = list(session.execute(query1).scalars())
            self.assertEqual(len(bright_detections), 3)  # Debería encontrar 3 detecciones
            
            # Filtrar por isdiffpos = True
            query2 = select(ZtfDetection).where(ZtfDetection.isdiffpos == True)
            positive_diff_detections = list(session.execute(query2).scalars())
            self.assertEqual(len(positive_diff_detections), 2)  # Debería encontrar 2 detecciones
            
            # Filtrar por rango de rb
            query3 = select(ZtfDetection).where(ZtfDetection.rb.between(0.8, 0.9))
            mid_rb_detections = list(session.execute(query3).scalars())
            self.assertEqual(len(mid_rb_detections), 3)  # Debería encontrar 3 detecciones
            
            # Filtrar por múltiples condiciones
            query4 = select(ZtfDetection).where(
                ZtfDetection.magpsf > 18.0,
                ZtfDetection.isdiffpos == False
            )
            combined_filter_detections = list(session.execute(query4).scalars())
            self.assertEqual(len(combined_filter_detections), 2)  # Debería encontrar 2 detecciones
            
            # Verificar que las detecciones filtradas son las esperadas (usando oid y measurement_id)
            self.assertEqual(
                sorted([(det.oid, det.measurement_id) for det in bright_detections]), 
                sorted([
                    (12345680, 987654321), 
                    (12345681, 987654323), 
                    (12345682, 987654324)
                ])
            )
            
            self.assertEqual(
                sorted([(det.oid, det.measurement_id) for det in positive_diff_detections]), 
                sorted([
                    (12345680, 987654321),
                    (12345680, 987654322)
                ])
            )
if __name__ == "__main__":
    unittest.main()