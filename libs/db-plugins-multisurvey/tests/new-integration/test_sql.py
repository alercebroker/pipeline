import unittest
import pytest
from sqlalchemy import (
    create_engine,
    inspect,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    select,
    text,
)
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

from db_plugins.db.sql.models_new import (
    Base,
    Object,
    ZtfObject,
    Detection,
    ZtfDetection,
    ForcedPhotometry,
    ZtfForcedPhotometry,
    NonDetection,
)
from test_data import (
    OBJECT_DATA,
    ZTF_OBJECT_DATA,
    DETECTION_DATA,
    ZTF_DETECTION_DATA,
    FORCED_PHOTOMETRY_DATA,
    ZTF_FORCED_PHOTOMETRY_DATA,
    NON_DETECTION_DATA,
)


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
            sessionmaker(autocommit=False, autoflush=True, bind=self._engine)
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
        self.assertIn("object", inspector.get_table_names())

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
        """Verificar que se pueden crear y consultar objetos en la base de datos"""
        # Crear un nuevo objeto
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.db.session() as session:
            session.add(test_object)

        # Verificar que el objeto se ha guardado
        with self.db.session() as session:
            query = select(Object)
            objects = list(session.execute(query).scalars())

            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0].oid, OBJECT_DATA["filter"][0]["oid"])

    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Crear instancias de Object para cada conjunto de datos
        objects = [Object(**data) for data in OBJECT_DATA["filter"]]

        # Agregar todos los objetos en una sola sesión
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)

        # Verificar que se han guardado todos los objetos
        with self.db.session() as session:
            query = select(Object)
            saved_objects = list(session.execute(query).scalars())

            # Verificar que el número de objetos guardados coincide con los creados
            self.assertEqual(len(saved_objects), len(OBJECT_DATA["filter"]))

    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Crear y agregar múltiples objetos a la base de datos
        objects = [Object(**data) for data in OBJECT_DATA["filter"]]
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
            self.assertEqual(
                len(tid1_stellar_objects), 2
            )  # Debería encontrar 2 objetos

            # Filtrar por n_det > 40
            query3 = select(Object).where(Object.n_det > 40)
            high_det_objects = list(session.execute(query3).scalars())
            self.assertEqual(len(high_det_objects), 2)  # Debería encontrar 2 objetos

            # Filtrar por rango de coordenadas
            query4 = select(Object).where(
                Object.meanra.between(291.0, 295.0), Object.meandec.between(73.0, 75.0)
            )
            coord_range_objects = list(session.execute(query4).scalars())
            self.assertEqual(len(coord_range_objects), 2)  # Debería encontrar 2 objetos

            # Verificar que los objetos filtrados son los esperados (usando OID)
            self.assertEqual(
                sorted([obj.oid for obj in stellar_objects]),
                sorted([12345680, 12345682]),
            )


@pytest.mark.usefixtures("psql_service")
class ZtfObjectModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo ZtfObject"""

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
        test_object = ZtfObject(**ZTF_OBJECT_DATA["filter"][0])

        with self.db.session() as session:
            session.add(test_object)

        # Verificar que el objeto se ha guardado
        with self.db.session() as session:
            query = select(ZtfObject)
            objects = list(session.execute(query))

            self.assertEqual(len(objects), 1)

    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Crear instancias de ZtfObject para cada conjunto de datos
        objects = [ZtfObject(**data) for data in ZTF_OBJECT_DATA["filter"]]

        # Agregar todos los objetos en una sola sesión
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)

        # Verificar que se han guardado todos los objetos
        with self.db.session() as session:
            query = select(ZtfObject)
            saved_objects = list(session.execute(query).scalars())

            # Verificar que el número de objetos guardados coincide con los creados
            self.assertEqual(len(saved_objects), len(ZTF_OBJECT_DATA["filter"]))

    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Crear y agregar múltiples objetos a la base de datos
        objects = [ZtfObject(**data) for data in ZTF_OBJECT_DATA["filter"]]
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
            self.assertEqual(
                len(high_gr_mean_objects), 2
            )  # Debería encontrar 2 objetos

            # Filtrar por rango de g_r_max_corr
            query3 = select(ZtfObject).where(ZtfObject.g_r_max_corr.between(1.1, 1.5))
            mid_corr_objects = list(session.execute(query3).scalars())
            self.assertEqual(len(mid_corr_objects), 3)  # Debería encontrar 3 objetos

            # Filtrar por múltiples condiciones
            query4 = select(ZtfObject).where(
                ZtfObject.g_r_max > 0.4, ZtfObject.g_r_mean < 3.0
            )
            combined_filter_objects = list(session.execute(query4).scalars())
            self.assertEqual(
                len(combined_filter_objects), 1
            )  # Debería encontrar 1 objeto

            # Verificar que los objetos filtrados son los esperados (usando oid)
            self.assertEqual(
                sorted([obj.oid for obj in high_gr_max_objects]),
                sorted([12345681, 12345682]),
            )

            self.assertEqual(
                sorted([obj.oid for obj in high_gr_mean_objects]),
                sorted([12345680, 12345682]),
            )


@pytest.mark.usefixtures("psql_service")
class DetectionModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo Detection"""

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
        """Verificar que se pueden realizar consultas a la tabla Detection vacía"""
        with self.db.session() as session:
            query = select(Detection)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_objects(self):
        """Verificar que se pueden crear y consultar objetos en la base de datos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(**OBJECT_DATA["filter"][0])

        # Guardar el objeto padre
        with self.db.session() as session:
            session.add(parent_object)

        # Crear un nuevo objeto de detección
        test_object = Detection(**DETECTION_DATA["filter"][0])

        with self.db.session() as session:
            session.add(test_object)

        # Verificar que el objeto se ha guardado
        with self.db.session() as session:
            query = select(Detection)
            objects = list(session.execute(query))

            self.assertEqual(len(objects), 1)

    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(**OBJECT_DATA["filter"][0])

        # Guardar el objeto padre
        with self.db.session() as session:
            session.add(parent_object)

        # Crear instancias de Detection para cada conjunto de datos
        objects = [Detection(**data) for data in DETECTION_DATA["filter"]]

        # Agregar todos los objetos en una sola sesión
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)

        # Verificar que se han guardado todos los objetos
        with self.db.session() as session:
            query = select(Detection)
            saved_objects = list(session.execute(query).scalars())

            # Verificar que el número de objetos guardado coincide con los creados
            self.assertEqual(len(saved_objects), len(DETECTION_DATA["filter"]))

    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(**OBJECT_DATA["filter"][0])

        # Guardar el objeto padre
        with self.db.session() as session:
            session.add(parent_object)

        # Crear y agregar múltiples objetos a la DB
        objects = [Detection(**data) for data in DETECTION_DATA["filter"]]
        with self.db.session() as session:
            for obj in objects:
                session.add(obj)

        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por band = 1
            query1 = select(Detection).where(Detection.band == 1)
            band1_detections = list(session.execute(query1).scalars())
            self.assertEqual(
                len(band1_detections), 2
            )  # Debería encontrar 2 detecciones

            # Filtrar por mjd > 60085.0
            query2 = select(Detection).where(Detection.mjd > 60085.0)
            recent_detections = list(session.execute(query2).scalars())
            self.assertEqual(
                len(recent_detections), 2
            )  # Debería encontrar 2 detecciones

            # Filtrar por rango de coordenadas
            query3 = select(Detection).where(
                Detection.ra.between(293.269, 293.270),
                Detection.dec.between(74.387, 74.388),
            )
            coord_detections = list(session.execute(query3).scalars())
            self.assertEqual(
                len(coord_detections), 3
            )  # Debería encontrar todas las detecciones

            # Filtro combinado
            query4 = select(Detection).where(
                Detection.band == 1, Detection.mjd > 60085.0
            )
            combined_detections = list(session.execute(query4).scalars())
            self.assertEqual(
                len(combined_detections), 1
            )  # Debería encontrar 1 detección

            # Verificar que los objetos filtrados son los esperados
            self.assertEqual(
                sorted([det.measurement_id for det in band1_detections]),
                sorted([1001, 1003]),
            )

            self.assertEqual(
                sorted([det.measurement_id for det in recent_detections]),
                sorted([1002, 1003]),
            )

            self.assertEqual(combined_detections[0].measurement_id, 1003)


@pytest.mark.usefixtures("psql_service")
class ZtfDetectionModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo ZtfDetection"""

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
        """Verificar que se pueden realizar consultas a la tabla ZtfDetection vacía"""
        with self.db.session() as session:
            query = select(ZtfDetection)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_detection(self):
        """Verificar que se pueden crear y consultar detecciones en la base de datos"""
        # Primero, crear el objeto al que hará referencia la detección
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.db.session() as session:
            session.add(test_object)
            session.commit()

        # Crear una nueva detección
        test_detection = ZtfDetection(**ZTF_DETECTION_DATA["filter"][0])

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
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear instancias de ZtfDetection para cada conjunto de datos
        detections = [ZtfDetection(**data) for data in ZTF_DETECTION_DATA["filter"]]

        # Agregar todas las detecciones en una sola sesión
        with self.db.session() as session:
            for det in detections:
                session.add(det)

        # Verificar que se han guardado todas las detecciones
        with self.db.session() as session:
            query = select(ZtfDetection)
            saved_detections = list(session.execute(query).scalars())

            # Verificar que el número de detecciones guardadas coincide con las creadas
            self.assertEqual(len(saved_detections), len(ZTF_DETECTION_DATA["filter"]))

    def test_filter_detections(self):
        """Verificar que se pueden filtrar detecciones con criterios específicos"""
        # Primero, crear los objetos a los que harán referencia las detecciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples detecciones a la base de datos
        detections = [ZtfDetection(**data) for data in ZTF_DETECTION_DATA["filter"]]
        with self.db.session() as session:
            for det in detections:
                session.add(det)

        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por magpsf > 18.0
            query1 = select(ZtfDetection).where(ZtfDetection.magpsf > 18.0)
            bright_detections = list(session.execute(query1).scalars())
            self.assertEqual(
                len(bright_detections), 3
            )  # Debería encontrar 3 detecciones

            # Filtrar por isdiffpos = True
            query2 = select(ZtfDetection).where(ZtfDetection.isdiffpos == True)
            positive_diff_detections = list(session.execute(query2).scalars())
            self.assertEqual(
                len(positive_diff_detections), 2
            )  # Debería encontrar 2 detecciones

            # Filtrar por rango de rb
            query3 = select(ZtfDetection).where(ZtfDetection.rb.between(0.8, 0.9))
            mid_rb_detections = list(session.execute(query3).scalars())
            self.assertEqual(
                len(mid_rb_detections), 3
            )  # Debería encontrar 3 detecciones

            # Filtrar por múltiples condiciones
            query4 = select(ZtfDetection).where(
                ZtfDetection.magpsf > 18.0, ZtfDetection.isdiffpos == False
            )
            combined_filter_detections = list(session.execute(query4).scalars())
            self.assertEqual(
                len(combined_filter_detections), 2
            )  # Debería encontrar 2 detecciones

            # Verificar que las detecciones filtradas son las esperadas (usando oid y measurement_id)
            self.assertEqual(
                sorted([(det.oid, det.measurement_id) for det in bright_detections]),
                sorted(
                    [
                        (12345680, 987654321),
                        (12345681, 987654323),
                        (12345682, 987654324),
                    ]
                ),
            )

            self.assertEqual(
                sorted(
                    [(det.oid, det.measurement_id) for det in positive_diff_detections]
                ),
                sorted([(12345680, 987654321), (12345680, 987654322)]),
            )


@pytest.mark.usefixtures("psql_service")
class ForcedPhotometryModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo ForcedPhotometry"""

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
        """Verificar que se pueden realizar consultas a la tabla ForcedPhotometry vacía"""
        with self.db.session() as session:
            query = select(ForcedPhotometry)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_forced_photometry(self):
        """Verificar que se pueden crear y consultar mediciones de fotometría forzada en la base de datos"""
        # Primero, crear el objeto al que hará referencia la fotometría forzada
        test_object = Object(**OBJECT_DATA["filter"][0])
        with self.db.session() as session:
            session.add(test_object)
            session.commit()
            object_oid = test_object.oid

        # Crear una nueva entrada de fotometría forzada
        test_forced_photometry = ForcedPhotometry(
            oid=object_oid,
            measurement_id=987654321,
            mjd=58765.4321,
            ra=150.123,
            dec=20.456,
            band=1,
        )

        with self.db.session() as session:
            session.add(test_forced_photometry)

        # Verificar que la medición se ha guardado
        with self.db.session() as session:
            query = select(ForcedPhotometry)
            measurements = list(session.execute(query))

            self.assertEqual(len(measurements), 1)

    def test_create_multiple_forced_photometry(self):
        """Verificar que se pueden crear y consultar múltiples mediciones de fotometría forzada"""
        # Primero, crear los objetos a los que harán referencia las mediciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear datos para varias mediciones de fotometría forzada
        photometry_data = FORCED_PHOTOMETRY_DATA["filter"]

        # Crear instancias de ForcedPhotometry para cada conjunto de datos
        measurements = [ForcedPhotometry(**data) for data in photometry_data]

        # Agregar todas las mediciones en una sola sesión
        with self.db.session() as session:
            for m in measurements:
                session.add(m)

        # Verificar que se han guardado todas las mediciones
        with self.db.session() as session:
            query = select(ForcedPhotometry)
            saved_measurements = list(session.execute(query).scalars())

            # Verificar que el número de mediciones guardadas coincide con las creadas
            self.assertEqual(len(saved_measurements), len(photometry_data))

    def test_filter_forced_photometry(self):
        """Verificar que se pueden filtrar mediciones de fotometría forzada con criterios específicos"""
        # Primero, crear los objetos a los que harán referencia las mediciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples mediciones a la base de datos
        measurements = [
            ForcedPhotometry(**data) for data in FORCED_PHOTOMETRY_DATA["filter"]
        ]
        with self.db.session() as session:
            for m in measurements:
                session.add(m)

        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por mjd > 58766
            query1 = select(ForcedPhotometry).where(ForcedPhotometry.mjd > 58766)
            later_measurements = list(session.execute(query1).scalars())
            self.assertEqual(
                len(later_measurements), 3
            )  # Debería encontrar 3 mediciones

            # Filtrar por band = 1
            query2 = select(ForcedPhotometry).where(ForcedPhotometry.band == 1)
            band1_measurements = list(session.execute(query2).scalars())
            self.assertEqual(
                len(band1_measurements), 2
            )  # Debería encontrar 2 mediciones

            # Filtrar por rango de dec
            query3 = select(ForcedPhotometry).where(
                ForcedPhotometry.dec.between(70, 73)
            )
            mid_dec_measurements = list(session.execute(query3).scalars())
            self.assertEqual(
                len(mid_dec_measurements), 1
            )  # Debería encontrar 1 medición

            # Filtrar por múltiples condiciones
            query4 = select(ForcedPhotometry).where(
                ForcedPhotometry.mjd > 58766, ForcedPhotometry.band == 1
            )
            combined_filter_measurements = list(session.execute(query4).scalars())
            self.assertEqual(
                len(combined_filter_measurements), 1
            )  # Debería encontrar 1 medición

            # Verificar que las mediciones filtradas son las esperadas (usando oid y measurement_id)
            self.assertEqual(
                sorted([(m.oid, m.measurement_id) for m in later_measurements]),
                sorted(
                    [
                        (12345680, 987654322),
                        (12345681, 987654323),
                        (12345682, 987654324),
                    ]
                ),
            )

            self.assertEqual(
                sorted([(m.oid, m.measurement_id) for m in band1_measurements]),
                sorted([(12345680, 987654321), (12345681, 987654323)]),
            )


@pytest.mark.usefixtures("psql_service")
class ZtfForcedPhotometryModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo ZtfForcedPhotometry"""

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
        """Verificar que se pueden realizar consultas a la tabla ZtfForcedPhotometry vacía"""
        with self.db.session() as session:
            query = select(ZtfForcedPhotometry)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_ztf_forced_photometry(self):
        """Verificar que se pueden crear y consultar mediciones de fotometría forzada ZTF en la base de datos"""
        # Primero, crear el objeto al que hará referencia la fotometría forzada
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.db.session() as session:
            session.add(test_object)
            session.commit()

        # Crear una nueva entrada de fotometría forzada ZTF
        test_ztf_photometry = ZtfForcedPhotometry(
            **ZTF_FORCED_PHOTOMETRY_DATA["filter"][0]
        )

        with self.db.session() as session:
            session.add(test_ztf_photometry)

        # Verificar que la medición se ha guardado
        with self.db.session() as session:
            query = select(ZtfForcedPhotometry)
            measurements = list(session.execute(query))

            self.assertEqual(len(measurements), 1)

    def test_create_multiple_ztf_forced_photometry(self):
        """Verificar que se pueden crear y consultar múltiples mediciones de fotometría forzada ZTF"""
        # Primero, crear los objetos a los que harán referencia las mediciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear instancias de ZtfForcedPhotometry para cada conjunto de datos
        photometry_data = ZTF_FORCED_PHOTOMETRY_DATA["filter"]
        measurements = [ZtfForcedPhotometry(**data) for data in photometry_data]

        # Agregar todas las mediciones en una sola sesión
        with self.db.session() as session:
            for m in measurements:
                session.add(m)

        # Verificar que se han guardado todas las mediciones
        with self.db.session() as session:
            query = select(ZtfForcedPhotometry)
            saved_measurements = list(session.execute(query).scalars())

            # Verificar que el número de mediciones guardadas coincide con las creadas
            self.assertEqual(len(saved_measurements), len(photometry_data))

    def test_filter_ztf_forced_photometry(self):
        """Verificar que se pueden filtrar mediciones de fotometría forzada ZTF con criterios específicos"""
        # Primero, crear los objetos a los que harán referencia las mediciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples mediciones a la base de datos
        measurements = [
            ZtfForcedPhotometry(**data) for data in ZTF_FORCED_PHOTOMETRY_DATA["filter"]
        ]
        with self.db.session() as session:
            for m in measurements:
                session.add(m)

        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por mag < 19.0
            query1 = select(ZtfForcedPhotometry).where(ZtfForcedPhotometry.mag < 19.0)
            bright_measurements = list(session.execute(query1).scalars())
            self.assertEqual(
                len(bright_measurements), 3
            )  # Debería encontrar 3 mediciones

            # Filtrar por rband = 1
            query2 = select(ZtfForcedPhotometry).where(ZtfForcedPhotometry.rband == 1)
            band1_measurements = list(session.execute(query2).scalars())
            self.assertEqual(
                len(band1_measurements), 2
            )  # Debería encontrar 2 mediciones

            # Filtrar por isdiffpos
            query3 = select(ZtfForcedPhotometry).where(
                ZtfForcedPhotometry.isdiffpos == 1
            )
            isdiffpos_measurements = list(session.execute(query3).scalars())
            self.assertEqual(
                len(isdiffpos_measurements), 2
            )  # Debería encontrar 2 mediciones

            # Filtrar por corrected = True
            query4 = select(ZtfForcedPhotometry).where(
                ZtfForcedPhotometry.corrected == True
            )
            corrected_measurements = list(session.execute(query4).scalars())
            self.assertEqual(
                len(corrected_measurements), 2
            )  # Debería encontrar 2 mediciones

            # Filtrar por rango de magnr
            query5 = select(ZtfForcedPhotometry).where(
                ZtfForcedPhotometry.magnr.between(17.0, 18.0)
            )
            mid_magnr_measurements = list(session.execute(query5).scalars())
            self.assertEqual(
                len(mid_magnr_measurements), 2
            )  # Debería encontrar 2 mediciones

            # Filtrar por múltiples condiciones
            query6 = select(ZtfForcedPhotometry).where(
                ZtfForcedPhotometry.mag < 19.0, ZtfForcedPhotometry.corrected == True
            )
            combined_filter_measurements = list(session.execute(query6).scalars())
            self.assertEqual(
                len(combined_filter_measurements), 1
            )  # Debería encontrar 1 medición

            # Verificar que las mediciones filtradas son las esperadas (usando oid y measurement_id)
            self.assertEqual(
                sorted([(m.oid, m.measurement_id) for m in bright_measurements]),
                sorted(
                    [
                        (12345680, 987654321),
                        (12345680, 987654322),
                        (12345682, 987654324),
                    ]
                ),
            )

            self.assertEqual(
                sorted([(m.oid, m.measurement_id) for m in band1_measurements]),
                sorted([(12345680, 987654321), (12345681, 987654323)]),
            )

            self.assertEqual(
                sorted([(m.oid, m.measurement_id) for m in corrected_measurements]),
                sorted([(12345681, 987654323), (12345682, 987654324)]),
            )


@pytest.mark.usefixtures("psql_service")
class NonDetectionModelTest(unittest.TestCase):
    """Pruebas específicas para el modelo NonDetection"""

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
        """Verificar que se pueden realizar consultas a la tabla NonDetection vacía"""
        with self.db.session() as session:
            query = select(NonDetection)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_non_detection(self):
        """Verificar que se pueden crear y consultar no-detecciones en la base de datos"""
        # Primero, crear el objeto al que hará referencia la no-detección
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.db.session() as session:
            session.add(test_object)
            session.commit()
            object_oid = test_object.oid

        # Crear una nueva entrada de no-detección
        test_non_detection = NonDetection(
            oid=object_oid, band=1, mjd=60030.12345, diffmaglim=19.5
        )

        with self.db.session() as session:
            session.add(test_non_detection)

        # Verificar que la no-detección se ha guardado
        with self.db.session() as session:
            query = select(NonDetection)
            non_detections = list(session.execute(query))

            self.assertEqual(len(non_detections), 1)

    def test_create_multiple_non_detections(self):
        """Verificar que se pueden crear y consultar múltiples no-detecciones"""
        # Primero, crear los objetos a los que harán referencia las no-detecciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear instancias de NonDetection para cada conjunto de datos
        non_detection_data = NON_DETECTION_DATA["filter"]
        non_detections = [NonDetection(**data) for data in non_detection_data]

        # Agregar todas las no-detecciones en una sola sesión
        with self.db.session() as session:
            for nd in non_detections:
                session.add(nd)

        # Verificar que se han guardado todas las no-detecciones
        with self.db.session() as session:
            query = select(NonDetection)
            saved_non_detections = list(session.execute(query).scalars())

            # Verificar que el número de no-detecciones guardadas coincide con las creadas
            self.assertEqual(len(saved_non_detections), len(non_detection_data))

    def test_filter_non_detections(self):
        """Verificar que se pueden filtrar no-detecciones con criterios específicos"""
        # Primero, crear los objetos a los que harán referencia las no-detecciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples no-detecciones a la base de datos
        non_detections = [NonDetection(**data) for data in NON_DETECTION_DATA["filter"]]
        with self.db.session() as session:
            for nd in non_detections:
                session.add(nd)

        # Probar diferentes filtros
        with self.db.session() as session:
            # Filtrar por diffmaglim > 20.0
            query1 = select(NonDetection).where(NonDetection.diffmaglim > 20.0)
            deep_non_detections = list(session.execute(query1).scalars())
            self.assertEqual(
                len(deep_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por band = 1
            query2 = select(NonDetection).where(NonDetection.band == 1)
            band1_non_detections = list(session.execute(query2).scalars())
            self.assertEqual(
                len(band1_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por rango de mjd
            query3 = select(NonDetection).where(NonDetection.mjd.between(60035, 60041))
            mid_mjd_non_detections = list(session.execute(query3).scalars())
            self.assertEqual(
                len(mid_mjd_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por un oid específico
            query4 = select(NonDetection).where(NonDetection.oid == 12345680)
            obj_non_detections = list(session.execute(query4).scalars())
            self.assertEqual(
                len(obj_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por múltiples condiciones
            query5 = select(NonDetection).where(
                NonDetection.diffmaglim > 19.5, NonDetection.band == 1
            )
            combined_filter_non_detections = list(session.execute(query5).scalars())
            self.assertEqual(
                len(combined_filter_non_detections), 1
            )  # Debería encontrar 1 no-detección

            # Verificar que las no-detecciones filtradas son las esperadas (usando oid y mjd)
            self.assertEqual(
                sorted([(nd.oid, nd.mjd) for nd in deep_non_detections]),
                sorted([(12345681, 60040.34567), (12345682, 60045.45678)]),
            )

            self.assertEqual(
                sorted([(nd.oid, nd.mjd) for nd in band1_non_detections]),
                sorted([(12345680, 60030.12345), (12345681, 60040.34567)]),
            )

            self.assertEqual(
                sorted([(nd.oid, nd.mjd) for nd in obj_non_detections]),
                sorted([(12345680, 60030.12345), (12345680, 60035.23456)]),
            )


if __name__ == "__main__":
    unittest.main()
