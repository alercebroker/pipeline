from base import BaseDbTest
from data import DETECTION_DATA, OBJECT_DATA, ZTF_DETECTION_DATA
from sqlalchemy.dialects.postgresql.base import select

from db_plugins.db.sql.models import Detection, Object, ZtfDetection


class DetectionModelTest(BaseDbTest):
    """Pruebas específicas para el modelo Detection"""

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla Detection vacía"""
        with self.psql_db.session() as session:
            query = select(Detection)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_objects(self):
        """Verificar que se pueden crear y consultar objetos en la base de datos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(**OBJECT_DATA["filter"][0])

        # Guardar el objeto padre
        with self.psql_db.session() as session:
            session.add(parent_object)

        # Crear un nuevo objeto de detección
        test_object = Detection(**DETECTION_DATA["filter"][0])

        with self.psql_db.session() as session:
            session.add(test_object)

        # Verificar que el objeto se ha guardado
        with self.psql_db.session() as session:
            query = select(Detection)
            objects = list(session.execute(query))

            self.assertEqual(len(objects), 1)

    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(**OBJECT_DATA["filter"][0])

        # Guardar el objeto padre
        with self.psql_db.session() as session:
            session.add(parent_object)

        # Crear instancias de Detection para cada conjunto de datos
        objects = [Detection(**data) for data in DETECTION_DATA["filter"]]

        # Agregar todos los objetos en una sola sesión
        with self.psql_db.session() as session:
            for obj in objects:
                session.add(obj)

        # Verificar que se han guardado todos los objetos
        with self.psql_db.session() as session:
            query = select(Detection)
            saved_objects = list(session.execute(query).scalars())

            # Verificar que el número de objetos guardado coincide con los creados
            self.assertEqual(len(saved_objects), len(DETECTION_DATA["filter"]))

    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Primero, crear un objeto al que referenciarán las detecciones
        parent_object = Object(**OBJECT_DATA["filter"][0])

        # Guardar el objeto padre
        with self.psql_db.session() as session:
            session.add(parent_object)

        # Crear y agregar múltiples objetos a la DB
        objects = [Detection(**data) for data in DETECTION_DATA["filter"]]
        with self.psql_db.session() as session:
            for obj in objects:
                session.add(obj)

        # Probar diferentes filtros
        with self.psql_db.session() as session:
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


class ZtfDetectionModelTest(BaseDbTest):
    """Pruebas específicas para el modelo ZtfDetection"""

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla ZtfDetection vacía"""
        with self.psql_db.session() as session:
            query = select(ZtfDetection)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_detection(self):
        """Verificar que se pueden crear y consultar detecciones en la base de datos"""
        # Primero, crear el objeto al que hará referencia la detección
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.psql_db.session() as session:
            session.add(test_object)
            session.commit()

        # Crear una nueva detección
        test_detection = ZtfDetection(**ZTF_DETECTION_DATA["filter"][0])

        with self.psql_db.session() as session:
            session.add(test_detection)

        # Verificar que la detección se ha guardado
        with self.psql_db.session() as session:
            query = select(ZtfDetection)
            detections = list(session.execute(query))

            self.assertEqual(len(detections), 1)

    def test_create_multiple_detections(self):
        """Verificar que se pueden crear y consultar múltiples detecciones"""
        # Primero, crear los objetos a los que harán referencia las detecciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear instancias de ZtfDetection para cada conjunto de datos
        detections = [ZtfDetection(**data) for data in ZTF_DETECTION_DATA["filter"]]

        # Agregar todas las detecciones en una sola sesión
        with self.psql_db.session() as session:
            for det in detections:
                session.add(det)

        # Verificar que se han guardado todas las detecciones
        with self.psql_db.session() as session:
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
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples detecciones a la base de datos
        detections = [ZtfDetection(**data) for data in ZTF_DETECTION_DATA["filter"]]
        with self.psql_db.session() as session:
            for det in detections:
                session.add(det)

        # Probar diferentes filtros
        with self.psql_db.session() as session:
            # Filtrar por magpsf > 18.0
            query1 = select(ZtfDetection).where(ZtfDetection.magpsf > 18.0)
            bright_detections = list(session.execute(query1).scalars())
            self.assertEqual(
                len(bright_detections), 3
            )  # Debería encontrar 3 detecciones

            # Filtrar por isdiffpos = True
            query2 = select(ZtfDetection).where(ZtfDetection.isdiffpos is True)
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
                ZtfDetection.magpsf > 18.0, ZtfDetection.isdiffpos is False
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
