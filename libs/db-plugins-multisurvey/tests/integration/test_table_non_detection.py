from base import BaseDbTest
from data import NON_DETECTION_DATA, OBJECT_DATA
from sqlalchemy.dialects.postgresql.base import select

from db_plugins.db.sql.models import Object, ZtfNonDetection


class NonDetectionModelTest(BaseDbTest):
    """Pruebas específicas para el modelo NonDetection"""

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla NonDetection vacía"""
        with self.psql_db.session() as session:
            query = select(ZtfNonDetection)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_non_detection(self):
        """Verificar que se pueden crear y consultar no-detecciones en la base de datos"""
        # Primero, crear el objeto al que hará referencia la no-detección
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.psql_db.session() as session:
            session.add(test_object)
            session.commit()
            object_oid = test_object.oid

        # Crear una nueva entrada de no-detección
        test_non_detection = ZtfNonDetection(
            oid=object_oid, band=1, mjd=60030.12345, diffmaglim=19.5
        )

        with self.psql_db.session() as session:
            session.add(test_non_detection)

        # Verificar que la no-detección se ha guardado
        with self.psql_db.session() as session:
            query = select(ZtfNonDetection)
            non_detections = list(session.execute(query))

            self.assertEqual(len(non_detections), 1)

    def test_create_multiple_non_detections(self):
        """Verificar que se pueden crear y consultar múltiples no-detecciones"""
        # Primero, crear los objetos a los que harán referencia las no-detecciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear instancias de NonDetection para cada conjunto de datos
        non_detection_data = NON_DETECTION_DATA["filter"]
        non_detections = [ZtfNonDetection(**data) for data in non_detection_data]

        # Agregar todas las no-detecciones en una sola sesión
        with self.psql_db.session() as session:
            for nd in non_detections:
                session.add(nd)

        # Verificar que se han guardado todas las no-detecciones
        with self.psql_db.session() as session:
            query = select(ZtfNonDetection)
            saved_non_detections = list(session.execute(query).scalars())

            # Verificar que el número de no-detecciones guardadas coincide con las creadas
            self.assertEqual(len(saved_non_detections), len(non_detection_data))

    def test_filter_non_detections(self):
        """Verificar que se pueden filtrar no-detecciones con criterios específicos"""
        # Primero, crear los objetos a los que harán referencia las no-detecciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples no-detecciones a la base de datos
        non_detections = [
            ZtfNonDetection(**data) for data in NON_DETECTION_DATA["filter"]
        ]
        with self.psql_db.session() as session:
            for nd in non_detections:
                session.add(nd)

        # Probar diferentes filtros
        with self.psql_db.session() as session:
            # Filtrar por diffmaglim > 20.0
            query1 = select(ZtfNonDetection).where(ZtfNonDetection.diffmaglim > 20.0)
            deep_non_detections = list(session.execute(query1).scalars())
            self.assertEqual(
                len(deep_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por band = 1
            query2 = select(ZtfNonDetection).where(ZtfNonDetection.band == 1)
            band1_non_detections = list(session.execute(query2).scalars())
            self.assertEqual(
                len(band1_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por rango de mjd
            query3 = select(ZtfNonDetection).where(
                ZtfNonDetection.mjd.between(60035, 60041)
            )
            mid_mjd_non_detections = list(session.execute(query3).scalars())
            self.assertEqual(
                len(mid_mjd_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por un oid específico
            query4 = select(ZtfNonDetection).where(ZtfNonDetection.oid == 12345680)
            obj_non_detections = list(session.execute(query4).scalars())
            self.assertEqual(
                len(obj_non_detections), 2
            )  # Debería encontrar 2 no-detecciones

            # Filtrar por múltiples condiciones
            query5 = select(ZtfNonDetection).where(
                ZtfNonDetection.diffmaglim > 19.5, ZtfNonDetection.band == 1
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
