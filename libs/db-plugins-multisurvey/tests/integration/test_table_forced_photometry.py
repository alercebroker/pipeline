from base import BaseDbTest
from data import FORCED_PHOTOMETRY_DATA, OBJECT_DATA, ZTF_FORCED_PHOTOMETRY_DATA
from sqlalchemy.dialects.postgresql.base import select

from db_plugins.db.sql.models import ForcedPhotometry, Object, ZtfForcedPhotometry


class ForcedPhotometryModelTest(BaseDbTest):
    """Pruebas específicas para el modelo ForcedPhotometry"""

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla ForcedPhotometry vacía"""
        with self.psql_db.session() as session:
            query = select(ForcedPhotometry)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_forced_photometry(self):
        """Verificar que se pueden crear y consultar mediciones de fotometría forzada en la base de datos"""
        # Primero, crear el objeto al que hará referencia la fotometría forzada
        test_object = Object(**OBJECT_DATA["filter"][0])
        with self.psql_db.session() as session:
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

        with self.psql_db.session() as session:
            session.add(test_forced_photometry)

        # Verificar que la medición se ha guardado
        with self.psql_db.session() as session:
            query = select(ForcedPhotometry)
            measurements = list(session.execute(query))

            self.assertEqual(len(measurements), 1)

    def test_create_multiple_forced_photometry(self):
        """Verificar que se pueden crear y consultar múltiples mediciones de fotometría forzada"""
        # Primero, crear los objetos a los que harán referencia las mediciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear datos para varias mediciones de fotometría forzada
        photometry_data = FORCED_PHOTOMETRY_DATA["filter"]

        # Crear instancias de ForcedPhotometry para cada conjunto de datos
        measurements = [ForcedPhotometry(**data) for data in photometry_data]

        # Agregar todas las mediciones en una sola sesión
        with self.psql_db.session() as session:
            for m in measurements:
                session.add(m)

        # Verificar que se han guardado todas las mediciones
        with self.psql_db.session() as session:
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
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples mediciones a la base de datos
        measurements = [
            ForcedPhotometry(**data) for data in FORCED_PHOTOMETRY_DATA["filter"]
        ]
        with self.psql_db.session() as session:
            for m in measurements:
                session.add(m)

        # Probar diferentes filtros
        with self.psql_db.session() as session:
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


class ZtfForcedPhotometryModelTest(BaseDbTest):
    """Pruebas específicas para el modelo ZtfForcedPhotometry"""

    def test_query_empty_tables(self):
        """Verificar que se pueden realizar consultas a la tabla ZtfForcedPhotometry vacía"""
        with self.psql_db.session() as session:
            query = select(ZtfForcedPhotometry)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_ztf_forced_photometry(self):
        """Verificar que se pueden crear y consultar mediciones de fotometría forzada ZTF en la base de datos"""
        # Primero, crear el objeto al que hará referencia la fotometría forzada
        test_object = Object(**OBJECT_DATA["filter"][0])

        with self.psql_db.session() as session:
            session.add(test_object)
            session.commit()

        # Crear una nueva entrada de fotometría forzada ZTF
        test_ztf_photometry = ZtfForcedPhotometry(
            **ZTF_FORCED_PHOTOMETRY_DATA["filter"][0]
        )

        with self.psql_db.session() as session:
            session.add(test_ztf_photometry)

        # Verificar que la medición se ha guardado
        with self.psql_db.session() as session:
            query = select(ZtfForcedPhotometry)
            measurements = list(session.execute(query))

            self.assertEqual(len(measurements), 1)

    def test_create_multiple_ztf_forced_photometry(self):
        """Verificar que se pueden crear y consultar múltiples mediciones de fotometría forzada ZTF"""
        # Primero, crear los objetos a los que harán referencia las mediciones
        for obj_data in OBJECT_DATA["filter"]:
            # Guardar el objeto
            obj = Object(**obj_data)
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear instancias de ZtfForcedPhotometry para cada conjunto de datos
        photometry_data = ZTF_FORCED_PHOTOMETRY_DATA["filter"]
        measurements = [ZtfForcedPhotometry(**data) for data in photometry_data]

        # Agregar todas las mediciones en una sola sesión
        with self.psql_db.session() as session:
            for m in measurements:
                session.add(m)

        # Verificar que se han guardado todas las mediciones
        with self.psql_db.session() as session:
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
            with self.psql_db.session() as session:
                session.add(obj)
                session.commit()

        # Crear y agregar múltiples mediciones a la base de datos
        measurements = [
            ZtfForcedPhotometry(**data) for data in ZTF_FORCED_PHOTOMETRY_DATA["filter"]
        ]
        with self.psql_db.session() as session:
            for m in measurements:
                session.add(m)

        # Probar diferentes filtros
        with self.psql_db.session() as session:
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
                ZtfForcedPhotometry.corrected is True
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
                ZtfForcedPhotometry.mag < 19.0, ZtfForcedPhotometry.corrected is True
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
