from base import BaseDbTest
from data import ZTF_OBJECT_DATA
from sqlalchemy.dialects.postgresql.base import select

from db_plugins.db.sql.models import ZtfObject


class ZtfObjectModelTest(BaseDbTest):
    """Pruebas específicas para el modelo ZtfObject"""

    def test_query_empty_table(self):
        """Verificar que se pueden realizar consultas a la tabla ZtfObject vacía"""
        with self.psql_db.session() as session:
            query = select(ZtfObject)
            obj = session.execute(query)
            # Verificar que no hay objetos (base de datos vacía)
            self.assertEqual(len(list(obj)), 0)

    def test_create_and_query_object(self):
        """Verificar que se pueden crear y consultar objetos en la base de datos"""
        # Crear un nuevo objeto
        test_object = ZtfObject(**ZTF_OBJECT_DATA["filter"][0])

        with self.psql_db.session() as session:
            session.add(test_object)

        # Verificar que el objeto se ha guardado
        with self.psql_db.session() as session:
            query = select(ZtfObject)
            objects = list(session.execute(query))

            self.assertEqual(len(objects), 1)

    def test_create_multiple_objects(self):
        """Verificar que se pueden crear y consultar múltiples objetos"""
        # Crear instancias de ZtfObject para cada conjunto de datos
        objects = [ZtfObject(**data) for data in ZTF_OBJECT_DATA["filter"]]

        # Agregar todos los objetos en una sola sesión
        with self.psql_db.session() as session:
            for obj in objects:
                session.add(obj)

        # Verificar que se han guardado todos los objetos
        with self.psql_db.session() as session:
            query = select(ZtfObject)
            saved_objects = list(session.execute(query).scalars())

            # Verificar que el número de objetos guardados coincide con los creados
            self.assertEqual(len(saved_objects), len(ZTF_OBJECT_DATA["filter"]))

    def test_filter_objects(self):
        """Verificar que se pueden filtrar objetos con criterios específicos"""
        # Crear y agregar múltiples objetos a la base de datos
        objects = [ZtfObject(**data) for data in ZTF_OBJECT_DATA["filter"]]
        with self.psql_db.session() as session:
            for obj in objects:
                session.add(obj)

        # Probar diferentes filtros
        with self.psql_db.session() as session:
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
