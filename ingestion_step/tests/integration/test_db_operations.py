import unittest
from typing import Any

import pytest
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import (
    DeclarativeBase,
    Detection,
    ForcedPhotometry,
    NonDetection,
    Object,
    ZtfDetection,
    ZtfForcedPhotometry,
    ZtfObject,
)
from sqlalchemy import insert, select

from ingestion_step.utils.database import (
    insert_detections,
    insert_forced_photometry,
    insert_non_detections,
    insert_objects,
)
from tests.integration.data import ztf_messages as msgs

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}


@pytest.mark.usefixtures("psql_service")
class BaseDbTests(unittest.TestCase):
    def setUp(self):
        # crear db
        self.psql_db = PsqlDatabase(psql_config)
        self.psql_db.create_db()

        # insertar datos existente

        with self.psql_db.session() as session:
            session.execute(insert(Object).values(msgs.existing_object_dict))
            session.execute(
                insert(Detection).values(msgs.existing_detections_dict)
            )
            session.execute(
                insert(ZtfDetection).values(msgs.existing_ztf_detections_dict)
            )
            session.execute(
                insert(ForcedPhotometry).values(msgs.existing_fp_dict)
            )
            session.execute(
                insert(ZtfForcedPhotometry).values(msgs.existing_ztf_fp_dict)
            )
            session.execute(
                insert(NonDetection).values(msgs.existing_non_detections_dict)
            )
            session.commit()

    def tearDown(self):
        # limpiar la db
        self.psql_db.drop_db()

    def query_data(self, model: type[DeclarativeBase]):
        with self.psql_db.session() as session:
            query_result = session.execute(select(model))
            result = []
            for row in query_result.all():
                new_data = row[0].__dict__
                new_data.pop("_sa_instance_state")
                result.append(new_data)

        return result

    def ziped_lists(
        self,
        first: list[dict[str, Any]],
        seccond: list[dict[str, Any]],
        field: str = "oid",
    ):
        result = zip(
            sorted(first, key=lambda x: x[field]),
            sorted(seccond, key=lambda x: x[field]),
        )
        return result

    def test_object(self):
        insert_objects(self.psql_db, msgs.new_objects_df)

        result = self.query_data(Object)
        result_ztf = self.query_data(ZtfObject)

        for res, exp in self.ziped_lists(result, msgs.objects_expected, "oid"):
            self.assertDictEqual(res, exp)

        for res, exp in self.ziped_lists(
            result_ztf, msgs.ztf_objects_expected, "oid"
        ):
            self.assertDictEqual(res, exp)

    def test_detection(self):
        insert_detections(self.psql_db, msgs.new_detections_df)

        result_detections = self.query_data(Detection)
        result_ztf_detections = self.query_data(ZtfDetection)

        for res, exp in self.ziped_lists(
            result_detections, msgs.detections_expected, "measurement_id"
        ):
            self.assertDictEqual(res, exp)

        for res, exp in self.ziped_lists(
            result_ztf_detections,
            msgs.ztf_detections_expected,
            "measurement_id",
        ):
            self.assertDictEqual(res, exp)

    def test_forced_photometry(self):
        insert_forced_photometry(self.psql_db, msgs.new_fp_df)

        result_fp = self.query_data(ForcedPhotometry)
        result_ztf_fp = self.query_data(ZtfForcedPhotometry)

        for res, exp in self.ziped_lists(
            result_fp, msgs.fp_expected, "measurement_id"
        ):
            self.assertDictEqual(res, exp)

        for res, exp in self.ziped_lists(
            result_ztf_fp, msgs.ztf_fp_expected, "measurement_id"
        ):
            print(f"HERE ---- \n {res} \n {exp} \n")

            for key in res.keys():
                if res[key] != exp[key]:
                    print(f"LLAVE MALA {key} {res[key]} != {exp[key]}f")
            self.assertDictEqual(res, exp)

    def test_non_detections(self):
        insert_non_detections(self.psql_db, msgs.new_non_detections_df)

        result = self.query_data(NonDetection)

        for res, exp in self.ziped_lists(
            result, msgs.non_detections_expected, "mjd"
        ):
            self.assertDictEqual(res, exp)
