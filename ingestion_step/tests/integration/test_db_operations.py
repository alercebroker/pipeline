import unittest

import pytest
from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import (
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

from .data.ztf_messages import *

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

        session = self.psql_db.session()
        with self.psql_db.session() as session:
            session.execute(insert(Object).values(existing_object_dict))
            session.execute(insert(NonDetection).values(existing_non_detections_dict))
            session.commit()

    def tearDown(self):
        # limpiar la db
        self.psql_db.drop_db()

    def query_data(self, model):
        with self.psql_db.session() as session:
            query_result = session.execute(select(model))
            result = []
            for row in query_result.all():
                new_data = row[0].__dict__
                new_data.pop("_sa_instance_state")
                result.append(new_data)

        return result

    def ziped_lists(self, first, seccond, field="oid"):
        result = zip(
            sorted(first, key=lambda x: x[field]),
            sorted(seccond, key=lambda x: x[field]),
        )
        return result

    def test_object(self):
        insert_objects(self.psql_db, new_objects_df)

        result = self.query_data(Object)
        result_ztf = self.query_data(ZtfObject)

        for res, exp in self.ziped_lists(result, objects_expected):
            self.assertDictEqual(res, exp)

        for res, exp in self.ziped_lists(result_ztf, ztf_objects_expected):
            self.assertDictEqual(res, exp)

    def test_detection(self):
        insert_detections(self.psql_db, new_detections_df)

        result_detections = self.query_data(Detection)
        result_ztf_detections = self.query_data(ZtfDetection)

        for res, exp in self.ziped_lists(result_detections, detections_expected):
            self.assertDictEqual(res, exp)

        for res, exp in self.ziped_lists(
            result_ztf_detections, ztf_detections_expected
        ):
            self.assertDictEqual(res, exp)

    def test_forced_photometry(self):
        insert_forced_photometry(self.psql_db, new_fp_df)

        result_fp = self.query_data(ForcedPhotometry)
        result_ztf_fp = self.query_data(ZtfForcedPhotometry)

        for res, exp in self.ziped_lists(result_fp, fp_expected):
            self.assertDictEqual(res, exp)

        for res, exp in self.ziped_lists(result_ztf_fp, ztf_fp_expected):
            self.assertDictEqual(res, exp)

    def test_non_detections(self):
        insert_non_detections(self.psql_db, new_non_detections_df)

        result = self.query_data(NonDetection)

        for res, exp in self.ziped_lists(result, non_detections_expected, "mjd"):
            self.assertDictEqual(res, exp)
