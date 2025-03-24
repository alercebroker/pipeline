from db_plugins.db.sql.models import (
    Object,
    Detection,
    ZTFDetection,
    ForcedPhotometry,
    ZTFForcedPhotometry,
    NonDetection
)
from db_plugins.db.sql._connection import PsqlDatabase

from sqlalchemy import select, insert
import pytest
import unittest

from .data.ztf_messages import *
from ingestion_step.utils.database import (
    insert_objects,
    insert_detections,
    insert_forced_photometry,
    insert_non_detections
)


psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}

@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("kafka_service")
class BaseDbTests(unittest.TestCase):
        
    def SetUp(self):
        # crear db
        self.psql_db = PsqlDatabase(psql_config)
        self.psql_db.create_db()

        # insertar datos existente

        session = self.psql_db.session()
        session.execute(insert(Object).values(existing_object_dict))
        session.commit()
        
    def TearDown(self):
        # limpiar la db
        self.psql_db.drop_db()

    def query_data(self, model):
        session = self.psql_db.session()
        result = session.execute(select(model))
        return list(result)
    
    def test_object(self):
        insert_objects(self.psql_db, new_objects_df)
        
        result = self.query_data(Object)
        self.assertDictEqual(result, objects_expected)

    def test_detection(self):
        insert_detections(self.psql_db, new_detections_df)

        result_detections = self.query_data(Detection)
        result_ztf_detections = self.query_data(ZTFDetection)
        self.assertDictEqual(result_detections, detections_expected)
        self.assertDictEqual(result_ztf_detections, detections_expected)

    def test_forced_photometry(self):
        insert_forced_photometry(self.psql_db, new_fp_df)
        
        result_fp = self.query_data(ForcedPhotometry)
        result_ztf_fp = self.query_data(ZTFForcedPhotometry)
        self.assertDictEqual(result_fp, fp_expected)
        self.assertDictEqual(result_ztf_fp, ztf_fp_expected)

    def test_non_detections(self):
        insert_non_detections(self.psql_db, new_non_detections_df)
        
        result = self.query_data(NonDetection)
        self.assertDictEqual(result, non_detections_expected)
