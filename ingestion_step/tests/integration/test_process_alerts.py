from typing import Any

import pytest
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.select_parser import select_parser
from ingestion_step.utils.database import (
    insert_detections,
    insert_forced_photometry,
    insert_non_detections,
    insert_objects,
)

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}


@pytest.mark.usefixtures("psql_db")
def test_process_alerts_ztf(ztf_alerts: list[dict[str, Any]], psql_db: PsqlDatabase):
    messages = ztf_alerts

    parser = select_parser("ztf")

    parsed_data = parser.parse(messages)

    insert_objects(psql_db, parsed_data["objects"])
    insert_detections(psql_db, parsed_data["detections"])
    insert_non_detections(psql_db, parsed_data["non_detections"])
    insert_forced_photometry(psql_db, parsed_data["forced_photometries"])
