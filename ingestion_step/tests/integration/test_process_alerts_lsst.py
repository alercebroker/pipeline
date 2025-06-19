import pytest
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.types import Message
from ingestion_step.lsst.strategy import LsstStrategy

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}


@pytest.mark.usefixtures("psql_db")
def test_process_alerts_ztf(lsst_alerts: list[Message], psql_db: PsqlDatabase):
    parsed_data = LsstStrategy.parse(lsst_alerts, psql_db)
    LsstStrategy.insert_into_db(psql_db, parsed_data)
