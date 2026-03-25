import pytest
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.types import Message
from ingestion_step.ztf.strategy import ZtfStrategy


@pytest.mark.usefixtures("psql_db")
def test_process_alerts_ztf(ztf_alerts: list[Message], psql_db: PsqlDatabase):
    parsed_data = ZtfStrategy.parse(ztf_alerts)
    ZtfStrategy.insert_into_db(psql_db, parsed_data)
