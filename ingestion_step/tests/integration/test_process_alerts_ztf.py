import asyncio

import pytest
from db_plugins.db.sql._connection import AsyncPsqlDatabase, PsqlDatabase

from ingestion_step.core.types import Message
from ingestion_step.ztf.strategy import ZtfStrategy

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}

DBs = tuple[PsqlDatabase, AsyncPsqlDatabase]


@pytest.mark.usefixtures("psql_db")
def test_process_alerts_ztf(ztf_alerts: list[Message], psql_db: DBs):
    parsed_data = ZtfStrategy.parse(ztf_alerts)
    asyncio.run(ZtfStrategy.insert_into_db(psql_db[1], parsed_data))
