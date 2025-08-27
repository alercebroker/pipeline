import asyncio

import pytest
from db_plugins.db.sql._connection import AsyncPsqlDatabase

from ingestion_step.core.types import Message
from ingestion_step.ztf.strategy import ZtfStrategy


@pytest.mark.usefixtures("async_psql_db")
def test_process_alerts_ztf(
    ztf_alerts: list[Message], async_psql_db: AsyncPsqlDatabase
):
    parsed_data = ZtfStrategy.parse(ztf_alerts)
    asyncio.run(ZtfStrategy.insert_into_db(async_psql_db, parsed_data))
