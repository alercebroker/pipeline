import asyncio

import pytest
from db_plugins.db.sql._connection import AsyncPsqlDatabase

from ingestion_step.core.types import Message
from ingestion_step.lsst.strategy import LsstStrategy


@pytest.mark.usefixtures("async_psql_db")
def test_process_alerts_lsst(
    lsst_alerts: list[Message], async_psql_db: AsyncPsqlDatabase
):
    parsed_data = LsstStrategy.parse(lsst_alerts)
    asyncio.run(LsstStrategy.insert_into_db(async_psql_db, parsed_data))
