import asyncio

import fastavro
import pytest
from db_plugins.db.sql._connection import AsyncPsqlDatabase, PsqlDatabase
from fastavro.schema import load_schema
from fastavro.types import Schema
from sqlalchemy import text
from sqlalchemy.orm import Session

from ingestion_step.core.types import Message
from ingestion_step.lsst.strategy import LsstStrategy

DBs = tuple[PsqlDatabase, AsyncPsqlDatabase]


@pytest.mark.usefixtures("psql_db")
def test_process_alerts(lsst_alerts: list[Message], psql_db: DBs):
    parsed_data = LsstStrategy.parse(lsst_alerts)
    asyncio.run(LsstStrategy.insert_into_db(psql_db[1], parsed_data))


@pytest.mark.usefixtures("psql_db")
def test_process_out_of_order(psql_db: DBs):
    schema: Schema = load_schema("../schemas/surveys/lsst_v9.0/lsst.v9_0.alert.avsc")
    with open("./tests/integration/data/lsst_generated_50.avro", "rb") as f:
        reader = fastavro.reader(f, schema)
        lsst_alerts: list[Message] = [msg for msg in reader]  # pyright: ignore

    lsst_alerts.sort(key=lambda msg: msg["diaSource"]["midpointMjdTai"], reverse=True)

    for alert in lsst_alerts:
        parsed_data = LsstStrategy.parse([alert])
        asyncio.run(LsstStrategy.insert_into_db(psql_db[1], parsed_data))

    with psql_db[0].session() as session:
        assert isinstance(session, Session)
        res = session.execute(
            text("SELECT count(*) FROM lsst_detection WHERE has_stamp = true;")
        )
        row = res.fetchone()
        assert row is not None and len(row) == 1

    n_alerts = row[0]

    assert n_alerts == len(lsst_alerts)


@pytest.mark.usefixtures("psql_db")
def test_process_duplicated(psql_db: DBs, lsst_alerts: list[Message]):
    lsst_alerts_duplicated = lsst_alerts + lsst_alerts

    parsed_data = LsstStrategy.parse(lsst_alerts_duplicated)
    asyncio.run(LsstStrategy.insert_into_db(psql_db[1], parsed_data))

    with psql_db[0].session() as session:
        assert isinstance(session, Session)
        res = session.execute(
            text("SELECT count(*) FROM lsst_detection WHERE has_stamp = true;")
        )
        row = res.fetchone()
        assert row is not None and len(row) == 1

    n_alerts = row[0]

    assert n_alerts == len(lsst_alerts)
