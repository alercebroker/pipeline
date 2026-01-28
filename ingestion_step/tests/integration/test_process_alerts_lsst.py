import pytest
from db_plugins.db.sql._connection import PsqlDatabase
from sqlalchemy import text
from sqlalchemy.orm import Session

from ingestion_step.core.types import Message
from ingestion_step.lsst.strategy import LsstStrategy


@pytest.mark.usefixtures("psql_db")
def test_process_alerts(lsst_alerts: list[Message], psql_db: PsqlDatabase):
    parsed_data = LsstStrategy.parse(lsst_alerts)
    LsstStrategy.insert_into_db(psql_db, parsed_data)


@pytest.mark.usefixtures("psql_db")
def test_process_out_of_order(psql_db: PsqlDatabase, lsst_alerts: list[Message]):
    lsst_alerts.sort(key=lambda msg: msg["diaSource"]["midpointMjdTai"], reverse=True)

    psql_db.drop_db()
    psql_db.create_db()

    for alert in lsst_alerts:
        parsed_data = LsstStrategy.parse([alert])
        LsstStrategy.insert_into_db(psql_db, parsed_data)

    with psql_db.session() as session:
        assert isinstance(session, Session)
        res = session.execute(
            text("SELECT count(*) FROM lsst_detection WHERE has_stamp = true;")
        )
        row = res.fetchone()
        assert row is not None and len(row) == 1

    n_alerts = row[0]

    assert n_alerts == len(lsst_alerts)


@pytest.mark.usefixtures("psql_db")
def test_process_duplicated(psql_db: PsqlDatabase, lsst_alerts: list[Message]):
    psql_db.drop_db()
    psql_db.create_db()

    lsst_alerts_duplicated = lsst_alerts + lsst_alerts

    parsed_data = LsstStrategy.parse(lsst_alerts_duplicated)
    LsstStrategy.insert_into_db(psql_db, parsed_data)

    with psql_db.session() as session:
        assert isinstance(session, Session)
        res = session.execute(
            text("SELECT count(*) FROM lsst_detection WHERE has_stamp = true;")
        )
        row = res.fetchone()
        assert row is not None and len(row) == 1

    n_alerts = row[0]

    assert n_alerts == len(lsst_alerts)
