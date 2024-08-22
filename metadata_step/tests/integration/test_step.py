from metadata_step.step import MetadataStep
from metadata_step.utils.database import PSQLConnection
from tests.data.messages import new_message_batch
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models import Detection, Gaia_ztf, Object, Ps1_ztf, Ss_ztf
from tests.data.mocks import *


def _populate_db(conn: PSQLConnection):
    with conn.session() as session:
        session.execute(insert(Object).values(object_mocks).on_conflict_do_nothing())
        session.execute(insert(Detection).values(detection_mocks).on_conflict_do_nothing())
        session.execute(insert(Gaia_ztf).values(gaia_mocks).on_conflict_do_nothing())
        session.execute(insert(Ps1_ztf).values(ps1_mocks).on_conflict_do_nothing())
        session.commit()


def _create_connection() -> PSQLConnection:
    config = {
        "USER": "postgres",
        "PASSWORD": "postgres",
        "HOST": "localhost",
        "PORT": "5432",
        "DB_NAME": "postgres",
    }
    return PSQLConnection(config)


def test_step(psql_service):
    # go step by step
    db = _create_connection()
    _populate_db(db)
    step = MetadataStep({"CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"}}, db)

    messages = new_message_batch()
    result = step.execute(messages)
    step.post_execute(result)

    with db.session() as session:
        # assert insertion
        ss_result = session.execute(select(Ss_ztf).where(Ss_ztf.oid == "ZTF21waka"))
        ps1_result = session.execute(
            select(Ps1_ztf).where(Ps1_ztf.oid == "ZTF21waka").where(Ps1_ztf.candid == 930930930)
        )
        gaia_result = session.execute(
            select(Gaia_ztf).where(Gaia_ztf.oid == "ZTF21waka").where(Gaia_ztf.candid == 930930930)
        )
        ss_result = list(ss_result)[0][0].__dict__
        ps1_result = list(ps1_result)[0][0].__dict__
        gaia_result = list(gaia_result)[0][0].__dict__

        assert ss_result["oid"] == "ZTF21waka"
        assert ss_result["candid"] == 930930930
        assert ss_result["ssdistnr"] == 100

        assert ps1_result["oid"] == "ZTF21waka"
        assert ps1_result["sgmag1"] == 100

        assert gaia_result["neargaia"] == 100

        # assert updating
        ps1_result = session.execute(
            select(Ps1_ztf).where(Ps1_ztf.oid == "ZTF00llmn").where(Ps1_ztf.candid == 1234567890)
        )
        ps1_result = list(ps1_result)[0][0].__dict__

        assert ps1_result["oid"] == "ZTF00llmn"

        # This assertion is failing. I don't understand the
        # expected behavior
        assert ps1_result["unique1"] is False
