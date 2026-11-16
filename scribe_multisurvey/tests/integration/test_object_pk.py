"""Object updates must target the full primary key (oid, sid).

The same oid can exist for more than one survey, so an update filtered on
oid alone touches rows from other surveys.
"""

import pytest
from db_plugins.db.sql._connection_pipeline import PsqlDatabase
from sqlalchemy import text

from sql_scribe.sql.command.commands import (
    LSSTMagstatCommand,
    ZTFMagstatCommand,
)

OID = 42
ZTF_SID = 0
LSST_SID = 1
INITIAL_N_DET = 1


@pytest.fixture(scope="module")
def db(psql_service):
    database = PsqlDatabase(
        {
            "ENGINE": "postgresql",
            "HOST": psql_service["host"],
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": psql_service["port"],
            "DB_NAME": "postgres",
        }
    )
    database.create_db()
    yield database
    database.drop_db()


@pytest.fixture
def shared_oid_objects(db):
    """Two object rows with the same oid, one per survey."""
    with db.session() as session:
        session.execute(text("DELETE FROM object"))
        session.execute(
            text(
                """
                INSERT INTO object (
                    oid, tid, sid, meanra, meandec, sigmara, sigmadec,
                    firstmjd, lastmjd, deltamjd, n_det, n_forced, n_non_det
                ) VALUES
                    (:oid, 0, :ztf_sid, 1, 1, 1, 1, 1, 1, 0, :n_det, 0, 0),
                    (:oid, 1, :lsst_sid, 1, 1, 1, 1, 1, 1, 0, :n_det, 0, 0)
                """
            ),
            {
                "oid": OID,
                "ztf_sid": ZTF_SID,
                "lsst_sid": LSST_SID,
                "n_det": INITIAL_N_DET,
            },
        )
        session.commit()
    return db


def magstat_payload(sid: int, n_det: int) -> dict:
    return {
        "oid": OID,
        "sid": sid,
        "meanra": 10.0,
        "meandec": 20.0,
        "sigmara": 0.1,
        "sigmadec": 0.1,
        "firstmjd": 60000.0,
        "lastmjd": 60010.0,
        "deltajd": 10.0,
        "n_det": n_det,
        "n_fphot": 0,
        "n_ndet": 0,
        "corrected": False,
        "stellar": False,
        "reference_change": False,
        "diffpos": False,
        "magstats": {"0": []},
    }


def n_det_by_sid(db) -> dict:
    with db.session() as session:
        rows = session.execute(
            text("SELECT sid, n_det FROM object WHERE oid = :oid"),
            {"oid": OID},
        ).all()
    return dict(rows)


@pytest.mark.parametrize(
    "command_cls, target_sid, other_sid",
    [
        (LSSTMagstatCommand, LSST_SID, ZTF_SID),
        (ZTFMagstatCommand, ZTF_SID, LSST_SID),
    ],
    ids=["lsst", "ztf"],
)
def test_magstat_updates_only_matching_sid(
    shared_oid_objects, command_cls, target_sid, other_sid
):
    db = shared_oid_objects
    command = command_cls(magstat_payload(target_sid, n_det=99))

    with db.session() as session:
        command_cls.db_operation(session, [command.data])
        session.commit()

    assert n_det_by_sid(db) == {target_sid: 99, other_sid: INITIAL_N_DET}
