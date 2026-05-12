from metadata_step.utils.database import (
    PSQLConnection,
    get_gaia_catalog,
    get_ps1_catalog,
    insert_metadata,
)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models import Detection, Gaia_ztf, Object, Ps1_ztf, Ss_ztf
from tests.data.mocks import object_mocks, detection_mocks, gaia_mocks, ps1_mocks

config = {
    "USER": "postgres",
    "PASSWORD": "postgres",
    "HOST": "localhost",
    "PORT": "5432",
    "DB_NAME": "postgres",
}

oids = ["ZTF00llmn"]


def _populate(conn: PSQLConnection):
    with conn.session() as session:
        session.execute(insert(Object).values(object_mocks).on_conflict_do_nothing())
        session.execute(
            insert(Detection).values(detection_mocks).on_conflict_do_nothing()
        )
        session.execute(insert(Gaia_ztf).values(gaia_mocks).on_conflict_do_nothing())
        session.execute(insert(Ps1_ztf).values(ps1_mocks).on_conflict_do_nothing())
        session.commit()


def test_selection_queries(psql_service):
    conn = PSQLConnection(config)
    _populate(conn)

    with conn.session() as session:
        gaia = get_gaia_catalog(session, oids)
        ps1 = get_ps1_catalog(session, oids)
        assert gaia is not None
        assert ps1 is not None


def test_metadata_insertion(psql_service):
    conn = PSQLConnection(config)
    _populate(conn)

    data = [
        {
            "oid": "ZTF00llmn",
            "ss": {
                "oid": "ZTF00llmn",
                "candid": 987654321,
                "ssdistnr": 93.00,
                "ssmagnr": 39.00,
                "ssnamenr": "neimu",
            },
            "reference": {
                "oid": "ZTF00llmn",
                "rfid": 99,
                "candid": 987654321,
                "fid": 1,
                "ranr": 11.11,
                "decnr": 22.22,
                "mjdstartref": 55500,
                "mjdendref": 56600,
                "nframesref": 1,
            },
            "dataquality": {
                "candid": 987654321,
                "oid": "ZTF00llmn",
                "fid": 1,
                "xpos": 0,
                "ypos": 0,
            },
            "gaia": {
                "oid": "ZTF00llmn",
                "candid": 987654321,
                "neargaia": 55.55,
                "unique1": False,
            },
            "ps1": {
                "oid": "ZTF00llmn",
                "candid": 987654321,
                "objectidps1": 15.11,
                "objectidps2": 25.22,
                "objectidps3": 35.33,
                "nmtchps": 1,
                "unique1": True,
                "unique2": True,
                "unique3": True,
            },
        }
    ]

    with conn.session() as session:
        insert_metadata(session, data, [])
        session.commit()
        # perform queries to make sure they were inserted/updated
        ss_result = session.execute(select(Ss_ztf).where(Ss_ztf.oid == "ZTF00llmn"))
        ps1_result = session.execute(
            select(Ps1_ztf)
            .where(Ps1_ztf.oid == "ZTF00llmn")
            .where(Ps1_ztf.candid == 1234567890)
        )
        gaia_result = session.execute(
            select(Gaia_ztf)
            .where(Gaia_ztf.oid == "ZTF00llmn")
            .where(Gaia_ztf.candid == 987654321)
        )
        ss_result = list(ss_result)[0][0].__dict__
        ps1_result = list(ps1_result)[0][0].__dict__
        gaia_result = list(gaia_result)[0][0].__dict__

        assert ss_result["ssnamenr"] == "neimu"
        assert ps1_result["unique1"] is True
        assert gaia_result["unique1"] is False
