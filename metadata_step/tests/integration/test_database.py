from metadata_step.utils.database import (
    PSQLConnection,
    get_gaia_catalog,
    get_ps1_catalog,
    insert_metadata,
)
from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models import Detection, Gaia_ztf, Object, Ps1_ztf
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
            "oid": "ZTF",
            "ss": {},
            "reference": {},
            "dataquality": {},
            "gaia": {},
            "ps1": {},
        }
    ]

    # with conn.session() as session:
    #    insert_metadata(session, data)
    #    # perform queries to make sure they were inserted/updated
