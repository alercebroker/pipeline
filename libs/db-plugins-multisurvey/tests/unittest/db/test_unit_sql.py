from unittest import mock

from sqlalchemy.orm import Session

from db_plugins.db.sql._connection_pipeline import PsqlDatabase


def test_session(psql_db: PsqlDatabase):
    with psql_db.session() as session:
        assert isinstance(session, Session)


@mock.patch("db_plugins.db.sql._connection.Base")
def test_create_db(base: Base, psql_db: PsqlDatabase):
    psql_db.create_db()
    base.metadata.create_all.assert_called()


@mock.patch("db_plugins.db.sql._connection.Base")
def test_drop_db(base: Base, psql_db: PsqlDatabase):
    psql_db.drop_db()
    base.metadata.drop_all.assert_called()
