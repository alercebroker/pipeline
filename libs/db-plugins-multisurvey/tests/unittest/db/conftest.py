from unittest import mock

import pytest
from sqlalchemy import Engine

from db_plugins.db.sql._connection import PsqlDatabase

config = {
    "USER": "nada",
    "PASSWORD": "nada",
    "HOST": "nada",
    "PORT": "nada",
    "DB_NAME": "nada",
}


@pytest.fixture(scope="session")
def mock_engine() -> Engine:
    mock_connection = mock.Mock()
    mock_connection.__enter__ = mock.Mock(return_value=mock.Mock())
    mock_connection.__exit__ = mock.Mock(return_value=None)

    mock_engine = mock.Mock()
    mock_engine.connect = mock.Mock(return_value=mock_connection)

    return mock_engine


@pytest.fixture(scope="session")
def psql_db(mock_engine: Engine):
    psql_db = PsqlDatabase(config, engine=mock_engine)

    return psql_db
