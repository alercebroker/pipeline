from db_plugins.db.sql.models import Object
from db_plugins.db.sql.connection import SQLConnection
import unittest
from unittest import mock
from mock_alchemy.mocking import UnifiedAlchemyMagicMock


class SQLConnectionTest(unittest.TestCase):
    def setUp(self):
        self.config = {}
        self.session_options = {
            "autocommit": False,
            "autoflush": True,
        }
        mock_engine = mock.Mock()
        self.db = SQLConnection(engine=mock_engine, Session=UnifiedAlchemyMagicMock)

    def tearDown(self):
        del self.db

    @mock.patch("db_plugins.db.sql.connection.SQLConnection.create_session")
    def test_connect_not_scoped(self, mock_create_session):
        self.db.connect(
            config=self.config, session_options=self.session_options, use_scoped=False
        )
        self.assertIsNotNone(self.db.engine)
        self.assertIsNotNone(self.db.Base)
        self.assertIs(self.db.Session, UnifiedAlchemyMagicMock)
        mock_create_session.assert_called_once()

    @mock.patch("db_plugins.db.sql.connection.SQLConnection.create_session")
    def test_connect_scoped(self, mock_create_session):
        self.db.connect(
            config=self.config, session_options=self.session_options, use_scoped=True
        )
        self.assertIsNotNone(self.db.engine)
        self.assertIsNotNone(self.db.Base)
        self.assertIs(self.db.Session, UnifiedAlchemyMagicMock)
        mock_create_session.assert_called_once()

    def test_create_session(self):
        self.db.create_session(use_scoped=False)
        self.assertIsInstance(self.db.session, UnifiedAlchemyMagicMock)

    def test_create_scoped_session(self):
        self.db.Base = mock.Mock()
        self.db.create_session(use_scoped=True)
        self.assertIsNotNone(self.db.session)
        self.assertIsNotNone(self.db.Base.query)

    def test_end_session(self):
        self.db.create_session(use_scoped=False)
        self.assertIsNotNone(self.db.session)
        self.db.end_session()
        self.assertIsNone(self.db.session)

    def test_create_db(self):
        self.db.Base = mock.Mock()
        self.db.create_db()
        self.db.Base.metadata.create_all.assert_called_once_with(bind=self.db.engine)

    def test_drop_db(self):
        self.db.Base = mock.Mock()
        self.db.drop_db()
        self.db.Base.metadata.drop_all.assert_called_once_with(bind=self.db.engine)

    def test_query(self):
        model = Object
        self.db.session = UnifiedAlchemyMagicMock()
        self.db.query(model)
        self.db.session.query.assert_called_once_with(model)
