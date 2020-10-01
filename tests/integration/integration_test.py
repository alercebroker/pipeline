from db_plugins.db.sql import (
    models,
    Pagination,
    SQLConnection,
    SQLQuery,
    Pagination,
    create_engine,
    Base,
)
from sqlalchemy.engine.reflection import Inspector
import unittest
import json
import time
import datetime


class SQLConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config = {
            "ENGINE": "postgresql",
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5432,
            "DB_NAME": "postgres",
        }
        self.config = {
            "SQLALCHEMY_DATABASE_URL": f"{config['ENGINE']}://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
        }
        self.db = SQLConnection()

    def tearDown(self):
        if self.db.Base and self.db.engine:
            self.db.Base.metadata.drop_all(bind=self.db.engine)

    def test_connect_not_scoped(self):
        pass

    def test_connect_scoped(self):
        pass

    def test_create_session(self):
        pass

    def test_create_scoped_session(self):
        pass

    def test_create_db(self):
        engine = create_engine(self.config["SQLALCHEMY_DATABASE_URL"])
        self.db.engine = engine
        self.db.Base = Base
        self.db.create_db()
        inspector = Inspector.from_engine(engine)
        print(inspector.get_table_names())

    def test_drop_db(self):
        pass

    def test_query(self):
        pass


class SQLQueryTest(unittest.TestCase):
    pass
