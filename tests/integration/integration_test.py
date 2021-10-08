from db_plugins.db.sql import (
    models,
    SQLConnection,
    SQLQuery,
    create_engine,
    Base,
    sessionmaker,
)
from db_plugins.db.generic import Pagination
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
        self.session_options = {
            "autocommit": False,
            "autoflush": True,
        }
        self.db = SQLConnection()

    def tearDown(self):
        if self.db.Base and self.db.engine:
            self.db.Base.metadata.drop_all(bind=self.db.engine)

    def test_connect_not_scoped(self):
        self.db.connect(
            self.config, session_options=self.session_options, use_scoped=False
        )
        self.assertIsNotNone(self.db.engine)
        self.assertIsNotNone(self.db.session)

    def test_connect_scoped(self):
        session_options = self.session_options
        session_options["autoflush"] = False
        self.db.connect(self.config, session_options=session_options, use_scoped=True)
        self.assertIsNotNone(self.db.engine)
        self.assertIsNotNone(self.db.session)

    def test_create_session(self):
        engine = create_engine(self.config["SQLALCHEMY_DATABASE_URL"])
        Session = sessionmaker(bind=engine, **self.session_options)
        self.db.Session = Session
        self.db.create_session()
        self.assertIsNotNone(self.db.session)

    def test_create_scoped_session(self):
        engine = create_engine(self.config["SQLALCHEMY_DATABASE_URL"])
        session_options = self.session_options
        session_options["autoflush"] = False
        Session = sessionmaker(bind=engine, **session_options)
        self.db.Session = Session
        self.db.Base = Base
        self.db.create_scoped_session()
        self.assertIsNotNone(self.db.session)

    def test_create_db(self):
        engine = create_engine(self.config["SQLALCHEMY_DATABASE_URL"])
        self.db.engine = engine
        self.db.Base = Base
        self.db.create_db()
        inspector = Inspector.from_engine(engine)
        self.assertGreater(len(inspector.get_table_names()), 0)

    def test_drop_db(self):
        engine = create_engine(self.config["SQLALCHEMY_DATABASE_URL"])
        self.db.engine = engine
        self.db.Base = Base
        self.db.Base.metadata.create_all(bind=self.db.engine)
        self.db.drop_db()
        inspector = Inspector.from_engine(engine)
        self.assertEqual(len(inspector.get_table_names()), 0)

    def test_query(self):
        query = self.db.query(models.Object)
        self.assertIsInstance(query, SQLQuery)


class SQLQueryTest(unittest.TestCase):
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
        self.session_options = {
            "autocommit": False,
            "autoflush": True,
        }
        self.db = SQLConnection()
        self.db.connect(
            self.config, session_options=self.session_options, use_scoped=False
        )

    @classmethod
    def tearDownClass(self):
        self.db.drop_db()
        self.db.session.close()

    def setUp(self):
        self.db.create_db()
        obj = models.Object(
            oid="ZTF1",
        )
        self.db.session.add(obj)
        self.db.session.commit()

    def tearDown(self):
        self.db.session.close()
        self.db.drop_db()

    def test_get_or_create(self):
        instance, created = self.db.query().get_or_create(
            models.Object, {"oid": "ZTF1"}
        )
        self.assertIsInstance(instance, models.Object)
        self.assertFalse(created)

    def test_get_or_create_created(self):
        instance, created = self.db.query().get_or_create(
            models.Object,
            {
                "oid": "ZTF2",
                "ndethist": 0,
                "ncovhist": 0,
                "mjdstarthist": 0.0,
                "mjdendhist": 0.0,
                "corrected": False,
                "stellar": False,
                "ndet": 0,
                "g_r_max": 0.0,
                "g_r_max_corr": 0.0,
                "g_r_mean": 0.0,
                "g_r_mean_corr": 0.0,
                "meanra": 0.0,
                "meandec": 0.0,
                "sigmara": 0.0,
                "sigmadec": 0.0,
                "deltajd": 0.0,
                "firstmjd": 0.0,
                "lastmjd": 0.0,
                "step_id_corr": "test",
            },
        )
        self.assertIsInstance(instance, models.Object)
        self.assertTrue(created)

    def test_check_exists(self):
        self.assertTrue(self.db.query().check_exists(models.Object, {"oid": "ZTF1"}))

    def test_update(self):
        instance = (
            self.db.session.query(models.Object)
            .filter(models.Object.oid == "ZTF1")
            .one_or_none()
        )
        updated = self.db.query().update(instance, {"oid": "ZTF2"})
        self.assertEqual(updated.oid, "ZTF2")

    def test_bulk_insert(self):
        objs = [{"oid": "ZTF2"}, {"oid": "ZTF3"}]
        self.db.query().bulk_insert(objs, models.Object)
        objects = self.db.session.query(models.Object).all()
        self.assertEqual(len(objects), 3)

    def test_paginate(self):
        pagination = self.db.query(models.Object).paginate()
        self.assertIsInstance(pagination, Pagination)
        self.assertEqual(pagination.total, 1)
        self.assertEqual(pagination.items[0].oid, "ZTF1")

    def test_find_one(self):
        obj = self.db.query(models.Object).find_one(filter_by={"oid": "ZTF1"})
        self.assertIsInstance(obj, models.Object)

    def test_find(self):
        obj_page = self.db.query(models.Object).find()
        self.assertIsInstance(obj_page, Pagination)
        self.assertEqual(obj_page.total, 1)
        self.assertEqual(obj_page.items[0].oid, "ZTF1")

    def test_find_paginate_false(self):
        obj_page = self.db.query(models.Object).find(paginate=False)
        self.assertIsInstance(obj_page, list)


class ScopedSQLQueryTest(unittest.TestCase):
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
        self.session_options = {
            "autocommit": False,
            "autoflush": False,
        }
        self.db = SQLConnection()
        self.db.connect(
            self.config, session_options=self.session_options, use_scoped=True
        )

    @classmethod
    def tearDownClass(self):
        self.db.drop_db()
        self.db.session.close()

    def setUp(self):
        self.db.create_db()
        obj = models.Object(
            oid="ZTF1",
        )
        self.db.session.add(obj)
        self.db.session.commit()

    def tearDown(self):
        self.db.session.close()
        self.db.drop_db()

    def test_query_property(self):
        self.assertEqual(len(models.Object.query.all()), 1)
        self.assertEqual(models.Object.query.first().oid, "ZTF1")

    def test_method_access_from_session(self):
        instance, created = self.db.session.query().get_or_create(
            model=models.Object, filter_by={"oid": "ZTF1"}
        )
        self.assertIsInstance(instance, models.Object)
        self.assertFalse(created)

    def test_method_access_from_query_property(self):
        instance, created = models.Object.query.get_or_create(filter_by={"oid": "ZTF1"})
        self.assertIsInstance(instance, models.Object)
        self.assertFalse(created)
