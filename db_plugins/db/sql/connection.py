from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

from .query import SQLQuery
from ..generic import DatabaseConnection, DatabaseCreator
from .models import Base

MAP_KEYS = {"HOST", "USER", "PASSWORD", "PORT", "DB_NAME", "ENGINE"}


def satisfy_keys(config_keys):
    return MAP_KEYS.difference(config_keys)


def settings_map(config):
    return f"{config['ENGINE']}://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


class SQLDatabaseCreator(DatabaseCreator):
    def create_database(self) -> DatabaseConnection:
        return SQLConnection()


class SQLConnection(DatabaseConnection):
    def __init__(self, config=None, engine=None, Base=None, Session=None, session=None):
        self.config = config
        self.engine = engine
        self.Base = Base
        self.Session = Session
        self.session = session

    def connect(
        self, config, base=None, session_options=None, use_scoped=False, scope_func=None
    ):
        """
        Establishes connection to a database and initializes a session.

        Parameters
        ----------
        config : dict
            Database configuration. For example:

            .. code-block:: python

                "SQL": {
                    "ENGINE": "postgresql",
                    "HOST": "host",
                    "USER": "username",
                    "PASSWORD": "pwd",
                    "PORT": 5432, # postgresql tipically runs on port 5432. Notice that we use an int here.
                    "DB_NAME": "database",
                }
        base : sqlalchemy.ext.declarative.declarative_base()
            Base class used by sqlalchemy to create tables
        session_options : dict
            Options passed to sessionmaker
        use_scoped : Boolean
            Whether to use scoped session or not. Use a scoped session if you are using it in a web service like an API.
            Read more about scoped sessions at: `<https://docs.sqlalchemy.org/en/13/orm/contextual.html?highlight=scoped>`_
        scope_func : function
            A function which serves as the scope for the session. The session will live only in the scope of that function.
        """

        self.config = config
        if len(satisfy_keys(set(config.keys()))) == 0:
            self.config["SQLALCHEMY_DATABASE_URL"] = settings_map(self.config)
        self.engine = self.engine or create_engine(
            self.config["SQLALCHEMY_DATABASE_URL"]
        )
        self.Base = base or Base
        session_options = session_options or {}
        session_options["query_cls"] = SQLQuery
        if self.Session is not None:
            self.Session = self.Session
        else:
            self.Session = sessionmaker(bind=self.engine, **session_options)
        if not use_scoped:
            self.create_session()
        else:
            self.create_scoped_session(scope_func)

    def create_session(self):
        self.session = self.Session()

    def create_scoped_session(self, scope_func=None):
        self.session = scoped_session(self.Session, scopefunc=scope_func)
        self.Base.query = self.session.query_property(query_cls=SQLQuery)

    def create_db(self):
        self.Base.metadata.create_all(bind=self.engine)

    def drop_db(self):
        self.Base.metadata.drop_all(bind=self.engine)

    def query(self, *args):
        """
        Creates a BaseQuery object that allows you to query the database using the SQLAlchemy API,
        or using the BaseQuery methods like ``get_or_create``

        Parameters
        ----------
        args : tuple
            Args you can pass to SQLALchemy Query class, for example a model.

        Examples
        --------
        .. code-block:: python

            # Using SQLAlchemy API
            db_conn.query(Probability).all()
            # Using db-plugins
            db_conn.query(Probability).find(filter_by=**filters)
            db_conn.query().get_or_create(model=Object, filter_by=**filters)
        """
        return self.session.query(*args)
