from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from .query import SQLQuery
from ..generic import DatabaseConnection, DatabaseCreator
from .models import Base

MAP_KEYS = {"HOST", "USER", "PASSWORD", "PORT", "DB_NAME", "ENGINE"}


def satisfy_keys(config_keys):
    return MAP_KEYS.difference(config_keys)


def settings_map(config):
    return f"{config['ENGINE']}://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


class SQLConnection(DatabaseConnection):
    def __init__(self, config=None, engine=None, Base=None, Session=None, session=None):
        self.config = config
        self.engine = engine
        self.Base = Base
        self.Session = Session
        self.session = session
        self.use_scoped = False

    def connect(
        self,
        config,
        base=None,
        session_options=None,
        create_session=True,
        use_scoped=False,
        scope_func=None,
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
        create_session : Boolean
            Whether to instantiate a session or not. The default value is True since this is the previous behavior.
            Proper usage should be to pass False and create / close the session on demand inside the application.
            You should call this method first and then call SQLConnection.create_session(use_scoped)

            .. code-block:: python

                # scoped session example
                conn = SQLConnection()
                conn.connect(config, create_session=False)
                conn.create_session(use_scoped=True)
                # use session
                conn.query()
                conn.session.remove()
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
        if self.Session is None:
            self.Session = sessionmaker(bind=self.engine, **session_options)
        if create_session:
            self.create_session(use_scoped, scope_func)

    def create_session(self, use_scoped, scope_func=None):
        """
        Creates a SQLAlchemy Session object to interact with the database.

        Parameters
        ----------
        use_scoped : Boolean
            Whether to use scoped session or not. Use a scoped session if you are using it in a web service like an API.
            Read more about scoped sessions at: `<https://docs.sqlalchemy.org/en/13/orm/contextual.html?highlight=scoped>`_
        scope_func : function
            A function which serves as the scope for the session. The session will live only in the scope of that function.
        """
        if not use_scoped:
            self._create_unscoped_session()
        else:
            self._create_scoped_session(scope_func)

    def _create_scoped_session(self, scope_func):
        self.session = scoped_session(self.Session, scopefunc=scope_func)
        self.Base.query = self.session.query_property(query_cls=SQLQuery)
        self.use_scoped = True

    def _create_unscoped_session(self):
        self.session = self.Session()

    def end_session(self):
        if self.use_scoped:
            self.session.remove()
        else:
            self.session.close()
        self.session = None

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


class SQLDatabaseCreator(DatabaseCreator):
    @classmethod
    def create_database(cls) -> SQLConnection:
        return SQLConnection()
