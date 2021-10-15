from pymongo import MongoClient
from .query import MongoQuery
from ..generic import DatabaseConnection, DatabaseCreator, BaseQuery
from .models import Base

MAP_KEYS = {"HOST", "USER", "PASSWORD", "PORT", "DATABASE"}


def satisfy_keys(config_keys):
    return MAP_KEYS.difference(config_keys)


class MongoDatabaseCreator(DatabaseCreator):
    @classmethod
    def create_database(cls) -> DatabaseConnection:
        return MongoConnection()


class MongoConnection(DatabaseConnection):
    def __init__(self, config=None, client=None, base=None):
        self.config = config
        self.client = client
        self.base = base or Base

    def connect(self, config):
        """
        Establishes connection to a database and initializes a session.

        Parameters
        ----------
        config : dict
            Database configuration. For example:

            .. code-block:: python

                "MONGO": {
                    "HOST": "host",
                    "USER": "username",
                    "PASSWORD": "pwd",
                    "PORT": 5432, # postgresql tipically runs on port 5432. Notice that we use an int here.
                    "DB_NAME": "database",
                }
        base : db_plugins.db.mongo.models.DeclarativeBase
            Base class to initialize the database
        """
        self.config = config
        invalid_keys = satisfy_keys(set(config.keys()))
        if len(invalid_keys) != 0:
            raise ValueError(f"Invalid config. Missing values {invalid_keys}")
        self.client = self.client or MongoClient(
            host=config["HOST"],  # <-- IP and port go here
            serverSelectionTimeoutMS=3000,  # 3 second timeout
            username=config["USER"],
            password=config["PASSWORD"],
            port=config["PORT"],
        )
        self.base.database = config["DATABASE"]

    def create_db(self):
        self.base.create_all(self.client)

    def drop_db(self):
        self.base.drop_all(self.client)

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
        return MongoQuery(self, *args)
