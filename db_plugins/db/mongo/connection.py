from pymongo import MongoClient
from db_plugins.db.mongo.query import mongo_query_creator
from db_plugins.db.generic import DatabaseConnection, DatabaseCreator
from db_plugins.db.mongo.models import Base

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

                config = {
                    "HOST": "host",
                    "USER": "username",
                    "PASSWORD": "pwd",
                    "PORT": 27017, # mongo tipically runs on port 27017.
                                   # Notice that we use an int here.
                    "DATABASE": "database",
                }
        base : db_plugins.db.mongo.models.Base
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
            authSource=config["DATABASE"]
        )
        self.base.set_database(config["DATABASE"])
        self.database = self.client[config["DATABASE"]]

    def create_db(self):
        self.base.metadata.create_all(self.client, self.config["DATABASE"])

    def drop_db(self):
        self.base.metadata.drop_all(self.client, self.config["DATABASE"])

    def query(self, query_class=None, *args, **kwargs):
        """Create a BaseQuery object that allows you to query the database using
        the PyMongo Collection API, or using the BaseQuery methods
        like ``get_or_create``.

        Parameters
        ----------
        args : tuple
            Args you can pass to pymongo.collection.Collection class.

        Examples
        --------
        .. code-block:: python

            # Using PyMongo API
            db_conn.query(
                database=my_db_instance,
                name='my_collection',
            ).find({'hello': 'world'})
            # Using db-plugins
            # These two statements are equivalent
            db_conn.query(Object).get_or_create(filter_by=**filters)
            db_conn.query().get_or_create(model=Object, filter_by=**filters)
        """
        if query_class:
            return mongo_query_creator(query_class)(
                self.database,
                *args,
                **kwargs,
            )
        return mongo_query_creator()(self.database, *args, **kwargs)
