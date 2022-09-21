from collections import UserDict

from pymongo import MongoClient
from db_plugins.db.mongo.query import MongoQuery
from db_plugins.db.generic import DatabaseConnection, DatabaseCreator
from db_plugins.db.mongo.models import BaseModel


class MongoConfig(UserDict):
    MAP_KEYS = {"host", "username", "password", "port", "database"}

    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)
        if self.MAP_KEYS.difference(self.keys()):
            missing = ", ".join(value.upper() for value in self.MAP_KEYS.difference(self.keys()))
            raise ValueError(f"Invalid configuration. Missing keys: {missing}")
        self._db_name = self.pop("database")

    @property
    def db_name(self):
        return self._db_name

    def __setitem__(self, key, value):
        """Converts keys from (case-insensitive) `snake_case` to `lowerCamelCase`"""
        klist = [w.lower() if i == 0 else w.title() for i, w in enumerate(key.split("_"))]
        super().__setitem__("".join(klist), value)


class MongoDatabaseCreator(DatabaseCreator):
    @classmethod
    def create_database(cls) -> DatabaseConnection:
        return MongoConnection()


class MongoConnection(DatabaseConnection):
    def __init__(self, config=None, client=None, base=None):
        self.config = MongoConfig(config) if config is not None else config
        self.client = client
        self.base = base or BaseModel
        self.database = None

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
                    "USERNAME": "username",
                    "PASSWORD": "pwd",
                    "PORT": 27017, # mongo tipically runs on port 27017.
                                   # Notice that we use an int here.
                    "DATABASE": "database",
                    "AUTH_SOURCE": "admin" # could be admin or the same as DATABASE
                }
        """
        self.config = MongoConfig(config)
        self.client = self.client or MongoClient(**self.config)
        self.base.set_database(self.config.db_name)
        self.database = self.client[self.config.db_name]

    def create_db(self):
        self.base.metadata.create_all(self.client, self.config.db_name)

    def drop_db(self):
        self.base.metadata.drop_all(self.client, self.config.db_name)

    def query(self, model=None, name=None):
        """Create a BaseQuery object that allows you to query the database using
        the PyMongo Collection API, or using the BaseQuery methods
        like ``get_or_create``.

        Parameters
        ----------
        model : Type[BaseModel]
            Model class to create the query for
        name : str
            Name of collection to use

        Examples
        --------
        .. code-block:: python

            # Using PyMongo API through collection attribute
            db_conn.query(
                name='my_collection',
            ).collection.find({'hello': 'world'})
            # Using db-plugins
            # These two statements are equivalent
            db_conn.query(model=Object).get_or_create(filter_by=filters)
            db_conn.query().get_or_create(model=Object, filter_by=filters)
        """
        return MongoQuery(self.database, model=model, name=name)
