from pymongo import MongoClient
from db_plugins.db.mongo.query import MongoQuery
from db_plugins.db.generic import DatabaseConnection, DatabaseCreator
from db_plugins.db.mongo.models import BaseModel


MAP_KEYS = {"HOST", "USERNAME", "PASSWORD", "PORT", "DATABASE"}
NOT_PYMONGO_KEYS = {"database"}


def satisfy_keys(config_keys):
    return MAP_KEYS.difference(config_keys)


def to_camel_case(config: dict):
    """Converts config keys to lowerCamelCase"""
    result_config = {}
    for key in config:
        lower_key = key.lower()
        camel_case_key = lower_key.split("_")
        camel_case_key = camel_case_key[0] + "".join(
            x.title() for x in camel_case_key[1:]
        )
        result_config[camel_case_key] = config[key]
    return result_config


class MongoDatabaseCreator(DatabaseCreator):
    @classmethod
    def create_database(cls) -> DatabaseConnection:
        return MongoConnection()


class MongoConnection(DatabaseConnection):
    def __init__(self, config=None, client=None, base=None):
        self.config = config
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
        base : db_plugins.db.mongo.models.BaseModel
            Base class to initialize the database
        """
        self.config = config
        invalid_keys = satisfy_keys(set(config.keys()))
        if len(invalid_keys) != 0:
            raise ValueError(f"Invalid config. Missing values {invalid_keys}")
        pymongo_arguments = to_camel_case(config)
        for key in NOT_PYMONGO_KEYS:
            del pymongo_arguments[key]
        self.client = self.client or MongoClient(**pymongo_arguments)
        self.base.set_database(config["DATABASE"])
        self.database = self.client[config["DATABASE"]]

    def create_db(self):
        self.base.metadata.create_all(self.client, self.config["DATABASE"])

    def drop_db(self):
        self.base.metadata.drop_all(self.client, self.config["DATABASE"])

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

            # Using PyMongo API
            db_conn.query(
                name='my_collection',
            ).collection.find({'hello': 'world'})
            # Using db-plugins
            # These two statements are equivalent
            db_conn.query(model=Object).get_or_create(filter_by=**filters)
            db_conn.query().get_or_create(model=Object, filter_by=**filters)
        """
        return MongoQuery(self.database, model=model, name=name)
