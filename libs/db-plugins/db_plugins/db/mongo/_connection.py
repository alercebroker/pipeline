from collections import UserDict
from pymongo import MongoClient
from .orm import ModelMetaClass


class _MongoConfig(UserDict):
    """Special dictionary used to parse configuration dictionaries for mongodb.

    The required keys are described in `REQUIRED_KEYS`, but can come with any
    capitalization. It is possible to add extra keys for other parameters used
    by `pymongo.MongoClient`.

    All keys are converted from `snake_case` to `lowerCamelCase` format, as
    used by `pymongo`. The special key `database` is removed from the dictionary
    proper, but can be accessed through the property `db_name`.
    """

    REQUIRED_KEYS = {"host", "username", "password", "port", "database"}

    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)
        if self.REQUIRED_KEYS.difference(self.keys()):
            missing = ", ".join(
                value.upper() for value in self.REQUIRED_KEYS.difference(self.keys())
            )
            raise ValueError(f"Invalid configuration. Missing keys: {missing}")
        self._db_name = self.pop("database")

    @property
    def db_name(self):
        return self._db_name

    def __setitem__(self, key, value):
        """Converts keys from (case-insensitive) `snake_case` to `lowerCamelCase`"""
        klist = [
            w.lower() if i == 0 else w.title() for i, w in enumerate(key.split("_"))
        ]
        super().__setitem__("".join(klist), value)


class MongoConnection:
    def __init__(self, config):
        """
        Establishes connection to a database.

        Parameters
        ----------
        config : dict
            Database configuration. For example:

            .. code-block:: python

                config = {
                    "HOST": "host",
                    "USERNAME": "username",
                    "PASSWORD": "pwd",
                    "PORT": 27017, # mongo typically runs on port 27017.
                                   # Notice that we use an int here.
                    "DATABASE": "database",
                    "AUTH_SOURCE": "admin" # could be admin or the same as DATABASE
                }
        """
        self.config = config
        self.client = MongoClient(**self.config)
        self.database = self.client[self.config.db_name]
        ModelMetaClass.set_database(self.config.db_name)

    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self, config):
        self.__config = config if config is None else _MongoConfig(config)

    def create_db(self):
        ModelMetaClass.metadata.create_all(self.client, self.config.db_name)

    def drop_db(self):
        ModelMetaClass.metadata.drop_all(self.client, self.config.db_name)
