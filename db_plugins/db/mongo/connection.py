from pymongo import MongoClient
from .query import MongoQuery
from ..generic import DatabaseConnection, DatabaseCreator, BaseQuery

MAP_KEYS = {"HOST", "USER", "PASSWORD", "PORT", "DATABASE"}

class MongoDatabaseCreator(DatabaseCreator):
    def create_database(self) -> DatabaseConnection:
        return MongoConnection()

class MongoConnection(DatabaseConnection):
    def __init__(self, config=None, client=None):
        self.config = config
        self.client = client

    def connect(self, config):
        self.config = config
        self.client = self.client or MongoClient(
            host=config['HOST'],  # <-- IP and port go here
            serverSelectionTimeoutMS=3000,  # 3 second timeout
            username=config['USER'],
            password=config['PASSWORD'],
            port=config['PORT'],
        )

    def create_db(self):
        self.db = self.client[self.config['DATABASE']]

    def drop_db(self):
        self.client.drop_database(self.config['DATABASE'])

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
