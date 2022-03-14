from db_plugins.db.generic import DatabaseConnection
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.sql import SQLConnection
from ingestion_step.utils.multi_driver import Mapper


class MultiDriverConnection(DatabaseConnection):
    def __init__(self, config: dict):
        self.mapper = Mapper()
        self.config = config
        self.psql_driver = SQLConnection()
        self.mongo_driver = MongoConnection()
        pass

    def connect(self):
        self.mongo_driver.connect(self.config)
        self.psql_driver.connect(self.config)

    def create_db(self):
        self.mongo_driver.create_db()
        self.psql_driver.create_db()

    def drop_db(self):
        self.mongo_driver.drop_db()
        self.psql_driver.drop_db()

    def query(self, query_class=None, *args, **kwargs):
        self.mongo_driver.query(query_class=query_class, *args, **kwargs)
        # mapper
        psql_object = self.mapper.convert(query_class)
        self.psql_driver.query(psql_object)
