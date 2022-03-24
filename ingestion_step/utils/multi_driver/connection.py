from db_plugins.db.generic import DatabaseConnection
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.sql import SQLConnection
from ingestion_step.utils.multi_driver.query import MultiQuery
from ingestion_step.utils.multi_driver.mapper import Mapper


class MultiDriverConnection(DatabaseConnection):
    def __init__(self, config: dict):
        # self.mapper = Mapper()
        self.config = config
        self.psql_driver = SQLConnection()
        self.mongo_driver = MongoConnection()

    def connect(self):
        self.mongo_driver.connect(self.config["MONGO"])
        self.psql_driver.connect(self.config["PSQL"])

    def create_db(self):
        self.mongo_driver.create_db()
        self.psql_driver.create_db()

    def drop_db(self):
        self.mongo_driver.drop_db()
        self.psql_driver.drop_db()
        self.psql_driver.session.close()

    def query(self, query_class=None, *args, **kwargs):
        return MultiQuery(self.psql_driver, self.mongo_driver, query_class, *args, **kwargs)
