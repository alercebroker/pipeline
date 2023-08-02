from pymongo import MongoClient


class DatabaseConnection:
    def __init__(self, config: dict):
        self.config = config
        self.client = MongoClient(**self.config)
        self.database = self.client[self.config.db_name]
