from pymongo import MongoClient


class DatabaseConnection:
    def __init__(self, config: dict):
        self.config = config
        _database = self.config.pop("database")
        self.client = MongoClient(**self.config)
        self.database = self.client[_database]
