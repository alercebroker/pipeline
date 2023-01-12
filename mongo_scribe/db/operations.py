from typing import List
from ..command.commands import DbCommand
from .update_one import update_one_factory
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Object
from pymongo.collection import Collection


class ScribeDbOperations:
    def __init__(self, config):
        connection = MongoConnection()
        connection.connect(config["MONGO"])
        self.collection: Collection = connection.query(Object).collection

    def bulk_execute(self, commands: List[DbCommand]):
        inserts = [
            command.data for command in commands if command.type == "insert"
        ]
        updates = [
            update_one_factory(command)
            for command in commands
            if command.type == "update"
        ]

        if len(inserts) > 0:
            self.collection.insert_many(inserts)

        if len(updates) > 0:
            self.collection.bulk_write(updates)
