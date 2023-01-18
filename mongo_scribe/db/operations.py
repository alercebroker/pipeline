from typing import List
from ..command.commands import DbCommand
from .models import get_model_collection
from .update_one import update_one_factory
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from pymongo.collection import Collection


class ScribeDbOperations:
    def __init__(self, config):
        connection = MongoConnection()
        connection.connect(config["MONGO"])
        self.connection = connection

    def _bulk_execute_in_collection(self, collection_name: str, commands: List[DbCommand]):
        inserts = [
            command.data for command in commands if command.type == "insert"
        ]
        updates = [
            update_one_factory(command)
            for command in commands
            if command.type == "update"
        ]

        collection = get_model_collection(self.connection, collection_name)

        if len(inserts) > 0:
            collection.insert_many(inserts)

        if len(updates) > 0:
            collection.bulk_write(updates)

    def bulk_execute(self, commands: List[DbCommand]):
        commands_by_collection = {}
        for command in commands:
            coll = command.collection
            try:
                commands_by_collection[coll].append(command)
            except KeyError:
                commands_by_collection[coll] = [command]

        if not commands_by_collection:
            return

        for coll, comm in commands_by_collection.items():
            self._bulk_execute_in_collection(coll, comm)
