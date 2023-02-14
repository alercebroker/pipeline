from typing import List
from db_plugins.db.mongo import MongoConnection
from mongo_scribe.command.commands import DbCommand
from mongo_scribe.db.models import get_model_collection
from mongo_scribe.db.update_one import update_one_factory


class ScribeDbOperations:
    """
    Class which contains all availible Scribe DB Operations
    """

    def __init__(self, config):
        connection = MongoConnection()
        connection.connect(config["MONGO"])
        self.connection = connection

    def bulk_execute(self, commands: List[DbCommand]):
        """
        Executes a list of commands obtained from a Kafka topic
        Does nothing when the command list is empty
        """

        if len(commands) == 0:
            return

        inserts = [
            command.data for command in commands if command.type == "insert"
        ]
        updates = [
            update_one_factory(command)
            for command in commands
            if command.type == "update"
        ]

        collection_name = commands[0].collection

        collection = get_model_collection(self.connection, collection_name)

        if len(inserts) > 0:
            collection.insert_many(inserts, ordered=False)

        if len(updates) > 0:
            collection.bulk_write(updates)
