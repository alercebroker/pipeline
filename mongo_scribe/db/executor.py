import os
from typing import List
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from ..command.commands import Command
from ..command.exceptions import NonExistentCollectionException


class ScribeCommandExecutor:
    """
    Class which contains all availible Scribe DB Operations
    """

    allowed = (
        Object.__tablename__,
        Detection.__tablename__,
        NonDetection.__tablename__,
    )

    def __init__(self, config):
        connection = MongoConnection()
        connection.connect(config["MONGO"])
        self.connection = connection

    def bulk_execute(self, collection_name: str, commands: List[Command]):
        """
        Executes a list of commands obtained from a Kafka topic
        Does nothing when the command list is empty
        """
        if collection_name not in self.allowed:
            raise NonExistentCollectionException()

        operations = []
        for command in commands:
            operations.extend(command.get_operations())

        if os.getenv("MOCK_DB_COLLECTION"):
            print(operations)
        elif operations:
            self.connection.database[collection_name].bulk_write(operations)
        else:
            return
