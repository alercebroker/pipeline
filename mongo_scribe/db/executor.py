from typing import List
from db_plugins.db.mongo import MongoConnection
from mongo_scribe.db.operations import create_operations, execute_operations
from mongo_scribe.command.commands import DbCommand

class ScribeCommandExecutor:
    """
    Class which contains all availible Scribe DB Operations
    """

    def __init__(self, config):
        connection = MongoConnection()
        connection.connect(config["MONGO"])
        self.connection = connection

    def bulk_execute(self, collection_name: str, commands: List[DbCommand]):
        """
        Executes a list of commands obtained from a Kafka topic
        Does nothing when the command list is empty
        """
        operations = create_operations(commands)
        execute_operations(self.connection, collection_name)(operations)
