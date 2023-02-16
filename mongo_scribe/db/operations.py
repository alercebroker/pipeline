from typing import List, TypedDict
from operator import itemgetter
from db_plugins.db.mongo import MongoConnection
from mongo_scribe.command.commands import DbCommand
from mongo_scribe.db.update_one import update_one_factory
from mongo_scribe.db.models import get_model_collection


class Operations(TypedDict):
    inserts: list
    updates: list
    update_probabilities: list


def create_operations(commands: List[DbCommand]) -> Operations:
    """
    Creates a structure that contains all the operations obtained from the command list
    """
    inserts = [
        command.get_raw_operation()
        for command in commands
        if command.type == "insert"
    ]
    updates = [
        update_one_factory(command)
        for command in commands
        if command.type == "update"
    ]

    return Operations(
        {"inserts": inserts, "updates": updates, "update_probabilities": []}
    )


def execute_operations(connection: MongoConnection, collection_name: str):
    """
    Returns a function that executes the operations in the provided collection
    """
    collection = get_model_collection(connection, collection_name)

    def execute(operations: Operations):
        inserts, updates = itemgetter("inserts", "updates")(operations)

        if len(inserts) > 0:
            collection.insert_many(inserts, ordered=False)

        if len(updates) > 0:
            collection.bulk_write(updates)

    return execute
