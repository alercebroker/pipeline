from typing import List, TypedDict
from operator import itemgetter
from db_plugins.db.mongo import MongoConnection

from mongo_scribe.command.commands import DbCommand
from mongo_scribe.db.factories.update_one import update_one_factory
from mongo_scribe.db.factories.update_probability import (
    UpdateProbabilitiesOperation,
)
from mongo_scribe.db.models import get_model_collection
from ..command.commons import ValidCommands


class Operations(TypedDict):
    inserts: list
    updates: list
    update_probabilities: UpdateProbabilitiesOperation


def create_operations(commands: List[DbCommand]) -> Operations:
    """
    Creates a structure that contains all the operations obtained from the command list
    """
    inserts = [
        command.get_raw_operation()
        for command in commands
        if command.type == ValidCommands.insert
    ]
    updates = [
        update_one_factory(command)
        for command in commands
        if command.type == ValidCommands.update
    ]

    update_probs = UpdateProbabilitiesOperation()
    for command in [
        command
        for command in commands
        if command.type == ValidCommands.update_probabilities
    ]:
        update_probs = update_probs.add_update(command)

    return Operations(
        {
            "inserts": inserts,
            "updates": updates,
            "update_probabilities": update_probs,
        }
    )


def execute_operations(connection: MongoConnection, collection_name: str):
    """
    Returns a function that executes the operations in the provided collection
    """
    collection = get_model_collection(connection, collection_name)

    def execute(operations: Operations):
        inserts, updates, update_probs = itemgetter(
            "inserts", "updates", "update_probabilities"
        )(operations)

        collection.insert_many(inserts, ordered=False)
        collection.bulk_write(updates)
        collection.update_probabilities(update_probs)

    return execute
