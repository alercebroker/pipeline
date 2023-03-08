from pymongo import UpdateOne
from mongo_scribe.command.commands import DbCommand, UpdateDbCommand


def update_one_factory(command: DbCommand) -> UpdateOne:
    """
    Creates a Pymongo operation UpdateOne based on a update command
    """
    if not isinstance(command, UpdateDbCommand):
        raise Exception(
            "Can't create an UpdateOne instance from a command that is not an update command."
        )

    criteria, data = command.get_raw_operation()
    upsert = command.options.upsert
    operation = {"$set": data}

    return UpdateOne(criteria, operation, upsert=upsert)
