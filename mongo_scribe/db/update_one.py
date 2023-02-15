from pymongo import UpdateOne
from mongo_scribe.command.commands import DbCommand


def update_one_factory(command: DbCommand) -> UpdateOne:
    """
    Creates a Pymongo operation UpdateOne based on a update command
    """
    if command.type != "update":
        raise Exception(
            "Can't create an UpdateOne instance from an insert command."
        )

    criteria = command.criteria
    operation = {"$set": command.data}

    return UpdateOne(criteria, operation, upsert=True)
