from mongo_scribe.command.commands import DbCommand, UpdateProbabilitiesDbCommand

class UpdateProbabilitiesOperation(TypedDict):
    classifier: dict
    updates: list

def update_probability_factory(command: DbCommand):
    if command.__class__ != UpdateProbabilitiesDbCommand:
        raise Exception("Invalid command to create an update probability operation")

    classifier, 