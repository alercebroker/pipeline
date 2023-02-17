from mongo_scribe.command.commands import (
    DbCommand,
    UpdateProbabilitiesDbCommand,
)


class UpdateProbabilitiesOperation:
    classifier: dict
    updates: list

    def __init__(self, classifier=None, updates=None):
        self.classifier = classifier
        self.updates = updates if updates is not None else []

    def add_update(self, command: DbCommand):
        if command.__class__ != UpdateProbabilitiesDbCommand:
            raise Exception(
                "Invalid command to create an update probability operation"
            )

        classifier, criteria, data = command.get_raw_operation()
        return UpdateProbabilitiesOperation(
            classifier, [*self.updates, (criteria["aid"], data)]
        )