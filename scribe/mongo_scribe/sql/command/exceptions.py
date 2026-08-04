class NoDataProvidedException(ValueError):
    """
    Exception to raise when the command doesn't have any data
    """

    def __init__(self):
        super().__init__("The command must contain data to insert or update")


class UpdateWithNoCriteriaException(ValueError):
    """
    Exception to raise when an update command doesn't have a filter (or criteria)
    """

    def __init__(self):
        super().__init__("The Update command must contain an update criteria")


class WrongFormatCommandException(ValueError):
    """
    Exception to raise when a dictionary doesn't have valid command fields
    """

    def __init__(self):
        super().__init__("Received a badly formatted message")


class NoTableProvidedException(ValueError):
    """
    Exception to raise when a command doesn't provide a collection to write on
    """

    def __init__(self):
        super().__init__("No table provided in the command")


class MongoDialectCommandException(ValueError):
    """
    Raised for a legacy Mongo-dialect command that the SQL scribe intentionally
    ignores -- currently the generic object update
    ``{"type": "update", "collection": "object"}`` (with no ``xmatch`` in data)
    that magstats_step still emits for the retired MongoDB backend.

    It subclasses ValueError so any existing ``except ValueError`` handler still
    treats it as a dropped command; the step catches it *first* to skip it
    quietly instead of logging it as an invalid drop. See ``../../step.py`` and
    the repo-root ``MONGODB-LEGACY.md``.
    """

    def __init__(self, type_, table):
        super().__init__(
            f"Ignoring legacy Mongo-dialect command: {type_} in table {table}."
        )
