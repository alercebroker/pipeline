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
