class NoDataProvidedException(Exception):
    """
    Exception to raise when the command doesn't have any data
    """

    def __init__(self):
        super().__init__("The command must contain data to insert or update")


class UpdateWithNoCriteriaException(Exception):
    """
    Exception to raise when an update command doesn't have a filter (or criteria)
    """

    def __init__(self):
        super().__init__("The Update command must contain an update criteria")


class MisformattedCommandExcepction(Exception):
    """
    Exception to raise when a dictionary doesn't have valid command fields
    """

    def __init__(self):
        super().__init__("Received a misformatted message")


class NonExistantCollectionException(Exception):
    """
    Exception to raise when trying to obtain a non-existant collection
    """

    def __init__(self):
        super().__init__("Collection doesn't exist")


class NoCollectionProvidedException(Exception):
    """
    Exception to raise when a command doesn't provide a collection to write on
    """

    def __init__(self):
        super().__init__("No collection provided in the command")

class NoClassifierInfoProvidedException(Exception):
    """
    Exception to raise when a command doesn't provide the classifier when 
    updating probabilities
    """

    def __init__(self):
        super().__init__("No classifier info provided in the command")