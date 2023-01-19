class NoDataProvidedException(Exception):
    def __init__(self):
        super().__init__("The command must contain data to insert or update")


class UpdateWithNoCriteriaException(Exception):
    def __init__(self):
        super().__init__("The Update command must contain an update criteria")


class MisformattedCommandExcepction(Exception):
    def __init__(self):
        super().__init__("Received a misformatted message")

class NonExistantCollectionException(Exception):
    def __init__(self):
        super().__init__("Collection doesn't exist")

class NoCollectionProvidedException(Exception):
    def __init__(self):
        super().__init__("No collection provided in the command")